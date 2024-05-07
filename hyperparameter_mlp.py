#!/usr/bin/env python

# Cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors,Crippen
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Arrays
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

# Plotting 
import matplotlib.pyplot as plt
import matplotlib as mpl

# Personal utility package
import package.plot
from package.plot import get_size_inches

from pathlib import Path

# Machine learning
import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger

# 10.1021/acs.jcim.3c01250
from chemprop import data, featurizers, models, nn
from chemprop.data.collate import NamedTuple, BatchMolGraph, Tensor, Iterable, Datum, collate_batch
from chemprop.data.dataloader import MoleculeDataset, ReactionDataset, MulticomponentDataset, DataLoader

# Utility
from datetime import datetime
from dataclasses import dataclass
import warnings
import sys

# Hyperparameter tuning
import optuna
from optuna_integration import PyTorchLightningPruningCallback

sys.stdout.flush()

from torch.utils.data import Dataset, DataLoader

@dataclass
class PropFeaturizer(featurizers.Featurizer):
    def __init__(self, features: list):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def MolWt(self, mol):
        return Descriptors.ExactMolWt(mol)

    def TPSA(self, mol):
        return Chem.rdMolDescriptors.CalcTPSA(mol)
    
    def LASA(self, mol):
        return Chem.rdMolDescriptors.CalcLabuteASA(mol)

    def NumRotatableBonds(self, mol):
        return Descriptors.NumRotatableBonds(mol)

    def NumHDonors(self, mol):
        return Descriptors.NumHDonors(mol)

    def NumHAcceptors(self, mol):
        return Descriptors.NumHAcceptors(mol)

    def MolLogP(self, mol):
        return Descriptors.MolLogP(mol)

    def MolMR(self, mol):
        return Descriptors.MolMR(mol)

    def AromProp(self, mol):
        if mol.GetNumHeavyAtoms() > 0 :
            return len(list(mol.GetAromaticAtoms())) / mol.GetNumHeavyAtoms()
        else:
            return 0

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([getattr(self,feature)(mol) for feature in self.features])

data_dir = Path.cwd() / "data"
train_file = data_dir / "solvation_train.csv"
test_file = data_dir / "solvation_test.csv"
prop_file = data_dir / "molecule_props.csv"
df = pd.read_csv(train_file) # load data
mol_prop = pd.read_csv(prop_file)

fpSize = 1024
features = ["TPSA", "LASA", "NumRotatableBonds", "NumHDonors", "NumHAcceptors", "MolLogP", "MolMR", "AromProp"]
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles) # load SMILES into RDKit
    fp = AllChem.GetMorganGenerator(radius=2, fpSize=fpSize)
    fp_array = np.array(fp.GetFingerprint(mol))
    featurizer = PropFeaturizer(features)
    return np.hstack([fp_array, featurizer(mol)])

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.array(X))  # store X as a pytorch Tensor
        self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
        self.len=len(self.X)                # number of samples in the data

    def __getitem__(self, index):
        return self.X[index], self.y[index] # get the appropriate item

    def __len__(self):
        return self.len

class FFN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=256, n_layers=1, dropout=0, lr=0.001):
        super().__init__()
        self.lr = lr
        model = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        model += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)] * n_layers
        model.append(torch.nn.Linear(hidden_dim, 1))  # Final layer
        self.model = torch.nn.Sequential(*model)  # Assign the correct variable to torch.nn.Sequential

    def forward(self, x):
        x = x.squeeze()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.nn.functional.mse_loss(output.squeeze(), y) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.nn.functional.mse_loss(output.squeeze(), y) 
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.nn.functional.mse_loss(output.squeeze(), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_r2", r2(output.squeeze(),y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

y = df["logK"]
X_solvent = np.stack(df["Solvent"].apply(featurize).values)
X_solute = np.stack(df["Solute"].apply(featurize).values)
X = np.hstack([X_solvent,X_solute])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,train_size=0.9)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.1/0.9,train_size=0.8/0.9)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

train_data = Dataset(X_train,y_train)
val_data = Dataset(X_val,y_val)
test_data = Dataset(X_test,y_test)

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


def objective(trial):
    params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 200, 2400),
        "n_layers": trial.suggest_int("n_layers", 1, 5),
        'dropout': trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-3, 0.1, log=True),
    }
    max_epochs =  trial.suggest_int("max_epochs", 50, 200)

    ffn = FFN(input_dim=fpSize*2+len(features)*2, **params)

    logger = CSVLogger("logs", name="model")

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs, # number of epochs to train for
    )

    hyperparameters = dict(max_epochs=max_epochs, **params)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(ffn, train_loader, val_loader)  # Define your dataloaders properly

    return trainer.callback_metrics["val_loss"].item()  # Or any other metric that you aim to minimize

# Create a study and execute optimization
pruner = optuna.pruners.PatientPruner(optuna.pruners.HyperbandPruner(), patience=3)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)  # You can adjust the number of trials

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")