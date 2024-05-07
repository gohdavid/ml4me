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
from sklearn.model_selection import cross_val_score, KFold
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
from chemprop.data.dataloader import MoleculeDataset, ReactionDataset, MulticomponentDataset, DataLoader, build_dataloader

# Utility
from datetime import datetime
from dataclasses import dataclass
import warnings
import sys

jid = sys.argv[1]

print(jid)
sys.stdout.flush()

def save_submission(prediction, filename):
    '''
    Utility function to dump a submission file.

    prediction (numpy.array): 1d numpy array contains your prediction
    filename (str): file path to where you want to save the result
    '''
    sub = pd.DataFrame( {'index': list(range(len(prediction))), 'logK': prediction } )
    sub.to_csv(filename, index=False)

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
        return len(list(mol.GetAromaticAtoms())) / mol.GetNumHeavyAtoms()

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return np.array([getattr(self,feature)(mol) for feature in self.features])

class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None

class MulticomponentMPNN:
    def __init__(self, smiles_columns, scaler, hidden_dim=1900, n_layers=2, dropout=0.008, depth=6):
        # Initialize the Multicomponent Message Passing Neural Network component
        self.mcmp = nn.MulticomponentMessagePassing(
            blocks=[nn.BondMessagePassing(depth=depth) for _ in range(len(smiles_columns))],
            n_components=len(smiles_columns),
        )

        # Initialize the aggregation method
        self.agg = nn.MeanAggregation()

        # Output transform setup
        self.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        # Fully connected feedforward network
        self.ffn = nn.RegressionFFN(
            input_dim=self.mcmp.output_dim,
            output_transform=self.output_transform,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Metric list setup
        self.metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric(), nn.metrics.R2Metric()]

        # Final MPNN model composition
        self.model = models.multi.MulticomponentMPNN(
            self.mcmp,
            self.agg,
            self.ffn,
            metrics=self.metric_list,
        )

    def get_model(self):
        return self.model

data_dir = Path.cwd() / "data"
train_file = data_dir / "solvation_train.csv"
test_file = data_dir / "solvation_test.csv"
prop_file = data_dir / "molecule_props.csv"
smiles_columns = ['Solute', 'Solvent'] # name of the column containing SMILES strings
target_columns = ['logK'] # list of names of the columns containing targets
df_input = pd.read_csv(train_file)
smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values
all_data = [[data.MoleculeDatapoint.from_smi(smis[0], y) for smis, y in zip(smiss, ys)]]
all_data += [[data.MoleculeDatapoint.from_smi(smis[i]) for smis in smiss] for i in range(1, len(smiles_columns))]

component_to_split_by = 0
mols = [d.mol for d in all_data[component_to_split_by]]

kf = KFold(n_splits=5)
for i, (train_indices, test_indices) in enumerate(kf.split(mols)):
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, [0], test_indices
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_datasets = [data.MoleculeDataset(train_data[i], featurizer) for i in range(len(smiles_columns))]
    train_mcdset = data.MulticomponentDataset(train_datasets)
    scaler = train_mcdset.normalize_targets()
    train_loader = build_dataloader(train_mcdset,num_workers=79)

    val_datasets = [data.MoleculeDataset(val_data[i], featurizer) for i in range(len(smiles_columns))]
    val_mcdset = data.MulticomponentDataset(val_datasets)
    val_mcdset.normalize_targets(scaler)
    val_loader = build_dataloader(val_mcdset, shuffle=False,num_workers=79)

    test_datasets = [data.MoleculeDataset(test_data[i], featurizer) for i in range(len(smiles_columns))]
    test_mcdset = data.MulticomponentDataset(test_datasets)
    test_loader = build_dataloader(test_mcdset, shuffle=False,num_workers=79)
        
    mcmpnn = MulticomponentMPNN(smiles_columns, scaler,
                                hidden_dim=2053, n_layers=1, dropout=0.30440412077533446,
                                depth=5)

    logger = CSVLogger("logs", name=f"mask_nodescriptor")

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=58, # number of epochs to train for
    )

    trainer.fit(mcmpnn.get_model(), train_loader, val_loader)

    formatted_time = datetime.now().strftime("%Y%m%d-%H%M")

    results = trainer.test(mcmpnn.get_model(), test_loader)

    with open(f"./mask/testr2_{formatted_time}_{jid}_nodescriptor_{i}.txt", "w") as fhandle:
        fhandle.write(f"{results}")

    kaggle_path = data_dir / "solvation_test.csv"
    df_kaggle = pd.read_csv(kaggle_path)
    smiss_kaggle = df_kaggle.loc[:, smiles_columns].values
    kaggle_data = [[data.MoleculeDatapoint.from_smi(smis[i]) for smis in smiss_kaggle] for i in range(len(smiles_columns))]
    kaggle_datasets = [data.MoleculeDataset(kaggle_data[i], featurizer) for i in range(len(smiles_columns))]
    kaggle_mcdset = data.MulticomponentDataset(kaggle_datasets)
    kaggle_loader = build_dataloader(kaggle_mcdset,shuffle=False)

    test_preds = trainer.predict(mcmpnn.get_model(), kaggle_loader)
    test_preds = np.concatenate(test_preds, axis=0)

    save_submission(test_preds.squeeze(),f"./mask/pred_{formatted_time}_{jid}_nodescriptor_{i}.csv")