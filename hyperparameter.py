#!/usr/bin/env python

print("Start")

# %% [markdown]
# #  <center> Problem Set 6 <center>
# 
# <center> 3.C01/3.C51, 10.C01/10.C51 <center>

# %% [markdown]
# ## Part 1: Baseline Regression Methods

# %% [markdown]
# ### Part 1.1: (5 points) Prepare Dataset

# %%
import pandas as pd

# load data
df = pd.read_csv('./solvation_train.csv')
mol_prop = pd.read_csv('./molecule_props.csv')

# %% [markdown]
# Some utility functions for you to generate features.

# %%
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors,Crippen
from rdkit import RDLogger

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import matplotlib as mpl
from package.plot import get_size_inches

from pathlib import Path

import torch

from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from chemprop import data, featurizers, models, nn

from datetime import datetime
from dataclasses import dataclass

import optuna
from optuna.integration import PyTorchLightningPruningCallback

RDLogger.DisableLog('rdApp.*')         

# %% [markdown]
# Generate fingerprints (e.g. a Morgan fingerprint).

# %% [markdown]
# ## Part 2: (50 points) Machine Learning Competition and Report

# %% [markdown]
# You can start a new notebook here to put all your models.

# %%
def save_submission(prediction, filename):
    '''
    Utility function to dump a submission file.
`1
    prediction (numpy.array): 1d numpy array contains your prediction
    filename (str): file path to where you want to save the result
    '''
    sub = pd.DataFrame( {'index': list(range(len(prediction))), 'logK': prediction } )
    sub.to_csv(filename, index=False)

# %%
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "solvation_train.csv"
smiles_columns = ['Solute', 'Solvent'] # name of the column containing SMILES strings
target_columns = ['logK'] # list of names of the columns containing targets
df_input = pd.read_csv(input_path)
smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values

# %%
@dataclass
class PropFeaturizer(featurizers.Featurizer):
    size = 6
    
    def __len__(self) -> int:
        """the length of the feature vector"""
        return self.size

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        """Featurize the molecule ``mol``"""

        MolWt = Descriptors.ExactMolWt(mol)
        TPSA = Chem.rdMolDescriptors.CalcTPSA(mol) #Topological Polar Surface Area
        nRotB = Descriptors.NumRotatableBonds(mol) #Number of rotable bonds
        HBD = Descriptors.NumHDonors(mol) #Number of H bond donors
        HBA = Descriptors.NumHAcceptors(mol) #Number of H bond acceptors
        logP = Descriptors.MolLogP(mol) #LogP
        
        return [MolWt, TPSA, nRotB, HBD, HBA, logP]
    
    # def __call__(self, mol: Chem.Mol) -> np.ndarray:
    #     """Featurize the molecule ``mol``"""
        
    #     # define Mol object
    #     mol = Chem.MolFromSmiles(smiles)
        
    #     # get morgan fingerprint
    #     # obtain a 512 bit fingperint, with radius 2
    #     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)

    #     # convert to numpy array
    #     fp_array = np.zeros((1,), int)
    #     DataStructs.ConvertToNumpyArray(fp, fp_array)

    #     MolWt = Descriptors.ExactMolWt(mol)
    #     TPSA = Chem.rdMolDescriptors.CalcTPSA(mol) #Topological Polar Surface Area
    #     nRotB = Descriptors.NumRotatableBonds(mol) #Number of rotable bonds
    #     HBD = Descriptors.NumHDonors(mol) #Number of H bond donors
    #     HBA = Descriptors.NumHAcceptors(mol) #Number of H bond acceptors
    #     logP = Descriptors.MolLogP(mol) #LogP
        
    #     return np.hstack([fp_array, [MolWt, TPSA, nRotB, HBD, HBA, logP]])

# %%
from chemprop.data.collate import *
from chemprop.data.dataloader import *

class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def custom_collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]
    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        torch.cat([tbs[0].X_d,tbs[1].X_d],axis=1),
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )

def custom_build_dataloader(
    dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    **kwargs,
):

    if class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = custom_collate_multicomponent
    else:
        collate_fn = collate_batch

    if len(dataset) % batch_size == 1:
        warnings.warn(
            f"Dropping last batch of size 1 to avoid issues with batch normalization \
(dataset size = {len(dataset)}, batch_size = {batch_size})"
        )
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )

# %%
mfs = [PropFeaturizer()]
all_data = [[data.MoleculeDatapoint.from_smi(smis[0], y, mfs=mfs) for smis, y in zip(smiss, ys)]]
all_data += [[data.MoleculeDatapoint.from_smi(smis[i], mfs=mfs) for smis in smiss] for i in range(1, len(smiles_columns))]
component_to_split_by = 0 # index of the component to use for structure based splits
mols = [d.mol for d in all_data[component_to_split_by]]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

# %%
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_datasets = [data.MoleculeDataset(train_data[i], featurizer) for i in range(len(smiles_columns))]
val_datasets = [data.MoleculeDataset(val_data[i], featurizer) for i in range(len(smiles_columns))]
test_datasets = [data.MoleculeDataset(test_data[i], featurizer) for i in range(len(smiles_columns))]

# %%
train_mcdset = data.MulticomponentDataset(train_datasets)
scaler = train_mcdset.normalize_targets()
val_mcdset = data.MulticomponentDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
test_mcdset = data.MulticomponentDataset(test_datasets)

# %%
train_loader = custom_build_dataloader(train_mcdset)
val_loader = custom_build_dataloader(val_mcdset, shuffle=False)
test_loader = custom_build_dataloader(test_mcdset, shuffle=False)

# %%
def objective(trial: optuna.Trial) -> float:
    # Define hyperparameters using trial object
    hidden_dim = trial.suggest_int("hidden_dim", 100, 2400)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    depth = trial.suggest_int("depth", 2, 6)
    max_epochs = trial.suggest_int("max_epochs", 50, 200)

    # Model setup (Using your existing setup)
    mcmp = nn.MulticomponentMessagePassing(
        blocks=[nn.BondMessagePassing(depth=depth) for _ in range(len(smiles_columns))],
        n_components=len(smiles_columns),
    )
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        input_dim=mcmp.output_dim + (len(smiles_columns)) * np.sum([i.__len__() for i in mfs]),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    model = models.multi.MulticomponentMPNN(mcmp, agg, ffn, metrics=[nn.metrics.RMSEMetric(), nn.metrics.MAEMetric(), nn.metrics.R2Metric()])

    # Logger and trainer setup
    logger = CSVLogger('logs', name=f'hyper_{trial.number}')
    trainer = pl.Trainer(
        logger=logger,
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
    )

    hyperparameters = dict(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, depth=depth, max_epochs=max_epochs)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_loader, val_loader)  # Define your dataloaders properly

    return trainer.callback_metrics["val_loss"].item()  # Or any other metric that you aim to minimize

# Create a study and execute optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)  # You can adjust the number of trials


# %%
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")



