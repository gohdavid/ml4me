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
from sklearn.model_selection import cross_val_score
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


features = ["TPSA", "LASA", "NumRotatableBonds", "NumHDonors", "NumHAcceptors", "MolLogP", "MolMR", "AromProp"]

data_dir = Path.cwd() / "data"
train_file = data_dir / "solvation_train.csv"
test_file = data_dir / "solvation_test.csv"
prop_file = data_dir / "molecule_props.csv"
smiles_columns = ['Solute', 'Solvent'] # name of the column containing SMILES strings
target_columns = ['logK'] # list of names of the columns containing targets
df_input = pd.read_csv(train_file)
smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values

split_type="random"
split = (0.9, 0.01, 0.09)

mfs = [PropFeaturizer(features)]
all_data = [[data.MoleculeDatapoint.from_smi(smis[0], y, mfs=mfs) for smis, y in zip(smiss, ys)]]
all_data += [[data.MoleculeDatapoint.from_smi(smis[i], mfs=mfs) for smis in smiss] for i in range(1, len(smiles_columns))]

component_to_split_by = 0
mols = [d.mol for d in all_data[component_to_split_by]]

train_indices, val_indices, test_indices = data.make_split_indices(mols, split_type, split)

train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_datasets = [data.MoleculeDataset(train_data[i], featurizer) for i in range(len(smiles_columns))]
train_mcdset = data.MulticomponentDataset(train_datasets)
scaler = train_mcdset.normalize_targets()
train_loader = custom_build_dataloader(train_mcdset)

val_datasets = [data.MoleculeDataset(val_data[i], featurizer) for i in range(len(smiles_columns))]
val_mcdset = data.MulticomponentDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
val_loader = custom_build_dataloader(val_mcdset, shuffle=False)

test_datasets = [data.MoleculeDataset(test_data[i], featurizer) for i in range(len(smiles_columns))]
test_mcdset = data.MulticomponentDataset(test_datasets)
test_loader = custom_build_dataloader(test_mcdset, shuffle=False)

class MulticomponentMPNN:
    def __init__(self, smiles_columns, scaler, features, hidden_dim=1900, n_layers=2, dropout=0.008, depth=6):
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
            input_dim=self.mcmp.output_dim + 2 * len(features),
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
    
# mcmpnn = MulticomponentMPNN(smiles_columns, scaler, features, hidden_dim=1904, n_layers=2, dropout=0.008, depth=6)
mcmpnn = MulticomponentMPNN(smiles_columns, scaler_target, features, hidden_dim=1923, n_layers=2,
                                dropout=0.016717180169380612, max_lr=0.006944491033377218, depth=5)

logger = CSVLogger("logs", name="ensemble3")

trainer = pl.Trainer(
    logger=logger,
    enable_checkpointing=True,
    enable_progress_bar=False,
    accelerator="gpu",
    devices=1,
    max_epochs=169, # number of epochs to train for
)

trainer.fit(mcmpnn.get_model(), train_loader, val_loader)

formatted_time = datetime.now().strftime("%Y%m%d-%H%M")

results = trainer.test(mcmpnn.get_model(), test_loader)

with open(f"./ensemble3/testr2_{formatted_time}_{jid}.txt", "w") as fhandle:
    fhandle.write(f"{results}")

kaggle_path = data_dir / "solvation_test.csv"
df_kaggle = pd.read_csv(kaggle_path)
smiss_kaggle = df_kaggle.loc[:, smiles_columns].values
kaggle_data = [[data.MoleculeDatapoint.from_smi(smis[i], mfs=mfs) for smis in smiss_kaggle] for i in range(len(smiles_columns))]
kaggle_datasets = [data.MoleculeDataset(kaggle_data[i], featurizer) for i in range(len(smiles_columns))]
kaggle_mcdset = data.MulticomponentDataset(kaggle_datasets)
kaggle_loader = custom_build_dataloader(kaggle_mcdset,shuffle=False)

test_preds = trainer.predict(mcmpnn.get_model(), kaggle_loader)
test_preds = np.concatenate(test_preds, axis=0)

save_submission(test_preds.squeeze(),f"./ensemble3/pred_{formatted_time}_{jid}.csv")