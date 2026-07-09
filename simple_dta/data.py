"""Data loading for the simplified DTA models.

Single source of truth: the dataset CSVs in ../data/ (columns:
compound_iso_smiles, target_smiles, target_sequence, affinity).

This module performs the *encoding* stage, which in the privacy-preserving
(MPC) setting happens in the clear, before secret sharing:

  - CNN model: label-encode the SMILES and protein strings (DeepDTA style).
  - GNN model: convert SMILES -> molecular graph via RDKit (DeepDTAGen style),
    while the protein still uses the same label-encoding.

Both models share the identical protein encoding so the comparison is clean.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Character vocabularies (from DeepDTA, hkmztrk/DeepDTA datahelper.py)
# ---------------------------------------------------------------------------
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
CHARPROTLEN = 25

# Per-dataset sequence lengths (DeepDTA defaults; davis proteins are longer).
DEFAULTS = {
    "davis": {"max_smi_len": 85, "max_seq_len": 1200},
    "kiba": {"max_smi_len": 100, "max_seq_len": 1000},
    "bindingdb": {"max_smi_len": 100, "max_seq_len": 1200},
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def label_encode(line, max_len, ch_ind):
    """Integer-encode a string; 0 is the pad index, vocab starts at 1."""
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        x[i] = ch_ind.get(ch, 0)
    return x


def load_csv(dataset, split):
    path = os.path.join(DATA_DIR, f"{dataset}_{split}.csv")
    df = pd.read_csv(path)
    return df["compound_iso_smiles"].tolist(), df["target_sequence"].tolist(), \
        df["affinity"].astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# CNN dataset: label-encoded SMILES + protein
# ---------------------------------------------------------------------------
class CNNDataset(Dataset):
    def __init__(self, dataset, split, max_smi_len=None, max_seq_len=None):
        d = DEFAULTS[dataset]
        self.max_smi_len = max_smi_len or d["max_smi_len"]
        self.max_seq_len = max_seq_len or d["max_seq_len"]
        smiles, prots, ys = load_csv(dataset, split)
        self.smiles = smiles
        self.prots = prots
        self.y = ys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xd = label_encode(self.smiles[idx], self.max_smi_len, CHARISOSMISET)
        xt = label_encode(self.prots[idx], self.max_seq_len, CHARPROTSET)
        return (torch.from_numpy(xd), torch.from_numpy(xt),
                torch.tensor(self.y[idx], dtype=torch.float32))


# ---------------------------------------------------------------------------
# GNN dataset: RDKit molecular graphs + label-encoded protein
# ---------------------------------------------------------------------------
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]


def atom_features(atom):
    from rdkit import Chem
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
             'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
             'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        one_of_k_encoding_unk(atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
             Chem.rdchem.HybridizationType.SP3D2]) +
        [atom.GetIsAromatic()] + [atom.IsInRing()])


NODE_FEATURE_DIM = 94  # length of atom_features() output (the "full" featurizer)


# ---------------------------------------------------------------------------
# Bond (edge) features -- previously unused: edge_index carried connectivity
# only, no edge_attr, so every GNN here (GCNConv) aggregated neighbors with
# purely structural weights and no notion of *what kind* of bond connects them.
# ---------------------------------------------------------------------------
def bond_features(bond):
    from rdkit import Chem
    bt = bond.GetBondType()
    return np.array(
        one_of_k_encoding_unk(bt,
            [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]) +
        [bond.GetIsConjugated(), bond.IsInRing()] +
        one_of_k_encoding_unk(bond.GetStereo(),
            [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
             Chem.rdchem.BondStereo.STEREOE]),
        dtype=np.float32)


EDGE_FEATURE_DIM = 11  # 4 bond-type one-hot + unk + conjugated + ring + 3 stereo one-hot + unk


# --- compact atom featurizers (for the small-GCN / MPC-friendly variant) ---
# Edges stay non-featured (GCNConv uses connectivity only); we only shrink the
# per-node descriptor. "small" = element one-hot + aromatic/ring flags (12 dims);
# "tiny" = element one-hot only (4 dims).
def small_atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I']) +  # 9 + unk = 10
        [atom.GetIsAromatic(), atom.IsInRing()],                # + 2 = 12
        dtype=np.float32)


def tiny_atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O']),  # 3 + unk = 4
        dtype=np.float32)


# name -> (feature_fn, dim). Used by build_graph_dataset / the GNN model.
FEATURIZERS = {
    "full": (atom_features, 94),
    "small": (small_atom_features, 12),
    "tiny": (tiny_atom_features, 4),
}


def smile_to_graph(smile, feat_fn=atom_features):
    """Return (num_nodes, node_features[N,D], edge_index[2,E], edge_features[E,11])
    for a SMILES. Edge features are always computed (cheap); models that don't
    use edge_attr (GCNConv) simply ignore it -- backward compatible."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smile}")
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        f = feat_fn(atom).astype(np.float32)
        s = f.sum()
        features.append(f / s if s > 0 else f)
    features = np.array(features, dtype=np.float32)

    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.append((a, b))
        edges.append((b, a))  # undirected -> both directions
        edge_feats.append(bf)
        edge_feats.append(bf)  # same bond, same feature vector both directions
    if edges:
        edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
        edge_attr = np.array(edge_feats, dtype=np.float32)  # [E, 11]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)
    return c_size, features, edge_index, edge_attr


def build_graph_dataset(dataset, split, max_seq_len=None, featurizer="full"):
    """Return a list of torch_geometric Data objects (cached per unique SMILES).

    `featurizer` selects the per-atom descriptor width: full(94)|small(12)|tiny(4).
    Every Data object carries `edge_attr` [E, 11] (bond type/conjugation/ring/
    stereo); GCNConv-based models ignore it, GATv2Conv-based ones can use it
    via `edge_dim=EDGE_FEATURE_DIM`.
    """
    from torch_geometric.data import Data
    feat_fn, _ = FEATURIZERS[featurizer]
    max_seq_len = max_seq_len or DEFAULTS[dataset]["max_seq_len"]
    smiles, prots, ys = load_csv(dataset, split)

    graph_cache = {}
    data_list = []
    for smi, prot, y in zip(smiles, prots, ys):
        if smi not in graph_cache:
            graph_cache[smi] = smile_to_graph(smi, feat_fn)
        c_size, features, edge_index, edge_attr = graph_cache[smi]
        xt = label_encode(prot, max_seq_len, CHARPROTSET)
        data = Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.tensor([y], dtype=torch.float32),
        )
        data.target = torch.from_numpy(xt).unsqueeze(0)  # [1, max_seq_len]
        data_list.append(data)
    return data_list
