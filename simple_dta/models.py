"""Two simplified affinity-prediction models, both far lighter than DeepDTAGen.

  CNNDTA : pure 1D-CNN on label-encoded SMILES + protein  (the DeepDTA design)
  GNNDTA : shallow GCN on the molecular graph + the same protein CNN tower

Shared by both:
  * ProteinCNN  -- 3-layer 1D conv tower over the protein sequence
  * a 1024->512->1 prediction head over the concatenated drug+protein vectors

Design notes for the downstream MPC port:
  * conv / matmul are linear (cheap in MPC); the cost is in the non-linearities.
  * `pool` is configurable (max|mean): mean-pool is free in MPC, max-pool needs
    comparisons. Default 'max' matches the published baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CHARISOSMILEN, CHARPROTLEN, NODE_FEATURE_DIM


def _pool1d(x, mode):
    # x: [batch, channels, length] -> [batch, channels] (or [batch, 2*channels] for maxmean)
    if mode == "max":
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)
    if mode == "mean":
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    if mode == "maxmean":
        mx = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        mn = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return torch.cat([mx, mn], dim=1)
    raise ValueError(f"unknown pool mode {mode}")


def _build_conv_stack(embed_dim, num_filters, channels, dilations):
    """channels: list of out-channels per layer (default [f,2f,3f]); dilations: per-layer."""
    if channels is None:
        channels = [num_filters, num_filters * 2, num_filters * 3]
    if dilations is None:
        dilations = [1] * len(channels)
    assert len(dilations) == len(channels), "dilations must match channels length"
    convs, in_c = [], embed_dim
    for out_c, kd, d in [(c, None, dd) for c, dd in zip(channels, dilations)]:
        convs.append((in_c, out_c, d))
        in_c = out_c
    return channels, dilations, convs


class ProteinCNN(nn.Module):
    """Embedding -> N Conv1d layers -> global pool. Output dim = channels[-1] (x2 if maxmean)."""

    def __init__(self, embed_dim=128, num_filters=32, kernel_size=8, pool="max",
                 channels=None, dilations=None):
        super().__init__()
        self.pool = pool
        self.embed = nn.Embedding(CHARPROTLEN + 1, embed_dim, padding_idx=0)
        channels, dilations, spec = _build_conv_stack(embed_dim, num_filters, channels, dilations)
        self.convs = nn.ModuleList(
            [nn.Conv1d(i, o, kernel_size, dilation=d) for i, o, d in spec])
        self.out_dim = channels[-1] * (2 if pool == "maxmean" else 1)

    def forward(self, target):
        x = self.embed(target).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        return _pool1d(x, self.pool)


class SmilesCNN(nn.Module):
    """Embedding -> N Conv1d layers -> global pool. Output dim = channels[-1] (x2 if maxmean)."""

    def __init__(self, embed_dim=128, num_filters=32, kernel_size=4, pool="max",
                 channels=None, dilations=None):
        super().__init__()
        self.pool = pool
        self.embed = nn.Embedding(CHARISOSMILEN + 1, embed_dim, padding_idx=0)
        channels, dilations, spec = _build_conv_stack(embed_dim, num_filters, channels, dilations)
        self.convs = nn.ModuleList(
            [nn.Conv1d(i, o, kernel_size, dilation=d) for i, o, d in spec])
        self.out_dim = channels[-1] * (2 if pool == "maxmean" else 1)

    def forward(self, smiles):
        x = self.embed(smiles).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        return _pool1d(x, self.pool)


class PredictionHead(nn.Module):
    """concat(drug, protein) -> [hidden (-> hidden//2)] -> 1.

    `hidden` is the width of the first FC layer (most of the model's params live
    here). Default 1024 matches the DeepDTA-style baseline; 512 halves the width.
    `layers` is the number of hidden FC layers: 2 = hidden, hidden//2 (baseline);
    1 = a single `hidden`-wide layer (drops the second layer's ~hidden*hidden//2
    params while keeping the wide first layer).
    """

    def __init__(self, in_dim, hidden=1024, layers=2, dropout=0.1):
        super().__init__()
        hidden_dims = [hidden] if layers == 1 else [hidden, hidden // 2]
        seq = []
        prev = in_dim
        for d in hidden_dims:
            seq += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        seq.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*seq)

    def forward(self, drug, protein):
        return self.net(torch.cat([drug, protein], dim=1))


class CNNDTA(nn.Module):
    """DeepDTA: two 1D-CNN towers + prediction head. No graph, no GNN."""

    def __init__(self, num_filters=32, drug_filters=None, prot_filters=None,
                 embed_dim=128, dropout=0.1, pool="max", head_dim=1024,
                 head_layers=2, proj_dim=0, drug_channels=None, prot_channels=None,
                 drug_dilations=None, prot_dilations=None,
                 drug_kernel=4, prot_kernel=8):
        super().__init__()
        drug_filters = drug_filters or num_filters
        prot_filters = prot_filters or num_filters
        self.drug = SmilesCNN(embed_dim, drug_filters, kernel_size=drug_kernel, pool=pool,
                              channels=drug_channels, dilations=drug_dilations)
        self.protein = ProteinCNN(embed_dim, prot_filters, kernel_size=prot_kernel, pool=pool,
                                  channels=prot_channels, dilations=prot_dilations)
        self.proj_dim = proj_dim
        if proj_dim:
            # compress each tower to a compact joint binding embedding before the head
            self.drug_proj = nn.Linear(self.drug.out_dim, proj_dim)
            self.prot_proj = nn.Linear(self.protein.out_dim, proj_dim)
            head_in = proj_dim * 2
        else:
            head_in = self.drug.out_dim + self.protein.out_dim
        self.head = PredictionHead(head_in, hidden=head_dim, layers=head_layers,
                                   dropout=dropout)

    def forward(self, batch):
        smiles, target, _ = batch
        d = self.drug(smiles)
        p = self.protein(target)
        if self.proj_dim:
            d = F.relu(self.drug_proj(d))
            p = F.relu(self.prot_proj(p))
        return self.head(d, p).squeeze(-1)


class GNNDTA(nn.Module):
    """Shallow GCN drug encoder + the same protein CNN tower + prediction head.

    `gcn_layers` controls depth (default 2 -- intentionally shallow for the MPC
    port). Topology (edge_index) is treated as data fed in at inference; in the
    MPC setting it may be public (cheap message passing) or secret.
    """

    def __init__(self, num_filters=32, embed_dim=128, gcn_dim=128, gcn_layers=2,
                 node_feat_dim=NODE_FEATURE_DIM, dropout=0.1, pool="max",
                 head_dim=1024, head_layers=2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.pool = pool
        dims = [node_feat_dim] + [gcn_dim] * gcn_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(gcn_layers)]
        )
        self.drug_out = nn.Linear(gcn_dim, gcn_dim)
        self.protein = ProteinCNN(embed_dim, num_filters, kernel_size=8, pool=pool)
        self.head = PredictionHead(gcn_dim + self.protein.out_dim,
                                   hidden=head_dim, layers=head_layers, dropout=dropout)

    def forward(self, data):
        from torch_geometric.nn import global_max_pool, global_mean_pool
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        pooled = (global_max_pool if self.pool == "max" else global_mean_pool)(x, batch)
        drug = F.relu(self.drug_out(pooled))
        protein = self.protein(data.target)
        return self.head(drug, protein).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
