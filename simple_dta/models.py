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

from data import CHARISOSMILEN, CHARPROTLEN, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from attention import CrossAttentionFusion, ProteinTransformer, DrugGraphTransformer, masked_pool


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

    def forward(self, target, return_seq=False):
        x = self.embed(target).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        if return_seq:
            return x.transpose(1, 2)  # [B, L', C] for cross-attention fusion
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

    def forward(self, smiles, return_seq=False):
        x = self.embed(smiles).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        if return_seq:
            return x.transpose(1, 2)  # [B, L', C] for cross-attention fusion
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

    def forward(self, *parts):
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.net(x)


class CNNDTA(nn.Module):
    """DeepDTA: two 1D-CNN towers + prediction head. No graph, no GNN.

    `ablate` drops a tower entirely (retrains a single-tower model from
    scratch, rather than just zeroing an input on the joint model) to
    measure that tower's marginal contribution to affinity prediction:
      "none"    -- both towers (baseline)
      "drug"    -- drop the drug/SMILES tower; head sees protein only
      "protein" -- drop the protein tower; head sees drug only
    """

    def __init__(self, num_filters=32, drug_filters=None, prot_filters=None,
                 embed_dim=128, dropout=0.1, pool="max", head_dim=1024,
                 head_layers=2, proj_dim=0, drug_channels=None, prot_channels=None,
                 drug_dilations=None, prot_dilations=None,
                 drug_kernel=4, prot_kernel=8, ablate="none"):
        super().__init__()
        assert ablate in ("none", "drug", "protein"), f"unknown ablate mode {ablate}"
        self.ablate = ablate
        drug_filters = drug_filters or num_filters
        prot_filters = prot_filters or num_filters
        self.drug = None if ablate == "drug" else SmilesCNN(
            embed_dim, drug_filters, kernel_size=drug_kernel, pool=pool,
            channels=drug_channels, dilations=drug_dilations)
        self.protein = None if ablate == "protein" else ProteinCNN(
            embed_dim, prot_filters, kernel_size=prot_kernel, pool=pool,
            channels=prot_channels, dilations=prot_dilations)
        self.proj_dim = proj_dim
        if proj_dim:
            # compress each surviving tower to a compact binding embedding before the head
            if self.drug is not None:
                self.drug_proj = nn.Linear(self.drug.out_dim, proj_dim)
            if self.protein is not None:
                self.prot_proj = nn.Linear(self.protein.out_dim, proj_dim)
            head_in = proj_dim * (2 if ablate == "none" else 1)
        else:
            head_in = ((self.drug.out_dim if self.drug is not None else 0) +
                      (self.protein.out_dim if self.protein is not None else 0))
        self.head = PredictionHead(head_in, hidden=head_dim, layers=head_layers,
                                   dropout=dropout)

    def forward(self, batch):
        smiles, target, _ = batch
        parts = []
        if self.drug is not None:
            d = self.drug(smiles)
            if self.proj_dim:
                d = F.relu(self.drug_proj(d))
            parts.append(d)
        if self.protein is not None:
            p = self.protein(target)
            if self.proj_dim:
                p = F.relu(self.prot_proj(p))
            parts.append(p)
        return self.head(*parts).squeeze(-1)


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


class _GCNStack(nn.Module):
    """Plain GCNConv stack -- same family as GNNDTA's drug encoder, reused here
    as AttnDTA's drug_encoder="gnn" control. Holding the drug encoder at this
    fixed baseline isolates what GAT / edge-features / attention-fusion /
    protein-attention each individually contribute, one flag at a time.
    """

    def __init__(self, node_feat_dim, gcn_dim, gcn_layers):
        super().__init__()
        from torch_geometric.nn import GCNConv
        dims = [node_feat_dim] + [gcn_dim] * gcn_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(gcn_layers)])
        self.out_dim = gcn_dim

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x


class GraphAttnEncoder(nn.Module):
    """Attention-based drug graph encoder: GATv2Conv stack, optionally
    conditioned on bond features (`edge_dim=EDGE_FEATURE_DIM`) instead of
    connectivity alone. Content-based, dynamic edge weighting -- contrast
    with `_GCNStack`/GNNDTA's GCNConv, which aggregates each atom's
    neighbors with fixed, degree-normalized coefficients regardless of what
    the neighboring atoms or bonds actually are.
    """

    def __init__(self, node_feat_dim, gat_dim=128, gat_layers=2, heads=4,
                 edge_dim=None, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        assert gat_dim % heads == 0, "gat_dim must be divisible by heads"
        self.edge_dim = edge_dim
        dims = [node_feat_dim] + [gat_dim] * gat_layers
        self.convs = nn.ModuleList([
            GATv2Conv(dims[i], dims[i + 1] // heads, heads=heads,
                      edge_dim=edge_dim, dropout=dropout, add_self_loops=True)
            for i in range(gat_layers)
        ])
        self.out_dim = gat_dim

    def forward(self, x, edge_index, edge_attr):
        ea = edge_attr if self.edge_dim is not None else None
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr=ea))
        return x  # per-node embeddings [N_total, gat_dim]


class AttnDTA(nn.Module):
    """One configurable model spanning every transformer/attention DTA
    architecture explored for this project. Each "proposal" discussed is a
    flag combination on this class, not a separate model -- they share
    tested code (towers, fusion, pooling, head) and are directly comparable
    the same way the existing CNN/GNN tower sweeps are. See EXPERIMENTS.md
    section 10 for the full design writeup.

      A: drug=cnn, protein=cnn,          fusion=cross               (AttentionDTA-style bilateral fusion)
      B: drug=cnn, protein=transformer,  fusion=concat              (protein self-attention tower swap)
      C: drug=gat, protein=cnn,          fusion=concat, edge_feats  (GAT + bond-feature drug tower)
      D: attn_kind="linear" on any of the above                     (softmax-free, MPC-friendly ablation)
      E: drug=gat, protein={cnn,transformer}, fusion=cross, edge_feats (graph-attention + cross-fusion)
      F: drug=gnn, fusion=cross                                    (plain GCN, no attention in the
                                                                     message passing, still cross-attended
                                                                     against the protein -- isolates
                                                                     whether GAT's *content-based* edge
                                                                     weighting matters, or whether having
                                                                     any per-atom sequence to cross-attend
                                                                     against is what helps)
      G: drug=graphformer, fusion={cross,concat}, edge_feats       (Graphormer-style: NO message passing
                                                                     at all -- raw per-atom features
                                                                     self-attend directly, bond features
                                                                     enter only as an additive attention-
                                                                     score bias. See DrugGraphTransformer.)

    IMPORTANT: protein_encoder="transformer" needs attn_kind="linear" in
    practice -- measured ~75-80 min/epoch on Davis with "softmax" (full O(n^2)
    self-attention over ~1000-1200 residues) vs. ~7 min/epoch with "linear".
    attn_kind="softmax" is fine for drug_encoder (short SMILES / molecular
    graphs) and for the cross-attention fusion step, both of which operate on
    much shorter sequences. drug_encoder="graphformer" always uses softmax
    internally regardless of attn_kind (its self-attention is over <100 atoms,
    so O(n^2) is cheap, and attn_bias -- the edge-feature bias -- isn't
    defined for the linear/kernelized reformulation, see MultiHeadAttention).
    """

    def __init__(self, drug_encoder="cnn", protein_encoder="cnn", fusion="concat",
                 attn_kind="softmax", attn_dim=128, attn_heads=4,
                 prot_attn_layers=2, prot_attn_window=64,
                 gat_dim=128, gat_layers=2, gat_heads=4, use_edge_feats=False,
                 drug_attn_dim=256, drug_attn_heads=8, drug_attn_layers=4,
                 node_feat_dim=NODE_FEATURE_DIM, cross_direction="both",
                 embed_dim=128, num_filters=32, drug_kernel=4, prot_kernel=8,
                 pool="max", head_dim=1024, head_layers=2, dropout=0.1):
        super().__init__()
        assert drug_encoder in ("cnn", "gnn", "gat", "graphformer"), \
            f"unknown drug_encoder {drug_encoder}"
        assert protein_encoder in ("cnn", "transformer"), f"unknown protein_encoder {protein_encoder}"
        assert fusion in ("concat", "cross"), f"unknown fusion {fusion}"
        assert cross_direction in ("both", "drug2prot", "prot2drug"), \
            f"unknown cross_direction {cross_direction}"
        if fusion == "concat" and drug_encoder in ("gnn", "gat", "graphformer") and pool == "maxmean":
            raise ValueError("pool='maxmean' isn't supported for graph global-pooling under "
                              "fusion='concat' (no dual max+mean graph pool implemented here) "
                              "-- use fusion='cross' (masked_pool supports maxmean) or pool in {max,mean}")
        self.drug_kind = drug_encoder
        self.protein_kind = protein_encoder
        self.fusion = fusion
        self.pool = pool

        # --- drug tower ---
        if drug_encoder == "cnn":
            self.drug = SmilesCNN(embed_dim, num_filters, kernel_size=drug_kernel, pool=pool)
            drug_out_dim = self.drug.out_dim
        elif drug_encoder == "gat":
            edge_dim = EDGE_FEATURE_DIM if use_edge_feats else None
            self.drug = GraphAttnEncoder(node_feat_dim, gat_dim=gat_dim, gat_layers=gat_layers,
                                         heads=gat_heads, edge_dim=edge_dim, dropout=dropout)
            drug_out_dim = gat_dim
        elif drug_encoder == "graphformer":
            edge_dim = EDGE_FEATURE_DIM if use_edge_feats else None
            self.drug = DrugGraphTransformer(node_feat_dim, embed_dim=drug_attn_dim,
                                             n_heads=drug_attn_heads, n_layers=drug_attn_layers,
                                             edge_dim=edge_dim, dropout=dropout)
            drug_out_dim = drug_attn_dim
        else:  # "gnn"
            self.drug = _GCNStack(node_feat_dim, gat_dim, gat_layers)
            drug_out_dim = gat_dim

        # --- protein tower ---
        if protein_encoder == "cnn":
            self.protein = ProteinCNN(embed_dim, num_filters, kernel_size=prot_kernel, pool=pool)
            prot_out_dim = self.protein.out_dim
        else:
            self.protein = ProteinTransformer(CHARPROTLEN, embed_dim=embed_dim,
                                              n_heads=attn_heads, n_layers=prot_attn_layers,
                                              window=prot_attn_window, kind=attn_kind,
                                              dropout=dropout, pool=pool)
            prot_out_dim = self.protein.out_dim

        # --- fusion + head ---
        if fusion == "cross":
            self.drug_proj = nn.Linear(drug_out_dim, attn_dim)
            self.prot_proj = nn.Linear(prot_out_dim, attn_dim)
            self.cross_attn = CrossAttentionFusion(attn_dim, n_heads=attn_heads,
                                                   kind=attn_kind, dropout=dropout,
                                                   direction=cross_direction)
            head_in = attn_dim * 2 * (2 if pool == "maxmean" else 1)
        else:
            head_in = drug_out_dim + prot_out_dim
        self.head = PredictionHead(head_in, hidden=head_dim, layers=head_layers, dropout=dropout)

    def _drug_forward(self, drug_input):
        if self.drug_kind == "cnn":
            smiles = drug_input
            if self.fusion == "cross":
                return self.drug_proj(self.drug(smiles, return_seq=True)), None
            return self.drug(smiles), None
        if self.drug_kind == "graphformer":
            x, edge_index, edge_attr, batch_vec = drug_input
            seq, pad_mask = self.drug(x, edge_index, edge_attr, batch_vec)
            if self.fusion == "cross":
                return self.drug_proj(seq), pad_mask
            return masked_pool(seq, pad_mask, self.pool), None
        x, edge_index, edge_attr, batch_vec = drug_input
        h = self.drug(x, edge_index, edge_attr) if self.drug_kind == "gat" else self.drug(x, edge_index)
        if self.fusion == "cross":
            from torch_geometric.utils import to_dense_batch
            seq, node_mask = to_dense_batch(h, batch_vec)  # node_mask: True = real node
            return self.drug_proj(seq), ~node_mask
        from torch_geometric.nn import global_max_pool, global_mean_pool
        pooled = (global_max_pool if self.pool == "max" else global_mean_pool)(h, batch_vec)
        return pooled, None

    def _protein_forward(self, target):
        if self.protein_kind == "cnn":
            if self.fusion == "cross":
                return self.prot_proj(self.protein(target, return_seq=True)), None
            return self.protein(target), None
        seq, pad_mask = self.protein(target, return_seq=True)
        if self.fusion == "cross":
            return self.prot_proj(seq), pad_mask
        return masked_pool(seq, pad_mask, self.pool), None

    def forward(self, batch):
        if self.drug_kind == "cnn":
            smiles, target, _ = batch
            drug_input = smiles
        else:
            data = batch
            drug_input = (data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.target

        drug_repr, drug_pad = self._drug_forward(drug_input)
        prot_repr, prot_pad = self._protein_forward(target)

        if self.fusion == "concat":
            return self.head(drug_repr, prot_repr).squeeze(-1)

        drug_seq, prot_seq = self.cross_attn(drug_repr, drug_pad, prot_repr, prot_pad)
        drug_vec = masked_pool(drug_seq, drug_pad, self.pool)
        prot_vec = masked_pool(prot_seq, prot_pad, self.pool)
        return self.head(drug_vec, prot_vec).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
