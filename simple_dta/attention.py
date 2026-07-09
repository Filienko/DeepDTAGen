"""Attention building blocks shared by every AttnDTA configuration (models.py).

MPC design notes (see EXPERIMENTS.md section 10 for the full writeup):
  * Q/K/V projections, Q.K^T, and the weighted-sum-with-V are all matmuls --
    linear, same cost class as the CNN towers' conv/FC layers.
  * softmax (exp + normalize) is the expensive part: no cheap MPC protocol for
    exp/division, and it's applied to an [Lq, Lk] matrix -- O(n^2) evaluations
    of that expensive op, vs. one ReLU per element in the CNN towers.
  * kind="linear" replaces softmax(QK^T)V with a softmax-free, ReLU-family
    kernel-feature-map reformulation (Katharopoulos et al. 2020) that is
    mathematically ~attention but reassociates the matmuls to be O(n) instead
    of O(n^2) in sequence length, and never needs exp/divide -- only the same
    elementwise nonlinearity class (ELU) already used elsewhere in the project.
    This is the piece that actually targets the MPC-efficiency goal; kind=
    "softmax" is the accuracy-reference / literature-standard configuration.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _elu_feature_map(x):
    return F.elu(x) + 1.0


def local_window_mask(length, window, device):
    """[L, L] bool mask, True = disallowed. window=0/None -> no mask (full attention).

    Bounds each position to its +/- window//2 neighbors -- the sequence-attention
    analogue of a conv's fixed receptive field. IMPORTANT: this only changes
    which pairs contribute to the softmax; it does NOT reduce the O(n^2)
    QK^T matmul cost, since the full score matrix is still computed and then
    masked (no sparse/chunked kernel implemented here). For long protein
    sequences (~1000-1200 residues here) this masking gives essentially no
    speedup over full attention -- measured ~75-80 min/epoch on Davis either
    way. Use kind="linear" (MultiHeadAttention) for an actual O(n) protein
    tower; that's the only practical option for full-length training runs.
    """
    if not window:
        return None
    idx = torch.arange(length, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    return dist > (window // 2)


def masked_pool(x, pad_mask, mode):
    """x: [B, L, D]; pad_mask: [B, L] bool, True = pad (or None = no padding)."""
    if pad_mask is None:
        pad_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    if mode == "max":
        xm = x.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        out = xm.max(dim=1).values
        return torch.nan_to_num(out, neginf=0.0)  # an all-pad row (shouldn't happen) -> 0
    if mode == "mean":
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    if mode == "maxmean":
        return torch.cat([masked_pool(x, pad_mask, "max"),
                           masked_pool(x, pad_mask, "mean")], dim=1)
    raise ValueError(f"unknown pool mode {mode}")


class MultiHeadAttention(nn.Module):
    """Multi-head attention for both self-attention (query_src is kv_src) and
    cross-attention (query_src != kv_src, e.g. drug queries protein).

    kind="softmax": standard scaled-dot-product attention.
    kind="linear":  softmax-free linear attention -- see module docstring.
    """

    def __init__(self, d_model, n_heads, kind="softmax", dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert kind in ("softmax", "linear"), f"unknown attention kind {kind}"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.kind = kind
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, length, _ = x.shape
        return x.view(b, length, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,Dh]

    def forward(self, query_src, kv_src, key_padding_mask=None, attn_mask=None, attn_bias=None):
        """query_src: [B, Lq, D]; kv_src: [B, Lk, D].
        key_padding_mask: [B, Lk] bool, True = pad (ignore that key/value).
        attn_mask: [Lq, Lk] bool, True = disallowed pair (softmax kind only).
        attn_bias: real-valued, broadcastable to [B, n_heads, Lq, Lk], added to the
            scores before masking/softmax (softmax kind only) -- e.g. a Graphormer-
            style structural bias derived from edge features. Not supported for
            kind="linear": kernelized attention never forms an explicit [Lq,Lk]
            score matrix, so there is nothing to add a per-pair bias to.
        """
        b, lq, _ = query_src.shape
        q = self._split_heads(self.q_proj(query_src))
        k = self._split_heads(self.k_proj(kv_src))
        v = self._split_heads(self.v_proj(kv_src))

        if self.kind == "softmax":
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if attn_bias is not None:
                scores = scores + attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            weights = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(weights, v)
        else:
            if attn_bias is not None:
                raise NotImplementedError(
                    "attn_bias requires kind='softmax' (see forward() docstring)")
            qp, kp = _elu_feature_map(q), _elu_feature_map(k)
            if key_padding_mask is not None:
                pad = key_padding_mask[:, None, :, None]  # [B,1,Lk,1]
                kp = kp.masked_fill(pad, 0.0)
                v = v.masked_fill(pad, 0.0)
            kv = torch.matmul(kp.transpose(-2, -1), v)               # [B,H,Dh,Dh]
            k_sum = kp.sum(dim=2, keepdim=True)                       # [B,H,1,Dh]
            denom = torch.matmul(qp, k_sum.transpose(-2, -1)) + 1e-6  # [B,H,Lq,1]
            out = torch.matmul(qp, kv) / denom

        out = out.transpose(1, 2).reshape(b, lq, self.d_model)
        return self.out_proj(out)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """Pre-LN self-attention block: attn -> residual+norm -> FFN -> residual+norm."""

    def __init__(self, d_model, n_heads, ff_dim=None, kind="softmax", dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 2
        self.attn = MultiHeadAttention(d_model, n_heads, kind=kind, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None, attn_bias=None):
        a = self.attn(x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                      attn_bias=attn_bias)
        x = self.norm1(x + self.dropout(a))
        f = self.ff(x)
        return self.norm2(x + self.dropout(f))


class ProteinTransformer(nn.Module):
    """Embedding -> positional encoding -> N self-attention blocks -> global pool.

    Full self-attention gives every residue a global receptive field from
    layer 1 (vs. the CNN tower's RF~22 residues over ~1000-1200, see
    EXPERIMENTS.md section 5) but costs O(L^2) per layer. Measured: with
    kind="softmax" this runs ~75-80 min/epoch on Davis (`window` masks
    results but doesn't cut the matmul cost -- see local_window_mask), vs.
    ~7 min/epoch with kind="linear" (genuinely O(n), no window needed).
    kind="linear" is the only practical choice for full-length training here.
    """

    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=2, window=64,
                 kind="softmax", dropout=0.1, pool="max"):
        super().__init__()
        self.pool = pool
        self.window = window
        self.kind = kind
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads, kind=kind, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_dim = embed_dim * (2 if pool == "maxmean" else 1)

    def forward(self, target, return_seq=False):
        pad_mask = (target == 0)  # [B, L] True = pad
        x = self.pos(self.embed(target))
        attn_mask = (local_window_mask(x.size(1), self.window, x.device)
                     if self.kind == "softmax" else None)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask, attn_mask=attn_mask)
        if return_seq:
            return x, pad_mask
        return masked_pool(x, pad_mask, self.pool)


class DrugGraphTransformer(nn.Module):
    """Graphormer-style drug encoder: NO message passing at all. Each atom's raw
    node-feature vector is projected straight into embed_dim and self-attended
    against every other atom in the same molecule (drug graphs are tiny --
    typically <100 atoms -- so full O(n^2) attention is cheap here, unlike the
    protein tower). Bond/edge features are injected as an additive per-head
    bias on the attention scores (Graphormer's edge encoding), rather than
    through explicit graph convolution -- contrast with `GraphAttnEncoder`
    (GATv2), which still does message passing, just with content-based
    edge weights. This is the "structure only as a bias, not a computation"
    end of the spectrum.

    No positional encoding: atom order within a molecule is arbitrary (unlike
    a protein's residue sequence), so there is no canonical position to encode.
    """

    def __init__(self, node_feat_dim, embed_dim=256, n_heads=8, n_layers=4,
                 edge_dim=None, ff_mult=2, dropout=0.1):
        super().__init__()
        self.edge_dim = edge_dim
        self.in_proj = nn.Linear(node_feat_dim, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads, ff_dim=embed_dim * ff_mult,
                                    kind="softmax", dropout=dropout)
            for _ in range(n_layers)
        ])
        if edge_dim is not None:
            self.edge_bias_proj = nn.Linear(edge_dim, n_heads)
        self.out_dim = embed_dim

    def forward(self, x, edge_index, edge_attr, batch_vec):
        """Returns (seq, pad_mask) -- pooling/projection is the caller's job,
        same contract as ProteinCNN/SmilesCNN's return_seq=True path."""
        from torch_geometric.utils import to_dense_batch, to_dense_adj
        seq, node_mask = to_dense_batch(x, batch_vec)  # [B, N, node_feat_dim], True=real
        h = self.in_proj(seq)
        pad_mask = ~node_mask

        attn_bias = None
        if self.edge_dim is not None:
            dense_edge = to_dense_adj(edge_index, batch_vec, edge_attr=edge_attr,
                                       max_num_nodes=seq.size(1))  # [B, N, N, edge_dim]
            attn_bias = self.edge_bias_proj(dense_edge).permute(0, 3, 1, 2)  # [B, heads, N, N]

        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask, attn_bias=attn_bias)
        return h, pad_mask


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between a drug and a protein sequence, each
    followed by a residual+norm -- mirrors AttentionDTA's two-sided fusion,
    generalized to run on top of any pair of sequence encoders (CNN,
    transformer, or graph-attention via a dense-batched node sequence).
    Runs on pre-pool sequences; pooling happens after.

    direction: "both" (default, bilateral -- drug queries protein AND
        protein queries drug, i.e. the original mechanism) | "drug2prot"
        (only drug queries protein; the protein sequence passes through
        unchanged) | "prot2drug" (only protein queries drug; the drug
        sequence passes through unchanged). Unilateral directions skip
        building the unused MultiHeadAttention entirely -- roughly half the
        params/compute of "both", not just a discarded computation -- since
        cross-attention is the single most expensive op in every top model
        here and MPC cost scales with it directly.
    """

    def __init__(self, d_model, n_heads=4, kind="softmax", dropout=0.1, direction="both"):
        super().__init__()
        assert direction in ("both", "drug2prot", "prot2drug"), f"unknown direction {direction}"
        self.direction = direction
        if direction in ("both", "drug2prot"):
            self.drug_from_prot = MultiHeadAttention(d_model, n_heads, kind=kind, dropout=dropout)
            self.norm_d = nn.LayerNorm(d_model)
        if direction in ("both", "prot2drug"):
            self.prot_from_drug = MultiHeadAttention(d_model, n_heads, kind=kind, dropout=dropout)
            self.norm_p = nn.LayerNorm(d_model)

    def forward(self, drug_seq, drug_pad, prot_seq, prot_pad):
        out_drug, out_prot = drug_seq, prot_seq
        if self.direction in ("both", "drug2prot"):
            d2 = self.drug_from_prot(drug_seq, prot_seq, key_padding_mask=prot_pad)
            out_drug = self.norm_d(drug_seq + d2)
        if self.direction in ("both", "prot2drug"):
            p2 = self.prot_from_drug(prot_seq, drug_seq, key_padding_mask=drug_pad)
            out_prot = self.norm_p(prot_seq + p2)
        return out_drug, out_prot
