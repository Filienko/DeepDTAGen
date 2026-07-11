"""Model "F" -- standalone reference implementation of the model architecture.

Our best MPC candidate (~1.08M params). Drug side: a plain 2-layer GCN over
the molecular graph (atoms=nodes, bonds=edges) -- no attention in the message
passing itself. Protein side: a plain 1D-CNN, same family as RegB/ConfigA's
protein tower. The two towers are fused with CROSS-ATTENTION (drug and
protein representations attend to each other before pooling, instead of just
concatenating), using LINEAR (softmax-free) attention: no exp/divide anywhere
in the model, which is the expensive part to compute under MPC. Reference
run: 0.831 mean balanced accuracy on Davis -- within ~1pp of our best
*softmax*-attention model while avoiding softmax entirely, which is why this
is the current MPC front-runner.

Everything that defines F -- every layer, every hyperparameter -- is in this
one file, with all unused branches from the general-purpose AttnDTA class in
../models.py and the attention building blocks in ../attention.py (softmax
attention, GAT/graphformer/GINE drug encoders, transformer protein encoder,
positional encoding, local-window masking, attn_bias, unilateral cross-
attention directions, other pooling modes, edge features -- GCNConv never
uses them) removed, and every architecture choice hardcoded as a constant
below. If you're porting this to MPC, the classes in this file + the
constants above them are the complete spec -- in particular, `LinearAttention`
is the ONLY attention primitive used anywhere in F, and it contains no
softmax/exp/division-by-a-data-dependent-value beyond the two `+ 1e-6`
denominators.

Data loading and metrics are NOT model-specific, so this file imports them
from ../data.py and ../metrics.py rather than duplicating them.

For day-to-day experimentation (sweeping hyperparameters, trying other
architectures) use the configurable version in ../models.py + ../attention.py
+ ../train.py instead -- that's the "development" codebase this file was
distilled from.

Usage:
    python standalone_F.py                          # Davis, seed 4221 (reference config)
    python standalone_F.py --dataset kiba
    python standalone_F.py --seed 7 --epochs 4       # quick smoke test

NOTE: this model is much slower to train than RegB/ConfigA (graph batching +
attention). Budget several hours on Davis; KIBA is much bigger again.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ../ (simple_dta/)
from data import build_graph_dataset, CHARPROTLEN, FEATURIZERS
from metrics import all_metrics

# =============================================================================
# Hyperparameters -- these ARE the F architecture. Change them and you are no
# longer running F (that's fine for experimentation, just know that's what
# you're doing).
# =============================================================================
NODE_FEAT_DIM = FEATURIZERS["full"][1]   # 94; per-atom descriptor width (RDKit featurizer)
GCN_DIM = 128                             # GCN hidden width, same at every layer
GCN_LAYERS = 2                            # number of GCNConv layers

EMBED_DIM = 128                           # protein tower token embedding width
PROT_CHANNELS = [32, 64, 96]              # protein tower Conv1d out-channels, one entry per layer
PROT_KERNEL = 8                           # protein tower conv kernel size

ATTN_DIM = 128                            # shared projection width both towers are cast to before fusion
ATTN_HEADS = 4                            # cross-attention heads

HEAD_HIDDEN = 1024                        # first FC layer width in the prediction head
DROPOUT = 0.1                             # dropout prob (head + attention)
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100
SEED = 4221                               # matches the reference 0.831-balAcc run
EVAL_INTERVAL = 5                         # epochs between test-set evaluations
SELECT_METRIC = "balacc_mean"             # checkpoint-selection metric (maximize)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


# =============================================================================
# Drug tower: plain 2-layer GCN over the molecular graph. Returns one
# embedding PER ATOM (no pooling here -- pooling happens after cross-
# attention, on the fused sequence).
# =============================================================================
class DrugGCN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [NODE_FEAT_DIM] + [GCN_DIM] * GCN_LAYERS
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(GCN_LAYERS)])

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x  # [N_total_atoms_in_batch, GCN_DIM]


# =============================================================================
# Protein tower: plain 1D-CNN, same design as RegB/ConfigA's protein tower,
# but returns the per-position conv feature sequence (not pooled) so it can
# be cross-attended against the drug atoms.
# =============================================================================
class ProteinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CHARPROTLEN + 1, EMBED_DIM, padding_idx=0)
        convs, in_c = [], EMBED_DIM
        for out_c in PROT_CHANNELS:
            convs.append(nn.Conv1d(in_c, out_c, PROT_KERNEL))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.out_dim = PROT_CHANNELS[-1]

    def forward(self, target):
        x = self.embed(target).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        return x.transpose(1, 2)  # [B, L', C] -- per-position sequence, not pooled


# =============================================================================
# Linear (softmax-free) multi-head attention. This is the ONLY attention
# primitive F uses -- no softmax, no exp, no data-dependent division besides
# the `+ 1e-6`-stabilized normalizers below.
#
#   softmax(QK^T)V  -- the standard, expensive-under-MPC formulation
#   phi(Q) (phi(K)^T V) / (phi(Q) sum(phi(K)))  -- the linear reformulation
#     used here (Katharopoulos et al. 2020), with phi(x) = elu(x) + 1. This
#     reassociates the matmuls to be O(n) instead of O(n^2) in sequence
#     length, and needs only ELU (same nonlinearity class as the CNN towers'
#     ReLU) -- no exp/divide, which is the part with no cheap MPC protocol.
# =============================================================================
def _feature_map(x):
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert ATTN_DIM % ATTN_HEADS == 0
        self.d_head = ATTN_DIM // ATTN_HEADS
        self.q_proj = nn.Linear(ATTN_DIM, ATTN_DIM)
        self.k_proj = nn.Linear(ATTN_DIM, ATTN_DIM)
        self.v_proj = nn.Linear(ATTN_DIM, ATTN_DIM)
        self.out_proj = nn.Linear(ATTN_DIM, ATTN_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def _split_heads(self, x):
        b, length, _ = x.shape
        return x.view(b, length, ATTN_HEADS, self.d_head).transpose(1, 2)  # [B,H,L,Dh]

    def forward(self, query_src, kv_src, key_padding_mask=None):
        """query_src: [B, Lq, D]; kv_src: [B, Lk, D].
        key_padding_mask: [B, Lk] bool, True = pad (ignore that key/value)."""
        b, lq, _ = query_src.shape
        q = self._split_heads(self.q_proj(query_src))
        k = self._split_heads(self.k_proj(kv_src))
        v = self._split_heads(self.v_proj(kv_src))

        qp, kp = _feature_map(q), _feature_map(k)
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, :, None]  # [B,1,Lk,1]
            kp = kp.masked_fill(pad, 0.0)
            v = v.masked_fill(pad, 0.0)
        kv = torch.matmul(kp.transpose(-2, -1), v)               # [B,H,Dh,Dh]
        k_sum = kp.sum(dim=2, keepdim=True)                       # [B,H,1,Dh]
        denom = torch.matmul(qp, k_sum.transpose(-2, -1)) + 1e-6  # [B,H,Lq,1]
        out = torch.matmul(qp, kv) / denom

        out = out.transpose(1, 2).reshape(b, lq, ATTN_DIM)
        return self.out_proj(out)


class CrossAttentionFusion(nn.Module):
    """Bilateral cross-attention: drug queries protein AND protein queries
    drug, each followed by a residual + LayerNorm."""

    def __init__(self):
        super().__init__()
        self.drug_from_prot = LinearAttention()
        self.norm_d = nn.LayerNorm(ATTN_DIM)
        self.prot_from_drug = LinearAttention()
        self.norm_p = nn.LayerNorm(ATTN_DIM)

    def forward(self, drug_seq, drug_pad, prot_seq, prot_pad):
        d2 = self.drug_from_prot(drug_seq, prot_seq, key_padding_mask=prot_pad)
        out_drug = self.norm_d(drug_seq + d2)
        p2 = self.prot_from_drug(prot_seq, drug_seq, key_padding_mask=drug_pad)
        out_prot = self.norm_p(prot_seq + p2)
        return out_drug, out_prot


def masked_max_pool(x, pad_mask):
    """x: [B, L, D]; pad_mask: [B, L] bool (True=pad) or None -> [B, D]."""
    if pad_mask is None:
        pad_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    xm = x.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
    out = xm.max(dim=1).values
    return torch.nan_to_num(out, neginf=0.0)  # an all-pad row (shouldn't happen) -> 0


class Head(nn.Module):
    """FC(ATTN_DIM*2 -> HEAD_HIDDEN -> HEAD_HIDDEN//2 -> 1). Layers live under
    a `.net` submodule (not inlined into this class) so this module's
    state_dict keys (head.net.0/3/6.*) match ../models.py's PredictionHead
    exactly -- the reference *_best.pth checkpoint in ../best_models/weights/
    loads directly into this class with no key remapping."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# F: DrugGCN + ProteinCNN -> project both to ATTN_DIM -> bilateral linear
# cross-attention -> max-pool each side -> concat -> FC(1024) -> FC(512) -> 1
# =============================================================================
class F_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug = DrugGCN()
        self.protein = ProteinCNN()
        self.drug_proj = nn.Linear(GCN_DIM, ATTN_DIM)
        self.prot_proj = nn.Linear(self.protein.out_dim, ATTN_DIM)
        self.cross_attn = CrossAttentionFusion()
        self.head = Head(ATTN_DIM * 2)

    def forward(self, data):
        x, edge_index, batch_vec, target = data.x, data.edge_index, data.batch, data.target

        h = self.drug(x, edge_index)                          # [N_total, GCN_DIM]
        seq, node_mask = to_dense_batch(h, batch_vec)          # [B, N, GCN_DIM], True=real atom
        drug_pad = ~node_mask                                  # [B, N], True=pad
        drug_seq = self.drug_proj(seq)                          # [B, N, ATTN_DIM]

        prot_seq = self.prot_proj(self.protein(target))        # [B, L', ATTN_DIM]
        prot_pad = None                                          # no padding mask on the conv output

        drug_seq, prot_seq = self.cross_attn(drug_seq, drug_pad, prot_seq, prot_pad)
        drug_vec = masked_max_pool(drug_seq, drug_pad)
        prot_vec = masked_max_pool(prot_seq, prot_pad)
        return self.head(torch.cat([drug_vec, prot_vec], dim=1)).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Train + evaluate
# =============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, dataset):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())
    P = np.concatenate(preds).flatten()
    G = np.concatenate(trues).flatten()
    return all_metrics(G, P, dataset=dataset), G, P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--out-dir", default=RUNS_DIR)
    ap.add_argument("--cuda", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None
                          and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"F_{args.dataset}_s{args.seed}{args.tag_suffix}"
    print(f"=== {tag} | device={device} | seed={args.seed} ===")

    print("Building molecular graphs (in-the-clear encoding, featurizer=full)...")
    train_list = build_graph_dataset(args.dataset, "train", featurizer="full")
    test_list = build_graph_dataset(args.dataset, "test", featurizer="full")
    train_loader = GeoLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = GeoLoader(test_list, batch_size=BATCH_SIZE, shuffle=False)

    model = F_Model().to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best = {SELECT_METRIC: float("-inf")}
    history = []
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        dt = time.time() - t0
        epoch_times.append(dt)
        train_mse = running / n

        if epoch % EVAL_INTERVAL == 0 or epoch == args.epochs:
            m, G, P = evaluate(model, test_loader, device, args.dataset)
            history.append({"epoch": epoch, "train_mse": train_mse, **m})
            print(f"[{tag}] epoch {epoch:3d} | {dt:5.1f}s | train_mse {train_mse:.4f} "
                  f"| test MSE {m['MSE']:.4f} CI {m['CI']:.4f} rm2 {m['rm2']:.4f} "
                  f"Pearson {m['Pearson']:.4f} balAcc {m['balacc_mean']:.4f}")
            if m[SELECT_METRIC] > best[SELECT_METRIC]:
                best = {"epoch": epoch, **m}
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"{tag}_best.pth"))
                np.savetxt(os.path.join(args.out_dir, f"{tag}_pred.txt"), P)
                np.savetxt(os.path.join(args.out_dir, f"{tag}_true.txt"), G)

    summary = {"tag": tag, "model": "F", "dataset": args.dataset, "epochs": args.epochs,
               "params": count_params(model), "seed": args.seed,
               "mean_epoch_sec": float(np.mean(epoch_times)),
               "total_train_sec": float(np.sum(epoch_times)),
               "best": best, "history": history}
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BEST [{tag}] === epoch {best.get('epoch')} | MSE {best['MSE']:.4f} "
          f"| CI {best['CI']:.4f} | rm2 {best['rm2']:.4f} | balAcc {best['balacc_mean']:.4f}")


if __name__ == "__main__":
    main()
