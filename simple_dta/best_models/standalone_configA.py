"""Model "ConfigA" -- standalone reference implementation of the model architecture.

Our small, single-layer-head CNN+CNN model (~267K params, about a third the
size of RegB). Same two-tower CNN recipe as RegB, but each tower is squeezed
to a compact 32-dim "binding embedding" (via a Linear+ReLU projection) before
a single wide FC layer, instead of RegB's 2-layer head. Reference run: 0.795
mean balanced accuracy on Davis -- within ~5pp of RegB at ~3x fewer params.
This is the model to reach for whenever "how cheap can we make it" matters
more than "how accurate."

Everything that defines ConfigA -- every layer, every hyperparameter -- is in
this one file, with all unused branches from the general-purpose CNNDTA
class in ../models.py (dilation configs, tower ablation, other pooling
modes) removed and every architecture choice hardcoded as a constant below.
If you're porting this to MPC, the `ConfigA` class + the constants above it
are the complete spec.

Data loading and metrics are NOT model-specific, so this file imports them
from ../data.py and ../metrics.py rather than duplicating them.

For day-to-day experimentation (sweeping hyperparameters, trying other
architectures) use the configurable version in ../models.py + ../train.py
instead -- that's the "development" codebase this file was distilled from.

Usage:
    python standalone_configA.py                       # Davis, seed 4221 (reference config)
    python standalone_configA.py --dataset kiba
    python standalone_configA.py --seed 13 --epochs 4   # quick smoke test
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ../ (simple_dta/)
from data import CNNDataset, CHARISOSMILEN, CHARPROTLEN
from metrics import all_metrics

# =============================================================================
# Hyperparameters -- these ARE the ConfigA architecture. Change them and you
# are no longer running ConfigA (that's fine for experimentation, just know
# that's what you're doing).
# =============================================================================
EMBED_DIM = 128                  # token embedding width, shared by both towers
DRUG_CHANNELS = [32, 64, 96]     # SMILES tower Conv1d out-channels, one entry per layer
DRUG_KERNEL = 4                  # SMILES tower conv kernel size
PROT_CHANNELS = [32, 64, 96]     # protein tower Conv1d out-channels, one entry per layer
PROT_KERNEL = 8                  # protein tower conv kernel size
PROJ_DIM = 32                    # each tower is compressed to this width before the head
HEAD_HIDDEN = 1536               # the single FC layer's width
DROPOUT = 0.1                     # dropout prob in the prediction head
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100
SEED = 4221                       # matches the reference 0.795-balAcc run
EVAL_INTERVAL = 5                 # epochs between test-set evaluations
SELECT_METRIC = "balacc_mean"     # checkpoint-selection metric (maximize)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


# =============================================================================
# Model: two 1D-CNN towers -> project each to 32-dim -> concat -> 1-layer FC head
# =============================================================================
class SmilesCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CHARISOSMILEN + 1, EMBED_DIM, padding_idx=0)
        convs, in_c = [], EMBED_DIM
        for out_c in DRUG_CHANNELS:
            convs.append(nn.Conv1d(in_c, out_c, DRUG_KERNEL))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.out_dim = DRUG_CHANNELS[-1]

    def forward(self, smiles):
        x = self.embed(smiles).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)  # global max pool -> [B, C]


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
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)


class Head(nn.Module):
    """FC(PROJ_DIM*2 -> HEAD_HIDDEN -> 1). Layers live under a `.net`
    submodule (not inlined into this class) so this module's state_dict keys
    (head.net.0/3.*) match ../models.py's PredictionHead exactly -- the
    reference *_best.pth checkpoint in ../best_models/weights/ loads
    directly into this class with no key remapping."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConfigA(nn.Module):
    """SmilesCNN/ProteinCNN -> project each to 32-dim -> concat -> FC(1536) -> 1."""

    def __init__(self):
        super().__init__()
        self.drug = SmilesCNN()
        self.protein = ProteinCNN()
        self.drug_proj = nn.Linear(self.drug.out_dim, PROJ_DIM)
        self.prot_proj = nn.Linear(self.protein.out_dim, PROJ_DIM)
        self.head = Head(PROJ_DIM * 2)

    def forward(self, smiles, target):
        d = F.relu(self.drug_proj(self.drug(smiles)))
        p = F.relu(self.prot_proj(self.protein(target)))
        return self.head(torch.cat([d, p], dim=1)).squeeze(-1)


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
        for xd, xt, y in loader:
            xd, xt = xd.to(device), xt.to(device)
            preds.append(model(xd, xt).cpu().numpy())
            trues.append(y.numpy())
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
    tag = f"configA_{args.dataset}_s{args.seed}{args.tag_suffix}"
    print(f"=== {tag} | device={device} | seed={args.seed} ===")

    train_loader = DataLoader(CNNDataset(args.dataset, "train"), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CNNDataset(args.dataset, "test"), batch_size=BATCH_SIZE, shuffle=False)

    model = ConfigA().to(device)
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
        for xd, xt, y in train_loader:
            xd, xt, y = xd.to(device), xt.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(xd, xt)
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

    summary = {"tag": tag, "model": "configA", "dataset": args.dataset, "epochs": args.epochs,
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
