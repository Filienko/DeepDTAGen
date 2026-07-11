"""Model "RegB" -- standalone reference implementation of the model architecture.

Our most accurate CNN+CNN model (~800K params). Two independent 1D-CNN
towers (drug SMILES string, protein amino-acid sequence), each global-max-
pooled to a vector, concatenated, and fed through a 2-layer FC head. No
attention, no molecular graph, no MPC-specific tricks -- this is the
"how good can a simple model get" accuracy ceiling everything else is
compared against. Reference run: 0.848 mean balanced accuracy on Davis
(clears the DeepDTAGen 0.820 target at every threshold).

Everything that defines RegB -- every layer, every hyperparameter -- is in
this one file, with all unused branches from the general-purpose CNNDTA
class in ../models.py (dilation configs, tower ablation, tower-output
projection, other pooling modes) removed and every architecture choice
hardcoded as a constant below. If you're porting this to MPC, the `RegB`
class + the constants above it are the complete spec.

Data loading and metrics are NOT model-specific, so this file imports them
from ../data.py and ../metrics.py rather than duplicating them.

For day-to-day experimentation (sweeping hyperparameters, trying other
architectures) use the configurable version in ../models.py + ../train.py
instead -- that's the "development" codebase this file was distilled from.

Usage:
    python standalone_regB.py                          # Davis, seed 4221 (reference config)
    python standalone_regB.py --dataset kiba
    python standalone_regB.py --seed 7 --epochs 4       # quick smoke test
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
# Hyperparameters -- these ARE the RegB architecture. Change them and you are
# no longer running RegB (that's fine for experimentation, just know that's
# what you're doing).
# =============================================================================
EMBED_DIM = 128                  # token embedding width, shared by both towers
DRUG_CHANNELS = [16, 32, 48]     # SMILES tower Conv1d out-channels, one entry per layer
DRUG_KERNEL = 4                  # SMILES tower conv kernel size
PROT_CHANNELS = [32, 64, 96]     # protein tower Conv1d out-channels, one entry per layer
PROT_DILATIONS = [1, 2, 4]       # protein tower conv dilation, one entry per layer
PROT_KERNEL = 8                  # protein tower conv kernel size
HEAD_HIDDEN = 1024                # first FC layer width in the prediction head
DROPOUT = 0.1                     # dropout prob in the prediction head
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100
SEED = 4221                       # matches the reference 0.848-balAcc run
EVAL_INTERVAL = 5                 # epochs between test-set evaluations
SELECT_METRIC = "balacc_mean"     # checkpoint-selection metric (maximize)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


# =============================================================================
# Model: two 1D-CNN towers -> concat -> 2-layer FC head
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
        for out_c, d in zip(PROT_CHANNELS, PROT_DILATIONS):
            convs.append(nn.Conv1d(in_c, out_c, PROT_KERNEL, dilation=d))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.out_dim = PROT_CHANNELS[-1]

    def forward(self, target):
        x = self.embed(target).transpose(1, 2)  # [B, embed, L]
        for conv in self.convs:
            x = F.relu(conv(x))
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)


class Head(nn.Module):
    """FC(in_dim -> HEAD_HIDDEN -> HEAD_HIDDEN//2 -> 1). Layers live under a
    `.net` submodule (not inlined into this class) so this module's state_dict
    keys (head.net.0/3/6.*) match ../models.py's PredictionHead exactly --
    the reference *_best.pth checkpoints in ../best_models/weights/ load
    directly into this class with no key remapping."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class RegB(nn.Module):
    """SmilesCNN(drug) + ProteinCNN(protein) -> concat -> FC(1024) -> FC(512) -> 1."""

    def __init__(self):
        super().__init__()
        self.drug = SmilesCNN()
        self.protein = ProteinCNN()
        self.head = Head(self.drug.out_dim + self.protein.out_dim)

    def forward(self, smiles, target):
        d = self.drug(smiles)
        p = self.protein(target)
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
    tag = f"regB_{args.dataset}_s{args.seed}{args.tag_suffix}"
    print(f"=== {tag} | device={device} | seed={args.seed} ===")

    train_loader = DataLoader(CNNDataset(args.dataset, "train"), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CNNDataset(args.dataset, "test"), batch_size=BATCH_SIZE, shuffle=False)

    model = RegB().to(device)
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

    summary = {"tag": tag, "model": "regB", "dataset": args.dataset, "epochs": args.epochs,
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
