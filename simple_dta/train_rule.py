"""Train the "rule-based protein tower" model: drug SmilesCNN (learned) +
a small fixed vector of PWM motif-scan scores (NOT learned, no protein CNN
at all) -> prediction head.

The PWMs come from `interpret_protein_tower.py --save-pwm`, fit from the
most-important channels of an already-trained joint CNNDTA model (see
runs/<tag>_pwm.json). This directly tests Steven's Q3: can a handful of
hand-scannable motifs (found to be HRD/DFG/APE-box kinase motifs) stand in
for the learned protein CNN tower?

Mirrors train.py's loop/summary format (same metrics, same checkpoint-
selection convention, same _pred.txt/_true.txt/_summary.json layout) so
gen_results.py's panel() can fold this tag into RESULTS_SUMMARY.txt exactly
like any other run.

Usage:
    python train_rule.py --dataset davis --pwm-tag cnn_davis_stable_pool_max --epochs 100
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import load_csv, label_encode, CHARISOSMISET, DEFAULTS
from metrics import all_metrics
from models import SmilesCNN, PredictionHead, count_params
from rule_features import load_pwms, RuleFeaturizer

SEED = 4221
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class RuleDataset(Dataset):
    """Drug: label-encoded SMILES (same as CNNDataset). Protein: precomputed,
    standardized k-dim rule-feature vector (shared cache across train/test so
    each unique protein sequence is only scanned once)."""

    def __init__(self, dataset, split, featurizer, max_smi_len, mean=None, std=None):
        smiles, prots, ys = load_csv(dataset, split)
        self.smiles = smiles
        self.y = ys
        self.max_smi_len = max_smi_len
        raw_feats = np.stack([featurizer(p) for p in prots], axis=0).astype(np.float32)
        self.mean = raw_feats.mean(axis=0) if mean is None else mean
        self.std = (raw_feats.std(axis=0) + 1e-6) if std is None else std
        self.feats = (raw_feats - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xd = label_encode(self.smiles[idx], self.max_smi_len, CHARISOSMISET)
        return (torch.from_numpy(xd), torch.from_numpy(self.feats[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32))


class RuleProteinDTA(nn.Module):
    """Learned drug tower + fixed rule-based protein features -> head. No
    protein CNN at all -- `rule_feats` is a precomputed, standardized k-dim
    vector, concatenated directly with the drug embedding."""

    def __init__(self, n_rule_feats, drug_filters=32, embed_dim=128, drug_kernel=4,
                dropout=0.1, head_dim=1024, head_layers=2, pool="max"):
        super().__init__()
        self.drug = SmilesCNN(embed_dim, drug_filters, kernel_size=drug_kernel, pool=pool)
        self.head = PredictionHead(self.drug.out_dim + n_rule_feats, hidden=head_dim,
                                   layers=head_layers, dropout=dropout)

    def forward(self, smiles, rule_feats):
        d = self.drug(smiles)
        return self.head(d, rule_feats).squeeze(-1)


def evaluate(model, loader, device, dataset):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xd, xf, y in loader:
            pred = model(xd.to(device), xf.to(device))
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
    P = np.concatenate(preds).flatten()
    G = np.concatenate(trues).flatten()
    return all_metrics(G, P, dataset=dataset), G, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], required=True)
    ap.add_argument("--pwm-tag", default="cnn_davis_stable_pool_max",
                    help="source run whose runs/<tag>_pwm.json to load motifs from")
    ap.add_argument("--top-k-motifs", type=int, default=None,
                    help="use only the top-k (by delta_MSE) PWMs from the file; default = all saved")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drug-filters", type=int, default=32)
    ap.add_argument("--drug-kernel", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--pool", choices=["max", "mean", "maxmean"], default="max")
    ap.add_argument("--head-dim", type=int, default=1024)
    ap.add_argument("--head-layers", type=int, choices=[1, 2], default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--select-metric", choices=["mse", "balacc"], default="balacc")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--out-dir", default=RUNS_DIR)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"rule_{args.dataset}{args.tag_suffix}"

    pwms, rf = load_pwms(os.path.join(RUNS_DIR, f"{args.pwm_tag}_pwm.json"))
    if args.top_k_motifs:
        pwms = pwms[:args.top_k_motifs]
    print(f"=== {tag} | device={device} | {len(pwms)} rule motifs from {args.pwm_tag} "
          f"(rf={rf}) | channels={[p['channel'] for p in pwms]} ===")
    featurizer = RuleFeaturizer(pwms)

    max_smi_len = DEFAULTS[args.dataset]["max_smi_len"]
    train_ds = RuleDataset(args.dataset, "train", featurizer, max_smi_len)
    test_ds = RuleDataset(args.dataset, "test", featurizer, max_smi_len,
                          mean=train_ds.mean, std=train_ds.std)  # standardize w/ TRAIN stats
    print(f"unique proteins scanned: {len(featurizer._cache)} "
          f"(train n={len(train_ds)}, test n={len(test_ds)})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = RuleProteinDTA(n_rule_feats=len(pwms), drug_filters=args.drug_filters,
                           embed_dim=args.embed_dim, drug_kernel=args.drug_kernel,
                           dropout=args.dropout, head_dim=args.head_dim,
                           head_layers=args.head_layers, pool=args.pool).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    sel_key = "balacc_mean" if args.select_metric == "balacc" else "MSE"
    sel_minimize = (sel_key == "MSE")
    best = {sel_key: float("inf") if sel_minimize else float("-inf")}
    history = []
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for xd, xf, y in train_loader:
            xd, xf, y = xd.to(device), xf.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(xd, xf)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        dt = time.time() - t0
        epoch_times.append(dt)
        train_mse = running / n

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m, G, P = evaluate(model, test_loader, device, args.dataset)
            history.append({"epoch": epoch, "train_mse": train_mse, **m})
            print(f"[{tag}] epoch {epoch:3d} | {dt:5.1f}s | train_mse {train_mse:.4f} "
                  f"| test MSE {m['MSE']:.4f} CI {m['CI']:.4f} rm2 {m['rm2']:.4f} "
                  f"Pearson {m['Pearson']:.4f} balAcc {m.get('balacc_mean', float('nan')):.4f}")
            cur = m.get(sel_key)
            improved = (cur is not None and not np.isnan(cur) and
                        (cur < best[sel_key] if sel_minimize else cur > best[sel_key]))
            if improved:
                best = {"epoch": epoch, **m}
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"{tag}_best.pth"))
                np.savetxt(os.path.join(args.out_dir, f"{tag}_pred.txt"), P)
                np.savetxt(os.path.join(args.out_dir, f"{tag}_true.txt"), G)

    summary = {
        "tag": tag, "model": "rule", "dataset": args.dataset,
        "pwm_tag": args.pwm_tag, "n_rule_motifs": len(pwms),
        "motif_channels": [p["channel"] for p in pwms],
        "epochs": args.epochs, "params": count_params(model),
        "mean_epoch_sec": float(np.mean(epoch_times)),
        "total_train_sec": float(np.sum(epoch_times)),
        "best": best, "pool": args.pool, "head_dim": args.head_dim,
        "head_layers": args.head_layers, "seed": args.seed,
        "embed_dim": args.embed_dim, "drug_filters": args.drug_filters,
        "drug_kernel": args.drug_kernel, "dropout": args.dropout,
        "weight_decay": args.weight_decay, "select_metric": args.select_metric,
        "history": history,
    }
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BEST [{tag}] === epoch {best.get('epoch')} | "
          f"MSE {best['MSE']:.4f} | CI {best['CI']:.4f} | rm2 {best['rm2']:.4f}")
    print(f"params {summary['params']:,} | mean {summary['mean_epoch_sec']:.1f}s/epoch "
          f"| total {summary['total_train_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
