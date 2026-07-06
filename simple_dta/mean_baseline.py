"""Trivial floor baseline: predict the train-set mean affinity for every test
example. No model, no learning -- just a sanity floor for the tower-ablation
study (Q1/Q2 in the drug-only/protein-only experiment): if a real model can't
beat this, it isn't using its input meaningfully.

Usage:
    python mean_baseline.py --dataset davis
    python mean_baseline.py --dataset davis --dataset kiba --dataset bindingdb
"""
import argparse
import json
import os
import numpy as np

from data import load_csv
from metrics import all_metrics


def run(dataset, out_dir):
    _, _, y_train = load_csv(dataset, "train")
    _, _, y_test = load_csv(dataset, "test")
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    mean_pred = np.full_like(y_test, y_train.mean())
    metrics = all_metrics(y_test, mean_pred, dataset=dataset)

    print(f"=== mean_baseline_{dataset} === train_mean={y_train.mean():.4f} "
          f"(n_train={len(y_train)}, n_test={len(y_test)})")
    print(f"MSE {metrics['MSE']:.4f} | CI {metrics['CI']:.4f} | rm2 {metrics['rm2']:.4f} "
          f"| Pearson {metrics['Pearson']:.4f} | balAcc_mean "
          f"{metrics.get('balacc_mean', float('nan')):.4f}")

    summary = {"tag": f"mean_baseline_{dataset}", "dataset": dataset,
               "train_mean": float(y_train.mean()), "n_train": len(y_train),
               "n_test": len(y_test), "metrics": metrics}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"mean_baseline_{dataset}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # same _pred.txt/_true.txt convention as train.py, so gen_results.py's
    # panel() can pick this up as a reference row in RESULTS_SUMMARY.txt
    np.savetxt(os.path.join(out_dir, f"mean_baseline_{dataset}_pred.txt"), mean_pred)
    np.savetxt(os.path.join(out_dir, f"mean_baseline_{dataset}_true.txt"), y_test)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"],
                    action="append", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()
    for dataset in args.dataset:
        run(dataset, args.out_dir)


if __name__ == "__main__":
    main()
