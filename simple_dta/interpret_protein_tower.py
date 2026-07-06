"""Post-hoc interpretability for the protein CNN tower of an already-trained
CNNDTA checkpoint. No retraining -- two analyses on a fixed model:

1. Channel knockout: zero each final-layer protein channel (post-ReLU,
   pre-pool) one at a time and measure how much test MSE gets worse. Ranks
   channels by importance without needing gradients or retraining.

2. Motif extraction: for the top-k channels from (1), find the protein
   subsequence that maximally activates that channel in each example (the
   argmax position under max-pooling *is* the window the channel "voted"
   on) and print the most-activating windows + a per-position consensus,
   the way DeepBind-style genomics CNNs are visualized as sequence motifs.

The receptive field of a stack of same-kernel Conv1d layers (stride 1, no
padding) maps a final-layer output index i exactly onto raw input window
[i, i + (kernel_size - 1) * sum(dilations) + 1) -- this only requires
reading `kernel_size`/`dilation` off the trained model's own conv modules,
so it works for any dilation schedule, not just the default 1/1/1.

Usage:
    python interpret_protein_tower.py --tag cnn_davis_stable_pool_max --top-k 8
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CNNDataset
from models import CNNDTA, _pool1d
from metrics import all_metrics

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def load_model(tag):
    with open(os.path.join(RUNS_DIR, f"{tag}_summary.json")) as f:
        cfg = json.load(f)
    assert cfg["model"] == "cnn", "interpretability script only supports CNNDTA"
    _il = lambda s: [int(x) for x in s.split(",")] if s else None
    model = CNNDTA(
        embed_dim=cfg["embed_dim"], pool=cfg["pool"], head_dim=cfg["head_dim"],
        head_layers=cfg["head_layers"], drug_filters=cfg["drug_filters"],
        prot_filters=cfg["prot_filters"], proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"], drug_channels=_il(cfg["drug_channels"]),
        prot_channels=_il(cfg["prot_channels"]), drug_dilations=_il(cfg["drug_dilations"]),
        prot_dilations=_il(cfg["prot_dilations"]), drug_kernel=cfg.get("drug_kernel", 4),
        prot_kernel=cfg.get("prot_kernel", 8), ablate=cfg.get("ablate", "none"))
    state = torch.load(os.path.join(RUNS_DIR, f"{tag}_best.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def protein_feature_map(model, target):
    """Replicate ProteinCNN.forward up to (but not including) the pool, so we
    can zero individual channels before pooling."""
    x = model.protein.embed(target).transpose(1, 2)
    for conv in model.protein.convs:
        x = F.relu(conv(x))
    return x  # [B, C, L]


def receptive_field(model):
    kernel = model.protein.convs[0].kernel_size[0]
    dilations = [c.dilation[0] for c in model.protein.convs]
    assert all(c.kernel_size[0] == kernel for c in model.protein.convs), \
        "receptive-field formula assumes identical kernel size across layers"
    return (kernel - 1) * sum(dilations) + 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_davis_stable_pool_max",
                    help="run tag whose runs/<tag>_summary.json + _best.pth to load")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--top-k", type=int, default=8,
                    help="how many top/important channels to extract motifs for")
    ap.add_argument("--motifs-per-channel", type=int, default=20,
                    help="how many top-activating windows to keep per channel")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=None, help="output JSON path (default runs/<tag>_interpret.json)")
    ap.add_argument("--save-pwm", action="store_true",
                    help="also fit + save a proper position weight matrix (log-odds "
                         "vs. train-set amino-acid background) per top channel, for "
                         "the rule-based motif-scan replacement (see rule_features.py)")
    ap.add_argument("--pwm-n", type=int, default=300,
                    help="how many top-activating (deduped) windows per channel to "
                         "align when fitting the PWM -- larger than --motifs-per-channel "
                         "(which is just for the printed preview) for a less noisy estimate")
    ap.add_argument("--pwm-pseudocount", type=float, default=1.0)
    ap.add_argument("--min-window-pos", type=int, default=15,
                    help="drop candidate windows starting before this position in "
                         "the sequence. Nearly every protein starts with Met and a "
                         "similar N-terminal/tag region, which is not a real internal "
                         "motif but can dominate the 'top activating window' pool for "
                         "channels that don't have many genuine strong hits -- verified "
                         "this happens for 4/5 channels on cnn_davis_stable_pool_max "
                         "(53-84%% of examples' argmax landed at position 0)")
    args = ap.parse_args()

    model, cfg = load_model(args.tag)
    dataset = cfg["dataset"]
    ds = CNNDataset(dataset, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    n = len(ds)
    n_channels = model.protein.out_dim if model.protein.pool != "maxmean" else model.protein.out_dim // 2
    rf = receptive_field(model)
    print(f"=== {args.tag} | dataset={dataset} split={args.split} n={n} | "
          f"protein pool={model.protein.pool} channels={n_channels} kernel={model.protein.convs[0].kernel_size[0]} "
          f"dilations={[c.dilation[0] for c in model.protein.convs]} => receptive_field={rf} residues ===")

    y_all = np.array(ds.y, dtype=np.float32)
    baseline_preds = np.zeros(n, dtype=np.float32)
    knockout_preds = np.zeros((n_channels, n), dtype=np.float32)

    # channel-argmax bookkeeping for motif extraction (max-pool only; still
    # computed for mean/maxmean but the "argmax window" is a heuristic there
    # since the pooled vector isn't literally that one position's value)
    argmax_pos = np.zeros((n, n_channels), dtype=np.int64)
    argmax_val = np.zeros((n, n_channels), dtype=np.float32)

    offset = 0
    for xd, xt, y in loader:
        b = xd.shape[0]
        d = model.drug(xd)
        if model.proj_dim:
            d = F.relu(model.drug_proj(d))

        feat = protein_feature_map(model, xt)  # [B, C, L]
        cmax, cpos = feat.max(dim=2)  # [B, C] each -- valid regardless of pool mode, used for motifs
        argmax_val[offset:offset + b] = cmax.numpy()
        argmax_pos[offset:offset + b] = cpos.numpy()

        pooled = _pool1d(feat, model.protein.pool)
        p = F.relu(model.prot_proj(pooled)) if model.proj_dim else pooled
        baseline_preds[offset:offset + b] = model.head(d, p).squeeze(-1).numpy()

        for c in range(n_channels):
            feat_ko = feat.clone()
            feat_ko[:, c, :] = 0.0
            pooled_ko = _pool1d(feat_ko, model.protein.pool)
            p_ko = F.relu(model.prot_proj(pooled_ko)) if model.proj_dim else pooled_ko
            knockout_preds[c, offset:offset + b] = model.head(d, p_ko).squeeze(-1).numpy()

        offset += b
        if offset % (args.batch_size * 4) == 0 or offset == n:
            print(f"  ...{offset}/{n} examples processed", flush=True)

    baseline_metrics = all_metrics(y_all, baseline_preds, dataset=dataset)
    print(f"baseline (no knockout): MSE {baseline_metrics['MSE']:.4f} "
          f"CI {baseline_metrics['CI']:.4f} (sanity check vs. training-time best "
          f"MSE {cfg['best']['MSE']:.4f} -- should match closely)")

    ranking = []
    for c in range(n_channels):
        m = all_metrics(y_all, knockout_preds[c], dataset=dataset)
        ranking.append({"channel": c, "knockout_MSE": m["MSE"],
                        "delta_MSE": m["MSE"] - baseline_metrics["MSE"],
                        "knockout_CI": m["CI"]})
    ranking.sort(key=lambda r: -r["delta_MSE"])

    print(f"\n=== channel importance ranking (top {min(15, n_channels)} of {n_channels}) ===")
    print(f"{'rank':>4} {'chan':>4} {'delta_MSE':>10} {'knockout_MSE':>13} {'knockout_CI':>12}")
    for i, r in enumerate(ranking[:15]):
        print(f"{i:>4} {r['channel']:>4} {r['delta_MSE']:>10.4f} "
              f"{r['knockout_MSE']:>13.4f} {r['knockout_CI']:>12.4f}")

    top_channels = [r["channel"] for r in ranking[:args.top_k]]
    print(f"\n=== motif extraction for top {len(top_channels)} channels "
          f"(window length = receptive field = {rf} residues) ===")
    motifs = {}
    pwm_collect_n = max(args.motifs_per_channel, args.pwm_n) if args.save_pwm else args.motifs_per_channel
    seq_lens = np.array([len(s) for s in ds.prots])
    for c in top_channels:
        pos = argmax_pos[:, c]
        val = argmax_val[:, c]
        valid = (pos + rf <= seq_lens) & (pos >= args.min_window_pos)  # drop
        # padding spillover AND N-terminal/tag-region hits (see --min-window-pos)
        idx_sorted = np.argsort(-val)
        idx_sorted = idx_sorted[valid[idx_sorted]]
        # Davis reuses each protein across ~dozens of drug pairs, so raw
        # top-N would just repeat one protein's window; dedupe by the window
        # text itself so the printed motifs reflect distinct proteins.
        windows, acts, seen = [], [], set()
        for i in idx_sorted:
            w = ds.prots[i][pos[i]:pos[i] + rf]
            if w in seen:
                continue
            seen.add(w)
            windows.append(w)
            acts.append(float(val[i]))
            if len(windows) == pwm_collect_n:
                break

        consensus = ""
        if windows:
            for pos_in_window in range(rf):
                letters = [w[pos_in_window] for w in windows if len(w) > pos_in_window]
                if letters:
                    vals, counts = np.unique(letters, return_counts=True)
                    consensus += vals[np.argmax(counts)]
        print(f"\n-- channel {c} (delta_MSE={next(r['delta_MSE'] for r in ranking if r['channel']==c):.4f}) --")
        print(f"consensus: {consensus}")
        for w, a in list(zip(windows, acts))[:10]:
            print(f"    {w}   (activation={a:.3f})")
        motifs[c] = {"consensus": consensus,
                    "windows": windows[:args.motifs_per_channel],
                    "activations": acts[:args.motifs_per_channel],
                    "n_windows_for_pwm": len(windows)}
        if args.save_pwm:
            motifs[c]["pwm_windows"] = windows  # full set, used below to fit the PWM

    if args.out is None:
        args.out = os.path.join(RUNS_DIR, f"{args.tag}_interpret.json")
    with open(args.out, "w") as f:
        json.dump({"tag": args.tag, "dataset": dataset, "split": args.split,
                   "receptive_field": rf, "baseline_metrics": baseline_metrics,
                   "ranking": ranking, "top_channels": top_channels,
                   "motifs": {c: {k: v for k, v in m.items() if k != "pwm_windows"}
                             for c, m in motifs.items()}}, f, indent=2)
    print(f"\nWrote {args.out}")

    if args.save_pwm:
        from rule_features import fit_pwm, background_frequencies
        # background = overall amino-acid frequency across the TRAIN split (not
        # test), so the log-odds score isn't fit and evaluated on the same data.
        train_prots = CNNDataset(dataset, "train").prots
        background = background_frequencies(train_prots)
        pwm_out = {"tag": args.tag, "dataset": dataset, "receptive_field": rf,
                   "pseudocount": args.pwm_pseudocount,
                   "background": {a: float(p) for a, p in background.items()},
                   "channels": []}
        for c in top_channels:
            alphabet, log_odds = fit_pwm(motifs[c]["pwm_windows"], rf, background,
                                         pseudocount=args.pwm_pseudocount)
            pwm_out["channels"].append({
                "channel": c,
                "delta_MSE": next(r["delta_MSE"] for r in ranking if r["channel"] == c),
                "consensus": motifs[c]["consensus"],
                "n_windows": motifs[c]["n_windows_for_pwm"],
                "alphabet": alphabet,
                "log_odds": log_odds.tolist(),  # [rf, len(alphabet)]
            })
        pwm_path = os.path.join(RUNS_DIR, f"{args.tag}_pwm.json")
        with open(pwm_path, "w") as f:
            json.dump(pwm_out, f, indent=2)
        print(f"Wrote {pwm_path} ({len(pwm_out['channels'])} channel PWMs, "
              f"rf={rf}, alphabet_size={len(pwm_out['channels'][0]['alphabet'])})")


if __name__ == "__main__":
    main()
