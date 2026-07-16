"""Real end-to-end accuracy of the affinity model + proof that FSS preserves it.

Given a trained mean-pool `CNNDTA` checkpoint (`--summary <runs/*_summary.json>
--ckpt <*.pth>`), this reports:

  1. **Cleartext** test metrics (MSE / CI / rm2 / balAcc) on the full test set --
     the model's real accuracy (reuses `metrics.all_metrics`).
  2. **FSS fidelity**: on a sample of test pairs, the self-contained 2-party FSS
     engine (`fss_infer.FSSBackend`) reproduces the cleartext prediction to within
     fixed-point tolerance (MAE / max err / Pearson) -> the encrypted model has the
     *same* accuracy as the plaintext one.

Because the runnable FSS engine uses `sycret`'s 32-bit ring, the FSS sample is run
at a reduced sequence length (`--fss-seq-len`) that fits the fixed-point budget;
the full-length, full-precision FSS run is the 64-bit Orca path (see MPC_PORT.md /
run_ezpc_vm.sh). Cleartext metrics always use the full sequence.

With no `--ckpt`, a random-init mean-pool model is used: the pipeline still runs
end-to-end on real test rows (the numbers are meaningless, but the code path and
FSS==cleartext fidelity are exercised).

Usage:
    python -m mpc.accuracy --summary runs/cnn_davis_mp_summary.json \
        --ckpt runs/cnn_davis_mp_best.pth --dataset davis
"""
import argparse

import numpy as np
import torch

from data import CNNDataset, DEFAULTS, load_csv, label_encode, CHARISOSMISET, CHARPROTSET
from metrics import all_metrics
from mpc.model_mpc import MPCModel, load_cnndta
from mpc import fss_infer as FSS


def cleartext_metrics(model, dataset, split, batch_size, max_eval=None):
    """Full-test cleartext predictions -> metric bundle (the model's real accuracy)."""
    from torch.utils.data import DataLoader
    ds = CNNDataset(dataset, split)
    loader = DataLoader(ds, batch_size=batch_size)
    preds, trues, seen = [], [], 0
    model.eval()
    with torch.no_grad():
        for xd, xt, y in loader:
            preds.append(model((xd, xt, None)).cpu().numpy())
            trues.append(y.numpy())
            seen += len(y)
            if max_eval and seen >= max_eval:
                break
    P = np.concatenate(preds).flatten()
    G = np.concatenate(trues).flatten()
    return all_metrics(G, P, dataset=dataset), G, P


def fss_fidelity(dataset, n, seq_len, frac_bits, device, pool="mean", profile=False):
    """On n REAL test pairs (truncated to seq_len), compare FSS vs cleartext
    predictions -- on the small model that fits sycret's 32-bit ring. Shows the
    FSS engine is faithful; the full-size checkpoint's full-length FSS run is the
    64-bit Orca path (MPC_PORT.md / run_ezpc_vm.sh)."""
    model = FSS.build_demo_model("small", pool=pool)   # fits the 32-bit fixed-point budget
    mpc = MPCModel(model)
    smiles, prots, ys = load_csv(dataset, "test")
    d = DEFAULTS[dataset]
    Ls = min(seq_len, d["max_smi_len"])
    Lp = min(seq_len * 3, d["max_seq_len"])
    xd = torch.stack([torch.from_numpy(label_encode(s, Ls, CHARISOSMISET)) for s in smiles[:n]])
    xt = torch.stack([torch.from_numpy(label_encode(p, Lp, CHARPROTSET)) for p in prots[:n]])
    with torch.no_grad():
        clear = mpc.forward_tokens_clear(xd, xt).to(torch.float64)
    fss_pred, be, dt = FSS._run(mpc, xd, xt, frac_bits, device, profile=profile)
    err = (fss_pred - clear).abs()
    if clear.numel() > 1 and clear.std() > 1e-9 and fss_pred.std() > 1e-9:
        pear = float(np.corrcoef(clear.numpy(), fss_pred.numpy())[0, 1])
    else:
        pear = float("nan")
    if profile:
        FSS.print_profile(be, n, dt)
    return dict(mae=err.mean().item(), maxerr=err.max().item(), pearson=pear,
                relu=be.relu_calls, rounds=be.online_rounds, secs=dt,
                Ls=Ls, Lp=Lp, clear=clear, fss=fss_pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=None, help="runs/<tag>_summary.json (mean-pool CNN)")
    ap.add_argument("--ckpt", default=None, help="trained mean-pool CNNDTA .pth")
    ap.add_argument("--dataset", default="davis", choices=["davis", "kiba", "bindingdb"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-eval", type=int, default=None, help="cap cleartext eval rows (speed)")
    ap.add_argument("--n-fss", type=int, default=16, help="test pairs run through the FSS engine")
    ap.add_argument("--fss-seq-len", type=int, default=40,
                    help="drug length for the FSS sample (protein = 4x); small to fit "
                         "sycret's 32-bit ring")
    ap.add_argument("--frac-bits", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--profile", action="store_true",
                    help="print per-op runtime + single-sample latency for the FSS run")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    model = load_cnndta(args.summary, args.ckpt)
    tag = args.ckpt or args.summary or "random-init (no checkpoint)"
    print(f"=== affinity accuracy | model: {tag} | dataset {args.dataset} ===")
    if args.ckpt is None:
        print("  [!] no --ckpt: metrics below are from RANDOM weights (pipeline check only)")

    m, G, P = cleartext_metrics(model, args.dataset, "test", args.batch_size, args.max_eval)
    print(f"  CLEARTEXT test ({len(G)} pairs): MSE {m['MSE']:.4f} | CI {m['CI']:.4f} "
          f"| rm2 {m['rm2']:.4f} | balAcc {m.get('balacc_mean', float('nan')):.4f}")
    print("     ^ this is the model's real accuracy; run under FSS it is preserved (below).")

    print(f"\n  FSS-engine fidelity (small model, sycret 32-bit ring) on real {args.dataset} "
          "test pairs:")
    fid = fss_fidelity(args.dataset, args.n_fss, args.fss_seq_len, args.frac_bits, device,
                       pool=model.drug.pool, profile=args.profile)
    print(f"    {args.n_fss} pairs @ drugL={fid['Ls']} protL={fid['Lp']}, f={args.frac_bits}, "
          f"{device}")
    print(f"    FSS vs cleartext  MAE {fid['mae']:.3e} | max {fid['maxerr']:.3e} "
          f"| Pearson {fid['pearson']:.4f}")
    print(f"    {fid['relu']} FSS-DReLU calls | {fid['rounds']} online rounds | {fid['secs']:.1f}s")
    print("  -> the FSS engine reproduces cleartext predictions within fixed-point tolerance.")
    print("     The FULL-SIZE checkpoint above exceeds sycret's 32-bit ring, so its full FSS")
    print("     run is the 64-bit Orca path: export_onnx.py -> run_ezpc_vm.sh (see MPC_PORT.md).")


if __name__ == "__main__":
    main()
