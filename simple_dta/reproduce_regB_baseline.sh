#!/usr/bin/env bash
# =============================================================================
# reproduce_regB_baseline.sh -- reproduce the RegB 0.848 Davis balAcc baseline
# from scratch, identically: same architecture, same hyperparameters, same seed,
# same CPU backend the reference was trained on.
#
# WHY THIS EXISTS / WHAT WAS VERIFIED
#   * Architecture + config are byte-identical to the reference (the reference
#     `state_dict` loads into models.py `CNNDTA` with strict=True, 800,417 params;
#     every RegB flag matches best_models/run_regB.sh and the committed
#     runs/cnn_davis_regB_d1_wd0_summary.json).
#   * Preprocessing (data.py) and the balAcc metric (metrics.py) are byte-identical
#     to the code that produced the reference -- and, decisively, the *frozen*
#     reference weights re-score to exactly balAcc 0.8481 / MSE 0.2905 through this
#     repo's current pipeline. So eval/preprocessing/metric are NOT a source of drift.
#   * The one thing that moves a *fresh training run* is the execution BACKEND:
#     the reference was trained on CPU (torch 1.12.1); training the same seed on GPU
#     diverges (different dropout RNG generator + non-associative float reduction
#     order), landing ~3-4 balAcc points off. Hence: this script pins CPU.
#
# HONEST CAVEAT ON "IDENTICALLY"
#   Bit-exact 0.848 also requires the reference's torch version (1.12.1). This box
#   has torch 2.13, so expect a result CLOSE to 0.848 (~0.82-0.85) but not
#   guaranteed to the digit -- the residual is the torch-version delta plus balAcc's
#   own best-epoch selection noise (~+-0.06 epoch-to-epoch). For a truly bit-exact
#   run, recreate the pinned environment first (see PINNED-ENV NOTE at the bottom),
#   then run this script -- it will pick up whichever torch is active.
#   If you only need the 0.848 numbers, don't retrain at all: just evaluate the
#   committed weights (see EVAL-ONLY NOTE at the bottom).
#
# USAGE
#   bash reproduce_regB_baseline.sh                 # CPU, seed 4221, 100 epochs
#   SEED=4221 EPOCHS=100 bash reproduce_regB_baseline.sh
#   EPOCHS=1 bash reproduce_regB_baseline.sh        # quick smoke test (~1 min)
# Env knobs: SEED (4221), EPOCHS (100), DATASET (davis).
# Runtime: ~60 s/epoch on this box's CPU -> ~100 min for the full 100-epoch run.
# =============================================================================
set -euo pipefail

SEED="${SEED:-4221}"           # 4221 = the reference run's seed
EPOCHS="${EPOCHS:-100}"
DATASET="${DATASET:-davis}"
TAG="_regB_${DATASET}_repro_s${SEED}_cpu"

# --- force the CPU backend (the reference's backend) -------------------------
# Empty CUDA_VISIBLE_DEVICES hides every GPU, so even if a flag slipped through,
# torch cannot use CUDA. We also pass NO --cuda flag, so train.py -> device=cpu.
export CUDA_VISIBLE_DEVICES=""
# Reproducibility hygiene (hash seed; PyTorch RNG is seeded inside train.py.set_seed).
export PYTHONHASHSEED=0

HERE="$(cd "$(dirname "$0")" && pwd)"   # simple_dta/ (holds train.py, ../data/)
cd "$HERE"

echo "=============================================================="
echo " Reproducing RegB baseline  (dataset=$DATASET seed=$SEED epochs=$EPOCHS)"
echo " backend: CPU  |  torch: $(python -c 'import torch;print(torch.__version__)')"
echo "=============================================================="

# RegB's EXACT flags -- identical to best_models/run_regB.sh and the committed
# reference config. Only difference from a GPU run: no --cuda (=> CPU).
python train.py \
  --model cnn \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --embed-dim 128 \
  --drug-channels 16,32,48 \
  --prot-channels 32,64,96 \
  --prot-dilations 1,2,4 \
  --head-dim 1024 \
  --head-layers 2 \
  --dropout 0.1 \
  --weight-decay 0 \
  --select-metric balacc \
  --seed "$SEED" \
  --tag-suffix "$TAG"

# --- compare the fresh run to the 0.848 reference, per threshold -------------
echo ""
echo "=== result vs the 0.848 reference (per threshold) ==="
DATASET="$DATASET" TAG="$TAG" python - <<'PY'
import json, os
ds  = os.environ["DATASET"]; tag = os.environ["TAG"]
runs = os.path.join(os.getcwd(), "runs")

def load(name):
    p = os.path.join(runs, f"{name}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None

mine = load(f"cnn_{ds}{tag}")   # tag already starts with '_'
ref  = load("cnn_davis_regB_d1_wd0")   # the committed 0.848 reference (Davis only)

if mine is None:
    print("  (no fresh summary found -- did training finish?)"); raise SystemExit

mb = mine["best"]
print(f"  reproduced balAcc_mean = {mb['balacc_mean']:.4f}   "
      f"(MSE {mb['MSE']:.4f}, best epoch {mb['epoch']})")
if ref is not None and ds == "davis":
    rb = ref["best"]
    print(f"  reference  balAcc_mean = {rb['balacc_mean']:.4f}   (MSE {rb['MSE']:.4f})")
    print(f"  delta                  = {mb['balacc_mean']-rb['balacc_mean']:+.4f}")
    thrs = sorted(mb["balacc"], key=lambda t: float(t))
    print(f"\n  {'thr':>5}{'reproduced':>12}{'reference':>11}{'delta':>9}")
    for t in thrs:
        m = mb["balacc"][t]; r = rb["balacc"].get(t)
        ms = f"{m:.4f}" if m == m else "nan"
        rs = f"{r:.4f}" if (r is not None and r == r) else "nan"
        dl = f"{m-r:+.4f}" if (m == m and r is not None and r == r) else "-"
        print(f"  {t:>5}{ms:>12}{rs:>11}{dl:>9}")
    print("\n  (small deltas = torch-version residual + balAcc best-epoch noise;")
    print("   the CPU backend removes the large GPU-vs-CPU divergence.)")
PY

# =============================================================================
# EVAL-ONLY NOTE -- if you just want the 0.848 numbers without retraining:
#   python - <<'PY'
#   import torch, numpy as np
#   from torch.utils.data import DataLoader
#   from data import CNNDataset; from models import CNNDTA; from metrics import all_metrics
#   m = CNNDTA(embed_dim=128, pool="max", head_dim=1024, head_layers=2,
#              drug_channels=[16,32,48], prot_channels=[32,64,96],
#              prot_dilations=[1,2,4], drug_kernel=4, prot_kernel=8, dropout=0.1)
#   m.load_state_dict(torch.load("best_models/weights/regB_davis_reference.pth",
#                                map_location="cpu")); m.eval()
#   P,G=[],[]
#   for xd,xt,y in DataLoader(CNNDataset("davis","test"), batch_size=256):
#       P.append(m((xd,xt,None)).detach().numpy()); G.append(y.numpy())
#   print(all_metrics(np.concatenate(G).ravel(), np.concatenate(P).ravel(), "davis")["balacc_mean"])
#   PY
#   -> 0.8481, exactly.
#
# PINNED-ENV NOTE -- for a BIT-EXACT retrain (matches torch 1.12.1 too):
#   conda env create -f ../environment.yml    # torch==1.12.1 (CPU), numpy 1.23, pandas 1.5
#   conda activate DeepDTAGen
#   bash reproduce_regB_baseline.sh
#   (torch 1.12 CPU installs fine on any box -- it does not need the GPU's CUDA.)
# =============================================================================
