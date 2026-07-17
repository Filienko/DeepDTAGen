#!/usr/bin/env bash
# =============================================================================
# retrain_regB_meanpool.sh -- retrain regB with MEAN pool and measure the accuracy
# drop vs the MAX-pool baseline (balAcc 0.848).
#
# Why: under FSS, max-pool is a log(L) DReLU tournament -- it is 114K of regB's
# 347K secure comparisons (~33%). Mean-pool is linear/local (FREE under FSS). So
# dropping max->mean removes that whole comparison stage; this script measures what
# accuracy that costs. (ReLU is intrinsic to the conv tower and can't be removed.)
#
# It trains TWO models with the SAME seed for a fair comparison:
#   1. regB max-pool  (the 0.848 baseline; SKIP_MAX=1 to reuse an existing run)
#   2. regB mean-pool (the FSS-cheaper candidate)
# then prints both balAcc and the drop. Only --pool differs between the two runs;
# every other flag matches mpc/configs.py CONFIGS["regB"].
#
# Env knobs:
#   EPOCHS   (100)     training epochs per run
#   SEED     (7)       RNG seed (same for both runs -> fair drop)
#   DATASET  (davis)   dataset
#   CUDA     (unset)   GPU id -> adds "--cuda $CUDA" (CPU if unset; ~2-5h/run on CPU)
#   SKIP_MAX (unset)   set to 1 to skip the max-pool baseline (reuse a prior run)
#
# Usage:
#   bash mpc/retrain_regB_meanpool.sh                 # both runs, CPU
#   CUDA=0 bash mpc/retrain_regB_meanpool.sh          # both runs on GPU 0
#   CUDA=0 SKIP_MAX=1 bash mpc/retrain_regB_meanpool.sh   # mean-pool only
# See mpc/RETRAIN_REGB.md for the full walkthrough.
# =============================================================================
set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-7}"
DATASET="${DATASET:-davis}"
SKIP_MAX="${SKIP_MAX:-}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # the simple_dta dir (holds train.py)
cd "$HERE"

CUDA_ARG=()
[ -n "${CUDA:-}" ] && CUDA_ARG=(--cuda "$CUDA")

log(){ echo -e "\n\033[1;36m[retrain-regB]\033[0m $*"; }

# regB's exact flags (mpc/configs.py CONFIGS["regB"]); $1 = pool mode.
train_regB(){
  local pool="$1"
  local suffix="_regB_${pool}_s${SEED}"
  log "training regB pool=$pool  (tag cnn_${DATASET}${suffix}, epochs=$EPOCHS, seed=$SEED)"
  python train.py --model cnn --dataset "$DATASET" \
    --drug-channels 16,32,48 --prot-channels 32,64,96 --prot-dilations 1,2,4 \
    --head-dim 1024 --head-layers 2 --dropout 0.1 --weight-decay 0 \
    --select-metric balacc --epochs "$EPOCHS" --seed "$SEED" \
    --pool "$pool" --tag-suffix "$suffix" "${CUDA_ARG[@]}"
}

if [ "$SKIP_MAX" = "1" ]; then
  log "SKIP_MAX=1 -> skipping the max-pool baseline (expecting an existing run)"
else
  train_regB max
fi
train_regB mean

log "comparison"
SEED="$SEED" DATASET="$DATASET" python - <<'PY'
import json, os, sys
seed = os.environ["SEED"]; dataset = os.environ["DATASET"]
runs = os.path.join(os.getcwd(), "runs")   # cwd is simple_dta (set by the shell script)

def balacc(pool):
    p = os.path.join(runs, f"cnn_{dataset}_regB_{pool}_s{seed}_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["best"].get("balacc_mean")

mx, mn = balacc("max"), balacc("mean")
print(f"\n  regB max-pool  balAcc = {mx if mx is None else f'{mx:.4f}'}")
print(f"  regB mean-pool balAcc = {mn if mn is None else f'{mn:.4f}'}")
if mx is not None and mn is not None:
    drop = mx - mn
    pct = 100.0 * drop / mx if mx else float("nan")
    print(f"  drop (max - mean)     = {drop:+.4f}  ({pct:+.2f}%)")
    print("\n  MPC payoff: mean-pool removes the ~114K/347K max-pool DReLU comparisons")
    print("  (regB 347K -> 233K secure comparisons). Confirm with:")
    print("    python mpc_cost.py --model cnn --dataset davis --drug-channels 16,32,48 \\")
    print("      --prot-channels 32,64,96 --prot-dilations 1,2,4 --pool mean \\")
    print("      --head-dim 1024 --head-layers 2")
else:
    missing = [p for p, v in (("max", mx), ("mean", mn)) if v is None]
    print(f"\n  (missing summary for: {', '.join(missing)} -- run without SKIP_MAX, or check runs/)")
PY

log "done. The mean-pool checkpoint (runs/cnn_${DATASET}_regB_mean_s${SEED}_best.pth)"
log "flows through the MPC tools with 114K fewer DReLUs -- see mpc/RETRAIN_REGB.md."
