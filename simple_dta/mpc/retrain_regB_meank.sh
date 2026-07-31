#!/usr/bin/env bash
# =============================================================================
# retrain_regB_meank.sh -- retrain regB with k-BIN AVERAGE pooling (--pool meank)
# and measure balAcc vs the max-pool baseline (~0.82) and the plain mean-pool floor
# (~0.758).
#
# Why: under FSS, max-pool is a log(L) DReLU tournament -- 114K of regB's 347K
# secure comparisons (~33%). Plain mean-pool deletes those (free/linear) but costs
# ~6 balAcc points because it averages the peak motif over the full length
# (SMILES 85, protein 1200). k-bin average pooling collapses L into `k` contiguous
# segments and flattens (adaptive_avg_pool1d(x,k).flatten(1)) -- still a fixed
# public linear map = FREE under FSS (0 comparisons, same 347K->233K win), but it
# keeps the coarse positional structure max captured. This sweeps k and reports the
# accuracy recovered for that same MPC saving.
#
# Env knobs:
#   EPOCHS  (100)          training epochs per run
#   SEEDS   ("7")          space-separated RNG seeds (seed 7 == the max baseline seed)
#   KS      ("2 4 8 16")   space-separated k (pool-bins) values to sweep
#   DATASET (davis)        dataset
#   CUDA    (unset)        GPU id -> adds "--cuda $CUDA" (CPU if unset; ~2-5h/run)
#
# Usage:
#   CUDA=0 bash mpc/retrain_regB_meank.sh                     # k in {2,4,8,16}, seed 7
#   CUDA=0 SEEDS="7 4221 42" KS="8" bash mpc/retrain_regB_meank.sh   # k=8, three seeds
# See mpc/RETRAIN_REGB.md for the max/mean baseline context.
# =============================================================================
set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-7}"
KS="${KS:-2 4 8 16}"
DATASET="${DATASET:-davis}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # the simple_dta dir (holds train.py)
cd "$HERE"

CUDA_ARG=()
[ -n "${CUDA:-}" ] && CUDA_ARG=(--cuda "$CUDA")

log(){ echo -e "\n\033[1;36m[regB-meank]\033[0m $*"; }

# regB's exact flags (mpc/configs.py CONFIGS["regB"]) with --pool meank; $1=k, $2=seed.
train_regB_meank(){
  local k="$1" seed="$2"
  local suffix="_regB_meank${k}_s${seed}"
  log "training regB pool=meank k=$k seed=$seed  (tag cnn_${DATASET}${suffix}, epochs=$EPOCHS)"
  python train.py --model cnn --dataset "$DATASET" \
    --drug-channels 16,32,48 --prot-channels 32,64,96 --prot-dilations 1,2,4 \
    --head-dim 1024 --head-layers 2 --dropout 0.1 --weight-decay 0 \
    --select-metric balacc --epochs "$EPOCHS" --seed "$seed" \
    --pool meank --pool-bins "$k" --tag-suffix "$suffix" "${CUDA_ARG[@]}"
}

for seed in $SEEDS; do
  for k in $KS; do
    train_regB_meank "$k" "$seed"
  done
done

log "summary"
SEEDS="$SEEDS" KS="$KS" DATASET="$DATASET" python - <<'PY'
import json, os
seeds = os.environ["SEEDS"].split(); ks = os.environ["KS"].split()
dataset = os.environ["DATASET"]; runs = os.path.join(os.getcwd(), "runs")

def balacc(tag):
    p = os.path.join(runs, f"{tag}_summary.json")
    if not os.path.exists(p): return None
    return json.load(open(p))["best"].get("balacc_mean")

def show(label, tag):
    v = balacc(tag)
    print(f"  {label:<28} {'(missing)' if v is None else f'{v:.4f}'}")

print("\n  baselines:")
for s in seeds:
    show(f"regB max     s{s}", f"cnn_{dataset}_regB_max_s{s}")
    show(f"regB mean    s{s}", f"cnn_{dataset}_regB_mean_s{s}")
print("\n  k-bin average pool (target: >=0.80 floor, >=0.831 stretch):")
for s in seeds:
    for k in ks:
        show(f"regB meank k={k} s{s}", f"cnn_{dataset}_regB_meank{k}_s{s}")
print("\n  MPC payoff (any k): 347K -> 233K secure comparisons (max-pool tree gone).")
print("  Confirm: python mpc_cost.py --model cnn --dataset davis --drug-channels 16,32,48 \\")
print("    --prot-channels 32,64,96 --prot-dilations 1,2,4 --pool meank --pool-bins <k> \\")
print("    --head-dim 1024 --head-layers 2")
PY

log "done -- meank checkpoints in runs/ flow through the MPC tools (bit-exact port,"
log "no max-pool DReLUs). See mpc/RETRAIN_REGB.md."
