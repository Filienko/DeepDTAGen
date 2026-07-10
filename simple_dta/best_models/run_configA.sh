#!/usr/bin/env bash
# ============================================================================
# Model "ConfigA" -- our small, single-layer-head CNN+CNN model. ~267K params
# (about a third the size of RegB). Same two-tower CNN family as RegB, but
# each tower is squeezed to 32 dims before a single wide FC layer instead of
# a deep head. Reference run: 0.795 mean balanced accuracy on Davis --
# within ~5pp of RegB at ~3x fewer params. See ../best_models/README.md.
#
# Usage:  ./run_configA.sh [dataset] [seed] [tag_suffix]
#   dataset     davis (default) | kiba | bindingdb
#   seed        4221 (default, matches the reference run) | any int
#   tag_suffix  name for this run's output files (default auto-built below)
# Example:  ./run_configA.sh davis 13
# ============================================================================
set -e
cd "$(dirname "$0")/.."   # run from simple_dta/, so train.py + ../data/ resolve

DATASET="${1:-davis}"
SEED="${2:-4221}"
TAG="${3:-_configA_${DATASET}_s${SEED}}"
EPOCHS="${EPOCHS:-100}"   # override with: EPOCHS=4 ./run_configA.sh   (quick smoke test)

python train.py \
  --model cnn \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --embed-dim 128 \
  --proj-dim 32 \
  --head-dim 1536 \
  --head-layers 1 \
  --select-metric balacc \
  --seed "$SEED" \
  --tag-suffix "$TAG"

echo ""
echo "Done. Results in runs/cnn_${DATASET}${TAG}_summary.json"
echo "Re-score with other metrics / rebuild the leaderboard: python gen_results.py"
