#!/usr/bin/env bash
# ============================================================================
# Model "RegB" -- our most accurate CNN+CNN model. ~800K params.
# Two plain 1D-CNN towers (drug + protein), no attention, no graph.
# Reference run: 0.848 mean balanced accuracy on Davis (beats the DeepDTAGen
# 0.820 target at every threshold). See ../best_models/README.md for details.
#
# Usage:  ./run_regB.sh [dataset] [seed] [tag_suffix]
#   dataset     davis (default) | kiba | bindingdb
#   seed        4221 (default, matches the reference run) | any int
#   tag_suffix  name for this run's output files (default auto-built below)
# Example:  ./run_regB.sh kiba 7
# ============================================================================
set -e
cd "$(dirname "$0")/.."   # run from simple_dta/, so train.py + ../data/ resolve

DATASET="${1:-davis}"
SEED="${2:-4221}"
TAG="${3:-_regB_${DATASET}_s${SEED}}"
EPOCHS="${EPOCHS:-100}"   # override with: EPOCHS=4 ./run_regB.sh   (quick smoke test)

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
  --select-metric balacc \
  --seed "$SEED" \
  --tag-suffix "$TAG"

echo ""
echo "Done. Results in runs/cnn_${DATASET}${TAG}_summary.json"
echo "Re-score with other metrics / rebuild the leaderboard: python gen_results.py"
