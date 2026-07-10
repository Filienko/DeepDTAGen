#!/usr/bin/env bash
# ============================================================================
# Model "F" -- our best MPC candidate. ~1.08M params.
# Drug side: a plain GCN over the molecular graph (no attention in the graph
# conv itself). Protein side: a plain CNN. Fusion: CROSS-ATTENTION between the
# two towers using LINEAR (softmax-free) attention -- no exp/divide anywhere
# in the model, which is what makes it MPC-friendly. Reference run: 0.831
# mean balanced accuracy on Davis, ~1pp below our best softmax-attention
# model but with none of softmax's MPC cost. See ../best_models/README.md.
#
# Usage:  ./run_F.sh [dataset] [seed] [tag_suffix]
#   dataset     davis (default) | kiba | bindingdb
#   seed        4221 (default, matches the reference run) | any int
#   tag_suffix  name for this run's output files (default auto-built below)
# Example:  ./run_F.sh davis 7
#
# NOTE: this model is much slower to train than RegB/ConfigA (graph batching +
# attention). Budget several hours on Davis; KIBA is much bigger again.
# ============================================================================
set -e
cd "$(dirname "$0")/.."   # run from simple_dta/, so train.py + ../data/ resolve

DATASET="${1:-davis}"
SEED="${2:-4221}"
TAG="${3:-_F_${DATASET}_s${SEED}}"
EPOCHS="${EPOCHS:-100}"   # override with: EPOCHS=4 ./run_F.sh   (quick smoke test)

python train.py \
  --model attn \
  --drug-encoder gnn \
  --protein-encoder cnn \
  --fusion cross \
  --attn-kind linear \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --select-metric balacc \
  --seed "$SEED" \
  --tag-suffix "$TAG"

echo ""
echo "Done. Results in runs/attn_${DATASET}${TAG}_summary.json"
echo "Re-score with other metrics / rebuild the leaderboard: python gen_results.py"
