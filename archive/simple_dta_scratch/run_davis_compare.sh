#!/bin/bash
# Full Davis comparison: CNN then GNN (sequential to avoid CPU contention).
# Designed to be launched fully detached (setsid/nohup) so it survives logout.
# Per-model logs: runs/cnn_davis.log, runs/gnn_davis.log
# Combined log : whatever you redirect this script's stdout to.
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepDTAGen

EPOCHS=${1:-100}
mkdir -p runs
echo "START $(date) | epochs=$EPOCHS | pid=$$"

echo "######## CNN davis (${EPOCHS} epochs) ########"
python -u train.py --model cnn --dataset davis --epochs "$EPOCHS" --eval-interval 10 \
    > runs/cnn_davis.log 2>&1
echo "CNN done $(date)"

echo "######## GNN davis (${EPOCHS} epochs) ########"
python -u train.py --model gnn --dataset davis --epochs "$EPOCHS" --eval-interval 10 \
    > runs/gnn_davis.log 2>&1
echo "GNN done $(date)"

echo "ALL DONE $(date)"
