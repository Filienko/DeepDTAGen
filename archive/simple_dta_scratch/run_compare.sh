#!/bin/bash
# Generic CNN-then-GNN comparison for one dataset.
#   bash run_compare.sh <dataset> [epochs]
# Designed to be launched fully detached (setsid/nohup) so it survives logout.
# Per-model logs: runs/cnn_<dataset>.log, runs/gnn_<dataset>.log
# Thread cap is inherited from OMP_NUM_THREADS/MKL_NUM_THREADS in the environment,
# so two datasets can run in parallel without oversubscribing the 48 cores.
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepDTAGen

DATASET=${1:?usage: run_compare.sh <dataset> [epochs]}
EPOCHS=${2:-100}
mkdir -p runs
echo "START $(date) | dataset=$DATASET | epochs=$EPOCHS | pid=$$ | OMP=${OMP_NUM_THREADS:-default}"

echo "######## CNN ${DATASET} (${EPOCHS} epochs) ########"
python -u train.py --model cnn --dataset "$DATASET" --epochs "$EPOCHS" --eval-interval 10 \
    > "runs/cnn_${DATASET}.log" 2>&1
echo "CNN done $(date)"

echo "######## GNN ${DATASET} (${EPOCHS} epochs) ########"
python -u train.py --model gnn --dataset "$DATASET" --epochs "$EPOCHS" --eval-interval 10 \
    > "runs/gnn_${DATASET}.log" 2>&1
echo "GNN done $(date)"

echo "ALL DONE $(date)"
