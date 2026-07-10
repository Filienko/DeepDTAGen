#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- reproduce the MPC/FSS PoC environment and run its checks.
# Idempotent: safe to re-run. From repo root or anywhere.
#   bash simple_dta/mpc/setup_env.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # simple_dta/mpc
SIMPLE_DTA="$(dirname "$HERE")"                # simple_dta
ROOT="$(dirname "$SIMPLE_DTA")"                # repo root

echo "[setup] 1/4 python deps"
python3 -m pip install -q -r "$HERE/requirements.txt"

echo "[setup] 2/4 extracting datasets from data.rar (if needed)"
if [ ! -f "$ROOT/data/davis_test.csv" ]; then
  command -v unrar >/dev/null || sudo apt-get install -y -qq unrar
  ( cd "$ROOT/data" && unrar x -o+ ../data.rar >/dev/null )
  # data.rar unpacks into data/data/*.csv -> flatten to data/*.csv (where data.py looks)
  [ -d "$ROOT/data/data" ] && mv -n "$ROOT/data/data/"*.csv "$ROOT/data/" 2>/dev/null || true
fi
python3 -c "import sys; sys.path.insert(0,'$SIMPLE_DTA'); from data import load_csv; \
  print('[setup] davis test rows:', len(load_csv('davis','test')[0]))"

echo "[setup] 3/4 FSS correctness checks"
( cd "$SIMPLE_DTA" && python3 -m mpc.test_fss )

echo "[setup] 4/4 smoke: op-cost + FSS sweep + ONNX export"
( cd "$SIMPLE_DTA" \
  && python3 mpc_cost.py --model cnn --dataset davis >/dev/null \
  && python3 -m mpc.fss_infer --frac-bits 6 >/dev/null \
  && python3 -m mpc.export_onnx --dataset davis >/dev/null )

echo "[setup] DONE -- environment ready."
echo "  next: python -m mpc.accuracy --summary runs/<mean-pool>_summary.json --ckpt <.pth>"
echo "  GPU FSS (Orca): bash simple_dta/mpc/run_ezpc_vm.sh   (see mpc/INSTALL.md)"
