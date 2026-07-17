#!/usr/bin/env bash
# =============================================================================
# run_orca_regB.sh -- benchmark regB's GPU-FSS online time under Orca, to compare
# against the ~2.6s MP-SPDZ CPU online baseline (see mpc/ONLINE_TIME.md).
#
# Orca does NOT ingest ONNX -- it runs C++ models from cnn.h on zeroed input (fine:
# timing is input-independent). So this: builds EzPC/GPU-MPC, adds regB to cnn.h
# (from mpc/orca/regB_cnn.h), `make orca`, runs the benchmark, prints the online time.
#
# Requires: NVIDIA GPU (Volta+ / sm_70+), CUDA toolkit ~11.7, CMake>=3.17, g++-9.
# The cnn.h edit + sytorch class names are version-specific -- do the integration by
# hand the first time (this script guides it). Run step-by-step (STOP_AFTER=clone|build).
# Docs: https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC/experiments/orca
# =============================================================================
set -euo pipefail

WORK="${WORK:-$HOME/orca-regB}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"     # the simple_dta dir
GPU_ARCH="${GPU_ARCH:-80}"                        # your GPU sm_XX (70 V100, 80 A100, 86 RTX30, 89 RTX40, 90 H100)
CUDA_VERSION="${CUDA_VERSION:-11.7}"
CUTLASS_BRANCH="${CUTLASS_BRANCH:-main}"
STOP_AFTER="${STOP_AFTER:-}"
log(){ echo -e "\n\033[1;36m[orca]\033[0m $*"; }

command -v nvidia-smi >/dev/null || { echo "Orca needs an NVIDIA GPU"; exit 1; }
nvidia-smi -L

log "1/4 clone EzPC + GPU-MPC"
[ -d "$WORK/EzPC" ] || git clone https://github.com/mpc-msri/EzPC "$WORK/EzPC"
cd "$WORK/EzPC/GPU-MPC"; git submodule update --init --recursive || true
[ "$STOP_AFTER" = clone ] && exit 0

log "2/4 add regB to experiments/orca/cnn.h  (MANUAL: version-specific)"
cat <<EOF
  Integrate regB into the sytorch model registry:
    1. Copy the RegB<T> class from:
         $REPO_DIR/mpc/orca/regB_cnn.h
       into  $WORK/EzPC/GPU-MPC/experiments/orca/cnn.h
    2. In getCNN<T>(name):   else if (name == "RegB") return new RegB<T>();
    3. Set the protein convs' dilations to {1,1},{1,2},{1,4} and confirm the head
       input dim (144 for regB's 48+96 towers) against your checkpoint's FC1 shape.
  Reconcile Conv2D/MaxPool2D/FC ctor signatures with the existing VGG/ResNet cases.
EOF
read -p "  Press ENTER once regB is added to cnn.h (or Ctrl-C to stop)... " _

log "3/4 build Orca"
export CUDA_VERSION GPU_ARCH
sh setup.sh "$CUTLASS_BRANCH"
make orca
[ "$STOP_AFTER" = build ] && exit 0

log "4/4 run the regB benchmark + read the ONLINE time"
cd experiments/orca
cat <<EOF
  Run one of (per the orca README):
     python run_experiment.py --party 0 --all true     # this host
     python run_experiment.py --party 1 --all true     # the peer
  or the raw inference binary with the regB model name + bitwidth 64 + scale 13:
     ./orca_inference RegB 64 13 <role 0=dealer/1=evaluator> <party> <keyDir>
  Read the printed per-op GPU-FSS timing; the ONLINE (evaluator) time is the number
  to compare against the ~2.6s MP-SPDZ CPU online. Predicted: Orca online << 2.6s,
  with DReLU + max-pool comparisons as the dominant (now GPU-parallel) cost.
EOF
log "done. See mpc/ONLINE_TIME.md for the 347K/114K comparison breakdown + analysis."
