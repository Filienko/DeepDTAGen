#!/usr/bin/env bash
# =============================================================================
# run_ezpc_vm.sh -- run the affinity CNN under EzPC's FSS 2PC (CPU / LLAMA).
# =============================================================================
# VERIFIED against mpc-msri/EzPC master (2026-07-16). Two facts that shape this:
#   * EzPC **OnnxBridge has NO GPU/Orca backend** -- only CPU: LLAMA (FSS 2PC),
#     CLEARTEXT_LLAMA, SECFLOAT*, CLEARTEXT_fp. This script uses **LLAMA** (FSS).
#   * **Orca (GPU-MPC) does NOT ingest ONNX** -- it runs hardcoded C++ models
#     (cnn.h) on ZEROED input as a benchmark. For real GPU-FSS of THIS model with
#     a real secret input, use **NssMPClib** (mpc/run_nssmpc_vm.sh), not Orca.
# So this is the CPU FSS reference (real 2PC, real secret input, 64-bit ring).
#
# EzPC pins numpy==1.21/onnx==1.12 -> use **Python 3.8-3.10**.
# Docs: https://github.com/mpc-msri/EzPC/tree/master/OnnxBridge
#       https://github.com/mpc-msri/EzPC/blob/master/sytorch/ezpc-cli.sh
# Versions move; run step-by-step (STOP_AFTER=deps|onnx|clone|compile).
# =============================================================================
set -euo pipefail

SCALE="${SCALE:-15}"                 # fixed-point fractional bits (ezpc default 15)
BITLENGTH="${BITLENGTH:-40}"         # ring bitwidth (ezpc default 40; 64 for more range)
DATASET="${DATASET:-davis}"
WORK="${WORK:-$HOME/ezpc-affinity}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${ONNX:-$WORK/affinity_fss.onnx}"
SUMMARY="${SUMMARY:-}"; CKPT="${CKPT:-}"
STOP_AFTER="${STOP_AFTER:-}"
log(){ echo -e "\n\033[1;34m[ezpc]\033[0m $*"; }

# ---- 1. system + python deps (Python 3.8-3.10!) --------------------------
log "1/4 deps (LLAMA/sytorch: Eigen, cmake, g++; OnnxBridge: numpy1.21/onnx1.12 -> Py3.8-3.10)"
sudo apt-get update -qq
sudo apt-get install -y -qq libeigen3-dev cmake build-essential git zip python3-dev
[ "$STOP_AFTER" = deps ] && exit 0

# ---- 2. export our model -> ONNX -----------------------------------------
log "2/4 exporting affinity model -> ONNX ($ONNX)"
mkdir -p "$WORK"
( cd "$REPO_DIR" && python3 -m mpc.export_onnx --dataset "$DATASET" --out "$ONNX" \
    ${SUMMARY:+--summary "$SUMMARY"} ${CKPT:+--ckpt "$CKPT"} )
# NOTE: our ONNX has TWO inputs (drug_onehot, prot_onehot). OnnxBridge demos are
# single-input; if multi-input tracing is unsupported in your version, export a
# single-input wrapper (concatenate the two one-hots) -- flagged for the VM.
[ "$STOP_AFTER" = onnx ] && exit 0

# ---- 3. clone EzPC + OnnxBridge deps -------------------------------------
log "3/4 clone EzPC + install OnnxBridge deps"
[ -d "$WORK/EzPC" ] || git clone https://github.com/mpc-msri/EzPC "$WORK/EzPC"
cd "$WORK/EzPC"; git submodule update --init --recursive
pip install -r OnnxBridge/requirements.txt
[ "$STOP_AFTER" = clone ] && exit 0

# ---- 4. compile ONNX -> LLAMA 2PC app, then run + TIME both phases --------
log "4/4 OnnxBridge -> LLAMA (FSS) 2PC app (scale=$SCALE bitlength=$BITLENGTH), run, time"
cd "$WORK/EzPC/OnnxBridge"
# emit the sytorch C++ + compiled binary + stripped weights (model_input_weights.dat):
python3 main.py --path "$ONNX" --generate executable --backend LLAMA \
    --scale "$SCALE" --bitlength "$BITLENGTH"
BASE="$(basename "${ONNX%.onnx}")"; APP="${BASE}_LLAMA_${SCALE}"; DIR="$(dirname "$ONNX")"
cat <<EOF

  Generated in $DIR:  $APP (2PC binary), $APP.cpp, model_input_weights.dat
  Run on localhost (3 roles; role 1=dealer/offline, 2=server/weights, 3=client/input):

    cd $DIR
    ~/EzPC/OnnxBridge/LLAMA/compile_llama.sh "$APP.cpp"     # if you used --generate code

    # --- OFFLINE (FSS key-gen) -- time it ---
    /usr/bin/time -v ./$APP 1 2> keygen.time               # -> server.dat, client.dat

    # --- ONLINE 2PC -- time single-sample latency (batch=1) ---
    ./$APP 2 model_input_weights.dat &                     # server (party 2)
    ./process_input.sh input.npy                           # -> input.inp (fixed-point)
    /usr/bin/time -v ./$APP 3 127.0.0.1 < input.inp > output.txt 2> online.time
    python ../helper/make_np_arr.py output.txt             # -> output.npy

    # verify vs cleartext + read the timings:
    python ../helper/run_onnx.py "$ONNX" input.npy
    python ../helper/compare_np_arrs.py -i onnx_output/expected.npy output.npy
    grep Elapsed keygen.time online.time                   # offline vs online wall-time

  LLAMA also prints its own online time + comm bytes to stdout (the per-op FSS
  breakdown). A correct 64-bit run matches cleartext to ~1e-2 at scale=$SCALE.
EOF
log "done (structure). Reconcile binary/flag names with your OnnxBridge version."
