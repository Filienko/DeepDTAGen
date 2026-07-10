#!/usr/bin/env bash
# =============================================================================
# run_ezpc_vm.sh -- build & run the affinity CNN under EzPC FSS 2PC on a larger VM
# =============================================================================
# Runs the *production* GPU-optimized FSS path (Orca / EzPC GPU-MPC) that this
# repo's sandbox can't (no GPU, heavy C++/CUDA build). It:
#   1. installs system + Python deps,
#   2. exports the FSS-ready affinity model to ONNX (mpc/export_onnx.py),
#   3. compiles it to a 2-party secure app with EzPC OnnxBridge,
#   4. runs both parties on localhost and diffs the secure output vs cleartext.
#
# Backends:
#   LLAMA  -- CPU FSS (64-bit ring). No GPU needed; good for a correctness run.
#   ORCA   -- GPU FSS (CUDA). The genuinely GPU-optimized path; needs an NVIDIA GPU.
#
# IMPORTANT: EzPC's exact commands/flags move over time. This script encodes the
# canonical flow and pins nothing upstream -- if a step drifts, consult the READMEs:
#   https://github.com/mpc-msri/EzPC/tree/master/OnnxBridge
#   https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC          (Orca/SIGMA, GPU)
#   https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC/ext/sytorch
# Run it step-by-step the first time (set STOP_AFTER=...) rather than blind.
# =============================================================================
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
BACKEND="${BACKEND:-LLAMA}"          # LLAMA (CPU) | ORCA (GPU)
SCALE="${SCALE:-13}"                 # fixed-point fractional bits (13-16 typical)
BITLENGTH="${BITLENGTH:-64}"         # ring bitwidth (64 recommended)
DATASET="${DATASET:-davis}"
WORK="${WORK:-$HOME/ezpc-affinity}"  # working dir for clones/artifacts
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # the simple_dta dir
ONNX="${ONNX:-$WORK/affinity_fss.onnx}"
SUMMARY="${SUMMARY:-}"               # runs/<tag>_summary.json (mean-pool CNN)
CKPT="${CKPT:-}"                     # trained mean-pool CNNDTA .pth
STOP_AFTER="${STOP_AFTER:-}"         # deps|onnx|compile|keygen|run (for stepping)

log(){ echo -e "\n\033[1;34m[ezpc]\033[0m $*"; }
step(){ [ "$STOP_AFTER" = "$1" ] && { log "STOP_AFTER=$1 reached"; exit 0; } || true; }

# ---- 1. system + python deps ------------------------------------------------
log "1/5 installing dependencies (backend=$BACKEND)"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake libeigen3-dev libssl-dev \
    git python3-dev python3-pip unrar
# Python deps for our exporter + OnnxBridge
pip install -q -r "$REPO_DIR/mpc/requirements.txt"
if [ "$BACKEND" = "ORCA" ]; then
  command -v nvidia-smi >/dev/null || { echo "ORCA needs an NVIDIA GPU"; exit 1; }
  log "GPU detected:"; nvidia-smi -L
  # Orca/GPU-MPC also needs: CUDA toolkit (>=11.7), CUTLASS, a Volta+ GPU (sm_70+).
  # Install the CUDA toolkit matching your driver, and let GPU-MPC fetch CUTLASS
  # via its own build (see GPU-MPC/README.md). Not scripted here -- host-specific.
fi
step deps

# ---- 2. export the FSS-ready model to ONNX ---------------------------------
log "2/5 exporting affinity model -> ONNX ($ONNX)"
mkdir -p "$WORK"
( cd "$REPO_DIR" && python3 -m mpc.export_onnx --dataset "$DATASET" --out "$ONNX" \
    ${SUMMARY:+--summary "$SUMMARY"} ${CKPT:+--ckpt "$CKPT"} )
# also dump a cleartext reference prediction for the final diff
python3 - "$ONNX" <<'PY'
import sys, numpy as np, onnxruntime as ort
sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
shp = {i.name: [d if isinstance(d,int) else 1 for d in i.shape] for i in sess.get_inputs()}
# one valid one-hot per position, saved as the secret client input + cleartext ref
inp = {}
for name, s in shp.items():
    x = np.zeros(s, dtype=np.float32); B,L,V = s
    x[:, np.arange(L), np.random.randint(0, V, L)] = 1.0
    inp[name] = x; np.save(name + ".npy", x)
ref = sess.run(None, inp)[0]
np.save("cleartext_ref.npy", ref)
print("cleartext ref:", ref.reshape(-1)[:4])
PY
step onnx

# ---- 3. clone + build EzPC / OnnxBridge ------------------------------------
log "3/5 fetching + building EzPC (backend=$BACKEND)"
[ -d "$WORK/EzPC" ] || git clone --depth 1 https://github.com/mpc-msri/EzPC "$WORK/EzPC"
cd "$WORK/EzPC"
git submodule update --init --recursive
pip install -q -r OnnxBridge/requirements.txt
# build the sytorch/LLAMA libraries (CPU FSS backend)
( cd GPU-MPC/ext/sytorch && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)" )
if [ "$BACKEND" = "ORCA" ]; then
  log "building GPU-MPC (Orca) -- see GPU-MPC/README.md for CUTLASS/CUDA setup"
  ( cd GPU-MPC && make -j"$(nproc)" )   # exact target per GPU-MPC/README.md
fi
step compile

# ---- 4. compile our ONNX -> secure 2PC app ---------------------------------
log "4/5 OnnxBridge: ONNX -> $BACKEND 2PC app (scale=$SCALE bitlength=$BITLENGTH)"
cd "$WORK/EzPC/OnnxBridge"
python3 main.py --path "$ONNX" --backend "$BACKEND" --scale "$SCALE" \
    --bitlength "$BITLENGTH" --generate code
# OnnxBridge emits the model .cpp + a build/run helper next to the ONNX; build it:
MODEL_DIR="$(dirname "$ONNX")"
log "generated app in $MODEL_DIR (see the emitted *.sh / README for the exact binary names)"
step keygen

# ---- 5. keygen (dealer) + 2-party run on localhost + diff -------------------
log "5/5 running 2PC inference (party0=model/server, party1=secret input/client)"
cat <<EOF
  The generated app exposes an offline (FSS key generation / dealer) phase and an
  online phase with two parties. Canonical localhost run:

     # offline: generate FSS keys
     ./<model> 1                       # dealer / key generation

     # online: two processes (party 0 has the public model, party 1 the secret input)
     ./<model> 2 0 <input_shares...> &  # server (party 0)
     ./<model> 2 1 <input_shares...>    # client (party 1), localhost

  Feed prot_onehot.npy / drug_onehot.npy (written in step 2) as party 1's secret
  input via the app's input-preprocessing step, and read back the output share.
  Then diff against cleartext_ref.npy:

     python3 -c "import numpy as np; \\
       o=np.load('secure_out.npy'); r=np.load('cleartext_ref.npy'); \\
       print('max abs err', abs(o.reshape(-1)-r.reshape(-1)).max())"

  A well-configured 64-bit FSS run reproduces the cleartext affinity to ~1e-2
  (fixed-point at scale=$SCALE). Match the exact binary/flag names to the version
  of OnnxBridge you cloned (its emitted README is authoritative).
EOF
log "done (structure). Consult the EzPC READMEs for version-specific binary names."
