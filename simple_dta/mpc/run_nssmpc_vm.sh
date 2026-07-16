#!/usr/bin/env bash
# =============================================================================
# run_nssmpc_vm.sh -- GPU-FSS 2PC inference of the CNN+CNN affinity model with
# NssMPClib (XidianNSS). Run on a CUDA VM. Records single-sample latency + per-op
# runtimes (mpc/nssmpc_infer.py does the timing).
#
# NssMPClib: PyTorch-native, genuinely FSS (DPF/DCF/DICF), 2PC semi-honest,
# public-weights/secret-input. GPU accelerates conv/matmul (CUTLASS); the FSS
# nonlinear eval is CPU-side (negligible for our ~800K model). Much lighter than
# EzPC/Orca. Docs: https://github.com/XidianNSS/NssMPClib
#
# Versions/paths move -- treat as a tested structure; reconcile with the repo you
# clone. Run step-by-step first (STOP_AFTER=install|params).
# =============================================================================
set -euo pipefail

WORK="${WORK:-$HOME/nssmpc-affinity}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"     # the simple_dta dir
SUMMARY="${SUMMARY:-}"                            # runs/<tag>_summary.json (CNN+CNN, mean OR max)
CKPT="${CKPT:-}"                                  # trained CNNDTA .pth
DATASET="${DATASET:-davis}"
BIT_LEN="${BIT_LEN:-64}"                          # ring bits (64 recommended)
SCALE_BIT="${SCALE_BIT:-16}"                      # fractional bits (16 for real numbers)
STOP_AFTER="${STOP_AFTER:-}"
log(){ echo -e "\n\033[1;35m[nssmpc]\033[0m $*"; }

# ---- 1. install NssMPClib -------------------------------------------------
log "1/4 installing NssMPClib (needs: NVIDIA GPU, CUDA>=11, gcc/g++, Python 3.10+, torch>=2.5)"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "  (no GPU -> CPU fallback; set DEVICE below to cpu)"
sudo apt-get update -qq && sudo apt-get install -y -qq git build-essential python3-dev
[ -d "$WORK/NssMPClib" ] || git clone --recursive https://github.com/XidianNSS/NssMPClib "$WORK/NssMPClib"
cd "$WORK/NssMPClib"
git submodule update --init --recursive            # CUTLASS + torchcsprng
pip install -e . --no-build-isolation
# our model deps (to build/share the affinity net)
pip install -q -r "$REPO_DIR/mpc/requirements.txt" || true
python3 scripts/installation_advice.py || true
[ "$STOP_AFTER" = install ] && { log "stop after install"; exit 0; }

# ---- 2. configure ring / device / security -------------------------------
log "2/4 configuring nssmpc/config/configs.json (BIT_LEN=$BIT_LEN SCALE_BIT=$SCALE_BIT DEVICE=cuda DEBUG_LEVEL=0)"
CFG="$WORK/NssMPClib/nssmpc/config/configs.json"
python3 - "$CFG" "$BIT_LEN" "$SCALE_BIT" <<'PY'
import json, sys
p, bl, sc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = json.load(open(p))
for k, v in [("BIT_LEN", bl), ("SCALE_BIT", sc), ("DEVICE", "cuda"), ("DEBUG_LEVEL", 0)]:
    # keys may be nested depending on version; set top-level and warn if absent
    if k in c: c[k] = v
    else: print(f"  [!] {k} not at top level of configs.json -- set it manually")
json.dump(c, open(p, "w"), indent=2)
print("  updated", p)
PY

# ---- 3. offline parameter (FSS key) generation ---------------------------
log "3/4 generating offline FSS parameters (DEBUG_LEVEL=0 => real, OTP-secure keys)"
python3 scripts/offline_parameter_generation.py
[ "$STOP_AFTER" = params ] && { log "stop after params"; exit 0; }

# ---- 4. run the 2 parties (localhost) ------------------------------------
log "4/4 running 2PC: party 0 = model/server, party 1 = secret input/client"
export PYTHONPATH="$REPO_DIR:$WORK/NssMPClib:${PYTHONPATH:-}"
ARGS="--summary '$SUMMARY' --ckpt '$CKPT' --dataset $DATASET --device cuda"
cat <<EOF
  Two terminals on this host (party 1 must start the peer; both need the config above):

     # terminal A (server / party 0, holds the CNN+CNN weights):
     cd $REPO_DIR && python -m mpc.nssmpc_infer --party 0 $ARGS

     # terminal B (client / party 1, holds the secret drug+protein, prints the result):
     cd $REPO_DIR && python -m mpc.nssmpc_infer --party 1 $ARGS

  The client prints the reconstructed affinity + single-sample latency; per-op
  runtimes come from mpc/nssmpc_infer.py's hooks (and NssMPClib's own timing log).
  First validate the model maps correctly (no GPU/nssmpc needed):
     cd $REPO_DIR && python -m mpc.nssmpc_infer $ARGS       # plaintext NssDTA==CNNDTA + timing
EOF
log "done (structure). If forward() taking two inputs (drug,prot) conflicts with your"
log "nssmpc SharedDataLoader version, see the NOTE in mpc/nssmpc_infer.py:run_secure()."
