# Install & run — MPC/FSS affinity port

Environment and steps for everything under `simple_dta/mpc/`: the runnable
Python FSS PoC, and the production GPU-FSS (Orca/EzPC) path.

**One-shot:** `bash simple_dta/mpc/setup_env.sh` does steps 1–3 below and runs the
checks. The rest of this file explains each piece and the EzPC/Orca build.

Verified with: Python 3.11, `torch 2.13 (CPU)`, `torch_geometric 2.8`, `rdkit
2026.3`, `sycret 0.2.8`, `onnx 1.22`, `onnxruntime 1.27` (see `requirements.txt`).

---

## 1. Python environment

```sh
python3 -m venv .venv && source .venv/bin/activate     # or conda
pip install -r simple_dta/mpc/requirements.txt
```
GPU box: install the CUDA build of torch instead of the CPU wheel, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu121`, then the
rest of `requirements.txt`.

`sycret` (the FSS DCF DReLU) ships a prebuilt Rust wheel — no toolchain needed. If
it fails to install on your platform, the FSS engine is the only thing that needs
it; the op-cost tool and ONNX export do not.

## 2. Data

```sh
sudo apt-get install -y unrar
cd data && unrar x -o+ ../data.rar          # unpacks to data/data/*.csv
mv -n data/*.csv .                          # flatten to data/*.csv (where data.py looks)
```
Check: `python -c "from data import load_csv; print(len(load_csv('davis','test')[0]))"` → `5010`.

## 3. Run the PoC (CPU, no GPU needed)

```sh
cd simple_dta
python -m mpc.test_fss                       # correctness checks (should end "ALL ... PASSED")
python -m mpc.fss_infer --sweep              # FSS engine + fixed-point precision sweep
python mpc_cost.py --compare                 # accuracy-vs-FSS-cost leaderboard
python -m mpc.export_onnx --dataset davis    # ONNX export (+ onnxruntime == torch check)
```

Real accuracy with your trained **mean-pool** checkpoint (train it with
`python train.py --model cnn --dataset davis --pool mean --select-metric balacc
--tag-suffix _mp`, which writes `runs/cnn_davis_mp_{summary.json,best.pth}`):

```sh
python -m mpc.accuracy --summary runs/cnn_davis_mp_summary.json \
                       --ckpt    runs/cnn_davis_mp_best.pth --dataset davis
```
Prints the model's real cleartext test metrics (MSE/CI/rm2/balAcc) and the FSS
engine's fidelity (FSS predictions vs cleartext, MAE/Pearson) on real test pairs.
The checkpoint can be **mean OR max pool** (`train.py --pool mean|max`; max works,
just costlier under FSS — no retrain needed). Add `--profile` for per-op +
single-sample timing. The full-size model's full-length FSS run is the GPU path
(§4); the Python engine's `sycret` core is 32-bit and handles small models only.

## 4. GPU-FSS with a real input — **NssMPClib** (recommended GPU path)

```sh
# on a CUDA VM (NVIDIA GPU, CUDA>=11, Python 3.10+, torch>=2.5):
SUMMARY=runs/<tag>_summary.json CKPT=runs/<tag>_best.pth DATASET=davis \
  bash simple_dta/mpc/run_nssmpc_vm.sh
```
Installs NssMPClib (`pip install -e .`, CUTLASS/csprng submodules), sets the ring
(`BIT_LEN=64 SCALE_BIT=16 DEVICE=cuda DEBUG_LEVEL=0`), generates offline FSS keys,
and runs both parties (server=weights, client=secret input). Records single-sample
latency + per-op runtimes. Validate the model mapping first (no GPU needed):
```sh
python -m mpc.nssmpc_infer --summary runs/<tag>_summary.json --ckpt <.pth>   # NssDTA==CNNDTA + timing
```
NssMPClib is genuinely FSS (DPF/DCF/DICF) and GPU-accelerates conv/matmul; the FSS
nonlinear eval is CPU-side (negligible for our ~800K model).

## 5. CPU-FSS reference of the exported ONNX — **EzPC / LLAMA**

```sh
sudo apt-get install -y build-essential cmake libeigen3-dev git zip python3-dev
# EzPC pins numpy==1.21/onnx==1.12 -> use Python 3.8-3.10
bash simple_dta/mpc/run_ezpc_vm.sh          # exports ONNX + drives OnnxBridge/LLAMA
# core command it runs:
#   python OnnxBridge/main.py --path affinity_fss.onnx --generate executable \
#          --backend LLAMA --scale 15 --bitlength 40
# then roles: 1=dealer/keygen, 2=server/weights, 3=client/secret input (localhost)
```
`/usr/bin/time -v` the offline vs online phases for the single-sample latency; LLAMA
prints its own online time + comm bytes. **CPU only** — OnnxBridge has no GPU
backend. Caveat: our ONNX has two inputs (drug/prot one-hot); OnnxBridge demos are
single-input, so a single-input wrapper may be needed.

## 6. Orca GPU-FSS — benchmark only (no ONNX / real input)

Orca (`EzPC/GPU-MPC`) gives true GPU-resident FSS timings but **does not ingest
ONNX** — it runs hardcoded C++ models (`cnn.h`) on **zeroed input** as a benchmark.
To time *our* CNN on it you must hand-write CNN+CNN into `experiments/orca/cnn.h`
and rebuild (`export CUDA_VERSION=11.7 GPU_ARCH=<sm>; sh setup.sh main; make orca`),
then `run_experiment.py` prints per-op GPU-FSS tables. Use only if you need headline
GPU-FSS throughput; for real secure inference of our model, use §4 (NssMPClib).

> Upstream binary/flag names move between versions — treat §4–6 and the scripts as
> tested *structure* and reconcile against the READMEs you
> clone. Links are in `run_ezpc_vm.sh` and `MPC_PORT.md`.
