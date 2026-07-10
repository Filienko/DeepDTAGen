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
The checkpoint **must be mean-pool** (max-pool costs comparison rounds; MPC rejects
it). The full-size model's *full-length* FSS run is the 64-bit Orca path (§5) — the
Python engine's `sycret` core is 32-bit and handles small models only.

## 4. EzPC CPU-FSS (LLAMA) build — 64-bit FSS, no GPU

```sh
sudo apt-get install -y build-essential cmake libeigen3-dev libssl-dev git python3-dev
git clone https://github.com/mpc-msri/EzPC && cd EzPC
git submodule update --init --recursive
pip install -r OnnxBridge/requirements.txt
( cd GPU-MPC/ext/sytorch && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j )
# compile our ONNX to a 2PC app:
python OnnxBridge/main.py --path /path/to/affinity_fss.onnx \
       --backend LLAMA --scale 13 --bitlength 64 --generate code
```
Then run the emitted app's offline (key-gen) + online (party 0 = model, party 1 =
secret input) phases on localhost and diff against cleartext. `mpc/run_ezpc_vm.sh`
scripts this end to end.

## 5. Orca GPU-FSS build — the GPU-optimized path

Additional requirements: an NVIDIA **Volta+ GPU (sm_70+)**, **CUDA toolkit ≥ 11.7**,
CUTLASS (fetched by the GPU-MPC build), a recent driver. Build under
`EzPC/GPU-MPC` per its `README.md`, then use `--backend` for the CUDA FSS backend
in OnnxBridge. Drive the whole thing with:

```sh
BACKEND=ORCA SCALE=13 BITLENGTH=64 \
  SUMMARY=runs/cnn_davis_mp_summary.json CKPT=runs/cnn_davis_mp_best.pth \
  bash simple_dta/mpc/run_ezpc_vm.sh
```
Benchmark: single-query latency is round-bound (model depth); throughput scales with
batch size (GPU absorbs the parallel work). See `MPC_PORT.md` §3.

> EzPC's exact binary/flag names move between versions — treat §4–5 and
> `run_ezpc_vm.sh` as the tested *structure* and reconcile against the READMEs you
> clone. Links are in `run_ezpc_vm.sh` and `MPC_PORT.md`.
