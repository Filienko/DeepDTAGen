# GPU-FSS port of the affinity predictor (Orca / EzPC)

How to run the affinity CNN as a **GPU-optimized, FSS-based, 2-party** secure
inference. Two tracks:

- **G1 — self-contained PoC (runs today, any box):** `fss_infer.py` executes the
  model under a real FSS protocol (fixed-point ring + `sycret` DCF DReLU +
  Beaver-triple select). Use it to validate correctness and fixed-point precision.
- **G2 — production GPU-FSS (this file):** export to ONNX → EzPC/GPU-MPC (**Orca**)
  → a CUDA 2-party app. This is where the *MPC implementation itself* is
  GPU-optimized: Orca implements the FSS DReLU/DCF and truncation as CUDA kernels
  over a 64-bit ring, so the nonlinear key-evaluation is GPU-parallel — which a
  pure-PyTorch engine (G1, `sycret` on CPU, 32-bit) can't be.

Why FSS and not CrypTen/secret-sharing: under GMW-style secret sharing each ReLU
costs several online rounds and (per CryptGPU) is *not* faster on GPU; FSS makes
ReLU a **single online round** with GPU-parallel per-element key eval. See
`../FSS_FRAMEWORKS.md`.

---

## 0. What gets exported

`export_onnx.py` exports `MPCReadyNet` (`model_mpc.py`) — the affinity CNN reshaped
so every op is an Orca kernel:

| plaintext op | FSS-ready form | why |
|---|---|---|
| `nn.Embedding` (secret token id) | **one-hot × table matmul** | no secret-index gather exists in FSS; tokens are the private input |
| global **max** pool | global **mean** pool | mean = local/free; max = a log(L)-round comparison tree |
| variable length | **static** padded length | FSS keys are sized per shape; ONNX tracing needs fixed shapes |
| Dropout | removed (`eval`) | no-op at inference |

Result: a graph of only **MatMul / Conv / Relu / ReduceMean / Gemm** — the exact
op set Orca accelerates (Conv1d maps to Conv2d with H=1; OnnxBridge handles this).

```sh
python -m mpc.export_onnx --dataset davis --out mpc/affinity_fss.onnx
# verifies onnxruntime output == torch (max abs err ~1e-9)
```

Train the deployable weights first with **mean pooling** (accuracy target):
```sh
python train.py --model cnn --dataset davis --pool mean --select-metric balacc \
  --tag-suffix _mpc_meanpool
```
then load that checkpoint into `MPCReadyNet` before export (pass the `CNNDTA` you
loaded weights into as `pool_model`).

---

## 1. Fixed-point / precision (GPU-MPC critical)

MPC runs over an integer ring, not floats.

- **Ring:** 64-bit (`Z_{2^64}`) on Orca — big enough that truncation wrap-error is
  negligible. (The G1 PoC is stuck at `sycret`'s **32-bit** ring, which is why its
  frac-bit sweep overflows above `f≈6`; 64-bit removes that ceiling.)
- **Fractional bits:** 13–16 for inference is the standard sweet spot. More bits =
  more precision but higher overflow risk.
- **Truncation:** every fixed-point multiply doubles the frac bits; Orca uses its
  GPU **stochastic-truncation** protocol (±1 LSB, accuracy-neutral) — no action
  needed beyond choosing `f`.
- **Normalize inputs and the regression target** so activations stay bounded
  (`|x| ≤ 2^(63-2f)`), else silent overflow wrecks accuracy. Validate your `f`
  choice cheaply first with `python -m mpc.fss_infer --sweep`.

---

## 2. Build & run Orca (on your CUDA box)

Orca lives in `mpc-msri/EzPC` under `GPU-MPC`; `OnnxBridge` compiles the ONNX into
a 2-party C++/CUDA app. Follow the upstream READMEs (versions move); the shape is:

```sh
# 1. clone + build EzPC / GPU-MPC (needs CUDA toolkit, CUTLASS, CMake, Eigen, OpenMP)
git clone https://github.com/mpc-msri/EzPC && cd EzPC/GPU-MPC
# follow GPU-MPC/README.md to build sytorch + the FSS backends (LLAMA + GPU)

# 2. compile our ONNX to a 2PC app via OnnxBridge
#    (CPU-FSS = LLAMA backend for a correctness check; GPU backend = Orca)
cd ../OnnxBridge
python main.py --path /path/to/affinity_fss.onnx --backend LLAMA --scale 13 --bitlength 64
# -> generates server (party 0) + client (party 1) binaries

# 3. offline: generate FSS keys (dealer);  online: run the two parties
#    party 0 holds the (public) model, party 1 secret-shares the drug+protein one-hot
./server-offline && ./client-offline          # key generation
./server LABEL 0 <ip> & ./client LABEL 1 <ip>  # 2PC inference
```

For the **GPU (Orca) backend**, build the `GPU-MPC` experiments per its README and
select the CUDA backend instead of `LLAMA`; the same ONNX and `--scale/--bitlength`
apply. Benchmark on the CUDA box: latency for batch=1 (round-bound) and throughput
at large batch (GPU-bound). Docs: `GPU-MPC/README.md`, `GPU-MPC/experiments/`.

---

## 3. GPU-optimization checklist (what actually moves wall-clock)

Ranked by impact (see `../FSS_FRAMEWORKS.md` for evidence):

1. **Depth = online rounds.** Each sequential ReLU/pool layer is a network
   round-trip the GPU can't parallelize away. Keep the tower shallow (our 3
   conv→ReLU stages). `python mpc_cost.py --compare` counts the comparisons; the
   DReLU count here is the round-bearing work.
2. **Mean pool, not max.** Already applied — removes a whole comparison stage.
3. **Total DReLU volume** sets FSS **offline key storage** (Orca's main bottleneck)
   — keep channel widths modest.
4. **Batch aggressively for throughput.** Online rounds are ~independent of batch,
   so amortize the round latency across many drug–protein pairs (GPU absorbs the
   parallel element work).
5. **Embedding-as-matmul is the dominant crypto cost** (secret one-hot × table).
   Keep vocab and `embed_dim` small; if the threat model permits *public* tokens,
   do the embedding gather in the clear and skip it entirely.

---

## 4. Files
- `model_mpc.py` — `MPCModel` (backend-agnostic) + `MPCReadyNet` (ONNX-exportable) + `load_cnndta` (checkpoint loader).
- `fss_infer.py` — self-contained FSS engine + `--sweep` precision tool.
- `accuracy.py` — load a trained mean-pool checkpoint → real cleartext metrics + FSS-vs-cleartext fidelity.
- `export_onnx.py` — ONNX export (`--summary/--ckpt`) + onnxruntime equivalence check.
- `run_ezpc_vm.sh` — build+run the EzPC LLAMA/Orca 2PC app on a larger (GPU) VM.
- `test_fss.py` — correctness checks (G0 equality, secure ReLU, end-to-end FSS).
- `setup_env.sh` / `requirements.txt` / `INSTALL.md` — environment + install steps.
