# FSS port of the affinity predictor — how to run it securely

Run the CNN+CNN affinity model as a **FSS-based, 2-party** secure inference. Three
tracks, in order of "runs today" → "GPU-optimized":

- **G1 — self-contained PoC (runs today, any box):** `fss_infer.py` executes the
  model under a real FSS protocol (fixed-point ring + `sycret` DCF DReLU +
  Beaver-triple select). Validates correctness, fixed-point precision, and prints
  per-op + single-sample timing (`--profile`). Small models only (sycret 32-bit).
- **GPU-FSS with a real secret input → NssMPClib** (`nssmpc_infer.py` +
  `run_nssmpc_vm.sh`). PyTorch-native, genuinely FSS (DPF/DCF/DICF), 2PC;
  GPU-accelerates conv/matmul. **This is the recommended GPU path for our CNN** —
  real input, full-size model, light install.

**Threat model: private model + private input.** Both the weights AND the drug/
protein are secret-shared — neither party learns the other's secret. Linear layers
are then secret×secret (Beaver matmul/conv), nonlinears are secure comparisons.
All the paths here run this: our engine defaults to `private_weights=True`;
NssMPClib's `share_model_param` secret-shares the model; EzPC/LLAMA holds the
weights secret in the 2PC. (A lighter *public-weights* variant, where linear layers
become local and only ReLU interacts, is available via `fss_infer --public-weights`.)
- **CPU-FSS reference of the exported ONNX → EzPC / LLAMA** (`run_ezpc_vm.sh`).
  Real 2PC, 64-bit ring, real secret input, but CPU (see the Orca note below).

**Two verified facts (EzPC master, 2026-07-16) that correct an earlier assumption:**
- EzPC **OnnxBridge has NO GPU backend** — it compiles ONNX only to **CPU** FSS
  (LLAMA) / SecFloat. There is no `--backend ORCA`.
- **Orca (GPU-MPC) does NOT ingest ONNX.** It runs *hardcoded C++ architectures*
  (`cnn.h`: VGG/ResNet/AlexNet…) on **zeroed input** as a GPU-FSS *benchmark*. To
  GPU-FSS *our* CNN with a real input you either (a) use **NssMPClib**, or (b)
  hand-port the model into sytorch C++ `cnn.h` and use Orca purely for timing.

Why FSS and not CrypTen/secret-sharing: under GMW-style secret sharing each ReLU
costs several online rounds and (per CryptGPU) is *not* faster on GPU; FSS makes
ReLU a **single online round** with GPU-parallel per-element key eval. See
`../FSS_FRAMEWORKS.md`. Note: the model supports **both mean- and max-pool** — mean
is free under FSS, max is a log(L) DReLU tournament (works, costlier; use whichever
your trained checkpoint has — no retrain needed).

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

## 2. Running it securely — pick a path

### (a) GPU-FSS, real input, our model → **NssMPClib** (recommended)
`bash mpc/run_nssmpc_vm.sh` on a CUDA VM. It installs NssMPClib, sets the ring
(`BIT_LEN=64 SCALE_BIT=16 DEVICE=cuda DEBUG_LEVEL=0`), generates offline FSS keys,
and runs two parties (server=weights, client=secret input). `mpc/nssmpc_infer.py`
rebuilds CNN+CNN in NssMPClib-native ops (embedding = 1×1 Conv2d; Conv1d→Conv2d
H=1; mean→AvgPool2d / max→MaxPool2d), loads your `CNNDTA` checkpoint, and records
single-sample latency + per-op runtimes. Validate the mapping first (no GPU):
```sh
python -m mpc.nssmpc_infer --summary runs/<tag>_summary.json --ckpt <.pth>   # NssDTA==CNNDTA + timing
```

### (b) CPU-FSS reference of the exported ONNX → **EzPC / LLAMA**
`bash mpc/run_ezpc_vm.sh` — exports our ONNX and drives EzPC OnnxBridge with the
**LLAMA** (FSS, CPU) backend, verified flow:
```sh
cd EzPC/OnnxBridge
python main.py --path affinity_fss.onnx --generate executable --backend LLAMA --scale 15 --bitlength 40
# role 1 = dealer/offline keygen, 2 = server/weights, 3 = client/secret input (localhost 127.0.0.1)
./model_LLAMA_15 1 ; ./model_LLAMA_15 2 model_input_weights.dat & ./model_LLAMA_15 3 127.0.0.1 < input.inp > output.txt
```
`/usr/bin/time -v` the offline vs online phases (single-sample = batch 1); LLAMA
prints its own online time + comm bytes. Caveat: our ONNX has **two inputs**
(drug/prot one-hot) — OnnxBridge demos are single-input, so you may need a
single-input wrapper. This path is CPU (no GPU backend exists in OnnxBridge).

### (c) True GPU-resident FSS numbers → **Orca** (benchmark only)
Orca (`EzPC/GPU-MPC`) gives real GPU-FSS timings but **only for models written in
its C++ `cnn.h`, on zeroed input** — not an ONNX/real-input path. Build:
`export CUDA_VERSION=11.7 GPU_ARCH=<sm> ; sh setup.sh main ; make orca`, add
CNN+CNN to `experiments/orca/cnn.h` (`getCNN`), then `run_experiment.py` prints
per-op GPU-FSS timing tables. Use it only if you need headline GPU-FSS throughput.

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
- `model_mpc.py` — `MPCModel` (backend-agnostic, mean+max pool) + `MPCReadyNet` (ONNX-exportable) + `load_cnndta` (checkpoint loader; `--summary/--ckpt/--config`).
- `configs.py` — named presets (`regB` = the 0.848 dilated model, `cfgA`); `--config regB` across the tools.
- `fss_infer.py` — self-contained FSS engine: **private-weight Beaver matmul/conv** (default; `--public-weights` for the light variant) + secure ReLU + secure max-pool + `--sweep` + `--profile` timing.
- `ONLINE_TIME.md` + `orca/regB_cnn.h` + `run_orca_regB.sh` — regB's 347K/114K comparison breakdown, why Orca (not SIGMA) cuts the 2.6s MP-SPDZ online time, and the Orca benchmark to measure it.
- `nssmpc_infer.py` — **NssMPClib GPU-FSS driver**: `NssDTA` (Conv2d-native CNN+CNN) + weight mapper from `CNNDTA` + `--party 0/1` 2PC + per-op/single-sample timing.
- `run_nssmpc_vm.sh` — install NssMPClib + run the 2PC inference on a CUDA VM (recommended GPU path).
- `accuracy.py` — load a trained checkpoint → real cleartext metrics + FSS-vs-cleartext fidelity (`--profile`).
- `export_onnx.py` — ONNX export (`--summary/--ckpt`) + onnxruntime equivalence check.
- `run_ezpc_vm.sh` — EzPC **LLAMA (CPU-FSS)** 2PC of the exported ONNX (verified flow).
- `test_fss.py` — correctness checks (G0 mean+max equality, secure ReLU, secure max-pool, end-to-end FSS).
- `setup_env.sh` / `requirements.txt` / `INSTALL.md` — environment + install steps.
