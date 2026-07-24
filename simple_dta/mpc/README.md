# `mpc/` — secure (FSS, GPU-optimizable) inference of the affinity CNN

A **2-party, Function-Secret-Sharing** secure inference of the DeepDTAGen
affinity predictor (drug–target regression **only** — no generative decoder).
This directory is the runnable port; the reasoning behind it lives in
[`../FSS_FRAMEWORKS.md`](../FSS_FRAMEWORKS.md) and [`../mpc_cost.py`](../mpc_cost.py).

**Threat model:** *private model + private input.* Both the weights and the
drug/protein are secret-shared; neither party learns the other's secret. Linear
layers are secret×secret (Beaver matmul/conv); nonlinears are secure comparisons.
(A lighter *public-weights* variant is available via `fss_infer --public-weights`.)

---

## Pick your path (decision tree)

```
Want to...                                          → use
──────────────────────────────────────────────────────────────────────────────
verify FSS correctness / precision, runs anywhere   → fss_infer.py  (self-contained)
count secure ops for a config (no crypto)           → ../mpc_cost.py --compare
real cleartext + FSS-fidelity accuracy on weights   → accuracy.py --summary/--ckpt
GPU-FSS, real Davis point, full-size regB           → nssmpc_infer.py + run_nssmpc_vm.sh
see how fixed-point quantization hits accuracy       → nssmpc_infer.py --fixed-point
true GPU-resident FSS online-time numbers           → run_orca_regB.sh  (benchmark)
CPU-FSS reference of the exported ONNX              → run_ezpc_vm.sh  (LLAMA)
drop max-pool (the GPU-hard op) & re-measure acc    → retrain_regB_meanpool.sh
```

**The honest one-liner on limits:** the self-contained `fss_infer` engine is on a
**32-bit** ring (`sycret`'s DCF), so it proves correctness on *small* models but
**overflows on full regB**. A genuine *secure full-regB* run is the **64-bit**
path — **NssMPClib** (GPU, `nssmpc_infer.py`) or **Orca** (`run_orca_regB.sh`).
No framework is simultaneously easy-to-extend-in-Python *and* GPU-accelerated on
the FSS nonlinears; see the extensibility discussion in `ONLINE_TIME.md`.

---

## Files

### Runnable here (CPU)
| file | what |
|---|---|
| **`fss_infer.py`** | self-contained FSS engine: private-weight Beaver matmul/conv + FSS DReLU (`sycret`) + secure max-pool tournament. `--sweep` (frac-bit precision), `--profile` (per-op + single-sample timing), `--public-weights`, `--config regB`. 32-bit ring → small models only. |
| **`model_mpc.py`** | `MPCModel` (backend-agnostic, mean **and** max pool) + `MPCReadyNet` (ONNX-exportable) + `load_cnndta(summary, ckpt, config)`. The op-reshaping (embedding→one-hot·table matmul, static shapes, Dropout removed). |
| **`accuracy.py`** | load a trained checkpoint → cleartext test MSE/CI/rm2/balAcc + FSS-vs-cleartext fidelity (MAE/Pearson). `--profile`. |
| **`nssmpc_infer.py`** | `NssDTA` (Conv2d-native CNN+CNN, dilation-aware) + weight mapper from `CNNDTA`. `run_plaintext` (NssDTA==CNNDTA check + timing), `run_fixedpoint` (**quantization sweep**), `--party 0/1` (the 2PC secure run, GPU VM), `--davis-index N` (real Davis point). |
| **`configs.py`** | named presets — `regB` (the 0.848 dilated ~800K CNN+CNN, max-pool) and `cfgA` (267K). `--config regB` across the tools; asserts param counts. |
| **`export_onnx.py`** | ONNX export of `MPCReadyNet` + onnxruntime==torch equivalence check. |
| **`test_fss.py`** | correctness suite (G0 mean+max equality, secure ReLU, secure max-pool, end-to-end, private-weights). `python -m mpc.test_fss`. |

### Run on a GPU VM (scripts, not run here)
| file | what |
|---|---|
| **`run_nssmpc_vm.sh`** | install NssMPClib, set 64-bit ring, keygen, run P0/P1 2PC — the **recommended GPU-FSS path** for our CNN. |
| **`run_orca_regB.sh`** + **`orca/regB_cnn.h`** | build EzPC/GPU-MPC, add regB to sytorch `cnn.h`, `make orca`, read the **online** time (vs the 2.6s MP-SPDZ CPU baseline). Benchmark only (zeroed input). |
| **`run_ezpc_vm.sh`** | EzPC **LLAMA (CPU-FSS)** 2PC of the exported ONNX (verified `ezpc-cli.sh` flow). |
| **`retrain_regB_meanpool.sh`** | retrain regB max→mean pool, print the balAcc drop (removes 114K/347K secure comparisons). See `RETRAIN_REGB.md`. |

### Docs
| file | what |
|---|---|
| **`MPC_PORT.md`** | the how-to-run-securely guide: three tracks, fixed-point/precision, GPU-optimization checklist, per-op export table. |
| **`ONLINE_TIME.md`** | regB's 347K/114K comparison breakdown; why Orca (not SIGMA/NssMPClib) cuts the 2.6s MP-SPDZ online time; extensibility contrast. |
| **`RETRAIN_REGB.md`** | why/how to trade max-pool for mean-pool and measure the accuracy cost. |
| **`INSTALL.md`** + `setup_env.sh` + `requirements.txt` | environment + install (Python PoC, EzPC LLAMA CPU build, Orca GPU build notes). |

---

## Quickstart (in-sandbox, CPU)

```sh
cd simple_dta
python -m mpc.test_fss                       # correctness suite (all green)
python -m mpc.fss_infer --profile            # FSS PoC: prediction + per-op timing
python mpc_cost.py --compare                 # op-cost leaderboard (regB cheapest)
```

With your trained weights (a `simple_dta` `CNNDTA` checkpoint):

```sh
# real accuracy + FSS fidelity
python -m mpc.accuracy --summary runs/<tag>_summary.json --ckpt runs/<tag>_best.pth --profile
# plaintext regB on a real Davis point + fixed-point sweep (how FP quantization bites)
python -m mpc.nssmpc_infer --config regB --ckpt runs/<regB>_best.pth --dataset davis \
  --davis-index 0 --fixed-point
```

The **secure** GPU run (64-bit, real 2PC) is `run_nssmpc_vm.sh` on a CUDA VM,
then `nssmpc_infer.py --party 0` / `--party 1` — see `MPC_PORT.md §2(a)`.
