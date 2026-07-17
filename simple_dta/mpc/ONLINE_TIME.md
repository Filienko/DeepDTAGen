# Does Orca/SIGMA cut regB's 2.6s MP-SPDZ online time?

Baseline: **MP-SPDZ, CPU, ~2.6s online** for one **regB** secure inference.
Question: would SIGMA or Orca reduce it? Short answer: **SIGMA no (wrong tool),
Orca yes — likely by 1–2 orders of magnitude**, because regB's entire online cost
is secure *comparisons*, which is exactly what Orca's GPU FSS kernels accelerate.

## What regB is
`cnn_davis_regB_d1_wd0` — the Davis accuracy winner (**balAcc 0.848**): ~**800,417**-
param dilated CNN+CNN. Drug SMILES-CNN 16→32→48 (k4); protein CNN 32→64→96 with
**dilations 1,2,4**; **max-pool**; head 1024→512→1.

## Threat model
Both the **model weights and the input are private** (secret-shared) — MP-SPDZ, Orca,
NssMPClib and our own `fss_infer.py` (`private_weights=True`) all run this setting:
linear layers are secret×secret (Beaver) and nonlinears are secure comparisons.
(Any Orca-vs-MP-SPDZ number must match protocol/threat model — assume **2PC
semi-honest, online-only** for both, else the comparison is apples-to-oranges.)

## Where regB's online cost goes (measured: `mpc_cost.py`)
| quantity | value | under FSS |
|---|---|---|
| secure **comparisons** / inference | **347K** = 233K ReLU-DReLU + **114K max-pool** | the online cost; GPU-parallel key eval in Orca |
| softmax / LayerNorm | **0 / 0** | — (pure CNN) |
| linear layers (Beaver rounds, private weights) | 11 (2 embed + 6 conv + 3 head) | cheap matmul/conv, 1 round each |
| nonlinear **round depth** (critical path) | ≈ 3 conv-ReLU + 11 (protein max-pool tree, L'=1151) + 2 head-ReLU | sets the latency floor (× RTT) |
| linear MACs | 117M | free/local-ish (GPU-crushed) |

So MP-SPDZ's 2.6s is spent almost entirely on the **347K secure comparisons** (CPU:
each comparison = several rounds + bit-level compute) plus the round-trip latency of
the ~16-deep nonlinear critical path.

## Framework verdicts
- **SIGMA — no.** Its value is GPU-FSS for **softmax / GeLU / LayerNorm**. regB has
  **none** of these (softmax=0, LN=0). SIGMA would add nothing over Orca; it's the
  transformer sibling of the same codebase. Skip it.
- **Orca — yes, the right tool.** regB's whole online cost is comparisons (DReLU +
  max-pool), which Orca evaluates as **CUDA DReLU/DCF kernels**: FSS makes each
  comparison ~1 online round, and the GPU evaluates all 347K keys in parallel. Orca
  reports ~26×+ vs GPU-MPC (Piranha) and far more vs CPU frameworks; for an 800K CNN
  the online phase should drop from 2.6s to the **tens-of-ms** range (comm/round-bound
  on a LAN, GPU-bound on compute).
- **NssMPClib — won't help *this* metric.** It's genuinely FSS and GPU-accelerates
  conv/matmul, but its **DReLU/DCF eval is CPU-side** — so on regB's 347K comparisons
  it won't beat MP-SPDZ's online. NssMPClib is the easy *real-input* GPU path; **Orca
  is the one that actually cuts online latency.**

## Levers
- **max-pool = 114K / 347K (~33%) of the comparisons.** regB is max-pool (needed for
  0.848). Orca runs max-pool on GPU so you keep the accuracy and still win; a mean-pool
  retrain would remove those 114K if you want to shave further (see `mpc_cost.py
  --pool mean`: 233K comparisons).
- **private weights** add 11 Beaver linear rounds — negligible vs the comparison cost.

## How to actually measure it (Orca, on the GPU box)
Orca does **not** ingest ONNX — it runs models defined in sytorch C++ `cnn.h` on
zeroed input (fine: timing is input-independent). Steps:
1. Add regB to `EzPC/GPU-MPC/experiments/orca/cnn.h` (`getCNN`) — we provide the
   definition in `mpc/orca/regB_cnn.h` (dilated Conv1d→Conv2d(H=1), max-pool).
2. Build: `cd EzPC/GPU-MPC && export CUDA_VERSION=11.7 GPU_ARCH=<sm> && sh setup.sh main && make orca`.
3. Run `experiments/orca/run_experiment.py` (or `orca_inference`) for regB; it prints
   per-op GPU-FSS timing + **online** time. Compare that to the 2.6s MP-SPDZ number.

`mpc/run_orca_regB.sh` scripts steps 2–3. Predicted result: Orca online ≪ 2.6s;
the per-op table will show the DReLU + max-pool comparisons as the dominant (now
GPU-parallel) cost, matching the 347K/114K breakdown above.
