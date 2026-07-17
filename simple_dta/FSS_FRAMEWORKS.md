# FSS / MPC frameworks for a secure affinity predictor

This note picks the framework to build a **GPU-based secure two-party (2PC)**
version of our affinity model, and records *why*. Scope: **affinity regression
only** — none of DeepDTAGen's generative decoder. Companion tool:
[`mpc_cost.py`](mpc_cost.py) quantifies the per-architecture cost that this note
argues about qualitatively.

**TL;DR.** Go **CNN + ReLU** and target **Orca** (the EzPC/GPU-MPC FSS stack) for
the real GPU port; prototype first in **CrypTen** for a fast PyTorch-native
latency baseline. Attention/transformer protein encoders are the wrong choice
under FSS — softmax and LayerNorm dominate the cost — and our own numbers show a
plain CNN both *beats* every attention model on accuracy and is the cheapest to
secure (see the leaderboard at the bottom).

---

## 1. Why FSS fits this problem

We want two parties (a model-holder and a data-holder, or two non-colluding
servers) to jointly compute an affinity prediction where **both the model weights
and the input are private** (secret-shared) — neither learns the other's secret.
Under secret-sharing MPC, with **private weights** the linear layers (conv/matmul)
are secret×secret and need a **Beaver-triple** multiply (one online round each, but
cheap — a big parallel matmul), and the **non-linearities** are the dominant online
cost. (If instead the weights are *public* to both parties, linear layers are purely
local and only the non-linearities interact — a lighter variant.)

**Function Secret Sharing (FSS)** makes the cheapest non-linearities cheap in the
best possible way: a ReLU becomes a single **DReLU** (secure "is x > 0?")
comparison with **one online round** and small communication, using
correlated-randomness keys generated offline. MaxPool is comparisons too. So a
model built from **Conv / Linear / ReLU / MaxPool** is close to the ideal shape
for a 2PC FSS port — which is exactly the DeepDTA-style CNN tower we already use.

The expensive operations — **softmax, GELU, exp, reciprocal/division, LayerNorm,
sigmoid** — have no cheap FSS gadget; each needs multiple rounds and large keys.
Order-of-magnitude figures from the private-transformer literature: a single
**softmax layer ≈ 10³× a same-size ReLU layer**, and **LayerNorm is more
expensive still** (secure mean/variance/rsqrt). This is the whole reason our
models were kept CNN-shaped and small.

---

## 2. Framework comparison (2PC, GPU focus)

| Framework | GPU FSS | PyTorch on-ramp | Ops it's good at | Maintenance | License |
|---|---|---|---|---|---|
| **Orca** (EzPC / GPU-MPC) | **Yes — native GPU FSS**, the flagship | via `sytorch` (PyTorch-like) | Conv, Linear, **ReLU/DReLU, MaxPool**; 2PC training **and** inference | active (mpc-msri); "PoC, not production" | MIT (EzPC) |
| **CrypTen** (Meta) | Yes (CUDA) | **Native — loads `nn.Module`** | Linear, Conv, ReLU, pooling; softmax etc. via approximations | **archived, read-only (May 2025)** | MIT |
| **SIGMA** (EzPC / GPU-MPC) | Yes | via `sytorch` | **Softmax, GeLU, SiLU, LayerNorm** + full transformer | active (same repo as Orca) | MIT (EzPC) |
| **AriaNN** | CPU + GPU (on PyTorch) | High (PySyft) | FSS ReLU/MaxPool/Conv/Linear | research prototype; pinned old PySyft (brittle install) | Apache/MIT (PySyft) |
| **Piranha** | Yes — GPU MPC (**not** FSS) | No (C++/CUDA app) | FC + ReLU; 2/3/4-PC linear-secret-sharing | research (USENIX'22); static | BSD-style |

Notes:
- **Orca** — *"Orca: FSS-based Secure Training and Inference with GPUs"*, IEEE S&P
  2024 (ePrint [2023/206](https://eprint.iacr.org/2023/206)). First system to run
  2PC FSS protocols natively on GPUs; reports ~**26× faster / ~98× less
  communication** than Piranha on CIFAR-scale CNNs. **Best match** once we commit
  to CNN+ReLU. Front-end is `sytorch` (a PyTorch-like graph builder), so we port
  the architecture into `sytorch`, not run `torch` directly.
- **CrypTen** — by far the easiest on-ramp (mirrors the PyTorch API, loads a
  trained `nn.Module`, offloads to CUDA), but the repo is **archived**. Use it for
  a quick PoC / latency baseline; **pin a commit**, don't build long-term on it.
- **SIGMA** — *"Secure GPT Inference with FSS"*, PoPETs 2024 (ePrint
  [2023/1269](https://eprint.iacr.org/2023/1269)). Only relevant **if** a
  transformer protein encoder ever becomes non-negotiable; it makes softmax/GELU
  *tractable*, not *cheap*.
- **Piranha** — great GPU-MPC engineering, but no FSS and no PyTorch front-end
  (you'd rewrite the model in C++). Orca supersedes it for our purpose.

---

## 3. Recommendation

1. **Architecture: CNN + ReLU**, mean- or max-pooled, small head. Drug side:
   either the SMILES-CNN tower, or a **GCN** (sparse matmul + ReLU, FSS-tractable).
   **Avoid GAT** (its attention softmax has the transformer cost problem) and
   avoid softmax cross-attention. If any attention is kept, use the already-built
   **`--attn-kind linear`** (softmax-free) — but note (§4, and `mpc_cost.py`) it
   still carries LayerNorm cost and does not beat the plain CNN.
2. **Prototype in CrypTen** (2PC, GPU): load a trained `CNNDTA` checkpoint, run
   encrypted inference on a few Davis pairs, confirm predictions match cleartext,
   measure latency. Fastest path to a real "MPC runs on our model" number.
3. **Production GPU port in Orca** (EzPC/GPU-MPC + `sytorch`): rebuild the CNN
   tower in `sytorch`, use its DReLU/MaxPool. This is the FSS-native, GPU,
   actively-maintained target.

---

## 4. Per-layer port notes (our CNN affinity model)

| Layer (`models.py`) | FSS cost | Note |
|---|---|---|
| `nn.Embedding` (protein/SMILES) | cheap | one-hot × table = a matmul, or a public-index gather if indices are public. Encoding happens **in the clear before sharing** (see `data.py` header). |
| `nn.Conv1d` (towers) | **free-ish** | linear; local computation on shares. Dilation adds **no** cost. |
| `nn.Linear` (head, projections) | **free-ish** | linear. The 1024/512 head is ~73% of params but cheap in MPC — param count ≠ MPC cost. |
| `F.relu` (towers, head, GCN) | **cheap** | one DReLU comparison per element — the FSS sweet spot. |
| `F.elu` (GAT, linear-attn kernel) | cheap | comparison-class, like ReLU. |
| max-pool (`pool=max`) | cheap-ish | comparisons; `mean`-pool is **free**. Worth a `--pool mean` run to price the accuracy trade (`mpc_cost.py` shows the op saving). |
| **softmax** (attention) | **expensive** | secure exp + reciprocal, O(Lq·Lk). Present only in `--fusion cross --attn-kind softmax` and the transformer protein encoder. Avoid. |
| **LayerNorm** (transformer/cross-attn) | **expensive** | secure mean/var/rsqrt. Present in every attention block — the reason even *linear* attention isn't free. |
| `sigmoid` (original DeepDTAGen `GatedCNN`) | expensive | another argument for our ReLU CNN over the upstream gated tower. |

---

## 5. Cost evidence (from `mpc_cost.py`, Davis, per inference)

Ranked by secure-inference cost, cheapest first. `sec.cmp` = secure DReLU
comparisons (relu+elu+maxpool); `softmax` = # exp/reciprocal elements;
`FSSscore` = a coarse weighted heuristic (cmp + 1000·softmax + 100·layernorm).

```
config                              balAcc   params   sec.cmp   softmax  FSSscore  free?
cnn_davis_regB_d1_wd0 (dilated CNN)  0.848  800.42K   347.15K         0   347.15K  yes
cnn_davis_cfgA_protDil123 (Config A) 0.793  267.07K   359.65K         0   359.65K  yes
cnn_davis_stable_pool_max (CNN base) 0.841  882.43K   364.06K         0   364.06K  yes
attn F  (GCN+cnn, linear cross)      0.831    1.08M   701.28K         0    16.27M  yes
attn D  (cnn+cnn, linear cross)      0.788    1.11M   725.44K         0    16.79M  yes
attn J  (raw-atom+cnn, linear cross) 0.803    1.79M     1.16M         0    32.29M  yes
attn A  (cnn+cnn, SOFTMAX cross)     0.840    1.11M   404.16K   716.83K  733.30M  NO
```

**The decision, in one line:** the dilated **CNN beats every attention model on
accuracy (0.848) *and* is the cheapest to secure** (fewest comparisons, zero
softmax). Softmax cross-attention (A) buys *no* accuracy over it yet costs ~2000×
more on the FSS heuristic; even softmax-free linear attention (F/D/J) roughly
doubles the comparison count and adds LayerNorm cost for lower accuracy. **CNN +
ReLU is the FSS port target.**

Regenerate this table any time with `python mpc_cost.py --compare`.

---

## 6. The port itself — `mpc/` (runnable)

The recommendation above is implemented in [`mpc/`](mpc/) as a **2-party, FSS,
GPU-optimizable** secure inference of the affinity CNN. Two tracks:

- **Runnable now:** [`mpc/fss_infer.py`](mpc/fss_infer.py) — a self-contained FSS
  engine (fixed-point ring + `sycret` DCF **DReLU** + Beaver-triple select) that
  runs `MPCReadyNet` end-to-end and verifies it matches cleartext. At `f=6` (its
  32-bit ring's headroom) it reproduces the plaintext prediction to ~0.1%; the
  `--sweep` shows the overflow ceiling above that — the empirical case for a
  64-bit GPU-FSS backend.
- **Production GPU-FSS:** [`mpc/MPC_PORT.md`](mpc/MPC_PORT.md) — export to ONNX
  ([`mpc/export_onnx.py`](mpc/export_onnx.py)) and run under **Orca** (EzPC/GPU-MPC),
  whose CUDA DReLU/truncation kernels over a 64-bit ring are where the *MPC itself*
  becomes GPU-optimized. `python -m mpc.test_fss` runs the correctness checks.

The model is reshaped so every op is an FSS kernel: embedding → one-hot·table
matmul, global **mean** pool (not max), static shapes, ReLU-only (see the per-layer
table in §4 and `MPC_PORT.md`).

## Sources
- Orca — ePrint [2023/206](https://eprint.iacr.org/2023/206); code: `mpc-msri/EzPC` (`GPU-MPC`, `sytorch`).
- SIGMA — ePrint [2023/1269](https://eprint.iacr.org/2023/1269); PoPETs 2024.
- LLAMA (FSS DNN, CPU predecessor) — ePrint [2022/793](https://eprint.iacr.org/2022/793).
- Piranha — USENIX Security 2022; `ucbrise/piranha`.
- CrypTen — arXiv [2109.00984](https://arxiv.org/abs/2109.00984); `facebookresearch/CrypTen` (archived).
- AriaNN — arXiv [2006.04593](https://arxiv.org/abs/2006.04593); `LaRiffle/ariann`.
- Surveys on private transformer inference — arXiv [2412.08145](https://arxiv.org/pdf/2412.08145), [2505.10315](https://arxiv.org/pdf/2505.10315).
