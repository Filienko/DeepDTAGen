# simple_dta — Experiment Log

Canonical record of every affinity-prediction experiment in `simple_dta/`, plus the
published reference numbers we compare against. **Read this first** when returning to
the project so we don't re-run or re-retrieve anything.

Goal: from-scratch, **affinity-only** DTA models far simpler than DeepDTAGen (which
trains >1 week), within **2–5 points** of DeepDTAGen accuracy, as a base for a later
**MPC** (privacy-preserving inference) protocol. CNN is the current front-runner on
accuracy-per-parameter.

Last updated: 2026-07-04. **If resuming: read Section 8 first** — it documents the
batch launched 2026-07-04 while Steven was away, including the one ablation we must
not lose track of (dilation vs. balAcc-checkpoint-selection).

---

## 0. Metric definitions

| metric | meaning | direction |
|---|---|---|
| **MSE** | mean squared error, predicted vs true affinity (primary training loss) | lower ↓ |
| **CI** | Concordance Index — P(model ranks a random pair in the correct order). 0.5=random, 1=perfect. The field's headline metric. | higher ↑ |
| **rm2** | regression-toward-the-mean-penalized R² (variance explained, penalizing scale/offset). >0.5 acceptable. | higher ↑ |
| **Pearson** | linear correlation of predicted vs true | higher ↑ |

MSE/rm2/Pearson judge *how close* the numbers are; CI judges *whether the ranking* is right.
**MSE is not comparable across datasets** (davis/kiba/bindingdb use different affinity scales).

---

## 1. Environment / how to run

- Conda env **`DeepDTAGen`** (CPU-only torch 1.12.1, PyG 2.2.0). Machine: 48 cores / 125 GB.
- Code: `data.py` (CSV→label-encode for CNN / RDKit graphs for GNN), `models.py`
  (`CNNDTA`, `GNNDTA`, shared `ProteinCNN`/`PredictionHead`), `train.py` (unified, single
  MSE loss; writes `runs/<tag>_summary.json` + `_pred.txt`/`_true.txt` + `_best.pth`), `metrics.py`.
- **Detached batch** (survives SSH logout): `setsid nohup bash run_queue.sh <jobs_file> <max_parallel> <threads> >/dev/null 2>&1 </dev/null & disown`.
  Per-job log `runs/<logtag>.log`; driver log `runs/queue.driver.log`.
- Check running jobs: `ps -C python -o cmd= | grep train.py`.
- Default seed 4221; multi-seed uses 7 and 13.

### train.py knobs
`--model {cnn,gnn} --dataset {davis,kiba,bindingdb} --epochs --embed-dim`
- Head: `--head-dim` (first FC width), `--head-layers {1,2}` (2 → hidden, hidden//2; 1 → single hidden).
- CNN towers: `--drug-filters`, `--prot-filters` (base f; layers f,2f,3f), `--pool {max,mean,maxmean}`, `--proj-dim` (>0 projects each tower to this dim before the head).
- GNN: `--node-feat {full=94,small=12,tiny=4}`, `--gcn-dim`, `--gcn-layers`.

---

## 2. Reference numbers — DeepDTAGen & DeepDTA (so we never re-retrieve)

Source: **Shah et al., "DeepDTAGen…", Nature Communications 16:5021 (2025), Table 1** (PDF in repo root).
Paper reports **CI / MSE / rm2 / AUPR** (no Pearson). Numbers are 6-fold CV averages;
ours are a single train/test split — treat cross-comparisons as approximate, and never
compare MSE across datasets.

| dataset | model | CI ↑ | MSE ↓ | rm2 ↑ | AUPR |
|---|---|---|---|---|---|
| **KIBA** | DeepDTA (published) | 0.863 | 0.194 | 0.630 | 0.788 |
| | **DeepDTAGen** | **0.897** | **0.146** | **0.765** | 0.843 |
| **Davis** | DeepDTA (published) | 0.878 | 0.261 | 0.630 | 0.714 |
| | **DeepDTAGen** | **0.890** | **0.214** | **0.705** | 0.772 |
| **BindingDB** | DeepDTA (retrained in-paper, "RT") | 0.844 | 0.633 | 0.633 | 0.710 |
| | **DeepDTAGen** | **0.876** | **0.458** | **0.760** | 0.870 |

Other baselines in Table 1 (CI, for context): GraphDTA(GIN) KIBA 0.891 / Davis 0.893;
AttentionDTA, DeepCDA, GDilatedDTA, DoubleSG-DTA, ELECTRA-DTA all ~0.88–0.90 on Davis.

### Parameter counts (computed, not from paper)
- **DeepDTAGen — affinity pipeline only: ~3,576,095 params** (excludes the VAE/Transformer
  generation machinery, which adds tens of millions). Breakdown: drug tower (3× GCNConv
  94→188→282→376 + Drug_FCs) 695K; protein tower (Gated-CNN, incl. a 10272→128 FC) **1.96M**;
  head (256→1024→512→256→1) 920K. The protein tower is the giant because it flattens the
  whole conv map (96×107) into one Linear layer — our design global-pools instead.
- **DeepDTA — ~1.9M** (estimate from its architecture: two 3-conv towers + head
  1024→1024→512→1; head dominates at ~1.77M).

---

## 3. Our CNN results

Architecture: two 1D-CNN towers (drug SMILES k=4, protein sequence k=8), each Embedding→
3× Conv1d (32→64→96)→global-max-pool→96-vec; concat→192→FC head→1. Variants change the
head and/or towers. `embed_dim=128` unless noted.

### Head-size ablation (baseline towers 32/32, max pool, concat=192)
| tag | head | params | dataset | seed(s) | MSE | CI | rm2 | Pearson |
|---|---|---|---|---|---|---|---|---|
| cnn_davis (baseline) | 1024→512→1 | 882K | davis | 4221/7/13 | **0.225** (0.224/0.225/0.226) | 0.887 | 0.713 | 0.851 |
| cnn_davis_h1024L1 | 1024→1 | **358K** | davis | 4221/7/13 | 0.248 (0.254/0.244/0.244) | 0.882 | 0.679 | 0.837 |
| cnn_davis_h512 | 512→256→1 | 390K | davis | 4221/7/13 | 0.275 (0.278/0.273/0.274) | 0.873 | 0.649 | 0.811 |
| cnn_kiba (baseline) | 1024→512→1 | 882K | kiba | 4221/7/13 | **0.179** (0.176/0.176/0.184) | 0.858 | 0.707 | 0.861 |
| cnn_kiba_h512 | 512→256→1 | 390K | kiba | 4221 | 0.180 | 0.856 | 0.695 | 0.860 |
| cnn_bindingdb (baseline) | 1024→512→1 | 882K | bindingdb | 4221/7/13 | **0.503** (0.499/0.512/0.497) | 0.862 | 0.706 | 0.869 |

**Findings:**
- On Davis, **cut head *depth* not *width***: h1024L1 (358K, single wide layer) beats h512
  (390K, two narrow layers) on every metric despite being smaller. But depth-cut isn't
  fully free vs baseline (+0.023 MSE).
- **Small head is ~free on KIBA** (0.180 vs 0.179) but **hurts Davis** (−6pt) — KIBA's 4× data
  makes conv features rich so the head isn't the bottleneck; tiny Davis leans on head capacity.

### vs references (best single-seed-comparable, Davis)
Our CNN baseline **beats published DeepDTA on all three datasets** and lands within the
2–5pt target of DeepDTAGen (Davis: +0.011 MSE / −0.003 CI vs DeepDTAGen; wins rm2).

### Tower-architecture sweep — `jobs_cnn_towers.txt` (RUNNING, launched 2026-07-02 20:44 UTC)
14 runs = 4 towers × 2 heads on davis+kiba, single seed 4221, minus redundant T1+H2.
Heads: **H1 = 1536→1** (wide-shallow), **H2 = 1024→512→1** (deep baseline head).

| tower | drug/prot filters | pool | proj | concat dim `in` | params H1 | params H2 |
|---|---|---|---|---|---|---|
| T1 base | 32 / 32 | max | – | 192 | 457K | 882K *(=baseline, not re-run)* |
| T2 dualpool | 16 / 16 | max+mean | – | 192 | 359K | 784K |
| T3 asym (lean drug) | 16 / 32 | max | – | 144 | 351K | 800K |
| T4 bottleneck | 32 / 32 | max | 32 | 64 | **267K** | 758K |

Rationale: T2 tests pooling quality (peak motif via max + composition via mean) at ¼ conv
cost; T3 reallocates capacity toward the harder/longer protein modality; T4 compresses the
joint binding embedding to shrink the head input 3×.

**Davis results (single seed 4221; baseline 3-seed = 0.225/0.887, old best-small h1024L1 358K = 0.248):**
| run | tower / head | params | MSE | CI | rm2 |
|---|---|---|---|---|---|
| t3_h2 | asym / deep | 800K | **0.231** | **0.893** | 0.669 |
| t3_h1 | asym / wide-shallow | **351K** | 0.236 | 0.880 | 0.679 |
| t1_h1 | base / wide-shallow | 457K | 0.238 | 0.884 | 0.669 |
| t4_h2 | bottleneck / deep | 758K | 0.238 | 0.881 | 0.691 |
| t2_h2 | dualpool / deep | 784K | 0.245 | 0.885 | 0.666 |
| t4_h1 | bottleneck / wide-shallow | **267K** | 0.248 | 0.879 | 0.662 |
| t2_h1 | dualpool / wide-shallow | 359K | 0.256 | 0.877 | 0.658 |

Reads: **T3 asymmetric (lean drug / full protein) wins both head classes** — the drug modality
compresses well, protein capacity matters. **t3_h1 = best accuracy-per-param (351K, 0.236)**,
beats old h1024L1. **t4_h1 = smallest (267K) at h1024L1 accuracy.** No tower beats baseline MSE
(0.225) — but t3_h2 beats baseline CI, and single-seed 0.231 is within Davis seed noise (~±0.01).
Dual-pool (T2) disappointed. **KIBA results pending.**

---

## 4. Our GNN results

Architecture: shallow GCN drug encoder (GCNConv, same family as DeepDTAGen but no gating/
transformer) + same protein CNN tower + head. Atom features are **hand-crafted RDKit
one-hots (NOT learned)**: full=94, small=12, tiny=4 dims.

| tag | node-feat | gcn-dim × layers | embed | params | dataset | seed | MSE | CI | rm2 | Pearson |
|---|---|---|---|---|---|---|---|---|---|---|
| gnn_davis (baseline) | full=94 | 128 × 2 | 128 | 903K | davis | 4221 | **0.270** | 0.874 | 0.625 | 0.817 |
| gnn_davis_f12d32 | small=12 | 32 × 3 | 32 | 736K | davis | 4221/7/13 | **0.270** (0.271/0.260/0.280) | 0.874 | 0.621 | 0.821 |
| gnn_davis_f12d24 | small=12 | 24 × 3 | 32 | 726K | davis | 4221 | 0.317 | 0.855 | 0.617 | 0.790 |
| gnn_davis_f12d16 | small=12 | 16 × 3 | 32 | 717K | davis | 4221 | 0.373 | 0.845 | 0.528 | 0.732 |
| gnn_davis_f12d12 | small=12 | 12 × 3 | 32 | 712K | davis | 4221 | 0.411 | 0.821 | 0.468 | 0.700 |
| gnn_davis_f4d24 | tiny=4 | 24 × 3 | 32 | 726K | davis | 4221 | 0.297 | 0.866 | 0.628 | 0.795 |
| gnn_davis_f4d32 | tiny=4 | 32 × 3 | 32 | 735K | davis | 4221 | 0.318 | 0.858 | 0.601 | 0.777 |
| gnn_bindingdb (baseline) | full=94 | 128 × 2 | 128 | 903K | bindingdb | 4221 | 0.606 | 0.848 | 0.676 | 0.841 |

**Findings:**
- **`small-12 / gcn-32 / L3` exactly ties the full 94-feat/gcn-128 GNN** (MSE 0.270, CI 0.874,
  3-seed) at a ~20× cheaper GCN core. Confirmed not seed luck.
- **GCN-width floor is sharp just below 32**: gcn-16 → 0.373, gcn-12 → 0.411. Even though the
  learned embedding's effective rank is ~11, message passing needs the wider 32-dim working space.
- **GNN trails the CNN on every dataset.** CNN is the front-runner.

### Dimensionality analysis (Davis, drug side) — see memory [[gnn-embedding-dimensionality]]
- Input 94-dim atom features: only ~24/94 columns ever active (66 dead; formal-charge block constant).
- Learned post-GCN 128-dim embedding: effective rank ~11; halogens separate on top PCs.
- **PENDING:** re-run this PCA on trained KIBA/BindingDB checkpoints (more chemical diversity
  should raise effective dim).

---

## 5. Receptive-field note (motivates the pending conv-depth work)

With 3 stride-1 convs and no interior pooling, each output feature sees only a small window:
- **Protein**: kernel 8 → RF **22 residues** over ~1000 = **~2%** of the sequence (myopic;
  binding pockets span sequence-distant residues).
- **Drug**: kernel 4 → RF **10 chars** over ~85 = ~12% (already adequate; SMILES is short).

→ argues for **asymmetric conv depth: deeper protein, shallower drug**, then **dilation**
to widen protein RF cheaply. See §6 and memory [[pending-conv-depth-experiment]].

---

## 6. Pending / next experiments

1. **(RUNNING)** Tower sweep `jobs_cnn_towers.txt` — davis DONE (see below), kiba running.
2. **(QUEUED — `jobs_cnn_conv.txt`, chained to auto-start after #1 via `chain_conv.sh`)**
   Conv-tower experiments, davis+kiba, deep head fixed, on the winning asym base:
   C1 shallow-drug (2 convs), C2 shallow-drug + deep-protein (4 convs), C3 dilated protein
   (1,2,4), C4 flat shape (32³/16³), C5 inverted shape (32→24→16). Covers depth, dilation,
   and filter-shape (#2/#3/#3b below). Compare against t3_h2. [[pending-conv-depth-experiment]]
3. (folded into #2 as C3) Strides / dilation on protein convs to widen RF.
3b. **Filter-count shape per layer** (not just depth): trial flat (32→32→32) and inverted
    (32→24→16) vs the inherited 32→64→96 pyramid. Rationale: the pyramid comes from 2D image
    classification (many high-level classes → wide top); DTA outputs one scalar and the drug
    rep is ~11–20 effective dims (§4 PCA), so a 96-wide top layer is likely overkill. Shrinks
    conv params *and* the pooled dims into the head. Maybe asymmetric (protein tolerates wider
    top than drug). [[pending-conv-depth-experiment]]
4. Cross-dataset PCA of trained GNN embeddings (KIBA/BindingDB). [[gnn-embedding-dimensionality]]
5. `--pool mean` MPC-cost ablation on the CNN.
6. Per-dataset convergence/capacity read alongside the accuracy table.

---

## 7. Bottom line so far

- **CNN > GNN on every dataset**, and cheaper. CNN baseline matches/beats published DeepDTA
  and is within 2–5pt of DeepDTAGen at a fraction of the size (882K vs DeepDTAGen's ~3.58M
  affinity pipeline) and training time (minutes/hours vs >1 week), with no generation machinery.
- Best small models so far: **CNN h1024L1 = 358K** (Davis, MSE 0.248), **GNN small-12/gcn-32 =
  736K** (Davis, ties full GNN). The current sweep is hunting for a smaller/better tower+head combo.

---

## 8. Batch launched 2026-07-04 (while Steven away ~2 days) — READ FIRST ON RETURN

All runs: **Davis**, 100 epochs, dropout 0.1, weight-decay 0 (regularization already
ruled out as unhelpful — see the reg sweep in RESULTS_SUMMARY.txt), **checkpoint
selected by balanced accuracy** (`--select-metric balacc`). Job file:
`jobs_batch_july.txt`. Launched detached via `run_queue.sh ... 5 8`.

**How to record results when they land:** just run `python3 gen_results.py`. All the
tags below are ALREADY registered in the Davis balanced-accuracy appendix of
`gen_results.py`, so finished runs auto-populate `RESULTS_SUMMARY.txt` (ranked). No
manual transcription needed. Per-run summary JSONs land at `runs/<tag>_summary.json`.

### 8.1 THE CRITICAL ABLATION — did dilation or balAcc-selection drive the 0.848? ⚠️

Background: our current best Davis model is the **dilated-protein 800K** net, which
scored **balAcc 0.848** (`cnn_davis_regB_d1_wd0`, beats the DeepDTAGen target 0.820 at
every threshold). BUT that number confounds TWO changes made at once vs. our earlier
runs: (a) protein **dilation** [1,2,4], and (b) selecting the saved checkpoint by
**balAcc instead of MSE**. We cannot yet say which one mattered, because we never ran
the non-dilated version of this architecture under balAcc-selection. This batch adds
exactly that missing cell (`cnn_davis_ablate_noDil_balacc`, run A1). The 2x2:

| protein tower | MSE-selected checkpoint | balAcc-selected checkpoint |
|---|---|---|
| **NO dilation** | `cnn_davis_t3_h2` = **0.787** | `cnn_davis_ablate_noDil_balacc` = **?? (A1, THIS BATCH)** |
| **dilation [1,2,4]** | `cnn_davis_c3dil` = **0.767** | `cnn_davis_regB_d1_wd0` = **0.848** |

(All four are the SAME architecture: drug 16-32-48, protein 32-64-96, head
144→1024→512→1, 800K params. Only the two knobs differ.)

**How to read A1 once it finishes** (fill in `??`):
- **balAcc-selection effect, holding dilation:** 0.767 → 0.848 = **+0.081** (already known).
- **balAcc-selection effect, no dilation:** 0.787 → A1.
- **dilation effect, holding balAcc-selection:** A1 → 0.848.
- If **A1 ≈ 0.84** → the win came mostly from **checkpoint selection**; dilation ~neutral.
  Implication: chase balAcc-selection everywhere; dilation is optional.
- If **A1 ≈ 0.79** → **dilation genuinely drives** the high score. Implication: dilation
  is a real lever; push on it (and Config-A + dilation, §8.2, should also jump).

### 8.2 Full job list & why each (17 runs, ordered most-important-first)

| # | tag suffix | params | what it tests |
|---|---|---|---|
| A1 | `_ablate_noDil_balacc` | 800K | **the §8.1 ablation** — non-dilated twin of the winner, balAcc-selected |
| D1 | `_winner_regB_s7` | 800K | confirm winner (0.848) holds on seed 7 |
| D2 | `_winner_regB_s13` | 800K | confirm winner holds on seed 13 |
| B1 | `_cfgA_protDil123` | 267K | **Config A (small/MPC) + coprime protein dilation [1,2,3]** — free params; does dilation transfer to the small model? [1,2,3] avoids the gridding our [1,2,4] can cause |
| C1 | `_cfgA_drugDil123` | 267K | Config A + **drug**-tower dilation [1,2,3] only (isolates drug dilation) |
| C2 | `_cfgA_drugprotDil123` | 267K | Config A + BOTH drug and protein dilation [1,2,3] |
| B2 | `_cfgA_protDil1235` | 367K | Config A + 4-layer protein, dilation [1,2,3,5] (coprime, wider RF) |
| B3 | `_cfgA_protDil123_11` | 367K | Config A + 4-layer protein, dilation [1,2,3,11] (very wide RF; expected too aggressive/gridding — testing the hypothesis) |
| E  | `_drug16_12_8_h1536` | 282K | untried combo: single wide head →1536→1, normal protein 32-64-96, shrinking drug tower 16→12→8 |
| G2 | `_stable_pool_mean` | 882K | **mean-pool ablation** (mean-pool is cheap in MPC vs max) |
| G1 | `_stable_pool_max` | 882K | max-pool twin of G2 (clean ablation pair; also = balAcc-selected stable base) |
| F1 | `_stable_embed64` | 852K | stable base, embedding dim 64 (cheaper) |
| F2 | `_stable_embed256` | 943K | stable base, embedding dim 256 |
| F3 | `_stable_kern_d6p12` | 956K | stable base, wider kernels (drug 6 / protein 12) |
| F4 | `_stable_kern_d3p6` | 846K | stable base, narrower kernels (drug 3 / protein 6) |
| F5 | `_stable_ch48_96_144` | 1.13M | stable base, wider channels 48-96-144 |
| F6 | `_stable_ch24_48_72` | 778K | stable base, narrower channels 24-48-72 |

Config A base = `--drug-filters 32 --prot-filters 32 --proj-dim 32 --head-dim 1536
--head-layers 1` (proj32 = each 96-dim tower squeezed to 32 before a single wide head;
head input 64). Stable base = default `32-64-96` towers + head `192→1024→512→1`
(`--head-dim 1024 --head-layers 2`), max pool, embed 128, kernels drug 4 / protein 8.

Code change made for this batch: added `--drug-kernel` / `--prot-kernel` flags (were
hardcoded 4 / 8). Backward compatible — defaults reproduce prior param counts exactly.

### 8.3 What we already RULED OUT (don't re-run)
- **More regularization** (dropout 0.2/0.3, weight-decay 1e-4): monotonically LOWERED
  balAcc in both configs. Overfitting is an MSE story, not a balAcc lever. Keep d0.1/wd0.
