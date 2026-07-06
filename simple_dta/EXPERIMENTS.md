# simple_dta — Experiment Log

Canonical record of every affinity-prediction experiment in `simple_dta/`, plus the
published reference numbers we compare against. **Read this first** when returning to
the project so we don't re-run or re-retrieve anything.

Goal: from-scratch, **affinity-only** DTA models far simpler than DeepDTAGen (which
trains >1 week), within **2–5 points** of DeepDTAGen accuracy, as a base for a later
**MPC** (privacy-preserving inference) protocol. CNN is the current front-runner on
accuracy-per-parameter.

Last updated: 2026-07-06. **If resuming: read Section 9 first** — it documents the
tower-ablation + protein-tower interpretability + rule-based motif-scan replacement
work (drug-only vs protein-only vs full model, plus swapping the learned protein
CNN for 5 hand-scored motifs), all now complete. Section 8 (the 2026-07-04 batch,
also complete) resolved the dilation-vs-balAcc-checkpoint-selection question.

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

---

## 9. Tower-ablation + protein-tower interpretability (2026-07-06, complete)

Motivating questions from Steven: (1) how much does the model lose if the protein
tower is removed and only the drug embedding predicts affinity? (2) same for
removing the drug tower — is protein-alone better than random guessing, and by how
much? (3) can the protein tower's learned filters be read out and replaced by a
hard-coded rule (e.g. a sliding-window motif match) instead of a learned CNN?

### 9.1 Code changes
- `models.py` / `train.py`: new `--ablate {none,drug,protein}` flag on `CNNDTA`.
  Unlike zeroing an input on the joint model, this **retrains a single-tower model
  from scratch** (head only sees the surviving tower), which is what actually
  answers "how good could a drug-only/protein-only model be" rather than "how much
  does the joint model rely on this input". `PredictionHead.forward` now takes
  `*parts` (1 or 2 tensors) instead of a hardcoded `(drug, protein)` pair.
- `mean_baseline.py` (new): trivial floor — predict the train-set mean affinity for
  every test example, report the same metric bundle, no model at all. Davis floor:
  **MSE 0.8015, CI 0.500, rm2 0.000, balAcc_mean 0.500** (exactly random, as expected
  for a constant predictor). Also dumps `_pred.txt`/`_true.txt` so it folds into
  `RESULTS_SUMMARY.txt` via the normal `gen_results.py` panel() mechanism.
- `interpret_protein_tower.py` (new): post-hoc, no retraining, works on any existing
  `runs/<tag>_best.pth` + its `_summary.json`. Two analyses:
  1. **Channel knockout** — zero each final-layer protein channel (post-ReLU,
     pre-pool) one at a time, measure ΔMSE on the test set. 96 channels × ~20
     batches, all forward-only — runs in well under a minute on CPU.
  2. **Motif extraction** — for the top-k channels from (1), use the argmax
     position under max-pool (the position that *is* the pooled value) to find
     each channel's most-activating window in the raw protein string. Window
     length = the tower's **receptive field**, computed exactly from the trained
     model's own `Conv1d.kernel_size`/`.dilation` via
     `RF = (kernel-1) * sum(dilations) + 1` (matches the RF=22 figure from §5 for
     the non-dilated tower; generalizes correctly to dilated towers too — verified
     RF=50 on the `regB_d1_wd0` [1,2,4]-dilated checkpoint). Dedupes by window text
     before ranking since Davis reuses each protein across ~68 drug pairs on
     average (raw top-N would otherwise just repeat one protein's window).

### 9.2 Interpretability finding (already run, no bugs, real signal)
Ran on `cnn_davis_stable_pool_max` (882K, plain 32-64-96 towers, no dilation) and
`cnn_davis_regB_d1_wd0` (800K, protein dilation [1,2,4]) — **both models' top
knockout-ranked channels correspond to real, textbook protein-kinase catalytic
motifs**, found with zero biological priors:
- `regB_d1_wd0` channel 50 (ΔMSE 0.0109): consensus
  `FQILRGLAYCHRRKILHRDLKPQNLLINERGELKLADFGLARAKSVPTKT` — contains **both the HRD
  motif** (catalytic-loop His-Arg-Asp) **and the DFG motif** (activation-loop
  start), the two most conserved motifs across the human kinome.
- `regB_d1_wd0` channel 57 (ΔMSE 0.0317, the single most important channel):
  spans DFG through the **APE motif** (activation-loop end):
  `...IADFGLARVIEDNEYTAREGAKFPIKWTAPEAINFG...`
- `stable_pool_max` channel 69: `...FPIKWTAPEA...` — same APE-region motif,
  found independently in the non-dilated model too.

Strong early evidence for Steven's Q3: a handful of channels dominate (knockout
ΔMSE drops off sharply after the top 2-3), and what they've learned is a small,
nameable set of conserved sequence motifs — exactly the kind of thing a
hard-coded PWM/regex sliding-window match could plausibly reproduce instead of a
learned conv stack. **The actual replacement-and-compare step is in §9.4** —
short answer: yes, largely (recovers ~58% of the drug-only-to-full-model gap).
Output JSON per run: `runs/<tag>_interpret.json`.

### 9.3 Tower-ablation training batch — DONE (launched + finished 2026-07-06)
`jobs_tower_ablation.txt`, launched via
`setsid nohup bash run_queue.sh jobs_tower_ablation.txt 4 10 > runs/tower_ablation_batch.log 2>&1 < /dev/null & disown`.
Davis, 100 epochs, balAcc-selected, dropout 0.1, wd 0, seeds 4221 + 7 (2 seeds
per ablation). All 4 tags registered in `gen_results.py`'s Davis appendix and
folded into `RESULTS_SUMMARY.txt`.

| tag | ablate | params | cost | best MSE | best CI | balAcc_mean |
|---|---|---|---|---|---|---|
| `cnn_davis_ablate_drugtower_only` (s4221) | `protein` (drug tower survives) | 682K | ~6s/epoch, ~13min | 0.6876 | 0.7408 | 0.589 |
| `cnn_davis_ablate_drugtower_only_s7` | `protein` | 682K | ~13min | 0.6952 | 0.7438 | 0.586 |
| `cnn_davis_ablate_prottower_only` (s4221) | `drug` (protein tower survives) | 726K | ~112s/epoch, ~186min | 0.7617 | 0.6507 | 0.541 |
| `cnn_davis_ablate_prottower_only_s7` | `drug` | 726K | ~186min | 0.7396 | 0.6524 | 0.539 |

Reference points: full model (both towers) `cnn_davis_stable_pool_max` =
balAcc_mean **0.841**, MSE 0.267; floor `mean_baseline_davis` = balAcc_mean
**0.500**, MSE 0.802.

**Answers Q1/Q2 directly:** drug-tower-only (avg 2 seeds) = balAcc **0.586**,
MSE ~0.69 — clears the floor (0.50/0.80) by a lot but nowhere near the full
model (0.841/0.267). protein-tower-only (avg 2 seeds) = balAcc **0.539**, MSE
~0.75 — barely above the floor. **On Davis, drug IDENTITY alone carries far
more signal than protein IDENTITY alone** (different drugs have systematically
different affinity distributions across the whole panel, independent of which
protein they're tested against; Davis's ~442 proteins are comparatively less
individually distinguishing for affinity than its drugs are). Protein tower
was confirmed ~18x more expensive per epoch than the drug tower (sequence
length ~1200 vs SMILES ~85) — matches the plan's estimate.

### 9.4 Rule-based motif-scan replacement for the protein tower (2026-07-06)

Direct test of the §9.2 hypothesis: replace the ENTIRE learned protein CNN
tower with a fixed, hand-scored feature vector — one scalar per top-ranked
channel, computed by scanning the raw sequence for the best match to that
channel's PWM (no learned protein-side parameters at all; only the drug
SmilesCNN is learned, same as the drug-tower-only ablation, plus 5 extra
fixed input dims to the head).

**New code:**
- `interpret_protein_tower.py --save-pwm [--pwm-n N] [--pwm-pseudocount P]`:
  fits a proper position weight matrix per top channel (frequency table over
  N=300 aligned top-activating windows, not just the majority-vote consensus
  string used for the printed preview) and writes `runs/<tag>_pwm.json`
  (log-odds vs. a TRAIN-split amino-acid background, so the score isn't fit
  and evaluated on the same data).
- `rule_features.py`: `fit_pwm`/`load_pwms`/`scan_max_score`/`RuleFeaturizer`.
  Scores every window of a sequence against a PWM (vectorized via
  `sliding_window_view`), keeps the max (mirrors the max-pool the real
  channel used). `RuleFeaturizer` caches by sequence text so Davis's protein
  reuse (~68 drug pairs/protein) only pays the scan cost once per unique
  protein (379 unique train proteins, not 25050 rows).
- `train_rule.py`: `RuleProteinDTA` = learned `SmilesCNN` (drug) + the fixed
  k-dim rule-feature vector (standardized with TRAIN mean/std) concatenated
  straight into the `PredictionHead`. Mirrors `train.py`'s loop/summary
  format exactly so results fold into `RESULTS_SUMMARY.txt` normally.

**Bug caught and fixed before trusting any of this:** the first PWM fit used
all 25 letters `CHARPROTSET` can encode (including ambiguous/rare codes
B/U/X/Z/O). `X` has background frequency ≈5×10⁻⁶ in the real training
corpus — so any position where a pseudocount alone gives it a nonzero
probability produces a log-odds ratio that blows up to a huge, meaningless
score. Verified this was live: every one of the 5 channels' "best-scoring
window" on a test sequence was an artifact region full of the letter used
for padding, not the real embedded motif. **Fix:** restrict the PWM/scoring
alphabet to the 20 standard amino acids only; any other character (in
sequence or padding) scores neutral (0), the same treatment already used for
out-of-vocabulary symbols. Re-validated after the fix: a PWM's own top hit
window, embedded in genuinely random 20-AA padding, scores 8.5 standard
deviations above the mean of 200 random 200-residue sequences, and other
channels' PWMs correctly score much lower on the same window. Lesson: always
re-derive background frequencies over the same restricted alphabet used for
the foreground counts, or rare symbols silently dominate log-odds scores.

**`rule_davis`/`rule_davis_s7` — DONE.** 100 epochs, balAcc-selected, 687K
params (drug tower 682K + ~5K for the 5 extra head input dims), using the 5
PWMs fit from `cnn_davis_stable_pool_max` (channels 74/84/69/56/2, §9.2).

**Second bug caught mid-flight (2026-07-06, same session): an N-terminal
artifact contaminated 4 of the 5 templates.** All 5 fitted templates showed
an identical, high-confidence `M` then `A` at positions 1-2 — implausible for
independent motifs. Checked directly: for channels 74/69/56/2, the
"best-matching window" landed at position 0 (the very start of the protein)
for **53-84% of test proteins**; only channel 84 was clean (32% at position 0,
median position 211). Nearly every protein starts with Met plus a similar
N-terminal/tag region (plausibly a shared expression-construct tag across
Davis's kinase clones, not real kinase biology) — for channels without many
genuinely strong internal hits, this generic boundary region became the
"tallest" activation by default, and our top-300-by-value window pool included
it. **Fix:** added `--min-window-pos` (default 15) to
`interpret_protein_tower.py`, dropping any candidate window that starts before
that position, for both the printed preview and the PWM fit. Re-ran on
`cnn_davis_stable_pool_max`: consensus strings are now clean again
(`...FPIKWTAPEA...` for channel 69, `...MAPEV...` for channel 2,
`...WSMGVIMYEMLC...` for channel 84 — the same real motifs as the original
small-preview find in §9.2, now confirmed at proper statistical support).
**Side finding:** channel 74 (the single highest knockout-importance channel)
only has **47 genuine non-artifact hits** across the whole test set, vs. 246
for channel 84 — meaning channel 74's apparent importance in the *original*
knockout ranking may be disproportionately driven by the N-terminal/tag
pattern rather than real kinase biology. Possible shortcut-learning signal,
worth another look if the model's importance ranking gets revisited.

**Ran both versions in parallel for a clean before/after comparison** (old
contaminated PWMs left running rather than killed, since they were already
>50% done when the second bug was found) — `rule_davis`/`rule_davis_s7`
(contaminated) vs. `rule_davis_v2clean`/`rule_davis_v2clean_s7`
(artifact-filtered), all 100 epochs, balAcc-selected. Both pairs registered
in `gen_results.py` and folded into `RESULTS_SUMMARY.txt`.

**Final results (§9.4 payoff):**

| model | params | protein-side compute | best MSE | best CI | balAcc_mean |
|---|---|---|---|---|---|
| Full model (both towers) | 882K | learned 96-channel CNN | 0.267 | — | **0.841** |
| **Rule v2 (clean)**, avg 2 seeds | 687K | 5 fixed PWM scans | 0.38-0.39 | 0.843/0.844 | **0.733** |
| Rule (N-term-contaminated), avg 2 seeds | 687K | 5 fixed PWM scans (buggy) | 0.38-0.41 | 0.834/0.840 | 0.723 |
| Drug-tower-only, avg 2 seeds (§9.3) | 682K | none | ~0.69 | ~0.74 | 0.586 |
| Protein-tower-only, avg 2 seeds (§9.3) | 726K | learned 96-channel CNN | ~0.75 | ~0.65 | 0.539 |
| Floor (mean-affinity) | — | none | 0.80 | 0.50 | 0.500 |

**Bottom line:** replacing the *entire* learned protein CNN tower with **5
fixed PWM motif-match scores** takes drug-only's balAcc from 0.586 to 0.733 —
recovering about **58% of the full drug-only-to-full-model gap** (0.841 −
0.586 = 0.255 total; the rule scan recovers 0.147 of it), at 687K params (vs
882K) with the entire protein side reduced to 5 sliding-window scans instead
of a 3-layer conv stack. The corrected (artifact-filtered) version beats the
contaminated one by +0.010 balAcc — confirming the N-terminal artifact was
net noise, not a useful (if biologically spurious) signal. The full learned
tower still leads by a real margin (0.841 vs 0.733), so 5 motifs don't fully
replace 96 channels — but they capture a large share of the value at a
fraction of the size and with zero learned protein-side parameters.

**Deferred (user's call, 2026-07-06):** whether the 96 protein-tower channels
are mostly redundant with each other vs. genuinely low-value — see memory
[[pending-channel-redundancy-check]]. Natural next steps if resuming this
thread: (a) try more than 5 motifs (does balAcc keep climbing toward 0.841,
or plateau?), (b) the redundancy check above, (c) repeat drug-only vs
protein-only vs rule-scan on KIBA (4x more data, far less protein reuse than
Davis's ~68x/protein) to see if the drug/protein signal balance shifts.
