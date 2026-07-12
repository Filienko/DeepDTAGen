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

---

## 10. Attention / transformer architectures (2026-07-06, built; batch pending)

Motivation: CNN/GNN have been the only two families tried. Attention is a
plausible third axis on both accuracy (global receptive field from layer 1 —
see the RF~22-residue bottleneck in section 5 — plus drug/protein interaction
modeling instead of late-fusion concat) and MPC-efficiency (round-complexity:
fewer layers needed for global RF *if* the softmax cost is dealt with). Full
design discussion (toy attention walkthrough, MPC cost breakdown, AttentionDTA
comparison) happened in conversation, not repeated here — this section is the
implementation record.

### 10.1 Design: one configurable model, not five

`AttnDTA` (`models.py`) spans every proposal discussed as a flag combination,
sharing tested code (towers/fusion/pooling/head) rather than duplicating
near-identical classes:

| flag | values | meaning |
|---|---|---|
| `--drug-encoder` | `cnn` \| `gnn` \| `gat` | `gnn` = plain GCNConv (GNNDTA-parity control); `gat` = GATv2Conv, content-based dynamic edge weighting, optional bond features |
| `--protein-encoder` | `cnn` \| `transformer` | `transformer` = self-attention tower, global RF from layer 1 |
| `--fusion` | `concat` \| `cross` | `cross` = bilateral cross-attention between tower sequences before pooling (AttentionDTA-style), replacing pool-then-concat |
| `--attn-kind` | `softmax` \| `linear` | `linear` = softmax-free linear attention (Katharopoulos-style), O(n) not O(n^2), no exp/divide -- the MPC-motivated variant |
| `--use-edge-feats` | flag | drug_encoder=gat only: condition attention on bond features (see 10.2) instead of connectivity alone |
| `--gat-dim/-layers/-heads`, `--attn-dim/-heads`, `--prot-attn-layers/-window` | | sizing knobs, same spirit as existing `--gcn-dim` etc. |

Proposal-letter mapping (from the design conversation):
- **A** (AttentionDTA-style bilateral fusion): `drug=cnn, protein=cnn, fusion=cross`
- **B** (protein self-attention tower swap): `drug=cnn, protein=transformer, fusion=concat`
- **C** (GAT + bond features drug tower): `drug=gat, protein=cnn, fusion=concat, use_edge_feats`
- **D** (softmax-free ablation): `--attn-kind linear` on any of the above
- **E** (graph-attention + cross-fusion, the "combine GNN with attention" idea): `drug=gat, protein={cnn,transformer}, fusion=cross, use_edge_feats`

New file `attention.py`: `MultiHeadAttention` (softmax/linear), `PositionalEncoding`,
`TransformerEncoderBlock`, `ProteinTransformer`, `CrossAttentionFusion`, `masked_pool`.
`models.py` adds `GraphAttnEncoder` (GATv2 stack), `_GCNStack` (drug_encoder="gnn"
control), and `AttnDTA`. `SmilesCNN`/`ProteinCNN` gained an additive `return_seq`
flag (pre-pool feature map, for cross-attention fusion) -- existing call sites
unaffected.

### 10.2 Bond (edge) features -- previously computed by nobody

`data.py` carried `edge_index` (connectivity) only; bonds had zero features, so
every GNN here (GCNConv) aggregated neighbors with purely structural,
degree-normalized weights, blind to *what kind* of bond connects two atoms.
Added `bond_features()`: bond-type one-hot (single/double/triple/aromatic +
unk), conjugation flag, ring-membership flag, stereo one-hot (none/Z/E + unk)
-- 11 dims (`EDGE_FEATURE_DIM`). Always computed in `smile_to_graph`/
`build_graph_dataset` (cheap; RDKit already parsed the molecule for atoms), so
it's backward compatible -- GCNConv-based models simply never read
`data.edge_attr`. `GATv2Conv(..., edge_dim=EDGE_FEATURE_DIM)` consumes it when
`--use-edge-feats` is set.

### 10.3 Important finding: naive "windowed" attention doesn't save compute

First implementation of `ProteinTransformer` bounded self-attention to a local
window via a boolean mask applied *after* computing the full `QK^T` score
matrix. This changes the result but **not the cost** -- the O(L^2) matmul still
runs in full regardless of window size. Measured on Davis (protein length
~1200, isolated single-process timing, not the batch-launch's per-job thread
budget):

| config | attn_kind | measured | projected epoch (Davis, ~25050 rows) |
|---|---|---|---|
| A (cnn+cnn, cross) | softmax | 0.40 s/batch (bs=64) | ~310 s |
| B (protein transformer, concat) | **softmax** | 5.84 s/batch (bs=64) | **~4573 s (~76 min)** |
| B (protein transformer, concat) | **linear** | ~0.94s/batch equiv | **~424 s (~7 min)** |
| C (GAT + edge feats, concat) | softmax | 0.05 s/batch (bs=64) | ~39 s |
| D (linear-attn ablation of A) | linear | 0.18 s/batch (bs=64) | ~139 s |
| E (GAT+edge feats+transformer+cross) | **softmax** | 6.13 s/batch (bs=64) | **~4799 s (~80 min)** |
| E (GAT+edge feats+transformer+cross) | **linear** | ~1.06s/batch equiv | **~527 s (~9 min)** |

**Conclusion: `--protein-encoder transformer` requires `--attn-kind linear` in
practice** -- ~11x speedup, and the only way a 100-epoch run finishes in hours
instead of ~5.5 days. This isn't just the MPC-motivated ablation (proposal D)
any more -- it's the only practically trainable configuration for B/E on this
CPU box. Genuinely useful result: the softmax-vs-linear cost gap the MPC
argument predicted (section on attention design) is real and large even in
plaintext, before any cryptographic protocol is involved -- the O(n^2) cost
of full attention over ~1200-residue sequences is expensive on ordinary
hardware too, not just under MPC. `attn_kind="softmax"` remains fine for A/C/D
(drug-side sequences ~85-length SMILES or small molecular graphs, and
cross-attention fusion where one side is always short) since the quadratic
term there is tiny. Docstrings in `attention.py`/`models.py`/`train.py` updated
to state this plainly rather than imply the `--prot-attn-window` flag saves
compute (it doesn't -- true chunked/sparse local attention was not
implemented; flagged as a possible follow-up in section 11 if window-limited
attention is ever needed as a third option between full-softmax and linear).

### 10.4 Smoke tests — all 5 proposals verified end-to-end (2026-07-06)

Full `train.py` CLI, 1 epoch, full Davis train/test split (not a subset),
`_smoke{A..E}` tag suffix. All produced `_best.pth` / `_pred.txt` / `_true.txt`
/ `_summary.json` correctly (same artifact set as `cnn`/`gnn`). Timings below
are **not** representative of per-config cost -- all 5 ran concurrently on
one 48-core box with no thread limits, so each suffered 5-way CPU/memory-
bandwidth contention (measured ~800-1150% CPU per process, i.e. genuinely
computing, not stuck -- confirmed via `ps`). Use section 10.3's isolated,
single-process numbers for actual cost comparisons; this table is a
correctness/pipeline check only.

| proposal | epoch time (5-way contended) | test MSE | CI | rm2 | balAcc |
|---|---|---|---|---|---|
| A (cross-attn fusion) | 1223 s | 0.636 | 0.751 | 0.202 | 0.556 |
| B (protein transformer, linear) | 1637 s | 0.686 | 0.746 | 0.196 | 0.544 |
| C (GAT + edge feats) | 425 s | 0.746 | 0.622 | 0.070 | 0.519 |
| D (linear-attn ablation of A) | 944 s | 0.648 | 0.745 | 0.192 | 0.550 |
| E (GAT+edge+transformer+cross, linear) | 1870 s | 0.818 | 0.575 | 0.037 | 0.508 |

(1-epoch numbers only -- these are sanity checks that the model trains at all,
not accuracy comparisons; e.g. C's lower CI after 1 epoch says nothing about
its 100-epoch ceiling.) All 5 clear random-guessing balAcc (0.500) after a
single epoch, confirming gradients are flowing correctly through every path
(cross-attention fusion, protein transformer, GAT + edge features).

### 10.5 Batch launched (2026-07-06) — `jobs_attn.txt`, 7 runs, Davis, 100 epochs

Launched via `run_queue.sh jobs_attn.txt 4 12` (bounded to 4 concurrent jobs,
12 threads each, same convention as prior batches). balAcc-selected, dropout
0.1, wd 0. Jobs: `A_crossfusion`, `D_crossfusion_linear`, `B_prottransformer`,
`C_gat_edgefeats`, `C2_gat_noedgefeats` (control: GAT without edge features,
isolates what bond features add), `E_gat_transformer_cross` (the composite
"graph attention + cross-fusion" idea), `E2_gat_cross_cnnprot` (E's cheaper
sibling -- same drug/fusion side, but CNN protein tower instead of a
transformer, isolating what cross-fusion + GAT add without also paying for
protein self-attention). **Results: TODO fill in when the batch completes** --
run `gen_results.py` (tags will need registering in its Davis panel list first,
same as every prior batch) and update this section + `RESULTS_SUMMARY.txt`.

Reference points already on the board (no training needed): full CNN baseline
`cnn_davis_stable_pool_max` = balAcc 0.841, MSE 0.267; GNN (plain GCN, no
edge features) `gnn_davis` = balAcc 0.779, MSE 0.270; floor
`mean_baseline_davis` = balAcc 0.500. C/C2 vs `gnn_davis` is the direct
"does GAT + edge features beat plain GCN" comparison (same node features,
same protein tower) this whole exploration was partly motivated by.

**Partial results as of 2026-07-07 (C and C2 complete, A/D/B/E still running):**

| tag | best epoch | MSE | CI | rm2 | balAcc@7.0 |
|---|---|---|---|---|---|
| C (GAT + edge feats) | 85/100 | 0.293 | 0.877 | 0.628 | 0.739 |
| C2 (GAT, no edge feats) | 70/100 | 0.249 | 0.879 | 0.634 | *(≥C on every metric)* |

Surprising early signal: **C2 (no edge features) beat C (with edge features)
on every metric** -- MSE, CI, and rm2 all slightly better without the bond
features. Tentative read: GATv2's learned, content-based attention over plain
connectivity may already capture most of what the hand-crafted bond features
would add, so the explicit edge-feature bias is redundant here rather than
additive. Both clearly beat plain GCN (`gnn_davis` 0.779 balAcc) at this point,
so "attention in the message passing" (GAT vs GCN) looks like the real lever,
not "edge features specifically" -- consistent with, and partly what
motivated, job **F** in section 11 (plain GCN + cross-attention, no GAT at
all) and the G-series (edge features as a pure attention bias, no message
passing at all). Not yet confirmed against A/D/B/E or the CNN baseline
(0.841) since those are still training.

## 11. Attention batch 2 — plain-GCN cross-attn + Graphormer-style raw-node attention (2026-07-07, built; batch launched)

Prompted directly by a question after section 10's early results: "did you
think of graph representation but NOT GATs — either GCN into attention, or
raw graph features straight into a large attention mechanism (no GCN/GAT at
all)?" Two genuinely different ideas, both new:

**F — plain GCN + cross-attention.** This combination was actually already
*possible* in the section-10 code (`--drug-encoder gnn --fusion cross` — the
generic graph branch of `_drug_forward` never assumed GAT specifically) but no
job in `jobs_attn.txt` ever used it: every launched drug-graph config was
`gat`. F fills that gap: `--drug-encoder gnn --protein-encoder cnn --fusion
cross --attn-kind linear`. Directly isolates the section-10.5 open question —
is GAT's *content-based* edge weighting what helps over the plain-GCN
baseline (`gnn_davis`, balAcc 0.779), or is it just having a per-atom sequence
available to cross-attend against the protein, regardless of which graph conv
produced it?

**G — Graphormer-style raw-node attention, no message passing at all.** Not
previously implemented. New class `DrugGraphTransformer` (`attention.py`):
each atom's raw feature vector (no GCN/GAT layer touches it) is linearly
projected to `embed_dim` and self-attended directly against every other atom
in the same molecule — cheap since drug graphs are tiny (<100 atoms, so full
O(n^2) softmax attention costs nothing like the protein-sequence case).
**No positional encoding** — atom order is arbitrary (unlike a protein's
residue sequence), so there's no canonical position to encode, unlike
`ProteinTransformer`. Bond/edge features are *not* fed through any
convolution; instead they're converted to a dense `[B, N, N, edge_dim]` tensor
via `to_dense_adj` and linearly projected to a per-head **additive bias on the
attention scores** (`Linear(edge_dim, n_heads)`, added pre-softmax) — the
actual Graphormer edge-encoding idea. Required adding `attn_bias` support to
`MultiHeadAttention`/`TransformerEncoderBlock` (softmax kind only — linear/
kernelized attention has no explicit `[Lq,Lk]` score matrix to bias, so
`attn_bias` + `kind="linear"` raises `NotImplementedError` by design).

This is a genuinely different point on the spectrum from GAT: GAT still does
message passing, just with content-based (not fixed) edge weights; the
Graphormer tower does *no* message passing — structure enters purely as a
bias term on otherwise-unstructured full attention. `--drug-encoder
graphformer` fed into the same bilateral cross-attention fusion already
working well for A/D (`--fusion cross`) is the main hypothesis; `--fusion
concat` (drug tower alone, no cross-attention) is a secondary control
expected to underperform, since a tower with zero structural inductive bias
in its own convolution likely leans hard on cross-attending against the
protein to do anything useful.

Per explicit request, sized generously for accuracy — "even if too large for
MPC, an interesting in-the-clear result in its own right" — 2x the width/depth
of every other attention component in this project:

| job | drug tower size | fusion | params |
|---|---|---|---|
| F | gnn (gcn_dim=128, 2 layers) | cross, linear | 1.08M |
| G1 | graphformer 256d / 8 heads / 4 layers | cross, softmax | 3.90M |
| G2 | graphformer 512d / 8 heads / 6 layers (xlarge) | cross, softmax | 15.5M |
| G1-concat | graphformer 256d / 8 heads / 4 layers (same as G1) | concat, softmax | 3.12M |
| H | graphformer 256d / 8 heads / **0 layers** (raw atoms, no self-attn at all) | cross, softmax | 1.79M |

For reference: the CNN baseline is 882K params, and the largest section-10
config (E) was 1.28M — G2 is ~17x the baseline and ~12x anything tried so far.

Smoke-tested (in-process forward/backward on a real 8-molecule Davis batch,
3 optimizer steps, loss decreasing, no NaN/Inf) before launching. Launched via
`run_queue.sh jobs_attn2.txt 2 6` (lower concurrency/threads than batch 1's
`4 12`, since batch 1's A/D/B/E were still running and already saturating the
48-core box — deliberately under-provisioned this second queue rather than
doubling total thread demand past capacity).

**H — added after a follow-up clarification.** The user's actual intent for
G was narrower than what got built: not intra-drug self-attention layers
followed by cross-attention, but the raw drug graph representation itself
participating directly in cross-attention with the protein — i.e. skip any
intra-drug processing (no GCN, no GAT, no self-attention) entirely, and let
cross-attention alone relate individual atoms to protein residues. This was
already expressible with the existing code via `--drug-attn-layers 0`
(`DrugGraphTransformer` with zero self-attention blocks): it linearly embeds
each atom's raw feature vector and returns that sequence unchanged, so no new
code was needed. Added as job **H**: identical to G1 in every setting
(`drug_attn_dim=256, attn_dim=256, attn_heads=8`) except `drug_attn_layers`:
0 vs G1's 4 — a clean, controlled A/B isolating whether *any* intra-drug
self-attention adds value over raw atoms + cross-attention alone. 1.79M
params. Smoke-tested the same way, launched standalone (not through the
`jobs_attn2.txt` queue, to avoid a file-append race with the already-running
`run_queue.sh` process — appended to the job file for the record regardless).

**Also flagged as a distinct, not-yet-tried idea (see project memory
`pending-unilateral-cross-attn-ablation`):** `CrossAttentionFusion` is already
bilateral (both `drug_from_prot` and `prot_from_drug` run every time
`fusion=cross` is used) — nobody has tried a **unilateral** version (only one
direction, with the other side staying a plain pooled vector). Cheap ablation,
halves cross-attention cost if one direction turns out to dominate — natural
follow-up once F/G1/G2/H are compared.

## 12. Batches 1+2 final results, and batch 3 — validation + MPC-efficiency follow-up (2026-07-09)

### 12.1 Final leaderboard (Davis, mean balAcc across thresholds 5.0-8.5)

| rank | tag | config | params | mean balAcc |
|---|---|---|---|---|
| 1 | A | cnn+cnn, cross-attn, **softmax** | 1.1M | **0.840** |
| 2 | F | gnn (plain GCN)+cnn, cross-attn, linear | 1.08M | 0.831 |
| 3 | H | raw atoms (0 self-attn)+cnn, cross-attn, softmax | 1.79M | 0.826 |
| 4 | F_s7 | F, seed replicate | 1.08M | 0.817 |
| 5 | E2 | gat+edge+cnn, cross-attn, softmax | 1.1M | 0.814 |
| 6 | J | graphformer(0L)+cnn, cross-attn, **linear** (fully softmax-free) | 1.09M | 0.803 |
| 7 | K2 | F, UNILATERAL `prot2drug` cross-attn | 1.01M | 0.800 |
| 8 | I | gnn+cnn, cross-attn, softmax | 1.08M | 0.798 |
| 9 | K1 | F, UNILATERAL `drug2prot` cross-attn | 1.01M | 0.793 |
| 10 | D | cnn+cnn, cross-attn, linear | 1.1M | 0.788 |
| 11 | E | gat+edge+protein-transformer, cross-attn, linear | 1.28M | 0.787 |
| 12 | C2 | gat (no edge feats), concat | 918K | 0.785 |
| 13 | B | cnn+protein-transformer, concat, linear | 1.08M | 0.773 |
| 14 | C | gat+edge, concat | 918K | 0.741 |
| — | G1/G2/G1-concat | Graphormer-style drug self-attn (4-6 layers) | 3.9-15.5M | **~0.50 (collapsed)** |

Reference: DeepDTAGen (paper-reported balAcc target) = 0.820, at 3.58M params
(affinity path only).

**Takeaways driving batch 3:**
1. **Cross-attention fusion is the winning ingredient regardless of what
   produces the per-token sequence** — CNN (A), plain GCN (F), or literally
   raw unprocessed atom features (H) all land in the top 3, well above every
   concat-fusion config (C/C2/B) and above the DeepDTAGen target. The specific
   drug-tower architecture underneath matters far less than *whether
   cross-attention fusion happens at all*.
2. **F is the most promising MPC candidate found so far**: linear (softmax-
   free) cross-attention, plain GCN drug tower, 0.831 — only 0.9pp below A's
   softmax ceiling, while avoiding the exp/divide cost of softmax entirely.
   D (the direct cnn+cnn linear-attention analog of A) only reached 0.788 —
   so the GCN drug tower isn't just tagging along, it's doing real work to
   close most of the gap that linear attention alone opens up.
3. **G1/G2/G1-concat all collapsed to random-guessing balAcc (~0.50)**,
   including G1-concat which has *no cross-attention at all* — ruling out
   cross-attention as the cause and isolating the failure to the Graphormer's
   own stacked self-attention layers (4-6 layers, no message passing, no
   positional/multi-hop structural encoding, trained from scratch with the
   same optimizer/LR schedule as every other config here, no warmup). Given
   this is also the direction *least* attractive for MPC (more attention
   layers = more expensive, not less), not pursued further for now — the
   inductive-bias argument (small molecules favor GCN/GAT's locality prior
   over unstructured full attention, especially at this data scale) looks
   right in hindsight.
4. **A vs. published AttentionDTA — not a literal reimplementation.**
   Checked against the actual repo (`zhaoqichang/AttentionDTA_TCBB`):
   AttentionDTA's own "attention" is a custom tanh-gated interaction map,
   mean-pooled across 8 heads at 768 total width (96/head) — not softmax
   attention at all. Our A uses genuine transformer-style scaled-dot-product
   softmax attention at 128 total width (32/head, ~1/6 the capacity), a
   shallower 2-hidden-layer head (vs their 3), and flat instead of graduated
   conv kernels. A converges on the same idea (cross-attention as fusion,
   replacing concat) independently, with a narrower, more standard attention
   block — and still edges out their reported number. Suggests the width/
   exact formulation of the attention block matters less than *having*
   cross-attention fusion.

### 12.2 Batch 3 (`jobs_attn3.txt`) — five jobs, validation + MPC efficiency

New model flag: `AttnDTA(..., cross_direction={both,drug2prot,prot2drug})`
threading through to `CrossAttentionFusion` (`attention.py`) — unilateral
directions build only one `MultiHeadAttention` module instead of two (not
just a discarded computation), so `drug2prot`/`prot2drug` are genuinely
~6% fewer params and roughly half the cross-attention compute vs `both`.
`train.py` gained `--cross-direction`. All 4 new configs smoke-tested
in-process (3 real optimizer steps on an 8-molecule Davis batch, loss
decreasing, no NaN) before launch.

| job | purpose | config | params |
|---|---|---|---|
| **I** | fills the missing {cnn,gnn}×{softmax,linear} grid quadrant (have A, D, F; missing gnn+softmax) — does GCN's edge over D grow or shrink under full softmax attention? | gnn+cnn, cross, softmax | 1.08M |
| **J** | the maximally MPC-cheap candidate: H's raw-atom drug tower (zero self-attention) + **linear** cross-attention instead of H's softmax — fully softmax-free model, single most important run for the MPC goal | graphformer (0 layers)+cnn, cross, linear | 1.09M |
| **F_s7** | seed replication of F (seed 7 vs F's 4221) — confirms the 4.3pp gap over D isn't seed noise before treating "GCN + linear cross-attn" as the reference MPC-efficient config | gnn+cnn, cross, linear, seed=7 | 1.08M |
| **K1** | unilateral cross-attn ablation on F's backbone, drug-queries-protein only (the direction the user predicted matters most: small drug graph reaching into the long protein) | gnn+cnn, cross, linear, `cross_direction=drug2prot` | 1.01M |
| **K2** | unilateral cross-attn ablation, protein-queries-drug only (reverse direction) | gnn+cnn, cross, linear, `cross_direction=prot2drug` | 1.01M |

Davis, 100 epochs, balAcc-selected, all 5 tags registered in `gen_results.py`.
Launched via `setsid nohup bash run_queue.sh jobs_attn3.txt 4 12 & disown`
(2026-07-09 07:46 UTC) — fully detached, independent of any SSH session.
G1/G2 (collapsed, batch 2) were killed early once confirmed diverged, freeing
the box for this batch; killing them required care to avoid disturbing E/H
(still training at the time) and triggered a brief duplicate-launch of H via
`run_queue.sh` re-reading its record-keeping line in `jobs_attn2.txt` when
both G1/G2 slots freed simultaneously — caught and killed within ~15s, only
cosmetic damage (H's plain-text log history before that point), the actual
training process/checkpoints were never touched.

### 12.3 Batch 3 results (final, all 5 jobs `DONE` as of 2026-07-10 03:12 UTC)

| tag | config | params | mean balAcc |
|---|---|---|---|
| F_s7 | gnn+cnn, cross, linear, seed=7 (replicates F) | 1.08M | 0.817 |
| J | graphformer(0L)+cnn, cross, **linear** (fully softmax-free) | 1.09M | **0.803** |
| K2 | gnn+cnn, cross, linear, `prot2drug`-only | 1.01M | 0.800 |
| I | gnn+cnn, cross, **softmax** | 1.08M | 0.798 |
| K1 | gnn+cnn, cross, linear, `drug2prot`-only | 1.01M | 0.793 |

Findings:
1. **J (fully softmax-free) is confirmed as a strong result**: 0.803 mean balAcc
   with *zero* softmax anywhere in the model — beats D, E, C2, B, C, I, and both
   unilateral configs (K1/K2), and trails only F (0.831) and H (0.826, its own
   softmax twin) among cross-attention configs. Best fully-softmax-free config
   found so far.
2. **Unilateral cross-attention costs real accuracy, but the direction that
   "should" matter more doesn't.** Both K1 (drug→prot only, 0.793) and K2
   (prot→drug only, 0.800) trail F's bilateral 0.831 by 3-4pp — but K2 (protein
   querying the drug) came out *ahead* of K1, the opposite of the pre-registered
   guess that drug→protein (small graph reaching into the long sequence) would
   dominate. Read: with linear attention's O(n) cost, the direction doesn't
   change which side "does the work" as much as expected; if only one direction
   is affordable under MPC, prefer `prot2drug` (K2 config) on this evidence.
3. **Softmax vs. linear on the GCN drug tower is now a near-tie, not a clean
   linear win.** With I's final number in (0.798, up from the provisional
   0.778), linear (F=0.831) still leads but by less than the mid-batch read
   suggested — I ≈ J ≈ K2 all land in the same 0.798-0.800 band regardless of
   softmax/linear. The clearest remaining softmax-vs-linear gap is still on the
   CNN drug tower (A=0.840 > D=0.788).
4. **Seed noise is ~1-4pp.** F_s7 (seed 7) landed at 0.817 vs F's original
   0.831 — treat gaps under ~2pp between configs as noise, not signal.

These results, plus the "why did H do so well" question, motivated batch 4
(section 13): shrink F, shrink+attention-ify the small CNN baseline, and
improve/diagnose the graph-aware family (E2, GINE, graphformer layer count).

## 13. Batch 4 (2026-07-10) — shrink F, upgrade the small CNN, edge-aware message passing

Three follow-up directions, requested directly after reviewing the batch 1-3
leaderboard.

### 13.1 Direction 1 (F1-F4): how small can F go?

F (gnn+cnn, cross-attn, linear, 0.831 balAcc) is the best MPC candidate found
so far, but 1.08M params has never been decomposed. A param audit of F's
actual layers: the GCN drug tower is tiny (~29K, node_feat_dim=94 -> 128 x2
layers), the protein CNN tower is ~101K, but **the 1024/512 prediction head
alone is ~788K -- roughly 73% of the whole model.** This mirrors exactly what
made Config A (section 3, `--proj-dim 32 --head-dim 1536 --head-layers 1`)
267K instead of 882K: cutting head width/depth is the single biggest lever,
not the tower. For F, cross-attention's `attn_dim` also enters the head's
input width (`attn_dim * 2`) and the cross-attention module's own params
scale quadratically with `attn_dim` -- so narrowing `attn_dim` compounds with
narrowing the head, unlike a plain concat model where they're independent
knobs.

| job | change from F | params |
|---|---|---|
| F1_lean | attn/gcn-dim 128->64, head 1024/512L2 -> 512L1 | 223K |
| F2_tiny | attn/gcn-dim 128->32 (heads 4->2), head 512L1 | 152K |
| F3_uni_head | + UNILATERAL `prot2drug` (K2's direction -- the one that scored *higher* in batch 3), head 512L1, full attn/gcn-dim kept | 358K |
| F4_uni_lean | UNILATERAL `prot2drug` + attn/gcn-dim 64 + head 512L1 (combines every lever) | 206K |

All smoke-tested (forward+backward on a real Davis batch; F4's exact sibling
config was run for 1 real epoch end-to-end, see 13.3). If F1/F3 hold close to
0.831 while F2/F4 drop off, that pinpoints whether the head or the
attention/GCN width is the safer place to cut for an MPC port.

### 13.2 Direction 2 (A2-A5): upgrade the 267K CNN+CNN baseline (`Config A`, section 3)

Config A (`--drug-filters 32 --prot-filters 32 --proj-dim 32 --head-dim 1536
--head-layers 1`, 267K params) is balAcc-selected at 0.795 -- within ~2.5pp of
DeepDTAGen and the smallest model on the whole board by a wide margin. Two of
its own findings were never combined with it: T3's asymmetric tower result
(lean drug / full protein wins, section 3) and every cross-attention-beats-
concat finding from sections 10-12 (batches 1-3). Also tries: does a
graph+cross-attention drug tower (the "F recipe") still win over plain CNN+
concat once compressed down to the same ~150-250K budget it usually competes
at 1M+?

| job | idea | params |
|---|---|---|
| A2_leantower | Config A + T3's lean-drug/full-protein asymmetry (`drug-filters 24`, never combined with proj/head compression before) | 248K |
| A3_smaller | Config A, compressed further (`proj-dim 16`, `head-dim 1024L1`) | 197K |
| A4_crossattn | cnn+cnn, but **cross-attention fusion (linear, dim 32) instead of concat+proj** -- does cross-attention help even at this tiny scale, or is it a large-model-only effect? | 208K |
| A5_minignn | "mini-F": gnn (gcn-dim 32) + cnn, cross-attn (dim 32, linear, `prot2drug`-only), head 512L1 -- the graph+cross-attention recipe compressed to CNN-baseline scale | 148K |

### 13.3 Direction 3 (E3-E5, L1-L2, H1L-H2L): graph-awareness, edge features, and the H puzzle

The user's framing: pure CNN+CNN towers are the least interesting result here
precisely *because* they ignore the molecular graph -- E2 (GAT+edge features+
cross-attn+softmax, 0.814) is the most appealing model on the board for that
reason even though it's not the top scorer, and the goal is to make *that*
family more accurate and more efficient, not to keep optimizing the
graph-blind CNN.

**E2's untried linear-attention twin, plus a control.** E2 has only ever been
run with `attn-kind softmax`; every other GNN-family drug tower (F, I, K1, K2)
has shown linear attention is at worst competitive there. E3 tests exactly
that swap; E4 drops `--use-edge-feats` under the same linear setting as a
control (E2 vs C2 already showed edge feats didn't help under softmax --
worth checking whether that holds under linear too). E5 combines E3's recipe
with the same shrink/unilateral levers as direction 1.

| job | change from E2 | params |
|---|---|---|
| E3_gat_edge_linear | `attn-kind softmax` -> `linear` | 1.11M |
| E4_gat_noedge_linear | E3 minus `--use-edge-feats` (isolates edge-feature contribution under linear attn) | 1.11M |
| E5_gat_lean | E3 + `gat-dim`/`attn-dim` 128->64, UNILATERAL `prot2drug`, head 512L1 | 218K |

**A genuinely new edge-aware drug tower: GINEConv (`--drug-encoder gine`).**
The taxonomy discussed earlier this session had a real gap: GCN ignores edge
features entirely; GAT uses them to *modulate an attention weight* over
already-adjacent neighbors; Graphormer uses them as a *bias on an
unstructured, non-adjacency-respecting attention score*. None of them sum
edge information directly into the message being passed. `GINEConv`
(Hu et al. 2019, "Strategies for Pre-training GNNs" -- well-established for
molecular property prediction, e.g. OGB) does exactly that:
`x_i' = MLP((1+eps)*x_i + sum_{j in N(i)} ReLU(x_j + edge_lin(edge_attr_ji)))`
-- adjacency is still hard-enforced (only real bonded neighbors contribute,
same as GCN/GAT), but the bond's own features are added directly into each
neighbor's contribution before the sum, not used to gate a coefficient.
New class `_GINEStack` in `models.py`; new `--drug-encoder gine` in
`train.py` (edge features always on for this encoder -- `--use-edge-feats` is
a no-op for it, since GINEConv's whole point requires them). Smoke-tested via
direct forward+backward on a synthetic batch, then a full real 1-epoch run on
Davis end-to-end (`L2_gine_lean`'s exact config: 215,977 params, 87.6s/epoch,
no NaN, loss and balAcc both sane) before launch.

| job | config | params |
|---|---|---|
| L1_gine_cross_linear | GINE (gcn-dim 128) + cnn, cross-attn, linear -- same budget class as F, direct comparison of "edges in the message" vs F's "no edges at all" | 1.12M |
| L2_gine_lean | GINE (gcn-dim 64) + cnn, cross-attn (dim 64, linear, `prot2drug`-only), head 512L1 -- same budget class as the direction-1/2 shrink jobs | 216K |

**The H puzzle: why does zero-structure raw-atom cross-attention (0.826) beat
GAT+edges (0.814) and nearly match F (0.831)?** H has no message passing and
no adjacency at all -- every atom cross-attends the protein directly from its
raw feature vector. G1 (same tower, but with 4 self-attention layers *before*
cross-attention) collapses to random guessing. That's a huge, unexplained
cliff between 0 layers (works great) and 4 layers (total collapse), and nobody
has tried the middle. H1L/H2L interpolate it directly -- same architecture as
H/G1, only `--drug-attn-layers` changes:

| job | drug-attn-layers | (H=0 and G1=4 already on the board) | params |
|---|---|---|---|
| H1L_graphformer1L | 1 | | 2.32M |
| H2L_graphformer2L | 2 | | 2.85M |

Working hypothesis this is meant to test: the collapse is a *training-depth/
optimization* failure specific to stacking multiple untrained self-attention
blocks with no positional encoding and no warmup (each added layer makes the
loss landscape harder to escape from a bad init), not evidence that "no
message passing" is inherently unstable -- since H itself has zero message
passing and trains fine. If H1L trains fine and H2L is where things start to
degrade, that pinpoints the depth at which it breaks and supports the
optimization-failure reading over a representational one. Params scale
steeply with layer count here (2.32M/2.85M vs H's 1.79M) because each added
`TransformerEncoderBlock` is a full 256-wide self-attention + FFN block over
up to ~100 atoms -- expensive to add, which is itself a reason to prefer H's
answer (0 layers) if it turns out 0 is in fact optimal, not just cheapest.

### 13.4 Batch launch

14 jobs total (`jobs_attn4.txt`), Davis, 100 epochs, balAcc-selected. All new
code paths (`gine` drug encoder, all width/direction combinations) smoke-
tested before launch -- either a real 1-epoch end-to-end run
(`L2_gine_lean`'s config: 215,977 params, 87.6s/epoch; `A4_crossattn`'s exact
config: 208,065 params, 79.8s/epoch, both on real Davis data, no NaN) or, for
combinations that only vary a width/direction flag already validated
end-to-end in batches 1-3 (F-family, GAT+linear, graphformer layer count),
a direct `AttnDTA` forward+backward smoke test on a synthetic batch. Launched
via `setsid nohup bash run_queue.sh jobs_attn4.txt 6 8 & disown` (bumped
concurrency to 6 parallel jobs x 8 threads = 48 threads total, up from batch
3's 4x12, since the box was measured near-idle beforehand -- load average
3.1 on 48 cores, other users' jobs using ~3 cores total).

### 13.5 Batch 4 results (final, all 15 jobs `DONE` as of 2026-07-12 00:25 UTC)

**Direction 1 -- shrink F (F=1.08M/0.831):**

| job | change from F | params | balAcc |
|---|---|---|---|
| F3_uni_head | unilateral `prot2drug` + head 512L1 only (full attn/gcn width kept) | 358K | **0.812** |
| F4_uni_lean | + attn/gcn-dim 64 | 206K | 0.782 |
| F1_lean | attn/gcn-dim 64, bilateral | 223K | 0.774 |
| F2_tiny | attn/gcn-dim 32 | 152K | 0.770 |

F3 wins by a wide margin: 3x smaller than F for -2pp. Narrowing attn/gcn width
costs much more than going unilateral does -- confirms the head-dominates-
params finding (13.1): once the head is already cut to 512L1, the next
cheapest lever is the cross-attention *direction*, not its *width*.

**Direction 2 -- upgrade Config A (267K/0.795):**

| job | idea | params | balAcc |
|---|---|---|---|
| A2_leantower | lean-drug/full-protein asymmetry | 248K | 0.788 |
| A4_crossattn | cross-attn (linear, dim32) instead of concat | 208K | 0.776 |
| A3_smaller | proj-dim 16, head 1024L1 | 197K | 0.759 |
| A5_minignn | "mini-F" -- gcn32 + unilateral cross-attn, head 512L1 | 148K | 0.743 |

**None beat Config A.** Every lever that helps at 1M+ params (asymmetric
towers, cross-attention fusion, graph towers) either hurts or does nothing at
the ~150-250K budget -- Config A's own compression (large single-layer head,
plain concat) already sits at this budget class's practical ceiling. Cross-
attention specifically needs a wider `attn_dim`/`gcn_dim` than 32 to pay for
itself; A4 (tiny cross-attn) loses to plain concat A2 at a comparable size.

**Direction 3 -- improve E2 (1.1M/0.814), keep graph-awareness:**

| job | config | params | balAcc |
|---|---|---|---|
| L1_gine_cross_linear | GINEConv (edges in the message) + cross, linear | 1.12M | **0.815** |
| E4_gat_noedge_linear | GAT no edge feats, linear | 1.11M | 0.814 |
| E3_gat_edge_linear | GAT + edge feats, linear (E2's softmax-free twin) | 1.11M | 0.813 |
| E5_gat_lean | GAT+edge, shrunk + unilateral + linear | 218K | 0.792 |
| L2_gine_lean | GINE, shrunk + unilateral + linear | 216K | 0.762 |
| H1L_graphformer1L | raw atoms + 1 self-attn layer, then cross | 2.32M | 0.663 |
| H2L_graphformer2L | raw atoms + 2 self-attn layers, then cross | 2.85M | 0.545 |

Findings:
1. **Linear attention matches softmax on the GAT/GINE drug tower.** E3/E4/L1
   all land 0.813-0.815, indistinguishable from E2's 0.814 softmax version --
   the softmax-vs-linear gap that's real on the CNN tower (A=0.840 vs
   D=0.788) essentially vanishes on graph towers. Good news for the MPC port:
   the most "interesting" (graph-aware, edge-aware) family can go fully
   linear for free.
2. **Edge features still don't help GAT** (E4 no-edge=0.814 vs E3 edge=0.813,
   a tie within noise) -- repeats the C/C2 and E2 pattern from batches 1-3.
   GATv2's content-based attention keeps capturing whatever hand-coded bond
   features would add, regardless of softmax/linear.
3. **GINEConv (edges genuinely summed into the message, not gating an
   attention weight) is the best result in this whole direction** -- 0.815,
   edging out both GAT variants and tying E2's original softmax score, while
   also being fully linear. This is the strongest evidence yet that *how*
   edge features enter the model matters more than *whether* they're used at
   all: GAT's gate-an-attention-weight mechanism gets no lift from edges,
   but GINE's sum-into-the-message mechanism does (however slightly).
4. **Shrinking direction-3 models costs more than shrinking F did**: E5
   (218K) drops 2.2pp from E3, L2 (216K) drops 5.3pp from L1 -- both bigger
   hits than F3's 1.9pp drop from F. Graph+edge towers appear to tolerate
   compression worse than the plain-GCN tower does.
5. **The H puzzle is resolved, and the answer is "fragile, not a cliff."**
   H (0 self-attn layers, raw atoms straight into cross-attention) = 0.826.
   H1L (1 layer) = 0.663 -- a 16pp drop from adding a *single* self-attention
   block. H2L (2 layers) = 0.545 -- already indistinguishable from G1's full
   collapse (4 layers, ~0.50). So the degradation is front-loaded: nearly all
   of the damage happens between 0 and 1 layers, not gradually across 4. This
   supports the training-instability reading over a representational one --
   stacking *any* untrained self-attention block on ~100 unordered atoms with
   no positional encoding destabilizes optimization almost immediately, it
   isn't a capacity/depth tradeoff that gradually degrades. Practical
   takeaway: if using the raw-atom + cross-attention recipe (H's family),
   zero self-attention layers isn't just cheapest, it's necessary.

**Updated overall leaderboard position:** F3 (358K/0.812) and L1
(1.12M/0.815) are both new additions to the top tier alongside A (0.840), F
(0.831), H (0.826), and E2 (0.814) -- L1 is now the best fully-linear,
graph-and-edge-aware model on the board, and F3 is the best small
(<400K) linear-attention model, beating Config A's non-attention 267K/0.795
by 1.7pp at a still-modest size.
