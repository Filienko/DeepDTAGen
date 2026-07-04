# Lightweight DTA — simplified drug–target affinity models

This repository is a **fork of [DeepDTAGen](#about-the-deepdtagen-fork)** that we use as a
baseline and reference. **Our own work lives entirely in [`simple_dta/`](simple_dta/)**: a
set of small, from-scratch models for predicting drug–target binding **affinity**, built to
be far simpler than DeepDTAGen and eventually run under a privacy-preserving (MPC) protocol.

> ## 👉 Start here
> - **Playing with the models / learning the code:** read **[`simple_dta/README.md`](simple_dta/README.md)** — the full guide.
> - **What we've tried and found:** **[`simple_dta/EXPERIMENTS.md`](simple_dta/EXPERIMENTS.md)** (lab notebook) and **[`simple_dta/RESULTS_SUMMARY.txt`](simple_dta/RESULTS_SUMMARY.txt)** (results table).
> - **The original DeepDTAGen docs:** [`DEEPDTAGEN_README.md`](DEEPDTAGEN_README.md).

---

## What is this project?

DeepDTAGen is a large (~3.6M-parameter affinity path, >1 week to train) multitask model that
both *predicts affinity* and *generates drugs*. We only care about the **affinity prediction**
half, and we want it **small and cheap** — because our end goal is secure multi-party
computation (MPC) inference, where every parameter and every non-linear operation costs a lot.

So in `simple_dta/` we built two clean, minimal models from scratch:
- a **1D-CNN** (DeepDTA-style, our best), and
- a **GNN** (graph-based drug encoder), for comparison.

They are **267K–882K parameters**, train in **hours on CPU** (not a week on GPU), and already
land within a few points of DeepDTAGen — and *beat* its balanced-accuracy target on Davis with
our best configuration. See `simple_dta/RESULTS_SUMMARY.txt`.

---

## Repository map — what's ours vs. what's the fork

| Path | Whose | What it is |
|---|---|---|
| **`simple_dta/`** | **Ours** | **All of our work.** Models, training, metrics, experiments, results. This is where you work. Has its own detailed README. |
| `DEEPDTAGEN_README.md` | Upstream | DeepDTAGen's original README (renamed so ours can take the top slot). |
| `model.py`, `training.py`, `test.py`, `generate.py`, `create_data.py`, `utils.py`, `FetterGrad.py`, `generation_eveluation.py` | Upstream | DeepDTAGen's original code. Untouched. We keep it to run DeepDTAGen as a reference baseline. |
| `models/`, `saved_models/` | Upstream | DeepDTAGen's trained weights (git-ignored; large). |
| `data/` | Shared | Dataset CSVs (`davis_*.csv`, `kiba_*.csv`, `bindingdb_*.csv`) used by **both** DeepDTAGen and our `simple_dta/` code. |
| `DEMO/`, `Affinities/`, `generated_results/`, `logs/` | Upstream | DeepDTAGen demo scripts and output artifacts. |
| `environment.yml`, `run_pipeline.sh` | Upstream | DeepDTAGen environment spec and pipeline runner. |
| `*.pdf`, `model.png` | Upstream | The DeepDTAGen paper and architecture figure. |
| `archive/` | Ours | Retired scratch: `simple_dta_scratch/` (superseded job files/scripts) and `deepdtagen_repro_notes/` (early reproduction notes). Nothing here is needed to run anything. |

**Rule of thumb:** if it's in `simple_dta/`, it's ours and it's current. Everything at the
repo root is DeepDTAGen's original code (kept as a reference baseline), except this `README.md`
and the `archive/` folder.

---

## Setup

Both our code and DeepDTAGen share one conda environment:
```sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepDTAGen        # PyTorch 1.12 (CPU), PyTorch-Geometric 2.2, RDKit
```
To build it from scratch, see `environment.yml` and the install steps in
`DEEPDTAGEN_README.md`.

**Data.** The dataset CSVs live in `data/` and are used by both our code and DeepDTAGen. They
are large (KIBA train ~80 MB) and **not** stored in git. On this machine they're already in
place; on a fresh clone, extract `data.rar` into `data/` first.

### Run our model (30-second version)
```sh
cd simple_dta
python train.py --model cnn --dataset davis --epochs 100 --select-metric balacc --tag-suffix _test
```
Full details, all the knobs, and worked examples are in
[`simple_dta/README.md`](simple_dta/README.md).

---

## About the DeepDTAGen fork

The upstream project is *DeepDTAGen: a multitask deep learning framework for drug-target
affinity prediction and target-aware drug generation* (Shah et al., *Nature Communications*
16:5021, 2025). Its code, demo, paper PDF, and original README (`DEEPDTAGEN_README.md`) are
preserved at the repo root and left unmodified. We compare our simplified models against its
published numbers throughout `simple_dta/EXPERIMENTS.md`.
