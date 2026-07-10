# The 3 flagship models

This folder is a pointer, not a copy: the actual model code lives one level up
in `../models.py` and `../train.py` (all three models share that code — they're
just different flag combinations). What's here is a **run script per model**
with the right flags already filled in, so you don't need to memorize anything
to reproduce or extend them.

If you only read one thing: run the script, look at the `_summary.json` it
produces, done. Everything below is reference detail.

## The 3 models at a glance

| | **RegB** | **ConfigA** | **F** |
|---|---|---|---|
| What it is | Plain CNN+CNN, 2 towers | Same family as RegB, shrunk | GCN drug + CNN protein, fused by **linear (softmax-free) cross-attention** |
| Role | Our accuracy ceiling | Our small/cheap reference | Our best **MPC candidate** |
| Params | ~800K | ~267K | ~1.08M |
| Davis balanced accuracy | **0.848** | 0.795 | 0.831 |
| Script | `run_regB.sh` | `run_configA.sh` | `run_F.sh` |
| Model class | `CNNDTA` (`../models.py`) | `CNNDTA` (`../models.py`) | `AttnDTA` (`../models.py`, `../attention.py`) |
| Full history | `EXPERIMENTS.md` §8 | `EXPERIMENTS.md` §8 | `EXPERIMENTS.md` §12 |

("Balanced accuracy" = our headline metric, averaged over 8 affinity
thresholds on Davis; 0.500 = random guessing, DeepDTAGen's published
reference is 0.820. See `../README.md` §6 for the full definition.)

## What each model actually is

**RegB** — two independent 1D-CNN towers, one over the drug's SMILES string,
one over the protein's amino-acid sequence, each pooled to a vector and fed
into a 2-layer head. No attention, no molecular graph, nothing MPC-specific —
this is the "how good can a simple model get" ceiling we compare everything
else against. The protein tower uses **dilated convolutions** (gaps between
kernel taps) to see more of the sequence per layer at no extra parameter cost.

**ConfigA** — the same two-CNN-tower recipe as RegB, but compressed: each
tower is squeezed to 32 dimensions before a single wide fully-connected layer
(instead of RegB's 2-layer head). ~3x fewer parameters than RegB for ~5
points less balanced accuracy — this is the model to reach for whenever
"how cheap can we make it" matters more than "how accurate."

**F** — our most promising design for an actual MPC (secure multi-party
computation) protocol. Drug side: a plain GCN over the molecular graph
(atoms=nodes, bonds=edges) — no attention in the message passing itself.
Protein side: the same plain CNN as RegB/ConfigA. The two towers are then
fused with **cross-attention** (drug and protein representations attend to
each other before pooling, instead of just concatenating), using **linear
attention** — a softmax-free formulation with no `exp`/divide anywhere in the
model, which is the expensive part to compute under MPC. F lands within ~1
point of our best *softmax*-attention model while avoiding softmax entirely,
which is why it's the current MPC front-runner. Full rationale for why
softmax is the thing to avoid: `EXPERIMENTS.md` §10.3 and §12.1.

## How to run one

```sh
cd simple_dta/best_models
./run_regB.sh            # trains RegB on Davis, seed 4221 (the reference config)
./run_configA.sh kiba    # same model, different dataset
./run_F.sh davis 7       # same model/dataset, different seed
```

Each script is just `train.py` with the model's exact flags hardcoded — open
any one of them to see precisely what it runs. All three accept:

```
./run_<model>.sh [dataset] [seed] [tag_suffix]
```

| arg | values | default |
|---|---|---|
| `dataset` | `davis`, `kiba`, `bindingdb` | `davis` |
| `seed` | any integer | `4221` (matches the reference numbers above) |
| `tag_suffix` | free text, names the output files | auto-built from dataset+seed |

Other things you can change:
- **Epochs** (for a quick test instead of a full run): `EPOCHS=4 ./run_regB.sh`
- **Everything else** (head size, dilation, attention width, ...) — these
  scripts intentionally hardcode the model definition. To try a *different*
  architecture, copy a script and edit the flags, or see `../README.md` §4
  for the full flag reference.

**Training time (Davis, this CPU box):** RegB/ConfigA ~2-3h each. F is much
slower (graph batching + attention) — budget several hours. KIBA is ~4x more
data than Davis; BindingDB is in between.

## Finding results

Every run writes to `../runs/`:
- `<tag>_summary.json` — final metrics (MSE, CI, rm2, Pearson, balanced
  accuracy per threshold) plus the exact config used.
- `<tag>_pred.txt` / `<tag>_true.txt` — raw predictions vs. truth, so you can
  compute **any other metric** yourself without retraining.
- `<tag>_best.pth` — the trained weights (not stored in git — see note below).

To re-score with different metrics or rebuild the ranked leaderboard:
```sh
cd simple_dta
python gen_results.py          # rewrites RESULTS_SUMMARY.txt
```
`metrics.py` has every metric we compute; add a new one there if needed.

**Already-trained reference runs** (the exact numbers in the table above,
no need to retrain to see them) live in `../runs/` under these tags:
`cnn_davis_regB_d1_wd0`, `cnn_davis_regA_d1_wd0`, `attn_davis_F_gnn_crossfusion`.
Open `../runs/<tag>_summary.json` directly, or run `gen_results.py` and look
them up in `RESULTS_SUMMARY.txt`. Note `*.pth` weight files are gitignored
(too large) — they exist locally on this machine but a fresh clone will need
to retrain via the scripts above to regenerate them (same seed = same result).

## Why these three specifically

Out of ~40 architectures explored (CNN, GNN, GAT, Graphormer, plain-attention
variants — full record in `EXPERIMENTS.md`), these three mark the three
corners we care about: **best accuracy** (RegB), **smallest/cheapest**
(ConfigA), and **best accuracy-per-MPC-cost** (F, the only one of the three
built with an eye toward an actual secure-computation protocol — no softmax,
minimal non-linearities). If you're picking one model to build an MPC
protocol around, start with F; RegB and ConfigA are accuracy/size reference
points to sanity-check against, not MPC candidates themselves (they don't use
attention at all, so there's nothing softmax-related to remove, but they also
don't have F's affinity-per-parameter efficiency).
