# Retrain regB with mean-pool — measure the FSS accuracy trade

**Goal:** find out how much accuracy regB loses if we swap its **max-pool** for
**mean-pool**, so we can drop the single most GPU-hostile op in the network.

## Why mean-pool matters for FSS

regB's secure-inference cost is **347K secure comparisons**, split as:

| stage | comparisons | FSS nature |
|---|---:|---|
| ReLU (3 conv stages + 2 head layers) | **233K** | intrinsic — one DReLU per element; can't remove |
| global **max-pool** (drug + protein towers) | **114K** | a `log(L)` DReLU **tournament** — removable |

Max-pool is ~**33%** of regB's comparisons and it is *sequential depth* (a
comparison tree ⇒ extra online rounds), which is exactly what a GPU can't
parallelize away. **Mean-pool is linear/local — completely FREE under FSS** (it's
just a scaled sum on the shares). So switching max→mean deletes that whole
comparison stage. ReLU is intrinsic to the conv tower and stays either way.

The only question is accuracy: regB's 0.848 balAcc was trained with max-pool. This
retrains it with mean-pool (same architecture, same seed, only `--pool` flipped)
and reports the drop.

## Run it

```sh
bash mpc/retrain_regB_meanpool.sh
```

Env knobs:

| var | default | meaning |
|---|---|---|
| `EPOCHS` | `100` | epochs per run |
| `SEED` | `7` | RNG seed — **same for both runs** so the drop is apples-to-apples |
| `DATASET` | `davis` | dataset |
| `CUDA` | *(unset)* | GPU id → adds `--cuda $CUDA`; CPU if unset |
| `SKIP_MAX` | *(unset)* | `1` = skip the max baseline (reuse an existing run) |

```sh
CUDA=0 bash mpc/retrain_regB_meanpool.sh              # both runs on GPU 0
CUDA=0 SKIP_MAX=1 bash mpc/retrain_regB_meanpool.sh   # mean-pool only (reuse the 0.848 max run)
```

The script trains **regB max-pool** (the 0.848 baseline) and **regB mean-pool**
with the same seed, then prints both balAcc and the drop. Every flag matches
`mpc/configs.py CONFIGS["regB"]` except `--pool`:

```
python train.py --model cnn --dataset davis \
  --drug-channels 16,32,48 --prot-channels 32,64,96 --prot-dilations 1,2,4 \
  --head-dim 1024 --head-layers 2 --dropout 0.1 --weight-decay 0 \
  --select-metric balacc --epochs 100 --seed 7 \
  --pool {max|mean} --tag-suffix _regB_{max|mean}_s7 [--cuda 0]
```

**Runtime:** ~2–5 h/run on CPU for Davis 100 epochs; minutes/run on a GPU. Two runs
unless `SKIP_MAX=1`. Outputs land in `runs/` (gitignored `.pth`/`.log`):
`cnn_davis_regB_{max,mean}_s7_summary.json` and `..._best.pth`.

## Read the results

The script's final block prints:

```
  regB max-pool  balAcc = 0.848x
  regB mean-pool balAcc = 0.8yyy
  drop (max - mean)     = +0.0zzz  (+p.pp%)
```

`balAcc` comes from `best.balacc_mean` in each `runs/*_summary.json` (the
checkpoint-selection metric, `--select-metric balacc`). For a per-threshold
breakdown run `python gen_results.py` on the summaries. If the drop is small, the
mean-pool checkpoint is the better FSS deployment target.

## MPC payoff

Confirm the comparison saving on the **mean-pool** config:

```sh
python mpc_cost.py --model cnn --dataset davis --drug-channels 16,32,48 \
  --prot-channels 32,64,96 --prot-dilations 1,2,4 --pool mean \
  --head-dim 1024 --head-layers 2
```

→ **347K → 233K** secure comparisons (the 114K max-pool tree is gone). That speeds
up **any** backend — the self-contained `fss_infer` engine, NssMPClib, Orca, and
the MP-SPDZ baseline (fewer comparisons ⇒ lower online time on all of them).

The mean-pool checkpoint then flows through the rest of the MPC tooling unchanged,
now with fewer DReLUs:

```sh
# cleartext + FSS-fidelity metrics on the mean-pool weights
python -m mpc.accuracy --summary runs/cnn_davis_regB_mean_s7_summary.json \
  --ckpt runs/cnn_davis_regB_mean_s7_best.pth --profile

# GPU-FSS single-sample latency + per-op timing (real Davis point)
python -m mpc.nssmpc_infer --summary runs/cnn_davis_regB_mean_s7_summary.json \
  --ckpt runs/cnn_davis_regB_mean_s7_best.pth --dataset davis --davis-index 0 --profile
```

The `--profile` table will show the max-pool DReLUs are gone versus the max-pool
run — the concrete FSS win from this retrain.

See also: `mpc/ONLINE_TIME.md` (the 347K/114K breakdown and the Orca online-time
analysis) and `../FSS_FRAMEWORKS.md` (why mean-pool is free and max-pool isn't).
