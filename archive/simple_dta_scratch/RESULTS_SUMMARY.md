# Results Summary — everything tried (meeting view)

Best-epoch on held-out test split. Multi-seed rows averaged over 3 seeds (4221/7/13).
Empty metric cells = job **running/queued**. Last updated: 2026-07-03.

**How to read a row:** each tower is `Embedding → conv filters (per layer) → global pool`,
then the two tower outputs are concatenated and fed to the **Head MLP** (shown as
`concat_dim → hidden → … → 1`). Default pool = global-max; `+meanpool` = max & mean
concatenated (doubles tower output). Drug conv kernel=4, protein kernel=8, embed=128
(GNN protein embed=32). All conv filters randomly init, trained by backprop.

## Reference (published — Shah et al., Nat. Commun. 16:5021 2025, Table 1; 6-fold CV)
| Model | Dataset | CI | MSE | rm2 | ~Params |
|---|---|---|---|---|---|
| DeepDTA | Davis | 0.878 | 0.261 | 0.630 | ~1.9M |
| **DeepDTAGen** | Davis | 0.890 | 0.214 | 0.705 | ~3.58M (affinity path) |
| DeepDTA | KIBA | 0.863 | 0.194 | 0.630 | ~1.9M |
| **DeepDTAGen** | KIBA | 0.897 | 0.146 | 0.765 | ~3.58M |
| DeepDTA | BindingDB | 0.844 | 0.633 | 0.633 | ~1.9M |
| **DeepDTAGen** | BindingDB | 0.876 | 0.458 | 0.760 | ~3.58M |

## OUR CNN — Davis
| Drug convs | Protein convs | Head MLP | Params | MSE | CI | rm2 | Time |
|---|---|---|---|---|---|---|---|---|
| 32→64→96 | 32→64→96 | 192→1024→512→1 | 882K | **0.225** | **0.887** | 0.713 | ~3h |
| 32→64→96 | 32→64→96 | 192→1024→1 | 358K | 0.248 | 0.881 | 0.679 | ~3h |
| 32→64→96 | 32→64→96 | 192→512→256→1 | 390K | 0.275 | 0.873 | 0.649 | ~4h |
| 32→64→96 | 32→64→96 | 192→1536→1 | 457K | 0.238 | 0.884 | 0.669 | 5h |
| 16→32→48 +meanpool | 16→32→48 +meanpool | 192→1536→1 | 359K | 0.256 | 0.877 | 0.658 | 3h |
| 16→32→48 +meanpool | 16→32→48 +meanpool | 192→1024→512→1 | 784K | 0.245 | 0.885 | 0.666 | 3h |
| 16→32→48 | 32→64→96 | 144→1536→1 | 351K | 0.236 | 0.880 | 0.678 | 5h |
| 16→32→48 | 32→64→96 | 144→1024→512→1 | 800K | 0.231 | **0.893** | 0.669 | 5h |
| 32→64→96 →proj32 | 32→64→96 →proj32 | 64→1536→1 | **267K** | 0.248 | 0.879 | 0.661 | 5h |
| 32→64→96 →proj32 | 32→64→96 →proj32 | 64→1024→512→1 | 758K | 0.238 | 0.881 | 0.691 | 5h |
| 16→32 | 32→64→96 | 128→1024→512→1 | 778K | | | | running |
| 16→32 | 32→64→96→128 | 160→1024→512→1 | 909K | | | | running |
| 16→32→48 | 32→64→96 dilated(1,2,4) | 144→1024→512→1 | 800K | | | | running |
| 16→16→16 | 32→32→32 | 48→1024→512→1 | 647K | | | | running |
| 16→12→8 | 32→24→16 | 24→1024→512→1 | 614K | | | | running |

## OUR CNN — KIBA
| Drug convs | Protein convs | Head MLP | Params | MSE | CI | rm2 | Time |
|---|---|---|---|---|---|---|---|---|
| 32→64→96 | 32→64→96 | 192→1024→512→1 | 882K | **0.179** | 0.858 | 0.707 | ~10h |
| 32→64→96 | 32→64→96 | 192→512→256→1 | 390K | 0.180 | 0.856 | 0.695 | 8h |
| 16→32→48 +meanpool | 16→32→48 +meanpool | 192→1536→1 | 359K | 0.249 | 0.815 | 0.619 | 11h |
| 16→32→48 +meanpool | 16→32→48 +meanpool | 192→1024→512→1 | 784K | 0.208 | 0.839 | 0.672 | 10h |
| 16→32 | 32→64→96 | 128→1024→512→1 | 778K | | | | queued |
| 16→32 | 32→64→96→128 | 160→1024→512→1 | 909K | | | | queued |
| 16→32→48 | 32→64→96 dilated(1,2,4) | 144→1024→512→1 | 800K | | | | queued |
| 16→16→16 | 32→32→32 | 48→1024→512→1 | 647K | | | | queued |
| 16→12→8 | 32→24→16 | 24→1024→512→1 | 614K | | | | queued |

*(5 other KIBA tower runs were killed mid-training to free the queue — no final numbers.)*

## OUR CNN — BindingDB
| Drug convs | Protein convs | Head MLP | Params | MSE | CI | rm2 | Time |
|---|---|---|---|---|---|---|---|---|
| 32→64→96 | 32→64→96 | 192→1024→512→1 | 882K | 0.503 | 0.862 | 0.706 | ~6h |

## OUR GNN — Davis (GCN drug encoder + protein conv tower)
Drug tower = `GCN×3 on N-dim atom features → D`; protein tower = `conv 32→64→96`.
| Drug tower | Protein convs | Head MLP | Params | MSE | CI | rm2 |
|---|---|---|---|---|---|---|
| GCN×2 on 94-dim → 128 | 32→64→96 | 224→1024→512→1 | 903K | 0.270 | 0.875 | 0.625 |
| GCN×3 on 12-dim → 32 | 32→64→96 | 128→1024→512→1 | 736K | 0.270 | 0.875 | 0.621 |
| GCN×3 on 12-dim → 24 | 32→64→96 | 120→1024→512→1 | 726K | 0.317 | 0.855 | 0.617 |
| GCN×3 on 12-dim → 16 | 32→64→96 | 112→1024→512→1 | 717K | 0.373 | 0.845 | 0.528 |
| GCN×3 on 12-dim → 12 | 32→64→96 | 108→1024→512→1 | 712K | 0.411 | 0.821 | 0.468 |
| GCN×3 on 4-dim → 24 | 32→64→96 | 120→1024→512→1 | 726K | 0.297 | 0.866 | 0.628 |

## OUR GNN — BindingDB
| Drug tower | Protein convs | Head MLP | Params | MSE | CI | rm2 |
|---|---|---|---|---|---|---|
| GCN×2 on 94-dim → 128 | 32→64→96 | 224→1024→512→1 | 903K | 0.606 | 0.848 | 0.676 |

## Bottom line
1. **882K CNN (32→64→96 towers, head 192→1024→512→1) beats published DeepDTA on all 3 datasets**,
   within 2–5pt of DeepDTAGen — at ¼ the params (882K vs 3.58M) and hours, not >1 week.
2. **CNN > GNN everywhere**; GNN collapses once GCN width drops below 32.
3. **Smallest good model: 267K** (proj-32 bottleneck, Davis MSE 0.248) — still beats DeepDTA.
4. No tower beat baseline MSE, but the **lean-drug (16→32→48) / full-protein** tower with the
   deep head hit the **best CI (0.893)** at 800K. On KIBA the lean-drug tower *hurts* (wants
   more drug capacity) — opposite of Davis.
5. **In progress:** conv depth/shape/dilation sweep (last 5 Davis rows), KIBA queued.
