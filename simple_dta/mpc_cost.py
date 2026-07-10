"""mpc_cost.py -- rank affinity models by *secure-inference* (FSS/MPC) cost.

Param count is the wrong yardstick for an MPC port: in secure two-party
computation the linear algebra (conv / matmul / linear) is cheap, and the whole
online cost is dominated by **non-linearities**. This tool counts the ops that
actually matter under Function Secret Sharing (FSS), so architectures can be
ranked by what a 2PC port would cost, not by parameters.

What it counts (per drug-protein pair, i.e. per inference):
  * secure comparisons -- the FSS "DReLU" primitive, one per element, that
    powers every cheap non-linearity:
      - relu       : # elements through every ReLU (conv towers, head, GCN/GINE)
      - elu        : # elements through every ELU (GAT, linear-attention kernel)
      - maxpool    : comparisons to collapse a sequence with max/maxmean pooling
                     (0 for mean pooling -- mean is *free* under FSS)
  * softmax  : # elements fed to softmax (exp + reciprocal); 0 for linear
               attention / concat CNN. THE expensive op -- see FSS_FRAMEWORKS.md.
  * layernorm: # elements normalized (mean/var/rsqrt); 0 unless a transformer
               block or cross-attention residual-norm is present.
  * linear_macs : multiply-accumulates in Linear/Conv/matmul -- CHEAP under FSS,
               reported only for context / a communication-volume proxy.

How: it does not re-derive shapes by hand. It builds the *real* model from the
same flags as train.py (train.build_model) and monkey-patches the low-level
torch ops for the duration of ONE forward pass on a shape-correct batch, tallying
from the actual tensors seen. That makes it automatically correct for every
--model / --drug-encoder / --protein-encoder / --fusion / --attn-kind / --pool
combination -- a new architecture flag is costed for free.

No dataset download needed: op counts depend only on tensor *shapes*, so protein
towers run at the dataset's full padded length (DEFAULTS) and drug graphs are
built from a handful of representative real SMILES via RDKit.

Usage:
  # cost one config (same flags as train.py):
  python mpc_cost.py --model cnn --dataset davis
  python mpc_cost.py --model attn --drug-encoder cnn --fusion cross --attn-kind softmax
  python mpc_cost.py --model attn --drug-encoder cnn --fusion cross --attn-kind linear

  # accuracy-vs-cost leaderboard, rebuilt from committed run summaries:
  python mpc_cost.py --compare
"""
import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

import attention
import models
import train
from data import (CHARISOSMILEN, CHARPROTLEN, DEFAULTS, FEATURIZERS,
                  label_encode, CHARPROTSET, smile_to_graph)

HERE = os.path.dirname(os.path.abspath(__file__))

# --- relative FSS cost weights (per element) --------------------------------
# COARSE order-of-magnitude proxies, NOT a protocol-exact cost. The robust,
# framework-agnostic signal is the raw secure-comparison count plus the
# `softmax_free` flag; the single weighted number below is a convenience for
# ranking. Literature (Orca S&P'24, SIGMA PoPETs'24, private-transformer
# surveys) puts a whole softmax layer at ~10^3x and LayerNorm even higher than
# an equal-size ReLU layer; we fold that into conservative per-element weights.
# Tune here; see FSS_FRAMEWORKS.md for the rationale.
W_CMP = 1          # one secure comparison (ReLU / ELU / max-pool compare)
W_SOFTMAX = 1000   # per softmax input element (secure exp + reciprocal)
W_LAYERNORM = 100  # per layernorm element (secure mean/var/rsqrt)

# Representative real drug SMILES (aspirin, caffeine, ibuprofen, naphthalene,
# imatinib, gefitinib, sorafenib, sunitinib) -- span small->large so the graph
# towers see a realistic atom-count distribution. Only used to shape the batch.
REP_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2ccccc2c1",
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
    "COc1cc2ncnc(c2cc1OCCCN3CCOCC3)Nc4ccc(c(c4)Cl)F",
    "CNC(=O)c1cc(ccn1)Oc2ccc(cc2)NC(=O)Nc3ccc(c(c3)C(F)(F)F)Cl",
    "CCN(CC)CCNC(=O)c1c(c([nH]c1C)/C=C\\2/c3cc(ccc3NC2=O)F)C",
]

# The accuracy-vs-cost leaderboard, keyed to committed run summaries in runs/.
# Each model is rebuilt FROM ITS OWN SUMMARY (config + params + balAcc all from
# the same source), so the table can't drift from the recorded results.
DEFAULT_COMPARE_TAGS = [
    "cnn_davis_cfgA_protDil123",       # Config A: smallest CNN, 267K, softmax-free
    "cnn_davis_stable_pool_max",       # DeepDTA-style dual-CNN concat baseline, 882K
    "cnn_davis_regB_d1_wd0",           # dilated-protein CNN, best small-family balAcc
    "attn_davis_F_gnn_crossfusion",    # F: GCN+cnn, linear cross-attn (best MPC candidate)
    "attn_davis_D_crossfusion_linear", # D: cnn+cnn, linear cross-attn (softmax-free twin of A)
    "attn_davis_J_rawnode_cross_linear",  # J: raw-atom+cnn, linear cross-attn (fully softmax-free)
    "attn_davis_A_crossfusion",        # A: cnn+cnn, SOFTMAX cross-attn (accuracy ceiling, costly)
]


class Counter:
    """Accumulates op counts across one forward pass."""

    def __init__(self):
        self.relu = 0
        self.elu = 0
        self.maxpool = 0
        self.softmax = 0
        self.layernorm = 0
        self.linear_macs = 0

    # -- derived views ------------------------------------------------------
    @property
    def comparisons(self):
        """Secure DReLU comparisons -- the clean, protocol-agnostic FSS cost."""
        return self.relu + self.elu + self.maxpool

    @property
    def softmax_free(self):
        return self.softmax == 0

    @property
    def fss_cost(self):
        """Single heuristic score (see weight caveat at top of file)."""
        return (W_CMP * self.comparisons
                + W_SOFTMAX * self.softmax
                + W_LAYERNORM * self.layernorm)

    def per_sample(self, n):
        out = Counter()
        for k in ("relu", "elu", "maxpool", "softmax", "layernorm", "linear_macs"):
            setattr(out, k, getattr(self, k) / n)
        return out


def _matmul_macs(a, b):
    """Best-effort MAC count for torch.matmul(a, b) -- context only."""
    if a.dim() < 2 or b.dim() < 2:
        return 0
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    lead_a = a.numel() // (m * k)
    lead_b = b.numel() // (b.shape[-2] * n)
    return max(lead_a, lead_b) * m * n * k


class count_ops:
    """Context manager that patches torch primitives to tally FSS-relevant ops.

    Patching the functional layer (F.relu / F.softmax / ...) catches BOTH
    functional calls (e.g. `F.relu(conv(x))` in the CNN towers) and nn.Module
    calls (nn.ReLU/nn.LayerNorm forward to the same functions), so nothing is
    missed regardless of how a layer is written. Pooling is wrapped at the two
    project helpers (models._pool1d / masked_pool) since max-pool's comparisons
    live inside a `.max()` reduction that has no single functional entry point.
    """

    def __init__(self, counter):
        self.c = counter

    def __enter__(self):
        c = self.c
        self._saved = {
            "relu": F.relu, "elu": F.elu, "softmax": F.softmax,
            "layer_norm": F.layer_norm, "matmul": torch.matmul,
            "m_pool1d": models._pool1d, "m_masked_pool": models.masked_pool,
            "a_masked_pool": attention.masked_pool,
        }
        s = self._saved

        def relu(x, *a, **k):
            c.relu += x.numel(); return s["relu"](x, *a, **k)

        def elu(x, *a, **k):
            c.elu += x.numel(); return s["elu"](x, *a, **k)

        def softmax(x, *a, **k):
            c.softmax += x.numel(); return s["softmax"](x, *a, **k)

        def layer_norm(x, *a, **k):
            c.layernorm += x.numel(); return s["layer_norm"](x, *a, **k)

        def matmul(a_, b_, *a, **k):
            try:
                c.linear_macs += _matmul_macs(a_, b_)
            except Exception:
                pass
            return s["matmul"](a_, b_, *a, **k)

        def _pool_cmps(x, mode, red_dim):
            # comparisons to max-reduce `red_dim`: (L-1) per surviving element.
            if mode in ("max", "maxmean"):
                surviving = x.numel() // x.shape[red_dim]
                return surviving * (x.shape[red_dim] - 1)
            return 0

        def pool1d(x, mode):  # x: [B, C, L], reduce L (dim 2)
            c.maxpool += _pool_cmps(x, mode, red_dim=2)
            return s["m_pool1d"](x, mode)

        def m_masked_pool(x, pad, mode):  # x: [B, L, D], reduce L (dim 1)
            c.maxpool += _pool_cmps(x, mode, red_dim=1)
            return s["m_masked_pool"](x, pad, mode)

        def a_masked_pool(x, pad, mode):
            c.maxpool += _pool_cmps(x, mode, red_dim=1)
            return s["a_masked_pool"](x, pad, mode)

        F.relu, F.elu, F.softmax, F.layer_norm = relu, elu, softmax, layer_norm
        torch.matmul = matmul
        models._pool1d = pool1d
        models.masked_pool = m_masked_pool
        attention.masked_pool = a_masked_pool
        # Linear/Conv MACs (F.linear uses addmm, not matmul, so count via hooks).
        self._hooks = []
        return self

    def hook_model(self, model):
        c = self.c

        def lin_hook(mod, inp, out):
            c.linear_macs += inp[0].numel() * mod.out_features

        def conv_hook(mod, inp, out):
            # out: [B, C_out, L_out]; MACs = numel(out) * (C_in/groups * kernel)
            c.linear_macs += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0]

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                self._hooks.append(m.register_forward_hook(lin_hook))
            elif isinstance(m, torch.nn.Conv1d):
                self._hooks.append(m.register_forward_hook(conv_hook))
        return self

    def __exit__(self, *exc):
        s = self._saved
        F.relu, F.elu, F.softmax, F.layer_norm = s["relu"], s["elu"], s["softmax"], s["layer_norm"]
        torch.matmul = s["matmul"]
        models._pool1d = s["m_pool1d"]
        models.masked_pool = s["m_masked_pool"]
        attention.masked_pool = s["a_masked_pool"]
        for h in self._hooks:
            h.remove()
        return False


def make_batch(args, n):
    """A shape-correct batch of n drug-protein pairs (no dataset download).

    Returns (batch_input, n_effective). Sequence models get random label-encoded
    strings at the dataset's full padded length; graph models get real molecular
    graphs from REP_SMILES (realistic atom counts) with random protein targets.
    Op counts are shape-driven, so random *values* are fine.
    """
    d = DEFAULTS[args.dataset]
    loader_kind = train.loader_kind_for(args)
    if loader_kind == "cnn":
        xd = torch.randint(1, CHARISOSMILEN + 1, (n, d["max_smi_len"]))
        xt = torch.randint(1, CHARPROTLEN + 1, (n, d["max_seq_len"]))
        return (xd, xt, None), n
    # graph drug tower
    from torch_geometric.data import Batch, Data
    feat_fn = FEATURIZERS[args.node_feat][0]
    L = d["max_seq_len"]
    smis = (REP_SMILES * ((n + len(REP_SMILES) - 1) // len(REP_SMILES)))[:n]
    data_list = []
    for smi in smis:
        _, feats, ei, ea = smile_to_graph(smi, feat_fn)
        g = Data(x=torch.from_numpy(feats),
                 edge_index=torch.from_numpy(ei),
                 edge_attr=torch.from_numpy(ea),
                 y=torch.tensor([0.0]))
        g.target = torch.randint(1, CHARPROTLEN + 1, (1, L))
        data_list.append(g)
    return Batch.from_data_list(data_list), len(data_list)


def count_config(args, n=8):
    """Build the model from args, run one forward pass, return (per-sample
    Counter, param_count)."""
    from models import count_params
    torch.manual_seed(0)
    model = train.build_model(args)
    model.eval()
    batch, n_eff = make_batch(args, n)
    counter = Counter()
    with torch.no_grad(), count_ops(counter) as ctx:
        ctx.hook_model(model)
        model(batch)
    return counter.per_sample(n_eff), count_params(model)


# ---------------------------------------------------------------------------
# Rebuild an args Namespace from a committed run summary (for --compare)
# ---------------------------------------------------------------------------
def args_from_summary(summary):
    """Parser defaults, overridden by every summary key that names a real flag.
    Missing keys (older summaries) fall back to the parser default -- harmless
    because they only matter for encoders that summary's model doesn't use."""
    args = train.make_parser().parse_args(
        ["--model", summary["model"], "--dataset", summary.get("dataset", "davis")])
    for k, v in summary.items():
        if k not in ("model", "dataset") and hasattr(args, k):
            setattr(args, k, v)
    return args


def _fmt(x):
    """Human-readable large-number formatting (12.3K / 4.56M / 1.20G)."""
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def report_single(args, n=8):
    c, params = count_config(args, n)
    desc = args.model
    if args.model == "attn":
        desc = (f"attn drug={args.drug_encoder} prot={args.protein_encoder} "
                f"fusion={args.fusion} attn={args.attn_kind} pool={args.pool}")
    else:
        desc = f"{args.model} pool={args.pool}"
    print(f"\n=== FSS op cost per inference | {desc} | {args.dataset} ===")
    print(f"  params            : {params:,}")
    print(f"  secure comparisons: {_fmt(c.comparisons):>10}   "
          f"(relu {_fmt(c.relu)} + elu {_fmt(c.elu)} + maxpool {_fmt(c.maxpool)})")
    print(f"  softmax elements  : {_fmt(c.softmax):>10}   "
          f"{'<- SOFTMAX-FREE' if c.softmax_free else '<- has softmax (expensive)'}")
    print(f"  layernorm elements: {_fmt(c.layernorm):>10}")
    print(f"  linear MACs (cheap): {_fmt(c.linear_macs):>9}")
    print(f"  weighted FSS score : {_fmt(c.fss_cost):>9}   "
          f"(cmp + {W_SOFTMAX}x softmax + {W_LAYERNORM}x layernorm)")


def load_summary(tag, out_dir):
    path = os.path.join(out_dir, f"{tag}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_compare(cfg):
    rows = []
    for tag in cfg.tags:
        summary = load_summary(tag, cfg.out_dir)
        if summary is None:
            print(f"  (skip {tag}: no summary in {cfg.out_dir})")
            continue
        args = args_from_summary(summary)
        args.dataset = cfg.dataset  # cost all configs on one dataset's shapes
        c, params = count_config(args, cfg.n)
        balacc = summary.get("best", {}).get("balacc_mean")
        rows.append({
            "tag": tag, "balacc": balacc, "params": params,
            "cmp": c.comparisons, "softmax": c.softmax,
            "ln": c.layernorm, "cost": c.fss_cost,
            "free": c.softmax_free,
        })
    if not rows:
        print("No summaries found to compare.")
        return
    rows.sort(key=lambda r: r["cost"])  # cheapest FSS first

    print(f"\n=== Accuracy vs FSS-cost (Davis shapes, per inference, cheapest first) ===")
    hdr = f"{'config':<34}{'balAcc':>8}{'params':>9}{'sec.cmp':>10}{'softmax':>10}{'FSSscore':>10}  free?"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ba = f"{r['balacc']:.3f}" if r["balacc"] is not None else "n/a"
        print(f"{r['tag']:<34}{ba:>8}{_fmt(r['params']):>9}{_fmt(r['cmp']):>10}"
              f"{_fmt(r['softmax']):>10}{_fmt(r['cost']):>10}  {'yes' if r['free'] else 'NO'}")
    print("\nsec.cmp = secure DReLU comparisons (relu+elu+maxpool); softmax = # exp/"
          "reciprocal elements.\nFSSscore = cmp + "
          f"{W_SOFTMAX}x softmax + {W_LAYERNORM}x layernorm (coarse; see FSS_FRAMEWORKS.md).")
    print("NOTE: GAT drug encoders route attention through torch_geometric's "
          "scatter-softmax,\nwhich this tool does not intercept -- a GAT config's "
          "softmax cost is therefore a lower bound.")


def main():
    import sys
    if "--compare" in sys.argv:
        ap = argparse.ArgumentParser(description="FSS accuracy-vs-cost leaderboard")
        ap.add_argument("--compare", action="store_true", required=True)
        ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
        ap.add_argument("--n", type=int, default=8, help="molecules used to average graph-tower shapes")
        ap.add_argument("--tags", nargs="*", default=DEFAULT_COMPARE_TAGS,
                        help="run-summary tags (without _summary.json) to include")
        ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
        run_compare(ap.parse_args())
    else:
        args = train.make_parser().parse_args()
        report_single(args)


if __name__ == "__main__":
    main()
