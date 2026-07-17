"""FSS-ready affinity CNN, written once against a `Backend` interface.

The *same* forward runs plaintext (`ClearBackend`, the reference) or under the
2-party FSS engine (`FSSBackend` in fss_infer.py) -- swap the backend, compare
the outputs. Everything here is deliberately shaped so every op maps to an
FSS/Orca kernel (see ../FSS_FRAMEWORKS.md and MPC_PORT.md):

  * nn.Embedding -> **one-hot x table matmul**. The drug/protein tokens are the
    *secret* input, and no FSS framework has a secret-index gather, so the
    embedding must be a matmul on a one-hot-encoded input. `to_onehot()` does
    the (public-structure) encoding; the table becomes the first matmul weight.
  * global **MEAN** pool, not max (mean is linear = free under MPC; max is a
    log(L) round comparison tree). This changes accuracy vs a max-pool model, so
    the FSS target must be a `--pool mean` checkpoint; the PoC's *correctness*
    reference is this same mean-pool cleartext model.
  * static shapes (pad to the dataset's max length), ReLU-only, no dropout.

The weights come straight from a trained `CNNDTA(pool="mean", proj_dim=0)`
(models.py); nothing is retrained here.
"""
import torch
import torch.nn.functional as F

from data import CHARISOSMILEN, CHARPROTLEN


def to_onehot(tokens, vocab_size):
    """[B, L] int token ids -> [B, L, vocab_size] float one-hot (index 0 = pad).

    This is the in-the-clear encoding the client does before secret-sharing; the
    one-hot *values* are the secret payload fed to the FSS engine."""
    return F.one_hot(tokens.long(), num_classes=vocab_size).float()


class Backend:
    """Tensor operations the model needs, so the forward is backend-agnostic.

    A 'tensor' is whatever the backend uses (a plaintext torch.Tensor for
    ClearBackend, an additively-shared pair for FSSBackend). Model weights are
    passed to the ops as plaintext torch.Tensors; the FSS backend decides whether
    to treat them as PRIVATE (secret-share them -> Beaver matmul/conv; threat model
    = private model + private data, neither party learns the other's secret) or
    public (local products). Default is private weights (see fss_infer.py).
    """

    def embed(self, x_onehot, table):
        """x_onehot: [B, L, V] (secret); table: [V, E] public -> [B, E, L]."""
        raise NotImplementedError

    def conv1d(self, x, weight, bias):
        """x: [B, Cin, L] (secret); weight [Cout,Cin,k], bias [Cout] public."""
        raise NotImplementedError

    def relu(self, x):
        raise NotImplementedError

    def mean_pool(self, x):
        """x: [B, C, L] -> [B, C] (mean over L). Free/linear under FSS."""
        raise NotImplementedError

    def max_pool(self, x):
        """x: [B, C, L] -> [B, C] (max over L). A log(L) DReLU tournament under FSS."""
        raise NotImplementedError

    def pool(self, x, mode):
        return self.mean_pool(x) if mode == "mean" else self.max_pool(x)

    def concat(self, a, b):
        """concat two [B, D] tensors along dim 1."""
        raise NotImplementedError

    def linear(self, x, weight, bias):
        """x: [B, in] (secret); weight [out,in], bias [out] public -> [B, out]."""
        raise NotImplementedError


class ClearBackend(Backend):
    """Plaintext float reference -- exactly the mean-pool CNNDTA computation."""

    def embed(self, x_onehot, table):
        return torch.matmul(x_onehot, table).transpose(1, 2)  # [B,E,L]

    def conv1d(self, x, weight, bias):
        return F.conv1d(x, weight, bias)

    def relu(self, x):
        return F.relu(x)

    def mean_pool(self, x):
        return x.mean(dim=2)

    def max_pool(self, x):
        return x.max(dim=2).values

    def concat(self, a, b):
        return torch.cat([a, b], dim=1)

    def linear(self, x, weight, bias):
        return F.linear(x, weight, bias)


class MPCModel:
    """FSS-ready affinity model: two CNN towers + head, all public weights.

    Build from a trained CNNDTA(pool="mean", proj_dim=0); run under any Backend.
    """

    def __init__(self, cnndta):
        assert cnndta.drug is not None and cnndta.protein is not None, \
            "MPCModel needs the full two-tower model (ablate='none')"
        assert cnndta.proj_dim == 0, "port the proj_dim=0 variant (no bottleneck)"
        assert cnndta.drug.pool in ("mean", "max") and cnndta.protein.pool == cnndta.drug.pool, (
            "MPC port supports pool in {mean,max} (same for both towers). mean is free "
            "under FSS; max is a log(L) DReLU tournament (works, just costlier).")
        self.pool_mode = cnndta.drug.pool
        cnndta.eval()
        self.drug_vocab = CHARISOSMILEN + 1
        self.prot_vocab = CHARPROTLEN + 1
        # embedding tables (become the first matmul weight of each tower)
        self.drug_table = cnndta.drug.embed.weight.detach()   # [Vd, E]
        self.prot_table = cnndta.protein.embed.weight.detach()
        self.drug_convs = [(c.weight.detach(), c.bias.detach()) for c in cnndta.drug.convs]
        self.prot_convs = [(c.weight.detach(), c.bias.detach()) for c in cnndta.protein.convs]
        # head: pull the Linear layers out of the Sequential (Dropout = no-op)
        self.head = [(m.weight.detach(), m.bias.detach())
                     for m in cnndta.head.net if isinstance(m, torch.nn.Linear)]

    def _tower(self, be, x_onehot, table, convs):
        x = be.embed(x_onehot, table)          # [B, E, L]
        for w, b in convs:
            x = be.relu(be.conv1d(x, w, b))
        return be.pool(x, self.pool_mode)       # [B, C]

    def forward(self, be, drug_onehot, prot_onehot):
        d = self._tower(be, drug_onehot, self.drug_table, self.drug_convs)
        p = self._tower(be, prot_onehot, self.prot_table, self.prot_convs)
        x = be.concat(d, p)
        for i, (w, b) in enumerate(self.head):
            x = be.linear(x, w, b)
            if i < len(self.head) - 1:          # ReLU between hidden layers, not after last
                x = be.relu(x)
        return x                                # [B, 1]

    def forward_tokens_clear(self, smiles, target):
        """Convenience: plaintext forward straight from integer token tensors
        (one-hot-encodes internally). Used to check equality with CNNDTA."""
        be = ClearBackend()
        d = to_onehot(smiles, self.drug_vocab)
        p = to_onehot(target, self.prot_vocab)
        return self.forward(be, d, p).squeeze(-1)


def load_cnndta(summary_path=None, ckpt_path=None, config=None):
    """Rebuild the exact CNNDTA a run was trained with (from its
    `runs/*_summary.json`) and load its weights. Reuses train.make_parser +
    train.build_model so the architecture always matches the checkpoint. With no
    summary, returns a default random-init mean-pool CNNDTA (pipeline testing).

    The checkpoint must be a `simple_dta` **CNN+CNN** `CNNDTA` (trained with
    `train.py --model cnn`, either `--pool mean` or the default `--pool max`), not
    the original DeepDTAGen model. mean-pool is free under FSS; max-pool works too
    (a log(L) DReLU tournament, just costlier -- see FSS_FRAMEWORKS.md)."""
    import json
    import train
    from models import CNNDTA
    if config is not None:
        from mpc.configs import build_cnndta
        model = build_cnndta(config)            # named preset (e.g. regB)
    elif summary_path is None:
        model = CNNDTA(pool="mean", proj_dim=0, head_layers=2, head_dim=128)
    else:
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary.get("model") == "cnn", "MPC port targets the CNN (CNN+CNN) model"
        args = train.make_parser().parse_args(
            ["--model", "cnn", "--dataset", summary.get("dataset", "davis")])
        for k, v in summary.items():
            if k not in ("model", "dataset") and hasattr(args, k):
                setattr(args, k, v)
        if args.pool not in ("mean", "max"):
            raise ValueError(f"checkpoint pool={args.pool!r}; MPC supports mean|max "
                             "(maxmean not supported).")
        model = train.build_model(args)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


class MPCReadyNet(torch.nn.Module):
    """`nn.Module` mirror of `MPCModel` that takes one-hot inputs -- an
    ONNX-exportable graph of only MatMul/Conv/Relu/mean/Gemm ops (no Embedding,
    no max-pool), i.e. exactly the ops Orca/EzPC GPU-FSS provides. Static input
    shapes are baked in at export (both FSS and ONNX tracing require fixed shapes).
    Used by export_onnx.py; see MPC_PORT.md for the Orca build."""

    def __init__(self, cnndta):
        super().__init__()
        mpc = MPCModel(cnndta)
        self.pool_mode = mpc.pool_mode
        self.drug_vocab, self.prot_vocab = mpc.drug_vocab, mpc.prot_vocab
        self.drug_table = torch.nn.Parameter(mpc.drug_table.clone(), requires_grad=False)
        self.prot_table = torch.nn.Parameter(mpc.prot_table.clone(), requires_grad=False)
        self.drug_convs = self._convs(mpc.drug_convs, kernel=cnndta.drug.convs[0].kernel_size[0])
        self.prot_convs = self._convs(mpc.prot_convs, kernel=cnndta.protein.convs[0].kernel_size[0])
        self.head = torch.nn.ModuleList()
        for w, b in mpc.head:
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight, lin.bias = torch.nn.Parameter(w.clone()), torch.nn.Parameter(b.clone())
            self.head.append(lin)

    @staticmethod
    def _convs(conv_weights, kernel):
        mods = torch.nn.ModuleList()
        for w, b in conv_weights:
            c = torch.nn.Conv1d(w.shape[1], w.shape[0], w.shape[2])
            c.weight, c.bias = torch.nn.Parameter(w.clone()), torch.nn.Parameter(b.clone())
            mods.append(c)
        return mods

    def _tower(self, x_onehot, table, convs):
        x = torch.matmul(x_onehot, table).transpose(1, 2)  # [B,E,L]
        for c in convs:
            x = torch.relu(c(x))
        return x.mean(dim=2) if self.pool_mode == "mean" else x.max(dim=2).values

    def forward(self, drug_onehot, prot_onehot):
        d = self._tower(drug_onehot, self.drug_table, self.drug_convs)
        p = self._tower(prot_onehot, self.prot_table, self.prot_convs)
        x = torch.cat([d, p], dim=1)
        for i, lin in enumerate(self.head):
            x = lin(x)
            if i < len(self.head) - 1:
                x = torch.relu(x)
        return x
