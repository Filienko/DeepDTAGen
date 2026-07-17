"""GPU-FSS 2-party inference of the CNN+CNN affinity model with **NssMPClib**.

NssMPClib (XidianNSS) is a PyTorch-native MPC library that is genuinely FSS
(DPF/DCF/DICF keys for the nonlinear ops), supports 2PC semi-honest, and
GPU-accelerates conv/matmul (CUTLASS). It's a much lighter install than EzPC/Orca
(`pip install -e .`) and takes a real secret input -- the pragmatic GPU-FSS path
for our ~800K CNN. (Its FSS *nonlinear* eval is CPU-side; negligible for a model
this small.)

**Private weights:** `nn.utils.share_model_param(model=...)` secret-shares the
model parameters between the two parties, so this path already runs under the
**private model + private input** threat model (linear layers become secret x
secret Beaver matmul -- neither party learns the other's secret). This matches the
self-contained engine's `private_weights=True` default (fss_infer.py).

Key idea that makes the whole net NssMPClib-native: **the embedding is a 1x1
Conv2d over one-hot channels**. So CNN+CNN maps entirely onto Conv2d / ReLU /
AvgPool2d(mean) or MaxPool2d(max) / Linear -- all supported layers. Conv1d[o,i,k]
-> Conv2d[o,i,1,k]; input one-hot [B,L,V] -> [B,V,1,L].

This module gives you:
  * `NssDTA` -- the Conv2d-based plaintext CNN+CNN (matches `CNNDTA` exactly).
  * `nss_from_cnndta(...)` -- load your trained `CNNDTA` (mean OR max pool)
    weights into it (verified here at the torch level, no GPU/NssMPClib needed).
  * `--party {0,1}` -- run the verified NssMPClib 2PC flow (server holds weights,
    client holds the secret drug+protein), timing single-sample latency + per-op.

Run it on the GPU VM via `mpc/run_nssmpc_vm.sh` (installs NssMPClib, sets configs,
launches both parties). The NssMPClib API calls below are quoted from its
`tests/application/neural_network/2pc/` example; reconcile names with the version
you install (flagged inline).
"""
import argparse
import time

import torch
import torch.nn as nn

from data import (CHARISOSMILEN, CHARPROTLEN, DEFAULTS, load_csv, label_encode,
                  CHARISOSMISET, CHARPROTSET)
from mpc.model_mpc import load_cnndta


def _input_tokens(dataset, batch, davis_index=None):
    """(xd, xt) token tensors [B, L]. If davis_index is set, load that real test
    row(s) from the dataset; else random tokens (shape-only, for timing)."""
    d = DEFAULTS[dataset]
    if davis_index is None:
        torch.manual_seed(1)
        xd = torch.randint(0, CHARISOSMILEN + 1, (batch, d["max_smi_len"]))
        xt = torch.randint(0, CHARPROTLEN + 1, (batch, d["max_seq_len"]))
        return xd, xt
    smiles, prots, _ = load_csv(dataset, "test")
    idx = [(davis_index + i) % len(smiles) for i in range(batch)]
    xd = torch.stack([torch.from_numpy(label_encode(smiles[i], d["max_smi_len"], CHARISOSMISET)) for i in idx])
    xt = torch.stack([torch.from_numpy(label_encode(prots[i], d["max_seq_len"], CHARPROTSET)) for i in idx])
    return xd, xt


class NssDTA(nn.Module):
    """CNN+CNN affinity model in NssMPClib-native ops (Conv2d/ReLU/pool/Linear).

    forward(drug_oh, prot_oh): one-hot inputs [B, V, 1, L] (V=vocab channels).
    `pool` in {"mean","max"} -> AvgPool2d / MaxPool2d over the length axis.
    """

    def __init__(self, drug_vocab, prot_vocab, embed_dim,
                 drug_channels, prot_channels, drug_kernel, prot_kernel,
                 head_dims, pool="mean", drug_dilations=None, prot_dilations=None):
        super().__init__()
        self.pool = pool
        self.drug_embed = nn.Conv2d(drug_vocab, embed_dim, kernel_size=1)   # one-hot @ table
        self.prot_embed = nn.Conv2d(prot_vocab, embed_dim, kernel_size=1)
        self.drug_convs = self._convs(embed_dim, drug_channels, drug_kernel, drug_dilations)
        self.prot_convs = self._convs(embed_dim, prot_channels, prot_kernel, prot_dilations)
        head_in = drug_channels[-1] + prot_channels[-1]
        layers, prev = [], head_in
        for i, h in enumerate(head_dims):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    @staticmethod
    def _convs(in_c, channels, k, dilations=None):
        dilations = dilations or [1] * len(channels)
        mods, prev = [], in_c
        for c, d in zip(channels, dilations):
            mods += [nn.Conv2d(prev, c, kernel_size=(1, k), dilation=(1, d)), nn.ReLU()]
            prev = c
        return nn.Sequential(*mods)

    def _pool(self, x):                                  # x: [B, C, 1, L']
        return (x.mean(dim=3) if self.pool == "mean" else x.amax(dim=3)).squeeze(2)

    def forward(self, drug_oh, prot_oh):
        d = self._pool(self.drug_convs(self.drug_embed(drug_oh)))
        p = self._pool(self.prot_convs(self.prot_embed(prot_oh)))
        return self.head(torch.cat([d, p], dim=1))


def nss_from_cnndta(summary=None, ckpt=None, config=None):
    """Build an NssDTA and copy a trained CNNDTA's weights into it.
    Conv1d[o,i,k]->Conv2d[o,i,1,k]; embedding table[V,E]->Conv2d(V->E,1x1)."""
    cnn = load_cnndta(summary, ckpt, config)
    drug_ch = [c.out_channels for c in cnn.drug.convs]
    prot_ch = [c.out_channels for c in cnn.protein.convs]
    head_dims = [m.out_features for m in cnn.head.net
                 if isinstance(m, nn.Linear)][:-1]
    drug_dil = [c.dilation[0] for c in cnn.drug.convs]
    prot_dil = [c.dilation[0] for c in cnn.protein.convs]
    net = NssDTA(CHARISOSMILEN + 1, CHARPROTLEN + 1, cnn.drug.embed.embedding_dim,
                 drug_ch, prot_ch, cnn.drug.convs[0].kernel_size[0],
                 cnn.protein.convs[0].kernel_size[0], head_dims, pool=cnn.drug.pool,
                 drug_dilations=drug_dil, prot_dilations=prot_dil)
    with torch.no_grad():
        # embedding table [V, E] -> 1x1 conv weight [E, V, 1, 1]
        net.drug_embed.weight.copy_(cnn.drug.embed.weight.t().reshape(
            net.drug_embed.weight.shape))
        net.drug_embed.bias.zero_()
        net.prot_embed.weight.copy_(cnn.protein.embed.weight.t().reshape(
            net.prot_embed.weight.shape))
        net.prot_embed.bias.zero_()
        for dst, src in zip([m for m in net.drug_convs if isinstance(m, nn.Conv2d)], cnn.drug.convs):
            dst.weight.copy_(src.weight.unsqueeze(2)); dst.bias.copy_(src.bias)  # [o,i,k]->[o,i,1,k]
        for dst, src in zip([m for m in net.prot_convs if isinstance(m, nn.Conv2d)], cnn.protein.convs):
            dst.weight.copy_(src.weight.unsqueeze(2)); dst.bias.copy_(src.bias)
        for dst, src in zip([m for m in net.head if isinstance(m, nn.Linear)],
                            [m for m in cnn.head.net if isinstance(m, nn.Linear)]):
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
    return net.eval(), cnn


def onehot4d(tokens, vocab):
    """[B, L] int -> [B, V, 1, L] one-hot (channels = vocab), NssDTA's input."""
    oh = torch.nn.functional.one_hot(tokens.long(), vocab).float()  # [B,L,V]
    return oh.permute(0, 2, 1).unsqueeze(2)                          # [B,V,1,L]


class _OpTimer:
    """Per-op wall-time via forward hooks (works plaintext; in the secure model it
    fires if convert_model preserves module boundaries -- else read NssMPClib's own
    timing log). CUDA-synced."""

    def __init__(self, model, device):
        self.t = {}; self.c = {}; self.cuda = device.type == "cuda"; self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.AvgPool2d, nn.MaxPool2d)):
                kind = type(m).__name__
                m._t0 = [0.0]
                self.handles.append(m.register_forward_pre_hook(self._pre(m)))
                self.handles.append(m.register_forward_hook(self._post(kind)))

    def _pre(self, m):
        def h(mod, inp):
            if self.cuda: torch.cuda.synchronize()
            m._t0[0] = time.perf_counter()
        return h

    def _post(self, kind):
        def h(mod, inp, out):
            if self.cuda: torch.cuda.synchronize()
            dt = time.perf_counter() - mod._t0[0]
            self.t[kind] = self.t.get(kind, 0.0) + dt; self.c[kind] = self.c.get(kind, 0) + 1
        return h

    def report(self, n, total):
        print(f"\n  per-op runtime (batch={n}):")
        print(f"    {'op':<10}{'calls':>7}{'total_ms':>11}{'ms/sample':>11}")
        for k in sorted(self.t, key=lambda k: -self.t[k]):
            print(f"    {k:<10}{self.c[k]:>7}{self.t[k]*1e3:>11.2f}{self.t[k]*1e3/n:>11.2f}")
        print(f"    {'END2END':<10}{'':>7}{total*1e3:>11.2f}{total*1e3/n:>11.2f}   <- single-sample latency")
        for h in self.handles: h.remove()


def run_plaintext(net, cnn, dataset, batch, device, davis_index=None):
    """Torch-level check + timing (no NssMPClib): NssDTA == CNNDTA, and the per-op
    timing table you'll also get under 2PC. This is what runs in THIS sandbox."""
    xd, xt = _input_tokens(dataset, batch, davis_index)
    drug, prot = onehot4d(xd, CHARISOSMILEN + 1).to(device), onehot4d(xt, CHARPROTLEN + 1).to(device)
    net = net.to(device)
    with torch.no_grad():
        ref = cnn((xd.to(device), xt.to(device), None))
        timer = _OpTimer(net, device)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        got = net(drug, prot).squeeze(-1)
        if device.type == "cuda": torch.cuda.synchronize()
        total = time.time() - t0
    err = (ref - got).abs().max().item()
    pt = f"Davis test #{davis_index}" if davis_index is not None else "random tokens"
    print(f"=== NssDTA plaintext check | pool={net.pool} | dev={device} | input={pt} ===")
    print(f"  NssDTA == CNNDTA(pool={net.pool}): max abs diff {err:.2e}  "
          f"({'OK' if err < 1e-4 else 'MISMATCH'})")
    print(f"  cleartext affinity: {[round(v, 4) for v in got.flatten().tolist()]}")
    timer.report(batch, total)
    return err < 1e-4


def _quant(x, s):
    """Round to `s` fractional bits (the fixed-point grid, in float64)."""
    scale = float(1 << s)
    return torch.round(x * scale) / scale


def run_fixedpoint(net, cnn, dataset, davis_index, batch, scales, device):
    """Emulate MPC fixed-point arithmetic on regB and show how the float prediction
    is affected: quantize weights to `s` fractional bits and truncate every
    Conv/Linear output to `s` bits (ReLU/max are exact), for a real Davis point.
    This is what the secure 64-bit FSS run (NssMPClib SCALE_BIT / Orca scale) computes
    numerically -- runnable here for full-size regB (no ring, so no overflow)."""
    import copy
    xd, xt = _input_tokens(dataset, batch, davis_index)
    drug = onehot4d(xd, CHARISOSMILEN + 1).double()
    prot = onehot4d(xt, CHARPROTLEN + 1).double()
    net = net.double().eval()
    with torch.no_grad():
        flt = net(drug, prot).squeeze(-1)
    pt = f"Davis test #{davis_index}" if davis_index is not None else "random tokens"
    print(f"=== regB fixed-point vs float | pool={net.pool} | input={pt} ===")
    print(f"  float (cleartext) affinity: {[round(v, 5) for v in flt.flatten().tolist()]}")
    print(f"  {'scale_bits':>10} {'fixed-point affinity':>22} {'abs_err':>11} {'rel_err':>9}")
    for s in scales:
        q = copy.deepcopy(net)
        for m in q.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.data = _quant(m.weight.data, s)
                if m.bias is not None:
                    m.bias.data = _quant(m.bias.data, s)
        hooks = [m.register_forward_hook(lambda mod, i, o, s=s: _quant(o, s))
                 for m in q.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        with torch.no_grad():
            fp = q(drug, prot).squeeze(-1)
        for h in hooks:
            h.remove()
        err = (fp - flt).abs()
        rel = (err / flt.abs().clamp(min=1e-9)).mean().item()
        val = [round(v, 5) for v in fp.flatten().tolist()]
        print(f"  {s:>10} {str(val):>22} {err.max().item():>11.2e} {rel:>9.2%}")
    print("  (weights quantized to s bits; every Conv/Linear output truncated to s bits;"
          "\n   ReLU/max-pool exact. This is the fixed-point effect the FSS scale controls.)")


def run_secure(net, dataset, role, device, davis_index=None, batch=1):
    """NssMPClib 2PC secure inference. Mirrors XidianNSS/NssMPClib
    tests/application/neural_network/2pc/. Runs on the GPU VM (needs nssmpc)."""
    import nssmpc.application.neural_network as nn_mpc               # noqa
    from nssmpc import PartyRuntime, Party2PC, SEMI_HONEST           # noqa

    party = Party2PC(role, SEMI_HONEST); party.online()
    with PartyRuntime(party):
        Sec = nn_mpc.utils.convert_model(type(net))                 # NssDTA -> secure
        cipher = Sec()
        if role == 0:                                               # server: holds weights
            shared = nn_mpc.utils.share_model_param(model=net)
            cipher = nn_mpc.utils.load_shared_param(cipher, shared)
            loader = nn_mpc.utils.SharedDataLoader(src_id=1)        # receives client input
        else:                                                       # client: holds secret input
            cipher = nn_mpc.utils.load_shared_param(cipher, nn_mpc.utils.share_model_param(model=net))
            xd, xt = _input_tokens(dataset, batch, davis_index)
            drug, prot = onehot4d(xd, CHARISOSMILEN + 1), onehot4d(xt, CHARPROTLEN + 1)
            # NOTE: NssDTA.forward takes TWO tensors; the example's SharedDataLoader
            # yields one. On the VM, feed drug+prot either as a tuple batch or two
            # loaders (confirm against your nssmpc version). Pack here:
            loader = nn_mpc.utils.SharedDataLoader(data_loader=[(drug, prot)])
        t0 = time.time()
        for data in loader:
            out = cipher(*data) if isinstance(data, (tuple, list)) else cipher(data)
            res = out.recon(target_id=1)
            if role == 1:
                pred = res.convert_to_real_field()
                print(f"  secure affinity (client): {pred.flatten().tolist()}")
        print(f"  single-sample secure latency: {time.time()-t0:.3f}s (role {role}, {device})")
    party.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=None); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None, help="named preset (e.g. regB) instead of --summary")
    ap.add_argument("--dataset", default="davis", choices=["davis", "kiba", "bindingdb"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--party", type=int, choices=[0, 1], default=None,
                    help="run NssMPClib 2PC as this party (needs nssmpc + a peer). "
                         "Omit for the plaintext torch check + timing.")
    ap.add_argument("--davis-index", type=int, default=None,
                    help="use this real test-set row as the secret input (else random tokens)")
    ap.add_argument("--fixed-point", action="store_true",
                    help="show the fixed-point-vs-float effect (scale-bit sweep) on the input")
    ap.add_argument("--scale-bits", default="8,10,13,16,20",
                    help="comma list of fractional bits for --fixed-point")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    net, cnn = nss_from_cnndta(args.summary, args.ckpt, args.config)
    if args.fixed_point:
        scales = [int(s) for s in args.scale_bits.split(",")]
        run_fixedpoint(net, cnn, args.dataset, args.davis_index, args.batch, scales, device)
        return
    if args.party is None:
        ok = run_plaintext(net, cnn, args.dataset, args.batch, device, args.davis_index)
        raise SystemExit(0 if ok else 1)
    run_secure(net, args.dataset, args.party, device, args.davis_index, args.batch)


if __name__ == "__main__":
    main()
