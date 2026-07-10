"""Self-contained 2-party FSS inference engine for the affinity CNN (PoC).

This is the runnable "MPC implementation itself is GPU-optimized, hence FSS"
artifact. It executes `MPCModel` (model_mpc.py) under a genuine Function Secret
Sharing protocol:

  * **Fixed-point** over the ring Z_{2^32} (`torch.int64` storage, natural
    wraparound), `f` fractional bits.
  * **Additive 2-of-2 secret sharing** of the (one-hot-encoded) private input.
    Model weights are **public** (server holds the model, client holds the
    private molecule+protein) -> every Conv/Linear/mean-pool is LOCAL (no
    interaction); only ReLU needs the network.
  * **ReLU via FSS**: DReLU is a real Distributed Comparison Function from the
    `sycret` library (AriaNN's Rust FSS core) -- one online round, evaluated
    element-wise (the GPU-parallel primitive). The `x * DReLU(x)` select is a
    Beaver-triple multiplication.

Honesty / scope (semi-honest, PoC):
  * Both parties are simulated in one process with a trusted dealer (offline
    correlated randomness). No sockets; "reveal" = reconstruct-in-sim. The
    computation each party does uses only its own shares + keys, so it faithfully
    models the protocol's cost structure.
  * `sycret`'s FSS core is compiled for a **32-bit** ring, which bounds the
    fixed-point budget: with many conv accumulations, small `f` is required to
    avoid overflow/truncation error. The frac-bit sweep below makes this precision
    ceiling explicit -- it is exactly why production GPU-FSS (Orca, 64-bit ring +
    CUDA DReLU kernels; see MPC_PORT.md) is the deployment target, not this PoC.
  * Integer conv/matmul run on `--device`; `sycret` key-eval is CPU. Orca is what
    makes the nonlinear key-eval itself GPU-parallel.

Usage:
    python -m mpc.fss_infer --device cpu --frac-bits 8
    python -m mpc.fss_infer --sweep            # frac-bit precision sweep
"""
import argparse
import time

import numpy as np
import torch

from data import DEFAULTS, CHARISOSMILEN, CHARPROTLEN
from models import CNNDTA
from mpc.model_mpc import MPCModel, Backend, ClearBackend, to_onehot

RING_BITS = 32
RING = 1 << RING_BITS
HALF = 1 << (RING_BITS - 1)
MASK = RING - 1


# --- fixed-point <-> ring -------------------------------------------------
def encode(x, f):
    return (torch.round(x * (1 << f)).to(torch.int64)) & MASK


def decode(v, f):
    v = v & MASK
    v = torch.where(v >= HALF, v - RING, v)
    return v.to(torch.float64) / (1 << f)


def rand_ring(shape, device):
    return torch.randint(0, RING, shape, dtype=torch.int64, device=device)


class Dealer:
    """Trusted offline party: sycret DReLU keys + Beaver triples."""

    def __init__(self, device):
        import sycret
        self.le = sycret.LeFactory(n_threads=1)
        self.device = device

    def drelu_keys(self, n):
        k0, k1 = self.le.keygen(n)
        alpha = np.asarray(self.le.alpha(k0, k1)).astype(np.int64) & MASK
        return k0, k1, torch.from_numpy(alpha).to(self.device)

    def beaver(self, shape, device):
        a = rand_ring(shape, device); b = rand_ring(shape, device)
        c = (a * b) & MASK
        return share(a), share(b), share(c)


def share(v):
    """Additive 2-of-2 secret sharing of ring tensor v."""
    s0 = rand_ring(v.shape, v.device)
    s1 = (v - s0) & MASK
    return s0, s1


def reconstruct(sh):
    return (sh[0] + sh[1]) & MASK


class FSSBackend(Backend):
    """Runs MPCModel under the FSS protocol. Weights public, activations shared."""

    def __init__(self, dealer, f, device):
        self.d = dealer
        self.f = f
        self.device = device
        self.relu_calls = 0
        self.online_rounds = 0  # sequential comm rounds (DReLU reveal + Beaver reveals)

    # weights are public floats -> fixed-point ring tensors
    def _w(self, weight):
        return encode(weight.to(torch.float64), self.f).to(self.device)

    def _trunc(self, sh):
        """SecureML 2-party local truncation by f bits (rescale after a mult)."""
        s0, s1 = sh
        t0 = (s0 & MASK) >> self.f
        t1 = (RING - (((RING - (s1 & MASK)) & MASK) >> self.f)) & MASK
        return t0 & MASK, t1 & MASK

    def _add_public(self, sh, pub):
        # add a public vector (bias) to a shared tensor -> add to one share only
        return (sh[0] + pub) & MASK, sh[1]

    def embed(self, x_onehot, table):
        # x_onehot is a *shared* [B,L,V]; table public [V,E]. matmul then truncate.
        W = self._w(table)
        out0 = torch.matmul(x_onehot[0], W) & MASK
        out1 = torch.matmul(x_onehot[1], W) & MASK
        out = self._trunc((out0, out1))
        return out[0].transpose(1, 2) & MASK, out[1].transpose(1, 2) & MASK

    def conv1d(self, x, weight, bias):
        W = self._w(weight)
        # int64 conv: linear, so apply per share locally; bias added once, post-trunc
        c0 = torch.nn.functional.conv1d(x[0], W) & MASK
        c1 = torch.nn.functional.conv1d(x[1], W) & MASK
        t0, t1 = self._trunc((c0, c1))
        b = self._w(bias).view(1, -1, 1)
        return self._add_public((t0, t1), b)

    def linear(self, x, weight, bias):
        W = self._w(weight)
        o0 = torch.matmul(x[0], W.t()) & MASK
        o1 = torch.matmul(x[1], W.t()) & MASK
        t0, t1 = self._trunc((o0, o1))
        return self._add_public((t0, t1), self._w(bias).view(1, -1))

    def mean_pool(self, x):
        L = x[0].shape[2]
        s0 = x[0].sum(dim=2) & MASK
        s1 = x[1].sum(dim=2) & MASK
        # multiply by public 1/L (fixed-point) then truncate
        inv = encode(torch.tensor(1.0 / L, dtype=torch.float64), self.f).to(self.device)
        return self._trunc(((s0 * inv) & MASK, (s1 * inv) & MASK))

    def concat(self, a, b):
        return (torch.cat([a[0], b[0]], dim=1) & MASK,
                torch.cat([a[1], b[1]], dim=1) & MASK)

    def relu(self, x):
        """ReLU(x) = x * DReLU(x). DReLU via sycret FSS DCF; select via Beaver."""
        self.relu_calls += 1
        shape = x[0].shape
        n = int(np.prod(shape))
        # --- FSS DReLU: reveal masked z = x + alpha, eval keys -> shares of (x<=0)
        k0, k1, alpha = self.d.drelu_keys(n)
        alpha = alpha.view(shape)
        z = (reconstruct(x) + alpha) & MASK           # one online round (masked reveal)
        self.online_rounds += 1
        z_np = (z.detach().cpu().numpy().astype(np.int64) & MASK).reshape(n)
        le0 = torch.from_numpy(np.asarray(self.d.le.eval(0, z_np, k0)).astype(np.int64)
                               ).to(self.device).view(shape) & MASK
        le1 = torch.from_numpy(np.asarray(self.d.le.eval(1, z_np, k1)).astype(np.int64)
                               ).to(self.device).view(shape) & MASK
        # b = (x > 0) = 1 - Le(x); as integer {0,1} shares (party0 holds the +1)
        b0 = (1 - le0) & MASK
        b1 = (-le1) & MASK
        # --- Beaver select: x (fixed-point) * b (0/1 integer) ; no truncation (b is 0/1)
        (a0, a1), (bb0, bb1), (c0, c1) = self.d.beaver(shape, self.device)
        d = (reconstruct(x) - reconstruct((a0, a1))) & MASK
        e = (reconstruct((b0, b1)) - reconstruct((bb0, bb1))) & MASK
        self.online_rounds += 1                        # Beaver reveal round
        r0 = (c0 + d * bb0 + e * a0 + d * e) & MASK
        r1 = (c1 + d * bb1 + e * a1) & MASK
        return r0, r1


def _run(mpc, smiles, target, f, device):
    dealer = Dealer(device)
    be = FSSBackend(dealer, f, device)
    xd = encode(to_onehot(smiles, mpc.drug_vocab).to(torch.float64), f).to(device)
    xt = encode(to_onehot(target, mpc.prot_vocab).to(torch.float64), f).to(device)
    t0 = time.time()
    out = mpc.forward(be, share(xd), share(xt))
    dt = time.time() - t0
    return decode(reconstruct(out), f).squeeze(-1), be, dt


def build_demo_model(scale):
    """A CNNDTA(pool=mean). `scale`='small' downsizes towers/head so the 32-bit
    fixed-point PoC has precision headroom; 'full' is the real architecture."""
    torch.manual_seed(0)
    if scale == "small":
        m = CNNDTA(pool="mean", proj_dim=0, head_layers=2, head_dim=64,
                   drug_channels=[8, 12], prot_channels=[8, 12],
                   drug_kernel=4, prot_kernel=8, embed_dim=32)
    else:
        m = CNNDTA(pool="mean", proj_dim=0, head_layers=2, head_dim=128)
    # Give a realistic affinity output scale (a trained model predicts ~5-10, not
    # ~0 like a random init) so the error metrics below are interpretable. Only the
    # final public bias changes; it does not affect the protocol.
    with torch.no_grad():
        last = [mod for mod in m.head.net if isinstance(mod, torch.nn.Linear)][-1]
        last.bias.fill_(7.0)
    m.eval()
    return m


def make_inputs(dataset, batch, seq_scale=1.0):
    d = DEFAULTS[dataset]
    Ls = max(16, int(d["max_smi_len"] * seq_scale))
    Lp = max(32, int(d["max_seq_len"] * seq_scale))
    torch.manual_seed(1)
    xd = torch.randint(0, CHARISOSMILEN + 1, (batch, Ls))
    xt = torch.randint(0, CHARPROTLEN + 1, (batch, Lp))
    return xd, xt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frac-bits", type=int, default=6,
                    help="fixed-point fractional bits (6 fits sycret's 32-bit ring "
                         "for this CNN; higher overflows -- see --sweep)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dataset", default="davis", choices=["davis", "kiba", "bindingdb"])
    ap.add_argument("--scale", default="small", choices=["small", "full"],
                    help="'small' fits 32-bit fixed-point; 'full' is the real arch")
    ap.add_argument("--seq-scale", type=float, default=0.1,
                    help="fraction of full padded sequence length (keeps the PoC fast)")
    ap.add_argument("--sweep", action="store_true", help="frac-bit precision sweep")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    m = build_demo_model(args.scale)
    mpc = MPCModel(m)
    xd, xt = make_inputs(args.dataset, args.batch, args.seq_scale)

    with torch.no_grad():
        clear = mpc.forward_tokens_clear(xd, xt).to(torch.float64)

    if args.sweep:
        print(f"=== FSS precision sweep | scale={args.scale} | dev={device} | "
              f"drug L={xd.shape[1]} prot L={xt.shape[1]} | batch={args.batch} ===")
        print(f"{'frac_bits':>9} {'max_abs_err':>12} {'mean_abs_err':>13} {'rel_err':>9}")
        for f in (6, 8, 10, 12, 14):
            pred, be, _ = _run(mpc, xd, xt, f, device)
            err = (pred - clear).abs()
            rel = (err / clear.abs().clamp(min=1e-6)).mean().item()
            print(f"{f:>9} {err.max().item():>12.2e} {err.mean().item():>13.2e} {rel:>9.2%}")
        return

    pred, be, dt = _run(mpc, xd, xt, args.frac_bits, device)
    err = (pred - clear).abs()
    print(f"=== FSS 2PC inference | scale={args.scale} | dev={device} | f={args.frac_bits} ===")
    print(f"  inputs: {args.batch} pairs | drug L={xd.shape[1]} prot L={xt.shape[1]}")
    print(f"  cleartext pred : {[round(v,4) for v in clear.tolist()]}")
    print(f"  FSS pred       : {[round(v,4) for v in pred.tolist()]}")
    print(f"  max abs err    : {err.max().item():.3e}   mean abs err: {err.mean().item():.3e}")
    print(f"  ReLU (FSS DReLU) calls: {be.relu_calls} | online rounds: {be.online_rounds}")
    print(f"  wall time      : {dt:.2f}s  ({device})")


if __name__ == "__main__":
    main()
