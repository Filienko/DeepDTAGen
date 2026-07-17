"""Reproducible correctness checks for the MPC/FSS port. Run: python -m mpc.test_fss

  1. G0: the FSS-ready one-hot `MPCModel` equals the original embedding `CNNDTA`.
  2. FSS primitives: fixed-point round-trip, additive sharing, secure ReLU.
  3. G1: end-to-end 2-party FSS inference matches cleartext within fixed-point
     tolerance at f=6 (sycret's 32-bit ring).
"""
import numpy as np
import torch

from data import DEFAULTS, CHARISOSMILEN, CHARPROTLEN
from models import CNNDTA
from mpc import fss_infer as F
from mpc.model_mpc import MPCModel


def test_g0_onehot_equals_embedding():
    d = DEFAULTS["davis"]
    for pool in ("mean", "max"):
        torch.manual_seed(0)
        m = CNNDTA(pool=pool, proj_dim=0, head_layers=2, head_dim=128).eval()
        mpc = MPCModel(m)
        xd = torch.randint(0, CHARISOSMILEN + 1, (4, d["max_smi_len"]))
        xt = torch.randint(0, CHARPROTLEN + 1, (4, d["max_seq_len"]))
        with torch.no_grad():
            ref = m((xd, xt, None))
            got = mpc.forward_tokens_clear(xd, xt)
        assert torch.allclose(ref, got, atol=1e-5), (pool, (ref - got).abs().max())
        print(f"G0 OK ({pool}-pool): one-hot MPCModel == embedding CNNDTA (max diff "
              f"{(ref-got).abs().max().item():.1e})")


def test_fixedpoint_roundtrip():
    x = torch.randn(1000, dtype=torch.float64) * 5
    for f in (6, 8, 12):
        y = F.decode(F.encode(x, f), f)
        assert (x - y).abs().max() < 2.0 ** (-f) * 2, f
    print("OK: fixed-point encode/decode round-trips within 1 LSB")


def test_secure_relu():
    dev = torch.device("cpu")
    d = F.Dealer(dev)
    be = F.FSSBackend(d, f=8, device=dev)
    x = torch.tensor([[-2.5, -0.1, 0.0, 0.3, 1.7, 4.0]], dtype=torch.float64)
    X = F.encode(x, 8)
    out = be.relu(F.share(X))
    got = F.decode(F.reconstruct(out), 8)
    exp = x.clamp(min=0)
    assert torch.allclose(got, exp, atol=1e-2), (got, exp)
    print("OK: secure FSS ReLU == max(x,0)")


def test_g1_end_to_end():
    dev = torch.device("cpu")
    m = F.build_demo_model("small")
    mpc = MPCModel(m)
    xd, xt = F.make_inputs("davis", batch=8, seq_scale=0.1)
    with torch.no_grad():
        clear = mpc.forward_tokens_clear(xd, xt).to(torch.float64)
    pred, be, _ = F._run(mpc, xd, xt, f=6, device=dev)
    mae = (pred - clear).abs().mean().item()
    assert mae < 5e-2, f"FSS inference too lossy at f=6: mae={mae}"
    print(f"G1 OK: FSS 2PC inference == cleartext at f=6 (mae {mae:.2e}, "
          f"{be.relu_calls} DReLUs, {be.online_rounds} online rounds)")


def test_secure_max_pool():
    """FSS global max-pool (secure_max tournament) == cleartext max over L."""
    dev = torch.device("cpu")
    d = F.Dealer(dev)
    be = F.FSSBackend(d, f=8, device=dev)
    x = torch.randn(2, 3, 17, dtype=torch.float64) * 4      # [B,C,L], odd L
    sh = F.share(F.encode(x, 8))
    out = be.max_pool(sh)
    got = F.decode(F.reconstruct(out), 8)
    exp = x.max(dim=2).values
    assert torch.allclose(got, exp, atol=2e-2), (got - exp).abs().max()
    print(f"OK: secure FSS max-pool == cleartext max (L=17 -> {be.relu_calls} "
          "tournament DReLUs)")


def test_g1_maxpool_end_to_end():
    """A MAX-pool CNN+CNN (the user's default checkpoint kind) runs under FSS."""
    dev = torch.device("cpu")
    m = F.build_demo_model("small", pool="max")
    mpc = MPCModel(m)
    assert mpc.pool_mode == "max"
    xd, xt = F.make_inputs("davis", batch=6, seq_scale=0.05)
    with torch.no_grad():
        clear = mpc.forward_tokens_clear(xd, xt).to(torch.float64)
    pred, be, _ = F._run(mpc, xd, xt, f=6, device=dev)
    mae = (pred - clear).abs().mean().item()
    assert mae < 5e-2, f"max-pool FSS too lossy at f=6: mae={mae}"
    print(f"G1-max OK: MAX-pool CNN+CNN FSS == cleartext at f=6 (mae {mae:.2e}, "
          f"{be.relu_calls} DReLUs, {be.online_rounds} rounds -- note the extra "
          "DReLUs vs mean-pool)")


def test_private_weights():
    """Private (secret-shared) weights via Beaver matmul/conv == cleartext, and
    cost more online rounds than public weights."""
    dev = torch.device("cpu")
    m = F.build_demo_model("small")
    mpc = MPCModel(m)
    xd, xt = F.make_inputs("davis", batch=4, seq_scale=0.1)
    with torch.no_grad():
        clear = mpc.forward_tokens_clear(xd, xt).to(torch.float64)
    priv, be_p, _ = F._run(mpc, xd, xt, f=6, device=dev, private_weights=True)
    pub, be_q, _ = F._run(mpc, xd, xt, f=6, device=dev, private_weights=False)
    assert (priv - clear).abs().mean() < 5e-2, "private-weight FSS too lossy"
    assert (pub - clear).abs().mean() < 5e-2, "public-weight FSS too lossy"
    assert be_p.online_rounds > be_q.online_rounds, "private weights should add rounds"
    assert be_p.linear_rounds == 9, be_p.linear_rounds   # 2 embed + 4 conv + 3 linear
    print(f"OK: PRIVATE-weight FSS == cleartext (mae {(priv-clear).abs().mean():.2e}); "
          f"{be_p.online_rounds} rounds vs {be_q.online_rounds} public "
          f"(+{be_p.linear_rounds} Beaver linear layers)")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    test_g0_onehot_equals_embedding()
    test_fixedpoint_roundtrip()
    test_secure_relu()
    test_secure_max_pool()
    test_g1_end_to_end()
    test_g1_maxpool_end_to_end()
    test_private_weights()
    print("\nALL MPC/FSS CHECKS PASSED")
