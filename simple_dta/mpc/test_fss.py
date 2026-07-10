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
    torch.manual_seed(0)
    m = CNNDTA(pool="mean", proj_dim=0, head_layers=2, head_dim=128).eval()
    mpc = MPCModel(m)
    d = DEFAULTS["davis"]
    xd = torch.randint(0, CHARISOSMILEN + 1, (4, d["max_smi_len"]))
    xt = torch.randint(0, CHARPROTLEN + 1, (4, d["max_seq_len"]))
    with torch.no_grad():
        ref = m((xd, xt, None))
        got = mpc.forward_tokens_clear(xd, xt)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()
    print("G0 OK: one-hot MPCModel == embedding CNNDTA (max diff "
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


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    test_g0_onehot_equals_embedding()
    test_fixedpoint_roundtrip()
    test_secure_relu()
    test_g1_end_to_end()
    print("\nALL MPC/FSS CHECKS PASSED")
