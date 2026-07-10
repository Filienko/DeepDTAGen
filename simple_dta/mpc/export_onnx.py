"""Export the FSS-ready affinity model to ONNX for the Orca / EzPC GPU-FSS port.

The production, genuinely-GPU-optimized-FSS route (see MPC_PORT.md) consumes an
ONNX graph: EzPC's OnnxBridge turns it into a 2-party C++ app with LLAMA (CPU FSS)
or GPU-MPC/Orca (CUDA FSS) backends. This script exports `MPCReadyNet` (one-hot
inputs -> MatMul/Conv/Relu/mean/Gemm only -- no Embedding, no max-pool) at a fixed
input shape, and verifies onnxruntime reproduces the torch output.

Usage:
    python -m mpc.export_onnx --dataset davis --out mpc/affinity_fss.onnx
"""
import argparse
import os

import numpy as np
import torch

from data import DEFAULTS, CHARISOSMILEN, CHARPROTLEN
from models import CNNDTA
from mpc.model_mpc import MPCReadyNet, to_onehot


def build_and_export(dataset, out_path, batch=1, pool_model=None):
    d = DEFAULTS[dataset]
    Ld, Lp = d["max_smi_len"], d["max_seq_len"]
    Vd, Vp = CHARISOSMILEN + 1, CHARPROTLEN + 1

    torch.manual_seed(0)
    cnn = pool_model or CNNDTA(pool="mean", proj_dim=0, head_layers=2, head_dim=128)
    net = MPCReadyNet(cnn).eval()

    drug = torch.zeros(batch, Ld, Vd)
    prot = torch.zeros(batch, Lp, Vp)
    # a valid one-hot dummy (fixed shape is what matters for tracing/FSS keys)
    drug[:, torch.arange(Ld), torch.randint(0, Vd, (Ld,))] = 1.0
    prot[:, torch.arange(Lp), torch.randint(0, Vp, (Lp,))] = 1.0

    with torch.no_grad():
        ref = net(drug, prot)

    torch.onnx.export(
        net, (drug, prot), out_path,
        input_names=["drug_onehot", "prot_onehot"], output_names=["affinity"],
        opset_version=13, dynamo=False)
    print(f"exported {out_path}  (drug [{batch},{Ld},{Vd}], prot [{batch},{Lp},{Vp}])")
    return out_path, (drug, prot), ref


def verify_onnx(out_path, inputs, ref):
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(out_path))
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    drug, prot = inputs
    got = sess.run(None, {"drug_onehot": drug.numpy(), "prot_onehot": prot.numpy()})[0]
    err = np.abs(got.reshape(-1) - ref.numpy().reshape(-1)).max()
    print(f"onnxruntime vs torch: max abs err {err:.2e}")
    assert err < 1e-4, "ONNX export does not match torch"
    print("ONNX export OK (numerics match torch)")
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="davis", choices=["davis", "kiba", "bindingdb"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "affinity_fss.onnx"))
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    path, inputs, ref = build_and_export(args.dataset, args.out, args.batch)
    verify_onnx(path, inputs, ref)
    print("\nNext: feed this ONNX to EzPC OnnxBridge for the GPU-FSS (Orca) 2PC app "
          "-- see mpc/MPC_PORT.md.")


if __name__ == "__main__":
    main()
