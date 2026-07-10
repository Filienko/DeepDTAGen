"""MPC / FSS port of the affinity-only DTA model.

Contents:
  model_mpc.py  -- FSS-ready CNN written once against a Backend interface
                   (embedding -> one-hot matmul, mean-pool, ReLU-only, static
                   shapes). `ClearBackend` is the plaintext reference.
  fss_infer.py  -- `FSSBackend`: a self-contained semi-honest 2-party FSS
                   inference engine (fixed-point ring + sycret DCF DReLU +
                   Beaver-triple select). Runnable, device-agnostic PoC.
  export_onnx.py-- export the FSS-ready model to ONNX for the Orca/EzPC GPU-FSS
                   pipeline (see MPC_PORT.md).

See ../FSS_FRAMEWORKS.md and MPC_PORT.md for the why and the production path.
"""
