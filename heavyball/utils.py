def set_torch(matmul_precision: str = "high", einsum_strategy: str = "auto-hq") -> None:
    """Opt-in torch setup for heavyball; call once at startup. Mutates PROCESS-WIDE state -- it
    affects every fp32 matmul in the process, your model's included, so it is not applied on import.

    - matmul_precision "high" lets fp32 matmuls use a fast algorithm (TF32 on tensor-core GPUs),
      ~2x throughput. For the optimizers' preconditioner contractions this is marginal for training:
      measured across SOAP/PSGD/KLSOAP, Hessian conditioning up to 1e6, default and 20x learning
      rates, the loss gap vs full fp32 ("highest") stays under ~1% (below seed noise) even though the
      parameter path can diverge far more. Pass "highest" for numerically sensitive work.
    - opt_einsum (a declared dependency) selects optimal contraction paths. The Kronecker-factored
      contractions are written pairwise, so they stay O(n^2) even if opt_einsum is absent; the
      dependency then affects contraction performance, not memory safety.

    Optimizer STATE precision (fp32 accumulation + stochastic rounding) is independent of this flag
    and is what keeps heavyball close to the fp64 trajectory.
    """
    import torch
    from torch.backends import cudnn, opt_einsum

    cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)  # "high" == TF32 for fp32 matmuls
    if opt_einsum.is_available():  # torch enables it by default when present; just pick the path strategy
        try:
            opt_einsum.strategy = einsum_strategy
        except (TypeError, ValueError):
            if einsum_strategy != "auto-hq":
                raise
            opt_einsum.strategy = "optimal"
