def set_torch(matmul_precision: str = "high", einsum_strategy: str = "auto-hq") -> None:
    """Opt into process-wide Torch matmul, cuDNN, and einsum settings."""
    import torch
    from torch.backends import cudnn, opt_einsum

    cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)
    if opt_einsum.is_available():
        try:
            opt_einsum.strategy = einsum_strategy
        except (TypeError, ValueError):
            if einsum_strategy != "auto-hq":
                raise
            opt_einsum.strategy = "optimal"
