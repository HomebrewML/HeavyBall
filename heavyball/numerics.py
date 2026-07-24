"""Shared numerics for slab-native HeavyBall implementations."""

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate


def _wide(value: Tensor) -> Tensor:
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def broadcast_leaf(scalar: Tensor, target: Tensor) -> Tensor:
    """Broadcast one scalar per slab row over ``target``'s trailing axes."""

    if isinstance(target, DTensor):
        scalar = DTensor.from_local(
            scalar,
            target.device_mesh,
            (Replicate(),),
            run_check=False,
            shape=scalar.shape,
            stride=scalar.stride(),
        ).redistribute(placements=target.placements)
    return scalar.reshape((scalar.shape[0],) + (1,) * (target.ndim - 1))


def balance_factors(factors: list[Tensor]) -> list[Tensor]:
    """Equalize factor log-magnitudes without changing their joint scale."""

    logs = []
    precise_logs = []
    for factor in factors:
        maximum = factor.abs().amax(dim=tuple(range(1, factor.ndim)))
        logs.append(maximum.log())
        precise_logs.append(maximum.double().log())
    mean_log = logs[0]
    precise_mean_log = precise_logs[0]
    for log in logs[1:]:
        mean_log = mean_log + log
    for log in precise_logs[1:]:
        precise_mean_log = precise_mean_log + log
    mean_log = mean_log * 0.5 if len(logs) == 2 else mean_log / len(logs)
    precise_mean_log = (
        precise_mean_log * 0.5
        if len(precise_logs) == 2
        else precise_mean_log / len(precise_logs)
    )

    scales = []
    valid = torch.ones_like(mean_log, dtype=torch.bool)
    reconstruct_valid = torch.isfinite(precise_mean_log)
    for log in logs:
        scale = (mean_log - log).exp()
        scales.append(scale)
        valid = valid & torch.isfinite(scale) & (scale > 0)
    for log in precise_logs:
        reconstruct_valid = reconstruct_valid & torch.isfinite(log)

    balanced = []
    for factor, scale, precise_log in zip(factors, scales, precise_logs, strict=True):
        direct = factor * broadcast_leaf(scale, factor)
        delta = precise_mean_log - precise_log
        reconstructed = (
            factor.double().sign()
            * (factor.double().abs().log() + broadcast_leaf(delta, factor)).exp()
        ).to(factor.dtype)
        reconstructed = torch.where(
            broadcast_leaf(reconstruct_valid, factor), reconstructed, factor
        )
        balanced.append(
            torch.where(broadcast_leaf(valid, factor), direct, reconstructed)
        )
    return balanced


def _strictly_aligned(left: Tensor, right: Tensor) -> Tensor:
    return ((left > 0) & (right > 0)) | ((left < 0) & (right < 0))


def _caution(grad: Tensor, update: Tensor) -> Tensor:
    """Apply legacy caution independently to every slab leaf."""

    aligned = _strictly_aligned(grad, update)
    aligned_flat = aligned.reshape(aligned.shape[0], -1)
    scale = aligned_flat.shape[1] / aligned_flat.sum(dim=1, keepdim=True).clamp_min(1).to(update.dtype)
    update = torch.where(aligned, update, torch.zeros_like(update))
    return update * broadcast_leaf(scale, update)


def _stochastic_keep_finite(bits: Tensor, noise: Tensor, keep_bits: int) -> Tensor:
    """Stochastically truncate float32 bits without rounding finite values to infinity."""

    discarded = 32 - keep_bits
    if discarded == 0:
        return bits
    rounded = (bits + noise).bitwise_and(-(1 << discarded))
    exponent_mask = 0x7F800000
    finite = bits.bitwise_and(exponent_mask) != exponent_mask
    rounded_to_nonfinite = rounded.bitwise_and(exponent_mask) == exponent_mask
    largest_finite = bits.bitwise_and(-0x80000000).bitwise_or(
        0x7F7FFFFF & -(1 << discarded)
    )
    return torch.where(finite & rounded_to_nonfinite, largest_finite, rounded)


def stochastic_round_bfloat16(value: Tensor, random: Tensor) -> Tensor:
    """Round to bfloat16 using explicit uniform noise and saturate finite overflow."""

    bits = value.float().view(torch.int32)
    noise = (random * (1 << 16)).to(torch.int32)
    return _stochastic_keep_finite(bits, noise, 16).view(torch.float32).bfloat16()


def stochastic_copy_(target: Tensor, source: Tensor, tempo, shared_noise=None) -> None:
    if target.dtype == torch.bfloat16 and source.dtype in (torch.float16, torch.float32, torch.float64):
        # Only reuse the parameter's noise when it is element-for-element aligned with the source:
        # same shape AND same sharding (DTensor placements match; None==None for plain tensors). An
        # owner-whole DTensor state whose global shape coincides with the param must NOT borrow the
        # param's batch-sharded noise -- it draws its own, correctly keyed per owner-whole leaf.
        aligned = (shared_noise is not None and shared_noise.shape == source.shape
                   and getattr(shared_noise, "placements", None) == getattr(source, "placements", None))
        noise = shared_noise if aligned else tempo.random_like(source)
        source = stochastic_round_bfloat16(source, noise)
    target.copy_(source)


def stable_l2_normalize(value: Tensor, *, dim: int | tuple[int, ...] | None, eps) -> Tensor:
    """Normalize stably along ``dim``, retaining legacy's small-value branch."""

    if value.numel() == 0:
        return value
    scale = value.abs().amax(dim=dim, keepdim=True)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    scaled = value / safe_scale
    scaled = torch.where(torch.isfinite(scaled), scaled, value.sign())
    norm = torch.linalg.vector_norm(scaled, dim=dim, keepdim=True)
    unit = scaled / torch.where(norm != 0, norm, torch.ones_like(norm))
    if eps is None:
        return unit
    epsilon = torch.as_tensor(eps, dtype=value.dtype, device=value.device).reshape(())
    direct = value / torch.where(epsilon != 0, epsilon, torch.ones_like(epsilon))
    return torch.where(norm * scale >= epsilon, unit, direct)
