"""Batched PSGD-PRO and QSGD for slab-native HeavyBall.

Both variants fit the same full per-dimension Q factors with legacy's
stochastic PRO/Procrustes update.  PSGD-PRO applies P = QᵀQ, while QSGD
applies each fitted Q factor once.
"""

import torch
from torch import Tensor

from .core import Recipe
from .kron import (
    _max_singular_value_power_iter,
    _next_lower_bound,
    _precondition,
    _precondition_mixed,
    psgd_kron_init,
)
from .matrix import merge_matrix_transform
from .numerics import _wide, balance_factors, broadcast_leaf
from .transforms import WHOLE, Tempo, sgd_commit


def _apply_once(update: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Apply one Q factor along each matrix dimension, as legacy QSGD does."""

    # Pairwise contraction avoids an O(n⁴) three-operand intermediate.
    return torch.einsum("nib,njb->nij", torch.einsum("nia,nab->nib", q0, update), q1)


def _apply_once_mixed(update: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Apply mixed diagonal/triangular Q factors once, as legacy QSGD does."""

    # Branching avoids an O(n³) three-operand intermediate.
    if q0.ndim == 2 and q1.ndim == 3:
        return torch.einsum("...ab,...Bb->...aB", q0.unsqueeze(-1) * update, q1)
    if q0.ndim == 3 and q1.ndim == 2:
        return torch.einsum("...Aa,...ab->...Ab", q0, q1.unsqueeze(-2) * update)
    return torch.einsum("...a,...b,...ab->...ab", q0, q1, update)


def _update_factor(
    q: Tensor,
    covariance: Tensor,
    target_energy: float,
    lower_bound: Tensor,
    tempo: Tempo,
    power_iterations: int,
) -> tuple[Tensor, Tensor]:
    """Port one full-matrix branch of legacy ``psgd_pro_update_precond``."""

    dtype = torch.promote_types(q.dtype, covariance.dtype)
    q = q.to(dtype)
    covariance = covariance.to(dtype)
    ell, lower_bound = _next_lower_bound(
        _max_singular_value_power_iter(covariance, power_iterations) + target_energy,
        lower_bound,
        tempo.hyper.lower_bound_beta,
    )
    candidate = q - (covariance @ q - target_energy * q) / broadcast_leaf(ell, q) * tempo.hyper.precond_lr

    rotation = (candidate.mT - candidate).contiguous()
    rotation_scale = _max_singular_value_power_iter(rotation, power_iterations).clamp_min(
        torch.finfo(rotation.dtype).smallest_normal
    )
    rotation = rotation / broadcast_leaf(rotation_scale, rotation)
    rotation_q = rotation @ candidate
    rotation_rotation_q = rotation @ rotation_q
    c1 = rotation_q.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    c2 = rotation_rotation_q.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    step = torch.where(c2 < 0, (-c1 / c2).clamp(min=0, max=1 / 8), 1 / 8)
    step = broadcast_leaf(step, candidate)
    return candidate + step * rotation_q + (0.5 * step * step) * rotation_rotation_q, lower_bound


def _update_diagonal_factor(
    q: Tensor,
    covariance: Tensor,
    target_energy: float,
    lower_bound: Tensor,
    tempo: Tempo,
) -> tuple[Tensor, Tensor]:
    """Port one diagonal branch of legacy ``psgd_pro_update_precond``."""

    dtype = torch.promote_types(q.dtype, covariance.dtype)
    q = q.to(dtype)
    covariance = covariance.to(dtype)
    ell, lower_bound = _next_lower_bound(
        covariance.amax(dim=-1) + target_energy,
        lower_bound,
        tempo.hyper.lower_bound_beta,
    )
    return (
        q
        - q
        * (covariance - target_energy)
        / broadcast_leaf(ell, q)
        * tempo.hyper.precond_lr,
        lower_bound,
    )


def _refresh_q(
    update: Tensor,
    q0: Tensor,
    q1: Tensor,
    lower0: Tensor,
    lower1: Tensor,
    tempo: Tempo,
    power_iterations: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run legacy's stochastic PRO Q update for a full two-factor slab."""

    q0, q1 = balance_factors([q0, q1])
    damping = tempo.hyper.dampening + torch.finfo(update.dtype).eps * update.abs()
    probe = tempo.randn_like(update)
    preconditioned = _precondition(update + damping * probe, q0, q1)

    covariance0 = torch.einsum("nab,ncb->nac", preconditioned, preconditioned)
    covariance1 = torch.einsum("nab,nac->nbc", preconditioned, preconditioned)
    total_numel = update.numel()
    q0, lower0 = _update_factor(
        q0,
        covariance0,
        total_numel / (q0.shape[0] * q0.shape[-1]),
        lower0,
        tempo,
        power_iterations,
    )
    q1, lower1 = _update_factor(
        q1,
        covariance1,
        total_numel / (q1.shape[0] * q1.shape[-1]),
        lower1,
        tempo,
        power_iterations,
    )
    return q0, q1, lower0, lower1


def _refresh_mixed_q(
    update: Tensor,
    q0: Tensor,
    q1: Tensor,
    lower0: Tensor,
    lower1: Tensor,
    tempo: Tempo,
    power_iterations: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run legacy's stochastic PRO Q update with at least one diagonal factor."""

    q0, q1 = balance_factors([q0, q1])
    damping = tempo.hyper.dampening + torch.finfo(update.dtype).eps * update.abs()
    probe = tempo.randn_like(update)
    preconditioned = _precondition_mixed(update + damping * probe, q0, q1)

    total_numel = update.numel()
    if q0.ndim == 2:
        covariance0 = torch.einsum("...ab,...ab->...a", preconditioned, preconditioned)
        q0, lower0 = _update_diagonal_factor(
            q0, covariance0, total_numel / q0.numel(), lower0, tempo
        )
    else:
        covariance0 = torch.einsum("nab,ncb->nac", preconditioned, preconditioned)
        q0, lower0 = _update_factor(
            q0,
            covariance0,
            total_numel / (q0.shape[0] * q0.shape[-1]),
            lower0,
            tempo,
            power_iterations,
        )
    if q1.ndim == 2:
        covariance1 = torch.einsum("...ab,...ab->...b", preconditioned, preconditioned)
        q1, lower1 = _update_diagonal_factor(
            q1, covariance1, total_numel / q1.numel(), lower1, tempo
        )
    else:
        covariance1 = torch.einsum("nab,nac->nbc", preconditioned, preconditioned)
        q1, lower1 = _update_factor(
            q1,
            covariance1,
            total_numel / (q1.shape[0] * q1.shape[-1]),
            lower1,
            tempo,
            power_iterations,
        )
    return q0, q1, lower0, lower1


def psgd_pro_init(ref_leaf: Tensor, *, max_size_triangular: Tensor | int = 2048) -> dict[str, Tensor]:
    """Allocate the same full two-factor Q state used by PSGD-Kron."""

    return psgd_kron_init(ref_leaf, max_size_triangular=max_size_triangular)


def psgd_nfactor_init(ref_leaf: Tensor, *, max_size_triangular: Tensor | int = 2048) -> dict[str, Tensor]:
    """Allocate one diagonal or triangular Q factor per original parameter axis."""

    limit = int(max_size_triangular)
    ref = _wide(ref_leaf)
    state = {}
    for index, size in enumerate(ref.shape):
        state[f"Q_{index}"] = (
            torch.ones(size, dtype=ref.dtype, device=ref.device)
            if size > limit
            else torch.eye(size, dtype=ref.dtype, device=ref.device)
        )
        state[f"running_lower_bound_{index}"] = torch.zeros(
            (), dtype=torch.float64, device=ref.device
        )
    return state


def _precondition_nfactor(update: Tensor, factors: list[Tensor]) -> Tensor:
    preconditioned = update
    for index, factor in enumerate(factors):
        moved = preconditioned.movedim(index + 1, 1)
        flat = moved.reshape(moved.shape[0], moved.shape[1], -1)
        if factor.ndim == 3:
            flat = torch.bmm(torch.bmm(factor.mT, factor), flat)
        else:
            flat = factor.square().unsqueeze(-1) * flat
        preconditioned = flat.reshape(moved.shape).movedim(1, index + 1)
    return preconditioned


def _refresh_nfactor(
    update: Tensor,
    factors: list[Tensor],
    lower_bounds: list[Tensor],
    tempo: Tempo,
    power_iterations: int,
) -> tuple[list[Tensor], list[Tensor]]:
    factors = balance_factors(factors)
    damping = tempo.hyper.dampening + torch.finfo(update.dtype).eps * update.abs()
    preconditioned = _precondition_nfactor(update + damping * tempo.randn_like(update), factors)
    next_factors = []
    next_lower_bounds = []
    for index, (factor, lower_bound) in enumerate(zip(factors, lower_bounds, strict=True)):
        triangular = factor.ndim == 3
        moved = preconditioned.movedim(index + 1, 1)
        flat = moved.reshape(moved.shape[0], moved.shape[1], -1)
        covariance = torch.bmm(flat, flat.mT) if triangular else flat.square().sum(dim=-1)
        target_energy = update.numel() / (factor.shape[0] * factor.shape[-1])
        if triangular:
            factor, lower_bound = _update_factor(
                factor,
                covariance,
                target_energy,
                lower_bound,
                tempo,
                power_iterations,
            )
        else:
            factor, lower_bound = _update_diagonal_factor(
                factor,
                covariance,
                target_energy,
                lower_bound,
                tempo,
            )
        next_factors.append(factor)
        next_lower_bounds.append(lower_bound)
    return next_factors, next_lower_bounds


def make_psgd_nfactor(power_iterations: int = 2):
    if type(power_iterations) is not int:
        raise TypeError("power_iterations must be a Python int")

    def psgd_nfactor_transform(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        del obs, param
        update = _wide(update)
        factor_count = len(state) // 2
        factors = [_wide(state[f"Q_{index}"]) for index in range(factor_count)]
        lower_bounds = [
            _wide(state[f"running_lower_bound_{index}"]) for index in range(factor_count)
        ]
        if tempo.refresh:
            factors, lower_bounds = _refresh_nfactor(
                update, factors, lower_bounds, tempo, power_iterations
            )
        preconditioned = _precondition_nfactor(update, factors)
        next_state = {}
        for index, (factor, lower_bound) in enumerate(zip(factors, lower_bounds, strict=True)):
            next_state[f"Q_{index}"] = factor
            next_state[f"running_lower_bound_{index}"] = lower_bound
        return preconditioned, next_state, tempo.live

    psgd_nfactor_transform.distributed_scope = WHOLE
    psgd_nfactor_transform.init = psgd_nfactor_init
    psgd_nfactor_transform.state_init_hyper = ("max_size_triangular",)
    psgd_nfactor_transform.config = {"power_iterations": power_iterations}
    return psgd_nfactor_transform


def make_psgd_pro(power_iterations: int = 2, *, sqrt: bool = False):
    """Build a PRO transform with trace-time iteration count and apply mode."""

    if type(power_iterations) is not int:
        raise TypeError("power_iterations must be a Python int")
    if type(sqrt) is not bool:
        raise TypeError("sqrt must be a Python bool")
    apply = _apply_once if sqrt else _precondition
    mixed_apply = _apply_once_mixed if sqrt else _precondition_mixed

    def psgd_pro_transform(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        del obs, param
        q0 = _wide(state["Q_0"])
        q1 = _wide(state["Q_1"])
        lower0 = _wide(state["running_lower_bound_0"])
        lower1 = _wide(state["running_lower_bound_1"])
        all_triangular = q0.ndim == 3 and q1.ndim == 3
        if tempo.refresh:
            if all_triangular:
                q0, q1, lower0, lower1 = _refresh_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations
                )
            else:
                q0, q1, lower0, lower1 = _refresh_mixed_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations
                )
        preconditioned = apply(update, q0, q1) if all_triangular else mixed_apply(update, q0, q1)
        return preconditioned, {
            "Q_0": q0,
            "Q_1": q1,
            "running_lower_bound_0": lower0,
            "running_lower_bound_1": lower1,
        }, tempo.live

    psgd_pro_transform.__name__ = "qsgd_transform" if sqrt else "psgd_pro_transform"
    if not sqrt:
        psgd_pro_transform.distributed_scope = WHOLE
    psgd_pro_transform = merge_matrix_transform(psgd_pro_transform)
    psgd_pro_transform.init = psgd_pro_init
    psgd_pro_transform.state_init_hyper = ("max_size_triangular",)
    psgd_pro_transform.config = {"power_iterations": power_iterations, "sqrt": sqrt}
    return psgd_pro_transform


psgd_pro_transform = make_psgd_pro()
qsgd_transform = make_psgd_pro(sqrt=True)
psgd_nfactor_transform = make_psgd_nfactor()


psgd_pro = Recipe(
    chain=(psgd_pro_transform,),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=1e-9,
        max_size_triangular=2048,
        weight_decay=0.0,
    ),
)

qsgd = Recipe(
    chain=(qsgd_transform,),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=1e-9,
        max_size_triangular=2048,
        weight_decay=0.0,
    ),
)


psgd_nfactor = Recipe(
    chain=(psgd_nfactor_transform,),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=1e-9,
        max_size_triangular=2048,
        weight_decay=0.0,
    ),
)


__all__ = [
    "make_psgd_nfactor",
    "make_psgd_pro",
    "psgd_nfactor",
    "psgd_nfactor_init",
    "psgd_nfactor_transform",
    "psgd_pro",
    "psgd_pro_init",
    "psgd_pro_transform",
    "qsgd",
    "qsgd_transform",
]
