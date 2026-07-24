"""KL-divergence matrix preconditioners for the slab-native optimizer.

The KL variants share SOAP's QR-maintained bases but estimate each Kronecker
factor after whitening the other factor by its current eigenvalue EMA. The
historical ``eigenvalues_*`` slots store the square roots of those EMAs.
"""

import torch
from torch import Tensor

from .core import Recipe
from .matrix import (
    _merged_matrix_dimensions,
    _project,
    _scaled_outer,
    _transport_exp_avg,
    _transport_exp_avg_sq,
    _update_gram,
    soap_init,
)
from .numerics import _wide, broadcast_leaf
from .transforms import Tempo, _second_moment, _second_moment_denom, sgd_commit

_DEFAULT_INIT_FACTOR = 0.1


def _initial_factor(ref: Tensor, init_factor: Tensor | float) -> Tensor:
    """Match legacy KL's finite, strictly positive eigenvalue initializer."""

    factor = torch.as_tensor(init_factor, dtype=ref.dtype, device=ref.device)
    if factor.numel() != 1 or not bool(torch.isfinite(factor) & (factor > 0)):
        raise ValueError("init_factor must be finite and positive")
    return factor.reshape(())


def _kl_state(
    ref_leaf: Tensor, *, init_factor: Tensor | float = _DEFAULT_INIT_FACTOR, max_precond_dim: Tensor
) -> dict[str, Tensor]:
    """Extend SOAP's matrix state with one KL eigenvalue-RMS per factor."""

    state = soap_init(ref_leaf, max_precond_dim=max_precond_dim)
    ref = _wide(ref_leaf)
    rows, columns = _merged_matrix_dimensions(tuple(ref_leaf.shape), int(max_precond_dim))
    factor = _initial_factor(ref, init_factor)
    if "GG_l" in state:
        state["eigenvalues_l"] = torch.ones(
            (rows,), dtype=ref.dtype, device=ref.device
        ) * factor.sqrt()
    if "GG_r" in state:
        state["eigenvalues_r"] = torch.ones(
            (columns,), dtype=ref.dtype, device=ref.device
        ) * factor.sqrt()
    return state


def kl_soap_init(
    ref_leaf: Tensor, *, init_factor: Tensor | float = _DEFAULT_INIT_FACTOR, max_precond_dim: Tensor
) -> dict[str, Tensor]:
    """Allocate KLSOAP's projected Adam and KL-factor state."""

    return _kl_state(ref_leaf, init_factor=init_factor, max_precond_dim=max_precond_dim)


def kl_shampoo_init(
    ref_leaf: Tensor, *, init_factor: Tensor | float = _DEFAULT_INIT_FACTOR, max_precond_dim: Tensor
) -> dict[str, Tensor]:
    """Allocate KLShampoo's parameter-space momentum and KL-factor state."""

    state = _kl_state(ref_leaf, init_factor=init_factor, max_precond_dim=max_precond_dim)
    del state["exp_avg_sq"]
    return state


def _factor_inverse(eigenvalues: Tensor, eps: Tensor) -> Tensor:
    """Apply an inverse standard deviation from the stored RMS factor state."""

    threshold = torch.as_tensor(
        eps, dtype=eigenvalues.dtype, device=eigenvalues.device
    ).sqrt()
    return eigenvalues.clamp_min(threshold).reciprocal()


def _heavy_factor_inverse(eigenvalues: Tensor, eps: Tensor) -> Tensor:
    """Apply the Moore-Penrose KL eigenvalue inverse."""

    threshold = torch.as_tensor(eps, dtype=eigenvalues.dtype, device=eigenvalues.device).sqrt()
    return torch.where(
        eigenvalues > threshold, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues)
    )


def _factor_rms(value: Tensor, dim: int) -> Tensor:
    """Compute a per-factor RMS without an overflow-prone square reduction."""

    scale = value.abs().amax(dim=dim, keepdim=True)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    rms = (value / safe_scale).square().mean(dim=dim).sqrt()
    return rms * scale.squeeze(dim)


def _kl_qr_basis(gram: Tensor, basis: Tensor) -> Tensor:
    """Run the authors' unsorted KL power iteration and QR refresh."""

    work = gram @ basis.to(gram.dtype)
    return torch.linalg.qr(work).Q.to(basis.dtype)


def _heavy_kl_qr_basis(gram: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalue-sorted KL power iteration and QR refresh; returns (sorted basis, sort order).

    Sorting the columns by Rayleigh quotient before QR orthogonalizes the dominant directions
    first (the Heavy refinement over the authors' unsorted qr). The basis stays in sorted order so
    the caller reorders the stored eigenvalues to match.
    """

    basis_work = basis.to(gram.dtype)
    work = gram @ basis_work
    eigenvalues = torch.einsum("nij,nij->nj", basis_work, work)
    order = torch.argsort(eigenvalues, dim=-1, descending=True)
    orthogonal, _ = torch.linalg.qr(work.gather(-1, order.unsqueeze(-2).expand_as(work)))
    return orthogonal.to(basis.dtype), order


def _apply_kl_preconditioner(
    update: Tensor,
    state: dict[str, Tensor],
    tempo: Tempo,
    beta: Tensor,
    exp_avg: Tensor | None = None,
) -> dict[str, Tensor]:
    """Port KL factor/eigenvalue updates and the legacy QR refresh order."""

    gg_left = _wide(state["GG_l"]) if "GG_l" in state else None
    gg_right = _wide(state["GG_r"]) if "GG_r" in state else None
    gg_scale_left = _wide(state["GG_l_scale"]) if gg_left is not None else None
    gg_scale_right = _wide(state["GG_r_scale"]) if gg_right is not None else None
    left = _wide(state["Q_l"]) if gg_left is not None else None
    right = _wide(state["Q_r"]) if gg_right is not None else None
    eigenvalues_left = _wide(state["eigenvalues_l"]) if gg_left is not None else None
    eigenvalues_right = _wide(state["eigenvalues_r"]) if gg_right is not None else None
    inverse_left = (
        _factor_inverse(eigenvalues_left, tempo.hyper.eps).to(update.dtype)
        if eigenvalues_left is not None
        else None
    )
    inverse_right = (
        _factor_inverse(eigenvalues_right, tempo.hyper.eps).to(update.dtype)
        if eigenvalues_right is not None
        else None
    )
    projected = _project(update, left, right, back=False)
    left_work = None
    if gg_left is not None:
        left_work = (
            update
            if right is None
            else (update @ right) * inverse_right.unsqueeze(-2)
        )
    right_work = None
    if gg_right is not None:
        right_work = (
            update
            if left is None
            else (left.mT @ update) * inverse_left.unsqueeze(-1)
        )
    rows, columns = update.shape[-2:]
    if left_work is not None:
        left_outer, left_outer_scale = _scaled_outer(left_work, left_work.mT)
        left_outer = left_outer / columns
    else:
        left_outer = left_outer_scale = None
    if right_work is not None:
        right_outer, right_outer_scale = _scaled_outer(right_work.mT, right_work)
        right_outer = right_outer / rows
    else:
        right_outer = right_outer_scale = None
    left_estimate = (
        _factor_rms(
            projected if inverse_right is None else projected * inverse_right.unsqueeze(-2),
            -1,
        )
        if gg_left is not None
        else None
    )
    right_estimate = (
        _factor_rms(
            projected if inverse_left is None else projected * inverse_left.unsqueeze(-1),
            -2,
        )
        if gg_right is not None
        else None
    )

    leaf_beta = beta.expand_as(tempo.age)
    if gg_left is not None:
        next_gg_left, next_scale_left = _update_gram(
            gg_left, gg_scale_left, left_outer, left_outer_scale, leaf_beta
        )
    else:
        next_gg_left = next_scale_left = None
    if gg_right is not None:
        next_gg_right, next_scale_right = _update_gram(
            gg_right, gg_scale_right, right_outer, right_outer_scale, leaf_beta
        )
    else:
        next_gg_right = next_scale_right = None
    left_beta = broadcast_leaf(leaf_beta, eigenvalues_left) if eigenvalues_left is not None else None
    right_beta = broadcast_leaf(leaf_beta, eigenvalues_right) if eigenvalues_right is not None else None
    next_eigenvalues_left = (
        _second_moment(eigenvalues_left, left_estimate, left_beta)
        if eigenvalues_left is not None
        else None
    )
    next_eigenvalues_right = (
        _second_moment(eigenvalues_right, right_estimate, right_beta)
        if eigenvalues_right is not None
        else None
    )

    next_left, next_right = left, right
    next_exp_avg = exp_avg
    if tempo.refresh:
        if next_gg_left is not None:
            next_left = _kl_qr_basis(next_gg_left, left)
        if next_gg_right is not None:
            next_right = _kl_qr_basis(next_gg_right, right)
        if next_exp_avg is not None:
            next_exp_avg = _transport_exp_avg(
                next_exp_avg,
                left,
                right,
                next_left,
                next_right,
            )

    next_state = {"GG_l": next_gg_left} if next_gg_left is not None else {}
    if next_gg_right is not None:
        next_state["GG_r"] = next_gg_right
    if next_gg_left is not None:
        next_state["GG_l_scale"] = next_scale_left
    if next_gg_right is not None:
        next_state["GG_r_scale"] = next_scale_right
    if next_gg_left is not None:
        next_state["Q_l"] = next_left
    if next_gg_right is not None:
        next_state["Q_r"] = next_right
    if next_gg_left is not None:
        next_state["eigenvalues_l"] = next_eigenvalues_left
    if next_gg_right is not None:
        next_state["eigenvalues_r"] = next_eigenvalues_right
    if next_exp_avg is not None:
        next_state["exp_avg"] = next_exp_avg
    return next_state


def _apply_heavy_kl_preconditioner(
    update: Tensor,
    state: dict[str, Tensor],
    tempo: Tempo,
    beta: Tensor,
    exp_avg: Tensor | None = None,
) -> dict[str, Tensor]:
    """Update KL factors with pseudoinverses and eigenvalue-sorted QR refreshes."""

    gg_left = _wide(state["GG_l"]) if "GG_l" in state else None
    gg_right = _wide(state["GG_r"]) if "GG_r" in state else None
    gg_scale_left = _wide(state["GG_l_scale"]) if gg_left is not None else None
    gg_scale_right = _wide(state["GG_r_scale"]) if gg_right is not None else None
    left = _wide(state["Q_l"]) if gg_left is not None else None
    right = _wide(state["Q_r"]) if gg_right is not None else None
    eigenvalues_left = _wide(state["eigenvalues_l"]) if gg_left is not None else None
    eigenvalues_right = _wide(state["eigenvalues_r"]) if gg_right is not None else None
    inverse_left = (
        _heavy_factor_inverse(eigenvalues_left, tempo.hyper.eps).to(update.dtype)
        if eigenvalues_left is not None
        else None
    )
    inverse_right = (
        _heavy_factor_inverse(eigenvalues_right, tempo.hyper.eps).to(update.dtype)
        if eigenvalues_right is not None
        else None
    )
    projected = _project(update, left, right, back=False)
    left_work = None
    if gg_left is not None:
        left_work = update if right is None else (update @ right) * inverse_right.unsqueeze(-2)
    right_work = None
    if gg_right is not None:
        right_work = update if left is None else (left.mT @ update) * inverse_left.unsqueeze(-1)
    rows, columns = update.shape[-2:]
    if left_work is not None:
        left_outer, left_outer_scale = _scaled_outer(left_work, left_work.mT)
        left_outer = left_outer / columns
    else:
        left_outer = left_outer_scale = None
    if right_work is not None:
        right_outer, right_outer_scale = _scaled_outer(right_work.mT, right_work)
        right_outer = right_outer / rows
    else:
        right_outer = right_outer_scale = None
    left_estimate = (
        _factor_rms(
            projected if inverse_right is None else projected * inverse_right.unsqueeze(-2),
            -1,
        )
        if gg_left is not None
        else None
    )
    right_estimate = (
        _factor_rms(
            projected if inverse_left is None else projected * inverse_left.unsqueeze(-1),
            -2,
        )
        if gg_right is not None
        else None
    )

    leaf_beta = beta.expand_as(tempo.age)
    if gg_left is not None:
        next_gg_left, next_scale_left = _update_gram(
            gg_left, gg_scale_left, left_outer, left_outer_scale, leaf_beta
        )
    else:
        next_gg_left = next_scale_left = None
    if gg_right is not None:
        next_gg_right, next_scale_right = _update_gram(
            gg_right, gg_scale_right, right_outer, right_outer_scale, leaf_beta
        )
    else:
        next_gg_right = next_scale_right = None
    left_beta = broadcast_leaf(leaf_beta, eigenvalues_left) if eigenvalues_left is not None else None
    right_beta = broadcast_leaf(leaf_beta, eigenvalues_right) if eigenvalues_right is not None else None
    next_eigenvalues_left = (
        _second_moment(eigenvalues_left, left_estimate, left_beta)
        if eigenvalues_left is not None
        else None
    )
    next_eigenvalues_right = (
        _second_moment(eigenvalues_right, right_estimate, right_beta)
        if eigenvalues_right is not None
        else None
    )

    next_left, next_right = left, right
    next_exp_avg = exp_avg
    if tempo.refresh:
        if next_gg_left is not None:
            next_left, left_order = _heavy_kl_qr_basis(next_gg_left, left)
            next_eigenvalues_left = next_eigenvalues_left.gather(-1, left_order)
        if next_gg_right is not None:
            next_right, right_order = _heavy_kl_qr_basis(next_gg_right, right)
            next_eigenvalues_right = next_eigenvalues_right.gather(-1, right_order)
        if next_exp_avg is not None:
            next_exp_avg = _transport_exp_avg(
                next_exp_avg,
                left,
                right,
                next_left,
                next_right,
            )

    next_state = {"GG_l": next_gg_left} if next_gg_left is not None else {}
    if next_gg_right is not None:
        next_state["GG_r"] = next_gg_right
    if next_gg_left is not None:
        next_state["GG_l_scale"] = next_scale_left
    if next_gg_right is not None:
        next_state["GG_r_scale"] = next_scale_right
    if next_gg_left is not None:
        next_state["Q_l"] = next_left
    if next_gg_right is not None:
        next_state["Q_r"] = next_right
    if next_gg_left is not None:
        next_state["eigenvalues_l"] = next_eigenvalues_left
    if next_gg_right is not None:
        next_state["eigenvalues_r"] = next_eigenvalues_right
    if next_exp_avg is not None:
        next_state["exp_avg"] = next_exp_avg
    return next_state


def kl_soap(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Run projected Adam after the KL factor update and optional QR refresh."""

    del obs, param
    original_shape = update.shape[1:]
    rows, columns = state["exp_avg"].shape[-2:]
    update = _wide(update.reshape(update.shape[0], rows, columns))
    next_state = _apply_kl_preconditioner(
        update,
        state,
        tempo,
        tempo.hyper.shampoo_beta,
        _wide(state["exp_avg"]),
    )
    left = next_state["Q_l"] if "GG_l" in next_state else None
    right = next_state["Q_r"] if "GG_r" in next_state else None
    projected = _project(update, left, right, back=False)
    beta1 = broadcast_leaf(tempo.hyper.beta1.expand_as(tempo.age), projected)
    beta2 = broadcast_leaf(tempo.hyper.beta2.expand_as(tempo.age), projected)
    exp_avg = next_state["exp_avg"] * beta1 + projected * (1 - beta1)
    prior_exp_avg_sq = _wide(state["exp_avg_sq"])
    if tempo.refresh:
        # The prior second moment lives in the old basis; rebase it before blending new-basis squares.
        old_left = _wide(state["Q_l"]) if "GG_l" in state else None
        old_right = _wide(state["Q_r"]) if "GG_r" in state else None
        prior_exp_avg_sq = _transport_exp_avg_sq(
            prior_exp_avg_sq,
            old_left,
            old_right,
            left,
            right,
        )
    exp_avg_sq = _second_moment(prior_exp_avg_sq, projected, beta2)
    started = tempo.age > 1
    exp_avg = torch.where(broadcast_leaf(started, exp_avg), exp_avg, _wide(state["exp_avg"]))
    exp_avg_sq = torch.where(broadcast_leaf(started, exp_avg_sq), exp_avg_sq, _wide(state["exp_avg_sq"]))
    preconditioned = _project(
        exp_avg / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, projected.dtype),
        left,
        right,
        back=True,
    )
    next_state.update(exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
    return preconditioned.reshape((preconditioned.shape[0], *original_shape)), next_state, tempo.live & started


kl_soap.init = kl_soap_init
kl_soap.state_init_hyper = ("init_factor", "max_precond_dim")
kl_soap.distributed_shard_separable = False


def heavy_kl_soap(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Run Heavy projected Adam after the KL factor update and sorted QR refresh."""

    del obs, param
    original_shape = update.shape[1:]
    rows, columns = state["exp_avg"].shape[-2:]
    update = _wide(update.reshape(update.shape[0], rows, columns))
    next_state = _apply_heavy_kl_preconditioner(
        update,
        state,
        tempo,
        tempo.hyper.shampoo_beta,
        _wide(state["exp_avg"]),
    )
    left = next_state["Q_l"] if "GG_l" in next_state else None
    right = next_state["Q_r"] if "GG_r" in next_state else None
    projected = _project(update, left, right, back=False)
    beta1 = broadcast_leaf(tempo.hyper.beta1.expand_as(tempo.age), projected)
    beta2 = broadcast_leaf(tempo.hyper.beta2.expand_as(tempo.age), projected)
    exp_avg = next_state["exp_avg"] * beta1 + projected * (1 - beta1)
    prior_exp_avg_sq = _wide(state["exp_avg_sq"])
    if tempo.refresh:
        old_left = _wide(state["Q_l"]) if "GG_l" in state else None
        old_right = _wide(state["Q_r"]) if "GG_r" in state else None
        prior_exp_avg_sq = _transport_exp_avg_sq(
            prior_exp_avg_sq,
            old_left,
            old_right,
            left,
            right,
        )
    exp_avg_sq = _second_moment(prior_exp_avg_sq, projected, beta2)
    started = tempo.age > 1
    exp_avg = torch.where(broadcast_leaf(started, exp_avg), exp_avg, _wide(state["exp_avg"]))
    exp_avg_sq = torch.where(broadcast_leaf(started, exp_avg_sq), exp_avg_sq, _wide(state["exp_avg_sq"]))
    preconditioned = _project(
        exp_avg / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, projected.dtype),
        left,
        right,
        back=True,
    )
    next_state.update(exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
    return preconditioned.reshape((preconditioned.shape[0], *original_shape)), next_state, tempo.live & started


heavy_kl_soap_init = kl_soap_init
heavy_kl_soap.init = kl_soap_init
heavy_kl_soap.state_init_hyper = ("init_factor", "max_precond_dim")
heavy_kl_soap.distributed_shard_separable = False


def kl_shampoo(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Apply KL-Shampoo's raw-gradient EMA in the updated KL eigensystem."""

    del obs, param
    original_shape = update.shape[1:]
    rows, columns = state["exp_avg"].shape[-2:]
    update = _wide(update.reshape(update.shape[0], rows, columns))
    prior_exp_avg = _wide(state["exp_avg"])
    exp_avg = prior_exp_avg * tempo.hyper.beta1 + update * (1 - tempo.hyper.beta1)
    started = tempo.age > 1
    exp_avg = torch.where(broadcast_leaf(started, exp_avg), exp_avg, prior_exp_avg)
    next_state = _apply_kl_preconditioner(update, state, tempo, tempo.hyper.beta2)
    left = next_state["Q_l"] if "GG_l" in next_state else None
    right = next_state["Q_r"] if "GG_r" in next_state else None
    projected = _project(exp_avg, left, right, back=False)
    inverse_left = (
        _factor_inverse(next_state["eigenvalues_l"], tempo.hyper.eps).to(projected.dtype)
        if "GG_l" in next_state
        else None
    )
    inverse_right = (
        _factor_inverse(next_state["eigenvalues_r"], tempo.hyper.eps).to(projected.dtype)
        if "GG_r" in next_state
        else None
    )
    preconditioned = projected
    if inverse_left is not None:
        preconditioned = preconditioned * inverse_left.unsqueeze(-1)
    if inverse_right is not None:
        preconditioned = preconditioned * inverse_right.unsqueeze(-2)
    preconditioned = _project(preconditioned, left, right, back=True)
    next_state["exp_avg"] = exp_avg
    return preconditioned.reshape((preconditioned.shape[0], *original_shape)), next_state, tempo.live & started


kl_shampoo.init = kl_shampoo_init
kl_shampoo.state_init_hyper = ("init_factor", "max_precond_dim")
kl_shampoo.distributed_shard_separable = False


def heavy_kl_shampoo(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Apply Heavy KL-Shampoo momentum in the updated KL eigensystem."""

    del obs, param
    original_shape = update.shape[1:]
    rows, columns = state["exp_avg"].shape[-2:]
    update = _wide(update.reshape(update.shape[0], rows, columns))
    prior_exp_avg = _wide(state["exp_avg"])
    exp_avg = prior_exp_avg * tempo.hyper.beta1 + update * (1 - tempo.hyper.beta1)
    started = tempo.age > 1
    exp_avg = torch.where(broadcast_leaf(started, exp_avg), exp_avg, prior_exp_avg)
    next_state = _apply_heavy_kl_preconditioner(update, state, tempo, tempo.hyper.beta2)
    left = next_state["Q_l"] if "GG_l" in next_state else None
    right = next_state["Q_r"] if "GG_r" in next_state else None
    projected = _project(exp_avg, left, right, back=False)
    inverse_left = (
        _heavy_factor_inverse(next_state["eigenvalues_l"], tempo.hyper.eps).to(projected.dtype)
        if "GG_l" in next_state
        else None
    )
    inverse_right = (
        _heavy_factor_inverse(next_state["eigenvalues_r"], tempo.hyper.eps).to(projected.dtype)
        if "GG_r" in next_state
        else None
    )
    preconditioned = projected
    if inverse_left is not None:
        preconditioned = preconditioned * inverse_left.unsqueeze(-1)
    if inverse_right is not None:
        preconditioned = preconditioned * inverse_right.unsqueeze(-2)
    preconditioned = _project(preconditioned, left, right, back=True)
    next_state["exp_avg"] = exp_avg
    return preconditioned.reshape((preconditioned.shape[0], *original_shape)), next_state, tempo.live & started


heavy_kl_shampoo_init = kl_shampoo_init
heavy_kl_shampoo.init = kl_shampoo_init
heavy_kl_shampoo.state_init_hyper = ("init_factor", "max_precond_dim")
heavy_kl_shampoo.distributed_shard_separable = False


kl_soap_recipe = Recipe(
    chain=(kl_soap,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.95,
        preconditioner_update_probability=0.5,
        eps=1e-8,
        weight_decay=0.01,
        max_precond_dim=2048,
        init_factor=_DEFAULT_INIT_FACTOR,
    ),
)

kl_shampoo_recipe = Recipe(
    chain=(kl_shampoo,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        preconditioner_update_probability=0.5,
        eps=1e-8,
        weight_decay=0.01,
        max_precond_dim=2048,
        init_factor=_DEFAULT_INIT_FACTOR,
    ),
)


heavy_kl_soap_recipe = Recipe(
    chain=(heavy_kl_soap,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.95,
        preconditioner_update_probability=0.5,
        eps=1e-8,
        weight_decay=0.01,
        max_precond_dim=2048,
        init_factor=_DEFAULT_INIT_FACTOR,
    ),
)

heavy_kl_shampoo_recipe = Recipe(
    chain=(heavy_kl_shampoo,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        preconditioner_update_probability=0.5,
        eps=1e-8,
        weight_decay=0.01,
        max_precond_dim=2048,
        init_factor=_DEFAULT_INIT_FACTOR,
    ),
)


__all__ = [
    "_apply_heavy_kl_preconditioner",
    "_heavy_factor_inverse",
    "_heavy_kl_qr_basis",
    "heavy_kl_shampoo",
    "heavy_kl_shampoo_init",
    "heavy_kl_shampoo_recipe",
    "heavy_kl_soap",
    "heavy_kl_soap_init",
    "heavy_kl_soap_recipe",
    "kl_shampoo",
    "kl_shampoo_init",
    "kl_shampoo_recipe",
    "kl_soap",
    "kl_soap_init",
    "kl_soap_recipe",
]
