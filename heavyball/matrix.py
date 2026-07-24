"""Batched matrix preconditioners for the slab-native optimizer.

This module keeps the SOAP state in the same coordinates as the legacy
implementation: ``exp_avg`` and the RMS stored in the historical
``exp_avg_sq`` slot live in the current Shampoo eigenbasis. Gram matrices use
``matrix / scale**2`` plus one scalar per factor, so finite fp32 gradients do
not require permanent fp64 matrices. Higher-rank leaves use legacy's exact
bind-time dimension merge when that produces a matrix; N-factor
preconditioning remains unsupported.
"""

import math
from functools import wraps

import torch
from torch import Tensor

from .core import Recipe, Transform
from .numerics import _wide, broadcast_leaf
from .transforms import (
    WHOLE,
    Tempo,
    _eigh_regularized_root,
    _eigh_scaled_gram_decomposition,
    _root_requires_spectral_application,
    adam,
    ademamix,
    beta_debias,
    laprop,
    nadam,
    sgd_commit,
)

_DEFAULT_MAX_PRECOND_DIM = 2048


def merged_matrix_shape(shape: tuple[int, ...], max_precond_dim: int) -> tuple[int, ...]:
    """Return legacy ``dim_merger(..., split=False)``'s shape without a slab axis."""

    if not shape:
        return shape

    new_shape = []
    cum_size = 1
    for size in shape[1:][::-1]:
        temp_size = cum_size * size
        if temp_size > max_precond_dim:
            if cum_size > 1:
                new_shape.append(cum_size)
                cum_size = size
            else:
                new_shape.append(size)
                cum_size = 1
        else:
            cum_size = temp_size
    if cum_size > 1:
        new_shape.append(cum_size)
    return (shape[0], *new_shape[::-1])


def matrix_route(info) -> bool:
    """Route only leaves whose default merge is a matrix with both dimensions > 1.

    A leading singleton (a 1-row leaf) merges to ``(1, N)`` whose 1x1 row factor is degenerate: the
    PSGD power iteration traces to a fake-tensor rank error under compile, and preconditioning a single
    row is pointless. Falling back to AdamW here is symmetric with ``(N, 1)``, whose trailing singleton
    ``merged_matrix_shape`` already drops.
    """

    merged = merged_matrix_shape(info.shape, _DEFAULT_MAX_PRECOND_DIM)
    return len(merged) == 2 and merged[0] > 1


def nfactor_route(info) -> bool:
    return info.ndim >= 3 and sum(size > 1 for size in info.shape) >= 3


def merge_matrix_transform(transform: Transform) -> Transform:
    """Run a matrix transform in merged coordinates and restore the leaf shape."""

    @wraps(transform)
    def merged(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        original_shape = update.shape[1:]
        trailing_size = math.prod(update.shape[2:])
        update = _wide(update.reshape(update.shape[0], update.shape[1], trailing_size))
        preconditioned, next_state, live = transform(update, obs, param, state, tempo)
        return preconditioned.reshape((preconditioned.shape[0], *original_shape)), next_state, live

    if hasattr(transform, "distributed_scope"):
        merged.distributed_scope = transform.distributed_scope
    else:
        merged.distributed_shard_separable = False
    return merged


def _merged_matrix_dimensions(shape: tuple[int, ...], max_precond_dim: int) -> tuple[int, int]:
    merged = merged_matrix_shape(shape, max_precond_dim)
    if len(merged) == 2:
        return merged
    if len(merged) > 2:
        raise ValueError(
            f"leaf merges to >2D at max_precond_dim={max_precond_dim}; N-factor preconditioning is a follow-up"
        )
    raise ValueError(f"leaf merges to {len(merged)}D at max_precond_dim={max_precond_dim}; matrix preconditioning requires 2D")


def _project(
    gradient: Tensor,
    left: Tensor | None,
    right: Tensor | None,
    *,
    back: bool,
) -> Tensor:
    """Project a batched matrix into, or out of, its two Shampoo bases."""

    if left is None and right is None:
        return gradient

    if left is None:
        return gradient @ (right.mT if back else right)
    if right is None:
        return (left if back else left.mT) @ gradient
    if back:
        return torch.einsum("nab,nia,njb->nij", gradient, left, right)
    return torch.einsum("nij,nia,njb->nab", gradient, left, right)


def _gram_value(gram: Tensor, scale: Tensor) -> Tensor:
    """Materialize a scaled Gram transiently in fp64 for a refresh."""

    scale = broadcast_leaf(scale.double(), gram)
    return gram.double() * scale.square()


def _combine_gram(
    gram: Tensor,
    scale: Tensor,
    outer: Tensor,
    outer_scale: Tensor,
    old_weight: Tensor,
    new_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Combine two ``matrix / scale**2`` representations without large squares."""

    next_scale = torch.maximum(scale, outer_scale)
    safe_scale = torch.where(next_scale != 0, next_scale, torch.ones_like(next_scale))
    old_ratio = broadcast_leaf((scale / safe_scale).to(gram.dtype), gram)
    new_ratio = broadcast_leaf((outer_scale / safe_scale).to(gram.dtype), gram)
    old_weight = broadcast_leaf(old_weight.to(gram.dtype), gram)
    new_weight = broadcast_leaf(new_weight.to(gram.dtype), gram)
    combined = (
        gram * old_ratio.square() * old_weight
        + outer.to(gram.dtype) * new_ratio.square() * new_weight
    )
    return combined, next_scale


def _update_gram(
    gram: Tensor,
    scale: Tensor,
    outer: Tensor,
    outer_scale: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor]:
    leaf_beta = beta.expand_as(scale)
    return _combine_gram(
        gram, scale, outer, outer_scale, leaf_beta, 1 - leaf_beta
    )


def _accumulate_gram(
    gram: Tensor, scale: Tensor, outer: Tensor, outer_scale: Tensor
) -> tuple[Tensor, Tensor]:
    ones = torch.ones_like(scale)
    return _combine_gram(gram, scale, outer, outer_scale, ones, ones)


def _outer(a: Tensor, b: Tensor) -> Tensor:
    """Batched outer product in fp64 for callers that need the physical value."""

    return a.double() @ b.double()


def _scaled_outer(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Form ``a @ b`` as ``outer / scale**2`` before a finite input can overflow.

    Ordinary inputs retain ``scale == 1`` and therefore the pre-hardening
    coordinates.  Only contractions whose conservative fp32 quadratic bound is
    unsafe are normalized.
    """

    dimensions = tuple(range(1, a.ndim))
    maximum = torch.maximum(
        a.abs().amax(dim=dimensions), b.abs().amax(dim=dimensions)
    )
    contraction = max(1, a.shape[-1])
    limit = math.sqrt(torch.finfo(a.dtype).max / contraction)
    needs_scale = maximum > limit
    scale = torch.where(needs_scale, maximum, torch.ones_like(maximum))
    scaled_a = a / broadcast_leaf(scale, a)
    scaled_b = b / broadcast_leaf(scale, b)
    direct = a @ b
    normalized = scaled_a @ scaled_b
    outer = torch.where(broadcast_leaf(needs_scale, direct), normalized, direct)
    return outer, scale


def _qr_basis(gram: Tensor, basis: Tensor) -> Tensor:
    """One legacy SOAP QR iteration, including its eigenvalue ordering rule."""

    basis_work = basis.to(gram.dtype)
    work = gram @ basis_work
    eigenvalues = torch.einsum("nij,nij->nj", basis_work, work)
    order = torch.argsort(eigenvalues, dim=-1, descending=True)
    index = order.unsqueeze(-2).expand_as(work)
    orthogonal, _ = torch.linalg.qr(work.gather(-1, index))
    return work.scatter(-1, index, orthogonal).to(basis.dtype)


def _transport_exp_avg(
    exp_avg: Tensor,
    old_left: Tensor | None,
    old_right: Tensor | None,
    new_left: Tensor | None,
    new_right: Tensor | None,
) -> Tensor:
    """Port legacy projected-state transport for the factors present in this leaf."""

    if old_left is None:
        if old_right is None:
            return exp_avg
        return torch.einsum("nab,nBb,nBd->nad", exp_avg, old_right, new_right)
    if old_right is None:
        return torch.einsum("nab,nAa,nAc->ncb", exp_avg, old_left, new_left)
    return torch.einsum(
        "nab,nAa,nBb,nAc,nBd->ncd",
        exp_avg,
        old_left,
        old_right,
        new_left,
        new_right,
    )


def _transport_exp_avg_sq(
    exp_avg_sq: Tensor,
    old_left: Tensor | None,
    old_right: Tensor | None,
    new_left: Tensor | None,
    new_right: Tensor | None,
) -> Tensor:
    """Re-express a stored RMS in the new basis as ``sqrt(rotate(rms**2))``."""

    dtype = exp_avg_sq.dtype
    variance = exp_avg_sq.double().square()

    if old_left is not None:
        old_left = old_left.double()
        new_left = new_left.double()
    if old_right is not None:
        old_right = old_right.double()
        new_right = new_right.double()
    if old_left is None:
        if old_right is None:
            return exp_avg_sq
        right = torch.einsum("nBb,nBd->nbd", old_right, new_right).square()
        transported = torch.einsum("nab,nbd->nad", variance, right)
        return transported.clamp_min(0).sqrt().to(dtype)
    if old_right is None:
        left = torch.einsum("nAa,nAc->nac", old_left, new_left).square()
        transported = torch.einsum("nab,nac->ncb", variance, left)
        return transported.clamp_min(0).sqrt().to(dtype)
    left = torch.einsum("nAa,nAc->nac", old_left, new_left).square()
    right = torch.einsum("nBb,nBd->nbd", old_right, new_right).square()
    transported = torch.einsum("nab,nac,nbd->ncd", variance, left, right)
    return transported.clamp_min(0).sqrt().to(dtype)


def _soap_basis_init(ref_leaf: Tensor, *, max_precond_dim: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    limit = int(max_precond_dim)
    rows, columns = _merged_matrix_dimensions(tuple(ref_leaf.shape), limit)
    ref = _wide(ref_leaf).reshape(rows, columns)
    state = {}
    if 1 < rows <= limit:
        left = torch.eye(rows, dtype=ref.dtype, device=ref.device)
        state["GG_l"] = torch.zeros_like(left)
        state["GG_l_scale"] = torch.ones((), dtype=ref.dtype, device=ref.device)
    if 1 < columns <= limit:
        right = torch.eye(columns, dtype=ref.dtype, device=ref.device)
        state["GG_r"] = torch.zeros_like(right)
        state["GG_r_scale"] = torch.ones((), dtype=ref.dtype, device=ref.device)
    if 1 < rows <= limit:
        state["Q_l"] = left
    if 1 < columns <= limit:
        state["Q_r"] = right
    return state, ref


def soap_init(ref_leaf: Tensor, *, max_precond_dim: Tensor) -> dict[str, Tensor]:
    """Allocate SOAP state in the legacy merged matrix coordinates.

    The engine adds the leading slab axis to every returned tensor.  The
    factors therefore become ``[N, m, m]`` and ``[N, n, n]`` in a group.
    """

    state, ref = _soap_basis_init(ref_leaf, max_precond_dim=max_precond_dim)
    state.update(adam.init(ref))
    return state


def _soap_factory(inner: Transform, transported=("exp_avg",), squared=("exp_avg_sq",)) -> Transform:
    def transform(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        old_left = _wide(state["Q_l"]) if "GG_l" in state else None
        old_right = _wide(state["Q_r"]) if "GG_r" in state else None
        projected = _project(update, old_left, old_right, back=False)
        # Inner bias-corrects by the per-leaf update count (age), matching SOAP (arXiv:2409.11321) and the
        # official soap.py, and consistent with the Gram/transport side below. (Legacy SOAP keyed the inner
        # on the global step; identical when always observed, wrong for leaves skipped via observed=.)
        preconditioned, inner_state, _ = inner(projected, obs, param, state, tempo)
        preconditioned = _project(preconditioned, old_left, old_right, back=True)

        shampoo_beta = beta_debias(tempo.hyper.shampoo_beta, tempo.age)
        old_gg_left = _wide(state["GG_l"]) if "GG_l" in state else None
        old_gg_right = _wide(state["GG_r"]) if "GG_r" in state else None
        old_scale_left = _wide(state["GG_l_scale"]) if old_gg_left is not None else None
        old_scale_right = _wide(state["GG_r_scale"]) if old_gg_right is not None else None
        if old_gg_left is not None:
            left_outer, left_outer_scale = _scaled_outer(update, update.mT)
            next_gg_left, next_scale_left = _update_gram(
                old_gg_left,
                old_scale_left,
                left_outer,
                left_outer_scale,
                shampoo_beta,
            )
        else:
            next_gg_left = next_scale_left = None
        if old_gg_right is not None:
            right_outer, right_outer_scale = _scaled_outer(update.mT, update)
            next_gg_right, next_scale_right = _update_gram(
                old_gg_right,
                old_scale_right,
                right_outer,
                right_outer_scale,
                shampoo_beta,
            )
        else:
            next_gg_right = next_scale_right = None

        next_left, next_right = old_left, old_right
        if tempo.refresh:
            if next_gg_left is not None:
                next_left = _qr_basis(next_gg_left, old_left)
            if next_gg_right is not None:
                next_right = _qr_basis(next_gg_right, old_right)
            # The set is configurable because multi-moment inners like AdEMAMix must rotate every first moment with the basis.
            for name in transported:
                inner_state[name] = _transport_exp_avg(
                    inner_state[name], old_left, old_right, next_left, next_right
                )
            # RMS slots represent diagonal variances; without this transport they stay in the stale basis.
            for name in squared:
                inner_state[name] = _transport_exp_avg_sq(
                    inner_state[name], old_left, old_right, next_left, next_right
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
        next_state.update(inner_state)
        return preconditioned, next_state, tempo.live

    transform.distributed_scope = WHOLE
    transform = merge_matrix_transform(transform)

    def init(ref_leaf: Tensor, *, max_precond_dim: Tensor) -> dict[str, Tensor]:
        state, ref = _soap_basis_init(ref_leaf, max_precond_dim=max_precond_dim)
        state.update(inner.init(ref))
        return state

    transform.init = init
    transform.state_init_hyper = ("max_precond_dim",)
    return transform


soap = _soap_factory(adam)
soap.__name__ = "soap"
solp = _soap_factory(laprop)
solp.__name__ = "solp"
soap_nadam = _soap_factory(nadam)
soap_nadam.__name__ = "soap_nadam"
soap_ademamix = _soap_factory(ademamix, transported=("exp_avg_fast", "exp_avg_slow"))
soap_ademamix.__name__ = "soap_ademamix"


def _inverse_fourth_root(gram: Tensor, eps: Tensor) -> Tensor:
    return _eigh_regularized_root(gram, eps, -0.25)


def shampoo_init(ref_leaf: Tensor, *, max_precond_dim: Tensor) -> dict[str, Tensor]:
    """Allocate a two-sided, inverse-fourth-root Shampoo preconditioner."""

    limit = int(max_precond_dim)
    rows, columns = _merged_matrix_dimensions(tuple(ref_leaf.shape), limit)
    ref = _wide(ref_leaf).reshape(rows, columns)
    state = {}
    if 1 < rows <= limit:
        left = torch.eye(rows, dtype=ref.dtype, device=ref.device)
        state["GG_l"] = torch.zeros_like(left)
        state["GG_l_scale"] = torch.ones((), dtype=ref.dtype, device=ref.device)
    if 1 < columns <= limit:
        right = torch.eye(columns, dtype=ref.dtype, device=ref.device)
        state["GG_r"] = torch.zeros_like(right)
        state["GG_r_scale"] = torch.ones((), dtype=ref.dtype, device=ref.device)
    if 1 < rows <= limit:
        state["L"] = left
    if 1 < columns <= limit:
        state["R"] = right
    return state


def shampoo(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Apply real two-sided Shampoo factors rebuilt on host-selected refreshes.

    The sign of each Gram scale is representation metadata: positive means the
    same-size factor slot stores a materialized root, negative means it stores
    the spectral factor whose ``B @ B.mT`` application preserves null modes.
    """

    del obs, param
    if not state:
        return _wide(update), {}, tempo.live
    old_gg_left = _wide(state["GG_l"]) if "GG_l" in state else None
    old_gg_right = _wide(state["GG_r"]) if "GG_r" in state else None
    stored_scale_left = _wide(state["GG_l_scale"]) if old_gg_left is not None else None
    stored_scale_right = _wide(state["GG_r_scale"]) if old_gg_right is not None else None
    left_spectral = stored_scale_left < 0 if stored_scale_left is not None else None
    right_spectral = stored_scale_right < 0 if stored_scale_right is not None else None
    old_scale_left = stored_scale_left.abs() if stored_scale_left is not None else None
    old_scale_right = stored_scale_right.abs() if stored_scale_right is not None else None
    if old_gg_left is not None:
        left_outer, left_outer_scale = _scaled_outer(update, update.mT)
        next_gg_left, next_scale_left = _accumulate_gram(
            old_gg_left, old_scale_left, left_outer, left_outer_scale
        )
    else:
        next_gg_left = next_scale_left = None
    if old_gg_right is not None:
        right_outer, right_outer_scale = _scaled_outer(update.mT, update)
        next_gg_right, next_scale_right = _accumulate_gram(
            old_gg_right, old_scale_right, right_outer, right_outer_scale
        )
    else:
        next_gg_right = next_scale_right = None

    left = _wide(state["L"]) if "GG_l" in state else None
    right = _wide(state["R"]) if "GG_r" in state else None
    if tempo.refresh:
        if next_gg_left is not None:
            left_vectors, left_roots = _eigh_scaled_gram_decomposition(
                next_gg_left,
                next_scale_left,
                tempo.hyper.eps,
                -0.25,
            )
            left_spectral = _root_requires_spectral_application(left_roots, left.dtype)
            left_direct = (
                (left_vectors * left_roots.unsqueeze(-2)) @ left_vectors.mT
            ).to(left.dtype)
            left_factored = (
                left_vectors * left_roots.sqrt().unsqueeze(-2)
            ).to(left.dtype)
            left = torch.where(
                broadcast_leaf(left_spectral, left_direct),
                left_factored,
                left_direct,
            )
        if next_gg_right is not None:
            right_vectors, right_roots = _eigh_scaled_gram_decomposition(
                next_gg_right,
                next_scale_right,
                tempo.hyper.eps,
                -0.25,
            )
            right_spectral = _root_requires_spectral_application(right_roots, right.dtype)
            right_direct = (
                (right_vectors * right_roots.unsqueeze(-2)) @ right_vectors.mT
            ).to(right.dtype)
            right_factored = (
                right_vectors * right_roots.sqrt().unsqueeze(-2)
            ).to(right.dtype)
            right = torch.where(
                broadcast_leaf(right_spectral, right_direct),
                right_factored,
                right_direct,
            )
    preconditioned = update
    if left is not None:
        direct = left @ preconditioned
        factored = left @ (left.mT @ preconditioned)
        preconditioned = torch.where(
            broadcast_leaf(left_spectral, direct), factored, direct
        )
    if right is not None:
        direct = preconditioned @ right
        factored = (preconditioned @ right) @ right.mT
        preconditioned = torch.where(
            broadcast_leaf(right_spectral, direct), factored, direct
        )
    next_state = {"GG_l": next_gg_left} if next_gg_left is not None else {}
    if next_gg_right is not None:
        next_state["GG_r"] = next_gg_right
    if next_gg_left is not None:
        next_state["GG_l_scale"] = torch.where(
            left_spectral, -next_scale_left, next_scale_left
        )
        next_state["L"] = left
    if next_gg_right is not None:
        next_state["GG_r_scale"] = torch.where(
            right_spectral, -next_scale_right, next_scale_right
        )
        next_state["R"] = right
    return preconditioned, next_state, tempo.live


shampoo.distributed_scope = WHOLE
shampoo = merge_matrix_transform(shampoo)
shampoo.init = shampoo_init
shampoo.state_init_hyper = ("max_precond_dim",)


soap_recipe = Recipe(
    chain=(soap,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.95,
        preconditioner_update_probability=0.5,
        eps=1e-8,
        weight_decay=0.01,
        max_precond_dim=_DEFAULT_MAX_PRECOND_DIM,
    ),
)

solp_recipe = Recipe(
    chain=(solp,),
    commit=sgd_commit,
    defaults=soap_recipe.defaults,
)

soap_nadam_recipe = Recipe(
    chain=(soap_nadam,),
    commit=sgd_commit,
    defaults={**soap_recipe.defaults, "beta2": 0.999, "momentum_decay": 4e-3},
)

soap_ademamix_recipe = Recipe(
    chain=(soap_ademamix,),
    commit=sgd_commit,
    defaults={
        **soap_recipe.defaults,
        "beta3": 0.999,
        "alpha": 2.0,
        "beta3_warmup": 0.0,
        "alpha_warmup": 0.0,
    },
)

shampoo_recipe = Recipe(
    chain=(shampoo,),
    commit=sgd_commit,
    defaults=dict(
        lr=3e-3,
        eps=1e-8,
        preconditioner_update_probability=0.5,
        weight_decay=0.01,
        max_precond_dim=_DEFAULT_MAX_PRECOND_DIM,
    ),
)

# Short recipe aliases keep transform names available for composition.
soapw = soap_recipe
solpw = solp_recipe
soap_nadamw = soap_nadam_recipe
soap_ademamixw = soap_ademamix_recipe
shampoow = shampoo_recipe


__all__ = [
    "matrix_route",
    "merge_matrix_transform",
    "merged_matrix_shape",
    "shampoo",
    "shampoo_init",
    "shampoo_recipe",
    "shampoow",
    "soap",
    "soap_ademamix",
    "soap_ademamix_recipe",
    "soap_ademamixw",
    "soap_init",
    "soap_nadam",
    "soap_nadam_recipe",
    "soap_nadamw",
    "soap_recipe",
    "soapw",
    "solp",
    "solp_recipe",
    "solpw",
]
