"""LATHER: Adam in PSGD-Kron's transported Q eigenbasis.

LATHER keeps Adam's moments in the orthogonal bases induced by PSGD-Kron's
triangular factors.  Oversized axes use diagonal factors without a rotation.
A host-selected refresh first fits those factors with the same stochastic
whitening update as :mod:`heavyball.kron`, then rotates the moment statistics
into the refreshed coordinates for the triangular axes.
"""

import torch
from torch import Tensor

from .core import Recipe
from .kron import _refresh_mixed_q, _refresh_q
from .matrix import (
    _DEFAULT_MAX_PRECOND_DIM,
    _project,
    _transport_exp_avg,
    _transport_exp_avg_sq,
    merge_matrix_transform,
    merged_matrix_shape,
)
from .numerics import _wide, broadcast_leaf
from .transforms import WHOLE, Tempo, beta_debias, sgd_commit


def _next_psgd_eigenbasis(q: Tensor, old_basis: Tensor | None) -> Tensor | None:
    """Port legacy ``get_psgd_eigenbasis`` for one PSGD-Kron factor."""

    q = _wide(q)
    if q.ndim < 3:
        return None
    if old_basis is None:
        raise ValueError(
            "_next_psgd_eigenbasis requires a previous basis for matrix blocks; "
            "use _initial_psgd_eigenbasis"
        )
    old_basis = _wide(old_basis)
    work = q.mT @ (q @ old_basis)
    basis = torch.linalg.qr(work, mode="reduced").Q
    projected = q @ _wide(basis)
    order = torch.argsort(torch.einsum("nij,nij->nj", projected, projected), dim=-1, descending=True)
    basis = basis.gather(-1, order.unsqueeze(-2).expand_as(basis))
    signs = torch.einsum("nij,nij->nj", old_basis, _wide(basis))
    signs = torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs)).to(dtype=basis.dtype)
    return basis * signs.unsqueeze(-2)


def _initial_psgd_eigenbasis(q: Tensor) -> Tensor:
    """Port legacy ``init_psgd_eigenbasis`` for one full matrix factor."""

    gram = _wide(q).detach().mT @ _wide(q).detach()
    scale = gram.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(gram.real.dtype).tiny)
    gram = gram / scale
    gram = (gram + gram.mH) * 0.5
    gram.diagonal(dim1=-2, dim2=-1).add_(torch.finfo(gram.real.dtype).eps)
    return torch.flip(torch.linalg.eigh(gram)[1], [-1]).contiguous()


def lather_init(ref_leaf: Tensor, *, max_size_triangular: Tensor | int = 2048) -> dict[str, Tensor]:
    """Allocate mixed Q factors and Adam state in the triangular factors' bases."""

    limit = int(max_size_triangular)
    merged = merged_matrix_shape(tuple(ref_leaf.shape), _DEFAULT_MAX_PRECOND_DIM)
    if len(merged) != 2:
        raise ValueError(
            f"lather requires a leaf whose dimensions merge to 2D at max_precond_dim={_DEFAULT_MAX_PRECOND_DIM}"
        )
    rows, columns = merged
    if rows == 0 or columns == 0:
        raise ValueError("lather requires nonempty merged 2D parameter leaves")
    ref = _wide(ref_leaf).reshape(rows, columns)

    def factor(size: int) -> Tensor:
        if size > limit:
            return torch.ones(size, dtype=ref.dtype, device=ref.device)
        return torch.eye(size, dtype=ref.dtype, device=ref.device)

    q0 = factor(rows)
    q1 = factor(columns)
    state = {
        "exp_avg": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
        "Q_0": q0,
        "Q_1": q1,
    }
    if q0.ndim == 2:
        state["Q_basis_0"] = _initial_psgd_eigenbasis(q0)
    if q1.ndim == 2:
        state["Q_basis_1"] = _initial_psgd_eigenbasis(q1)
    state["running_lower_bound_0"] = torch.zeros((), dtype=torch.float64, device=ref.device)
    state["running_lower_bound_1"] = torch.zeros((), dtype=torch.float64, device=ref.device)
    return state


def make_lather(power_iterations: int = 2):
    """Build LATHER with a trace-time PSGD spectral power-iteration count."""

    if type(power_iterations) is not int:
        raise TypeError("power_iterations must be a Python int")

    def lather_transform(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        """Run Adam in the current Q eigenbasis and refresh it only when selected."""

        del obs, param
        update = _wide(update)
        old_left = state.get("Q_basis_0")
        old_right = state.get("Q_basis_1")
        old_left = None if old_left is None else _wide(old_left)
        old_right = None if old_right is None else _wide(old_right)
        exp_avg = _wide(state["exp_avg"])
        exp_avg_sq = _wide(state["exp_avg_sq"])

        projected = _project(update, old_left, old_right, back=False)
        beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), projected)
        beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), projected)
        next_exp_avg = exp_avg * beta1 + projected * (1 - beta1)
        next_exp_avg_sq = exp_avg_sq * beta2 + projected.square() * (1 - beta2)
        preconditioned = _project(
            next_exp_avg / next_exp_avg_sq.sqrt().clamp_min(tempo.hyper.eps), old_left, old_right, back=True
        )

        q0 = _wide(state["Q_0"])
        q1 = _wide(state["Q_1"])
        lower0 = _wide(state["running_lower_bound_0"])
        lower1 = _wide(state["running_lower_bound_1"])
        all_triangular = q0.ndim == 3 and q1.ndim == 3
        next_left, next_right = old_left, old_right
        if tempo.refresh:
            vector = tempo.randn_like(update)
            if all_triangular:
                q0, q1, lower0, lower1 = _refresh_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations, vector
                )
            else:
                q0, q1, lower0, lower1 = _refresh_mixed_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations, vector
                )
            next_left = _next_psgd_eigenbasis(q0, old_left)
            next_right = _next_psgd_eigenbasis(q1, old_right)
            next_exp_avg = _transport_exp_avg(next_exp_avg, old_left, old_right, next_left, next_right)
            next_exp_avg_sq = _transport_exp_avg_sq(
                next_exp_avg_sq, old_left, old_right, next_left, next_right
            )

        next_state = {
            "exp_avg": next_exp_avg,
            "exp_avg_sq": next_exp_avg_sq,
            "Q_0": q0,
            "Q_1": q1,
            "running_lower_bound_0": lower0,
            "running_lower_bound_1": lower1,
        }
        if next_left is not None:
            next_state["Q_basis_0"] = next_left
        if next_right is not None:
            next_state["Q_basis_1"] = next_right
        return preconditioned, next_state, tempo.live

    lather_transform.distributed_scope = WHOLE
    lather_transform = merge_matrix_transform(lather_transform)
    lather_transform.init = lather_init
    lather_transform.state_init_hyper = ("max_size_triangular",)
    lather_transform.config = {"power_iterations": power_iterations}
    return lather_transform


lather_transform = make_lather()


lather = Recipe(
    chain=(lather_transform,),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=1e-9,
        max_size_triangular=2048,
        weight_decay=0.0,
    ),
)


__all__ = [
    "lather",
    "lather_init",
    "lather_transform",
    "make_lather",
]
