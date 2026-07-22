"""Batched diagonal-plus-low-rank PSGD for slab-native HeavyBall."""

import torch
from torch import Tensor

from .core import Recipe, RefreshCadence
from .numerics import _wide, broadcast_leaf
from .transforms import Tempo, sgd_commit


def _lra_low_rank_mm(A: Tensor, B: Tensor, x: Tensor) -> Tensor:
    """Apply ``I + A B^T`` independently to every slab leaf."""

    return x + (A @ (B.mT @ x.unsqueeze(-1))).squeeze(-1)


def _lra_precond(update_flat: Tensor, U: Tensor, V: Tensor, d: Tensor) -> Tensor:
    """Apply the fitted ``P = Q^T Q`` preconditioner."""

    update_flat = _wide(update_flat)
    U = _wide(U)
    V = _wide(V)
    d = _wide(d)
    q_update = _lra_low_rank_mm(U, V, d * update_flat)
    return d * _lra_low_rank_mm(V, U, q_update)


def _balance_lra(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """Balance the low-rank factor maxima without changing their product."""

    log_u = U.abs().amax(dim=(-2, -1)).log()
    log_v = V.abs().amax(dim=(-2, -1)).log()
    mean_log = (log_u + log_v) * 0.5
    scale_u = (mean_log - log_u).exp()
    scale_v = (mean_log - log_v).exp()
    valid = torch.isfinite(scale_u) & torch.isfinite(scale_v) & (scale_u > 0) & (scale_v > 0)
    scale_u = torch.where(valid, scale_u, torch.ones_like(scale_u))
    scale_v = torch.where(valid, scale_v, torch.ones_like(scale_v))
    return U * broadcast_leaf(scale_u, U), V * broadcast_leaf(scale_v, V)


def _mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _refresh_lra(
    update_flat: Tensor,
    U: Tensor,
    V: Tensor,
    d: Tensor,
    tempo: Tempo,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one batched Lie-group update of the LRA preconditioner."""

    U, V = _balance_lra(U, V)
    vector = tempo.randn_like(update_flat)
    eps = torch.finfo(update_flat.dtype).eps
    hessian_vector = update_flat + (tempo.hyper.dampening + eps * update_flat.abs()) * vector

    qh = _lra_low_rank_mm(U, V, d * hessian_vector)
    ph = d * _lra_low_rank_mm(V, U, qh)
    rank = U.shape[-1]
    eye = torch.eye(rank, dtype=U.dtype, device=U.device)

    scaled_vector = vector / d
    utv = U.mT @ V
    inv_qt_vector = scaled_vector - _mv(
        V,
        torch.linalg.solve(
            eye + utv + eps * eye,
            _mv(U.mT, scaled_vector).unsqueeze(-1),
        ).squeeze(-1),
    )

    scaled_inv_qt_vector = inv_qt_vector / d
    vtu = V.mT @ U
    inv_p_vector = scaled_inv_qt_vector - _mv(
        U,
        torch.linalg.solve(
            eye + vtu + eps * eye,
            _mv(V.mT, scaled_inv_qt_vector).unsqueeze(-1),
        ).squeeze(-1),
    )

    inverse_difference = inv_qt_vector - inv_p_vector
    nabla_d = d.square() * ph * hessian_vector - vector * inverse_difference
    a0 = d * ph
    a1 = vector
    b0 = inverse_difference / d
    b1 = hessian_vector
    score = torch.hypot(a0, a1).log2() + torch.hypot(b0, b1).log2()
    index = score.argmax(dim=-1, keepdim=True)
    a = a0.gather(-1, index).square() + a1.gather(-1, index).square()
    b = b0.gather(-1, index).square() + b1.gather(-1, index).square()
    divisor = (a * b).sqrt().clamp_min(eps).squeeze(-1)
    d = d - broadcast_leaf(tempo.hyper.precond_lr / divisor, d) * nabla_d

    at_u = _mv(U.mT, qh)
    bt_u = _mv(U.mT, inv_qt_vector)
    at_uut = _mv(U, at_u)
    bt_uut = _mv(U, bt_u)
    divisor_v = (
        qh.norm(dim=-1) * at_uut.norm(dim=-1)
        + inv_qt_vector.norm(dim=-1) * bt_uut.norm(dim=-1)
    ).clamp_min(eps)
    a_projected = qh + _mv(V, at_u)
    b_projected = inv_qt_vector + _mv(V, bt_u)
    v_update = b_projected.unsqueeze(-1) * bt_u.unsqueeze(-2) - a_projected.unsqueeze(-1) * at_u.unsqueeze(-2)
    V = V + broadcast_leaf(tempo.hyper.precond_lr / divisor_v, V) * v_update

    at_v = _mv(V.mT, qh)
    bt_v = _mv(V.mT, inv_qt_vector)
    at_vvt = _mv(V, at_v)
    bt_vvt = _mv(V, bt_v)
    divisor_u = (
        qh.norm(dim=-1) * at_vvt.norm(dim=-1)
        + inv_qt_vector.norm(dim=-1) * bt_vvt.norm(dim=-1)
    ).clamp_min(eps)
    ip_vtu = eye + V.mT @ U
    a_coeff = (at_v.unsqueeze(-2) @ ip_vtu).squeeze(-2)
    b_coeff = (bt_v.unsqueeze(-2) @ ip_vtu).squeeze(-2)
    u_update = inv_qt_vector.unsqueeze(-1) * b_coeff.unsqueeze(-2) - qh.unsqueeze(-1) * a_coeff.unsqueeze(-2)
    U = U + broadcast_leaf(tempo.hyper.precond_lr / divisor_u, U) * u_update
    return U, V, d


def psgd_lra_init(ref_leaf: Tensor, *, rank: Tensor | int = 10) -> dict[str, Tensor]:
    """Allocate diagonal and low-rank state in flattened leaf coordinates."""

    rank = int(rank)
    ref = _wide(ref_leaf).reshape(-1)
    scale = (ref.numel() * (rank + 10)) ** -0.5
    return {
        "U": torch.randn(ref.numel(), rank, dtype=ref.dtype, device=ref.device) * scale,
        "V": torch.randn(ref.numel(), rank, dtype=ref.dtype, device=ref.device) * scale,
        "d": torch.ones(ref.numel(), dtype=ref.dtype, device=ref.device),
    }


def make_psgd_lra(rank: int = 10):
    """Build a flat PSGD-LRA transform with a default approximation rank."""

    if type(rank) is not int:
        raise TypeError("rank must be a Python int")

    def lra_init(ref_leaf: Tensor, *, rank: Tensor | int = rank) -> dict[str, Tensor]:
        return psgd_lra_init(ref_leaf, rank=rank)

    def psgd_lra(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        del obs, param
        original_shape = update.shape
        update_flat = _wide(update).reshape(update.shape[0], -1)
        U = _wide(state["U"])
        V = _wide(state["V"])
        d = _wide(state["d"])
        if tempo.refresh:
            U, V, d = _refresh_lra(update_flat, U, V, d, tempo)
        preconditioned = _lra_precond(update_flat, U, V, d).reshape(original_shape)
        return preconditioned, {"U": U, "V": V, "d": d}, tempo.live

    psgd_lra.init = lra_init
    psgd_lra.state_init_hyper = ("rank",)
    psgd_lra.config = {"rank": rank}
    return psgd_lra


psgd_lra = make_psgd_lra()


lra = Recipe(
    chain=(psgd_lra,),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        dampening=1e-9,
        rank=10,
        weight_decay=0.0,
    ),
)


__all__ = ["RefreshCadence", "lra", "make_psgd_lra", "psgd_lra", "psgd_lra_init"]
