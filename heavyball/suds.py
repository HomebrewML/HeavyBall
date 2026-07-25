"""Rank-1 SUDS preconditioning for slab-native HeavyBall."""

import torch
from torch import Tensor

from .core import Recipe
from .numerics import _wide, broadcast_leaf, stable_l2_normalize as _stable_l2_normalize
from .transforms import Tempo, adam, sgd_commit


def stable_l2_normalize(x: Tensor, dim: int | tuple[int, ...] | None = None, eps=0) -> Tensor:
    """Legacy stable-L2 normalization through the shared primitive."""

    return _stable_l2_normalize(_wide(x), dim=dim, eps=eps)


def _householder_vec_e1_to_v(v: Tensor, eps: float = 1e-12) -> Tensor:
    """Return the batched reflector vector whose Householder map sends e1 to ``v``."""

    v = stable_l2_normalize(v, dim=-1, eps=eps)
    e1 = torch.zeros_like(v)
    e1[..., 0] = 1.0
    w = e1 - v
    scale = w.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(scale != 0, scale, torch.ones_like(scale))
    norm = torch.linalg.vector_norm(w / safe, dim=-1, keepdim=True) * scale
    return torch.where(
        norm >= eps,
        stable_l2_normalize(w, dim=-1),
        torch.zeros_like(w),
    )


def eigvecs_product_rank1(G: Tensor, v: Tensor, w: Tensor | None = None, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Apply the rank-1 Fisher Householder basis without materializing a matrix."""

    if w is None:
        w = _householder_vec_e1_to_v(v, eps)
    dtype = torch.promote_types(_wide(G).dtype, _wide(w).dtype)
    G, w = _wide(G).to(dtype), _wide(w).to(dtype)
    return G - 2.0 * (G * w).sum(dim=-1, keepdim=True) * w, w


def oja_update(v: Tensor, g: Tensor, lr: float | Tensor = 1e-2, eps: float = 1e-12) -> Tensor:
    """One numerically stable, batched legacy Oja power-iteration update."""

    dtype = torch.promote_types(_wide(v).dtype, _wide(g).dtype)
    v = stable_l2_normalize(_wide(v).to(dtype), dim=-1, eps=eps)
    g = _wide(g).to(dtype)
    scale = g.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(scale != 0, scale, torch.ones_like(scale))
    g = g / safe
    gv = (g * v).sum(dim=-1, keepdim=True)
    residual = g - (gv / (v * v).sum(dim=-1, keepdim=True)) * v
    roundoff = 8 * torch.finfo(dtype).eps * torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    stationary = (gv.abs() <= roundoff) | (
        torch.linalg.vector_norm(residual, dim=-1, keepdim=True) <= roundoff.clamp_min(eps)
    )
    coefficient = lr * gv
    log_coefficient = coefficient.abs().log() + 2 * scale.log()
    sign = coefficient.sign()
    small = v + sign * log_coefficient.clamp_max(0).exp() * residual
    large = sign * residual + (-log_coefficient).clamp_max(0).exp() * v
    out = torch.where(log_coefficient > 0, large, small)
    return stable_l2_normalize(torch.where(stationary, v, out), dim=-1, eps=eps)


def _transport_rank1_first_moment(m, old_w, new_w):
    m_phys = m - 2.0 * (m * old_w).sum(dim=-1, keepdim=True) * old_w
    return m_phys - 2.0 * (m_phys * new_w).sum(dim=-1, keepdim=True) * new_w


def _transport_rank1_second_moment(q, old_w, new_w):
    dtype = q.dtype
    q = q.double().square()
    old_w = old_w.double()
    new_w = new_w.double()
    c = (old_w * new_w).sum(dim=-1, keepdim=True)
    p = -2.0 * old_w + 4.0 * c * new_w
    r = -2.0 * new_w
    qaa = (q * old_w.square()).sum(dim=-1, keepdim=True)
    qbb = (q * new_w.square()).sum(dim=-1, keepdim=True)
    qab = (q * old_w * new_w).sum(dim=-1, keepdim=True)
    variance = (q + 2.0 * q * (old_w * p + new_w * r)
                + p.square() * qaa + r.square() * qbb + 2.0 * p * r * qab).clamp_min(0)
    return variance.sqrt().to(dtype)


def suds_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    ref = _wide(ref_leaf)
    return {
        "exp_avg": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
        "fisher_approx": torch.zeros_like(ref),
        "seen": torch.zeros((), dtype=torch.bool, device=ref.device),
    }


def suds(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Adam in SUDS's rank-1 Fisher Householder basis, one basis per slab leaf."""

    update = _wide(update)
    seen = state["seen"]
    first = seen.logical_not()
    update_flat = update.reshape(update.shape[0], -1)
    stored_fisher = _wide(state["fisher_approx"])
    fisher_flat = stored_fisher.reshape(stored_fisher.shape[0], -1)
    seed = stable_l2_normalize(update_flat, dim=-1, eps=1e-8)
    fisher = torch.where(fisher_flat.abs().amax(dim=-1, keepdim=True) == 0, seed, fisher_flat)

    rotated, w = eigvecs_product_rank1(update_flat, fisher)
    adam_tempo = tempo._replace(age=tempo.age - seen.to(tempo.age.dtype))
    preconditioned, adam_state, _ = adam(
        rotated.reshape_as(update),
        obs,
        param,
        {"exp_avg": state["exp_avg"], "exp_avg_sq": state["exp_avg_sq"]},
        adam_tempo,
    )
    preconditioned, _ = eigvecs_product_rank1(preconditioned.reshape(preconditioned.shape[0], -1), fisher, w)
    next_fisher = oja_update(fisher, update_flat, tempo.hyper.precond_lr)

    new_w = _householder_vec_e1_to_v(next_fisher)
    transported_avg = _transport_rank1_first_moment(
        _wide(adam_state["exp_avg"]).reshape(update_flat.shape), w, new_w
    ).reshape_as(update)
    transported_avg_sq = _transport_rank1_second_moment(
        _wide(adam_state["exp_avg_sq"]).reshape(update_flat.shape), w, new_w
    ).reshape_as(update)

    return (
        preconditioned.reshape_as(update),
        {
            "exp_avg": torch.where(
                broadcast_leaf(first, transported_avg), _wide(state["exp_avg"]), transported_avg
            ),
            "exp_avg_sq": torch.where(
                broadcast_leaf(first, transported_avg_sq), _wide(state["exp_avg_sq"]), transported_avg_sq
            ),
            "fisher_approx": torch.where(broadcast_leaf(first, next_fisher), fisher, next_fisher).reshape_as(stored_fisher),
            "seen": torch.ones_like(seen),
        },
        tempo.live & seen,
    )


suds.init = suds_init
suds.distributed_shard_separable = False


suds_adamw = Recipe(
    chain=(suds,),
    commit=sgd_commit,
    defaults=dict(
        lr=0.0025,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.0,
        precond_lr=1e-2,
    ),
)


__all__ = ["eigvecs_product_rank1", "oja_update", "stable_l2_normalize", "suds", "suds_adamw"]
