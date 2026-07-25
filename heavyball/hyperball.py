"""Stateful HyperBall terminal commit for HeavyBall 4.0."""

import torch
from torch import Tensor

from .numerics import _caution, _wide, broadcast_leaf, stable_l2_normalize

def _stable_l2_components(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return legacy stable-L2 scale and normalized norm for every slab row."""

    value = _wide(value)
    if value.numel() == 0:
        scale = torch.zeros(value.shape[0], dtype=value.dtype, device=value.device)
        return scale, torch.ones_like(scale)
    flat = value.reshape(value.shape[0], -1)
    scale = flat.abs().amax(dim=1)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    norm = torch.linalg.vector_norm(flat / safe_scale.unsqueeze(1), dim=1)
    return scale, norm

def _unpack_init_norm(init_norm: Tensor) -> tuple[Tensor, Tensor]:
    """Accept legacy's one- or two-scalar init-norm representations."""

    if init_norm.shape[1] == 2:
        return init_norm[:, 0], init_norm[:, 1]
    norm_scale = init_norm.reshape(init_norm.shape[0])
    return norm_scale, torch.ones_like(norm_scale)


def hyperball_commit_init(reference: Tensor) -> dict[str, Tensor]:
    """Allocate the legacy two-scalar norm template and first-step latch."""

    reference = _wide(reference)
    return {
        "init_norm": torch.zeros(2, dtype=reference.dtype, device=reference.device),
        "seen": torch.zeros((), dtype=torch.bool, device=reference.device),
    }


def hyperball_commit(param: Tensor, update: Tensor, state: dict[str, Tensor], tempo):
    """Apply HyperBall's decay, caution, normalized step, and sphere projection."""

    param = _wide(param)
    update = _wide(update)
    stored_init_norm = _wide(state["init_norm"])
    seen = state["seen"]
    first = seen.logical_not()
    scale, norm = _stable_l2_components(param)
    first_init_norm = torch.cat((scale.unsqueeze(1), norm.unsqueeze(1)), dim=1)
    init_norm = torch.where(broadcast_leaf(first, first_init_norm), first_init_norm, stored_init_norm)

    lr = _wide(tempo.hyper.lr)
    caution = _wide(tempo.hyper.caution)

    raw_grad = torch.zeros_like(update) if tempo.raw_grad is None else _wide(tempo.raw_grad)
    update = torch.where(caution != 0, _caution(raw_grad, update), update)

    norm_scale, scaled_norm = _unpack_init_norm(init_norm)
    step_scale = broadcast_leaf((lr * scaled_norm) * norm_scale, param)
    radius = broadcast_leaf(scaled_norm * norm_scale, param)
    normalized_update = stable_l2_normalize(
        update.reshape(update.shape[0], -1), dim=1, eps=torch.finfo(update.dtype).tiny
    ).reshape_as(update)
    candidate_param = param - normalized_update * step_scale
    candidate_param = stable_l2_normalize(
        candidate_param.reshape(candidate_param.shape[0], -1),
        dim=1,
        eps=torch.finfo(candidate_param.dtype).tiny,
    ).reshape_as(candidate_param) * radius
    return candidate_param, {"init_norm": init_norm, "seen": torch.ones_like(seen)}


hyperball_commit.init = hyperball_commit_init
hyperball_commit.distributed_shard_separable = False
