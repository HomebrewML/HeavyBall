"""Stateful schedule-free terminal commit for HeavyBall 4.0."""

import torch
from torch import Tensor

from .numerics import _caution, _strictly_aligned, _wide, broadcast_leaf


def schedule_free_commit_init(reference: Tensor) -> dict[str, Tensor]:
    """Seed the legacy schedule-free averaging iterate and cumulative weight."""

    return {
        "z": _wide(reference).clone(),
        "weight_sum": torch.zeros((), dtype=_wide(reference).dtype, device=reference.device),
        "lr_max": torch.zeros((), dtype=_wide(reference).dtype, device=reference.device),
    }


def schedule_free_commit(param: Tensor, update: Tensor, state: dict[str, Tensor], tempo):
    """Apply legacy SFAdamW's schedule-free terminal update over a full slab."""

    param = _wide(param)
    update = _wide(update)
    z = _wide(state["z"])
    weight_sum = _wide(state["weight_sum"])
    lr = _wide(tempo.hyper.lr)
    beta1 = _wide(tempo.hyper.beta1)
    decay = _wide(tempo.hyper.weight_decay)
    weight_lr_power = _wide(tempo.hyper.weight_lr_power)
    r = _wide(tempo.hyper.r)
    caution = _wide(tempo.hyper.caution)
    cautious_decay = _wide(tempo.hyper.cautious_weight_decay)

    lr_max = _wide(state["lr_max"])
    lr_max = torch.maximum(lr.abs(), lr_max)

    age = tempo.age.to(weight_sum.dtype)
    age_weight = torch.where(
        r != 0,
        age.pow(r),
        torch.ones_like(age),
    )
    weight = lr_max.pow(weight_lr_power) * age_weight
    weight_sum_new = weight_sum + weight
    ckp1 = torch.where(
        weight_sum_new != 0,
        weight / weight_sum_new,
        torch.zeros_like(weight_sum_new),
    )
    ckp1 = broadcast_leaf(ckp1, param)

    decay_param = torch.where(_strictly_aligned(param, update), param, torch.zeros_like(param))
    decay_param = torch.where(cautious_decay != 0, decay_param, param)
    update = torch.where(decay != 0, update + decay_param * decay, update)
    raw_grad = torch.zeros_like(update) if tempo.raw_grad is None else _wide(tempo.raw_grad)
    update = torch.where(caution != 0, _caution(raw_grad, update), update)
    candidate_param = torch.lerp(param, z, ckp1)
    candidate_param = candidate_param + update * (lr * (beta1 * (1 - ckp1)) - lr)
    z_new = z - update * lr
    return candidate_param, {"z": z_new, "weight_sum": weight_sum_new, "lr_max": lr_max}


def schedule_free_eval_swap(
    param: Tensor, state: dict[str, Tensor], hyper, entering_train: bool
) -> tuple[Tensor, dict[str, Tensor]]:
    """Swap schedule-free's train/eval parameter representation without touching state."""

    param = _wide(param)
    z = _wide(state["z"])
    beta1 = _wide(hyper.beta1)
    active = beta1 > 0
    safe_beta1 = torch.where(active, beta1, torch.ones_like(beta1))
    weight = 1 - safe_beta1 if entering_train else 1 - 1 / safe_beta1
    return torch.where(active, param + (z - param) * weight, param), state


schedule_free_commit.init = schedule_free_commit_init
schedule_free_commit.eval_swap = schedule_free_eval_swap
