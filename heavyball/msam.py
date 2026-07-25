"""Momentum-SAM terminal commit for HeavyBall's slab-native optimizer core."""

import torch
from torch import Tensor

from .numerics import _caution, _strictly_aligned, _wide, broadcast_leaf, stable_l2_normalize
from .transforms import beta_debias


def msam_commit_init(reference: Tensor) -> dict[str, Tensor]:
    """Seed MSAM's unperturbed master, momentum, and the perturbed iterate saved across eval."""

    ref = _wide(reference)
    # The fp32 master must not be quantized through a low-precision parameter.
    return {"z": ref.clone(), "exp_avg": torch.zeros_like(ref), "saved": reference.clone()}


def msam_commit(param: Tensor, update: Tensor, state: dict[str, Tensor], tempo):
    """Apply legacy MSAMLaProp's terminal momentum-SAM update over a full slab."""

    del param
    update = _wide(update)
    z = _wide(state["z"])
    exp_avg = _wide(state["exp_avg"])
    lr = _wide(tempo.hyper.lr)
    beta1 = broadcast_leaf(beta_debias(_wide(tempo.hyper.beta1), tempo.age), update)
    decay = _wide(tempo.hyper.weight_decay)
    sam_step_size = _wide(tempo.hyper.sam_step_size)
    caution = _wide(tempo.hyper.caution)
    cautious_decay = _wide(tempo.hyper.cautious_weight_decay)

    exp_avg_new = exp_avg * beta1 + update * (1 - beta1)
    raw_grad = torch.zeros_like(exp_avg_new) if tempo.raw_grad is None else _wide(tempo.raw_grad)
    filtered_exp_avg = torch.where(caution != 0, _caution(raw_grad, exp_avg_new), exp_avg_new)
    decay = torch.where(
        cautious_decay != 0,
        decay * _strictly_aligned(z, filtered_exp_avg).to(z.dtype),
        decay,
    )
    z_new = z * (1 - decay * lr) + filtered_exp_avg * -lr
    flat_exp_avg = filtered_exp_avg.flatten(start_dim=1)
    normalized_exp_avg = stable_l2_normalize(flat_exp_avg, dim=1, eps=1e-8).reshape_as(filtered_exp_avg)
    perturbed_param = z_new - normalized_exp_avg * sam_step_size
    return perturbed_param, {"z": z_new, "exp_avg": exp_avg_new, "saved": state["saved"]}


def msam_eval_swap(param: Tensor, state: dict[str, Tensor], hyper, entering_train: bool) -> tuple[Tensor, dict[str, Tensor]]:
    """Show the unperturbed master in eval and restore the exact perturbed iterate in train, leaving the
    fp32 master untouched -- exchanging it with the low-precision parameter would quantize it."""

    del hyper
    if entering_train:
        return state["saved"], {"z": state["z"], "exp_avg": state["exp_avg"], "saved": state["saved"]}
    return state["z"], {"z": state["z"], "exp_avg": state["exp_avg"], "saved": param}


msam_commit.init = msam_commit_init
msam_commit.eval_swap = msam_eval_swap
msam_commit.distributed_shard_separable = False
