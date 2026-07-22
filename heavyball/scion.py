"""Scion's spectral-norm LMO and seeded parameter initialization."""

import itertools
import math

import torch
from torch import Tensor

from .core import Recipe
from .numerics import _wide, stable_l2_normalize
from .transforms import WHOLE, Tempo, first_moment, orthogonalize, sgd_commit


def _stable_l2_normalize(value: Tensor, *, dim: int, eps: Tensor) -> Tensor:
    """Normalize through the shared stable-L2 primitive."""

    return stable_l2_normalize(value, dim=dim, eps=eps)


def _scion_bias_rms_direction(update: Tensor, eps: Tensor) -> Tensor:
    if update.ndim == 1:
        return update / update.abs().clamp(min=eps)
    dimension = update.shape[1]
    scale = math.sqrt(dimension)
    return _stable_l2_normalize(update, dim=1, eps=eps * scale) * scale


def _orthogonal_direction(update: Tensor, tempo: Tempo) -> Tensor:
    flat = update.reshape(update.shape[0], update.shape[1], -1)
    direction, _, _ = orthogonalize(flat, None, None, {}, tempo)
    return direction.reshape_as(update)


def _scion_spectral_direction(update: Tensor, tempo: Tempo) -> Tensor:
    direction = _orthogonal_direction(update, tempo)
    in_dimension = max(update.reshape(update.shape[0], update.shape[1], -1).shape[2], 1)
    return direction * math.sqrt(update.shape[1] / in_dimension)


def _scion_spectral_conv_direction(update: Tensor, tempo: Tempo) -> Tensor:
    direction = _orthogonal_direction(update, tempo)
    spatial = math.prod(update.shape[3:])
    return direction * (math.sqrt(update.shape[1] / max(update.shape[2], 1)) / max(spatial, 1))


def scion_lmo(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
    """Apply Scion's leaf-ndim-selected LMO to a full ``[N, *shape]`` slab."""

    del obs, param, state
    update = _wide(update)
    if update.ndim >= 4:
        direction = _scion_spectral_conv_direction(update, tempo)
    elif update.ndim == 3:
        direction = _scion_spectral_direction(update, tempo)
    else:
        direction = _scion_bias_rms_direction(update, tempo.hyper.eps)
    scale = tempo.hyper.scale.to(dtype=direction.dtype, device=direction.device)
    return direction * scale, {}, tempo.live


def scion_lmo_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def _copy_initialized_(target: Tensor, source: Tensor, generator: torch.Generator) -> None:
    """Match legacy seeded bfloat16 stochastic rounding at build time."""

    if target.dtype == torch.bfloat16 and source.dtype in (torch.float16, torch.float32, torch.float64):
        bits = source.float().view(torch.int32)
        noise = torch.randint(
            0,
            1 << 16,
            bits.shape,
            dtype=bits.dtype,
            device=bits.device,
            generator=generator,
        )
        source = (bits + noise).bitwise_and(-65536).view(torch.float32).bfloat16()
    target.copy_(source)


@torch.no_grad()
def scion_param_init(slab_row: Tensor, seed: int, *, scale: Tensor | None = None) -> None:
    """Port legacy Scion's seed-indexed orthogonal parameter initialization."""

    promoted = _wide(slab_row)
    generator = torch.Generator(device=slab_row.device)
    generator.manual_seed(seed)
    if slab_row.ndim >= 2:
        initialized = promoted.clone().double()
        for spatial_index in itertools.product(*(range(size) for size in initialized.shape[2:])):
            torch.nn.init.orthogonal_(initialized[(slice(None), slice(None), *spatial_index)], generator=generator)
        fan_out, fan_in = initialized.shape[:2]
        spatial = math.prod(initialized.shape[2:])
        initialized.mul_(math.sqrt(fan_out / max(fan_in, 1)) / max(spatial, 1))
        initialized = initialized.to(dtype=promoted.dtype)
    else:
        initialized = torch.zeros_like(promoted)
    if scale is not None:
        initialized.mul_(scale.to(dtype=initialized.dtype, device=initialized.device))
    _copy_initialized_(slab_row, initialized, generator)


scion_lmo.init = scion_lmo_init
scion_lmo.distributed_scope = WHOLE
scion_lmo.param_init = scion_param_init
scion_lmo.param_init_hyper = ("scale",)

scion = Recipe(
    chain=(first_moment, scion_lmo),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, eps=1e-8, scale=1.0, weight_decay=0.0),
)
scion_route = scion


__all__ = ["scion", "scion_lmo", "scion_lmo_init", "scion_param_init", "scion_route"]
