from typing import Callable, List

import torch
from torch import Tensor
from torch.utils import _pytree as tree_util


def scalar_like(x):
    return torch.zeros((), dtype=x.dtype, device=x.device)


def _upcast_value(x: Tensor):
    if x.dtype.is_complex:
        return x.to(torch.cdouble)
    if x.dtype.is_floating_point:
        return x.to(torch.double)
    return x.to(torch.int64)


def _upcast(fn: Callable[[...], Tensor]) -> Callable[[...], float]:
    def _fn(*args, **kwargs):
        args, kwargs = tree_util.tree_map(_upcast_value, (args, kwargs))
        return fn(*args, **kwargs).item()

    return _fn


@_upcast
def _local_l2_norm(x):
    return x.square().sum().sqrt()


@_upcast
def _local_rms_norm(x):
    return x.square().mean().sqrt()


@_upcast
def _global_l2_norm(xs: List[Tensor]) -> Tensor:
    return sum((x.square().sum() for x in xs), start=scalar_like(xs[0])) ** 0.5


@_upcast
def _global_rms_norm(xs: List[Tensor]) -> Tensor:
    norm = sum((x.square().sum() for x in xs), start=scalar_like(xs[0]))
    numel = sum(x.numel() for x in xs)
    return (norm / numel) ** 0.5
