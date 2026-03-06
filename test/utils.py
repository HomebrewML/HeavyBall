import functools
import warnings
from typing import Callable, List

import torch
from torch import Tensor
from torch.utils import _pytree as tree_util

import heavyball
from heavyball.chainable import FunctionTransform

# Optimizers incompatible with the standard get_optim(betas=(0.9, 0.999)) call:
#   AdEMAMix variants require 3 betas, SplitOpt requires dict param specs,
#   SAMWrapper and Newton variants require closures
_SKIP_GET_OPTIM = {"ForeachAdEMAMix", "ForeachSOAPAdEMAMix", "SOAPAdEMAMix", "SplitOpt", "SAMWrapper",
    "ForeachCachedNewtonPSGD", "NewtonHybrid2PSGDKron", "ForeachNewtonPSGDLRA", "NewtonHybrid2PSGDLRA", "NewtonPSGDLRA",
    "NewtonPSGDKron", }


def _fn_key(f):
    if isinstance(f, FunctionTransform):
        return f.fn_name
    if isinstance(f, functools.partial):
        return (_fn_key(f.func), f.args) + tuple(sorted(f.keywords.items()))
    if hasattr(f, "__name__"):
        return f.__name__
    return repr(f)


def _deduplicate_by_chain(names):
    """Keep one optimizer per unique chain of functions.

    Two optimizers that differ only by foreach=True/False have identical
    chains and test the same code paths — keep whichever appears first.
    """
    seen = set()
    out = []
    for name in names:
        dummy = [torch.nn.Parameter(torch.randn(4, 4))]
        cls = getattr(heavyball, name)
        try:
            opt = cls(dummy, lr=1e-3)
            key = tuple(_fn_key(f) for f in opt._fns)
        except Exception as e:
            warnings.warn(f"Failed to instantiate {name} for dedup: {e}")
            continue
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


REPRESENTATIVE_OPTS = _deduplicate_by_chain([name for name in heavyball.__all__ if name not in _SKIP_GET_OPTIM])

@torch.no_grad()
def set_grad(model: torch.nn.Module, *, dtype: torch.dtype = None):
    for p in model.parameters():
        g = torch.randn(p.shape, device=p.device, dtype=dtype or p.dtype, requires_grad=False)
        p.grad = g.to(p.dtype)

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
