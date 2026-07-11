import functools
import inspect
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils import _pytree as tree_util

import heavyball
from heavyball.chainable import FunctionTransform, _walk_fns

# Optimizers incompatible with the standard get_optim(betas=(0.9, 0.999)) call:
#   AdEMAMix variants require 3 betas, SplitOpt requires dict param specs,
#   SAMWrapper and Newton variants require closures
_SKIP_GET_OPTIM = {
    "AdEMAMix",
    "SOAPAdEMAMix",
    "HeavySOAPAdEMAMix",
    "SplitOpt",
    "SAMWrapper",
}

_OPTIM_DEFAULTS = {
    "betas": (0.9, 0.999),
    "precondition_frequency": 16,
    "merge_dims": True,
    "warmup_steps": 100,
    "max_precond_dim": 2**16,
    "beta": 0.9,
    "max_size_triangular": 2**16,
    "split": False,
    "precond_grad_accum": False,
    "momentum_into_precond_update": True,
    "eps": 1e-8,
    "weight_decay": 0,
    "dampening": 2**-24,
    "preconditioner_update_probability": 1.0,
    "precond_init_scale": 1.0,
    "store_triu_as_line": False,
    "update_clipping": None,
    "delayed": True,
}


def get_optim(optim, params, **kwargs):
    sig = inspect.signature(optim)
    defaults = {key: value for key, value in _OPTIM_DEFAULTS.items() if key in sig.parameters}
    defaults.update(kwargs)
    return optim(params, **defaults)


def _fn_key(f):
    if isinstance(f, FunctionTransform):
        return f.fn_name
    if isinstance(f, functools.partial):
        return (_fn_key(f.func), f.args) + tuple(sorted(f.keywords.items()))
    if hasattr(f, "__name__"):
        return f.__name__
    return repr(f)


def _deduplicate_by_chain(names):
    """Keep one optimizer per unique chain of functions; also report which are bucket-aware."""
    seen = set()
    out = []
    bucket_aware = set()
    for name in names:
        dummy = [torch.nn.Parameter(torch.randn(4, 4))]
        cls = getattr(heavyball, name)
        opt = cls(dummy, lr=1e-3)
        key = tuple(_fn_key(f) for f in opt._fns), opt.param_groups[0].get("sqrt", False)
        if any(ft._under_bucket for ft in _walk_fns(opt._fns)):
            bucket_aware.add(name)
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out, bucket_aware


REPRESENTATIVE_OPTS, BUCKET_AWARE_OPTS = _deduplicate_by_chain(
    [
        name
        for name in heavyball.__all__
        if name not in _SKIP_GET_OPTIM
        and isinstance(getattr(heavyball, name), type)
        and issubclass(getattr(heavyball, name), torch.optim.Optimizer)
    ]
)


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
def _global_l2_norm(xs: list[Tensor]) -> Tensor:
    return sum((x.square().sum() for x in xs), start=scalar_like(xs[0])) ** 0.5


@_upcast
def _global_rms_norm(xs: list[Tensor]) -> Tensor:
    norm = sum((x.square().sum() for x in xs), start=scalar_like(xs[0]))
    numel = sum(x.numel() for x in xs)
    return (norm / numel) ** 0.5
