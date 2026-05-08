import contextlib
import copy
import functools
import math
import warnings
from collections.abc import Iterable as _Iterable
from typing import Iterable, List, Literal, Optional, Union

import torch
from torch import Tensor

from . import utils

use_default = utils.use_default


def _key_in_state(state, key):
    if isinstance(key, str):
        return key in state
    for k in key:
        if isinstance(k, (tuple, list)):
            continue
        if k not in state:
            return False
    return True


def _guard_in_state(state, key, template_fn):
    if not _key_in_state(state, key):
        state[key] = template_fn()
    return state[key]


_SKIP = object()


class FunctionTransform:
    def __init__(self, fn, names: list[str] | None = None):
        if names is None:
            names = []
        self.fn = fn
        self.fn_name = self.get_fn().__name__
        self.transform_idx = None
        self.names = names
        self._under_bucket = False

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        raise NotImplementedError

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        skip_update = False
        for st, a in zip(states, zip(update, grad, param, *args)):
            if self.transform_idx not in st.get("is_initialized", set()):
                if self._init(st, group, *a, **kwargs) is _SKIP:
                    skip_update = True
                if "is_initialized" not in st:
                    st["is_initialized"] = set()
                st["is_initialized"].add(self.transform_idx)
        if skip_update:
            return _SKIP
        vars = [[st.get(self.val_name(name), None) for st in states] for name in self.names]
        return self._call(state, group, update, grad, param, vars, *args, **kwargs)

    def get_fn(self):
        if utils.hasattr_none(self.fn, "get_fn"):
            return self.fn.get_fn()
        return self.fn

    def _build_val_names(self):
        self._val_names = {name: f"{self.fn_name}_{name}_{self.transform_idx}" for name in self.names}

    def val_name(self, name):
        return self._val_names[name]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fn}, transform_idx={self.transform_idx})"


def _enforce_uniform_skip(results):
    skips = [skip for _, skip, _ in results]
    if not skips:
        return False
    if all(skips):
        return True
    if not any(skips):
        return False
    raise ValueError("All branches must uniformly skip or not skip updates")


def _normalize_chain(fns):
    if fns is None:
        return None
    return fns if isinstance(fns, (list, tuple)) else (fns,)


class Parallel:
    def __init__(self, branches: List[List[callable]], merge_fn: callable):
        self.branches = branches
        self.merge_fn = merge_fn

    def __call__(self, state, group, update, grad, param):
        results = []
        for branch in self.branches:
            branch_update = [torch.clone(u, memory_format=torch.preserve_format) for u in update]
            u, skip = _inner_chain(state, group, branch_update, grad, param, *branch)
            results.append((u, skip, None))
        if _enforce_uniform_skip(results):
            return _SKIP
        return self.merge_fn([u for u, _, _ in results])


class Route:
    """Route params by predicate through different fn chains.

    Takes arbitrary (predicate, fns) pairs. Each param is assigned to the first
    matching route; unmatched params use the default chain (None = passthrough).
    All routes must uniformly either skip or not skip updates.
    """

    def __init__(self, *routes, default=None):
        self.routes = [(pred, _normalize_chain(fns)) for pred, fns in routes]
        self.default = _normalize_chain(default)

    def __call__(self, state, group, update, grad, param):
        buckets = {}
        assigned = set()
        for j, (pred, _) in enumerate(self.routes):
            for i, p in enumerate(param):
                if i not in assigned and pred(p):
                    buckets.setdefault(j, []).append(i)
                    assigned.add(i)
        default_idx = [i for i in range(len(param)) if i not in assigned]

        def _sel(lst, idx):
            return [lst[i] for i in idx]

        results = []

        all_chains = [(buckets.get(j), fns) for j, (_, fns) in enumerate(self.routes)]
        if default_idx:
            all_chains.append((default_idx, self.default))

        for idx, fns in all_chains:
            if not idx:
                continue
            if fns is not None:
                u, skip = _inner_chain(
                    _sel(state, idx), group, _sel(update, idx), _sel(grad, idx), _sel(param, idx), *fns
                )
            else:
                u, skip = _sel(update, idx), False
            results.append((u, skip, idx))

        if _enforce_uniform_skip(results):
            return _SKIP

        out = [None] * len(param)
        for u_list, _, idx in results:
            if u_list is not None:
                for i, u in zip(idx, u_list):
                    out[i] = u
        return out


def route(*routes, default=None):
    return Route(*routes, default=default)


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key, lambda: torch.zeros_like(ref, dtype=dtype, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get("storage_dtype", "float32")
    return getattr(torch, dtype)


_PASSTHROUGH_KWARGS = {"orig_shapes"}

_RENAMED_KWARGS = {"foreach": "multi_tensor"}

_REMOVED_KWARGS = frozenset(
    {
        "stochastic_schedule",
        "normalize_grads",
        "correct_bias",
        "inverse_free",
        "adaptive",
    }
)


def _build_defaults(locals_dict):
    d = locals_dict.copy()
    d.pop("self", None)
    params = d.pop("params")
    kwargs = d.pop("kwargs")

    for old, new in _RENAMED_KWARGS.items():
        if old in kwargs:
            warnings.warn(
                f"'{old}' was renamed to '{new}' in HeavyBall 3.0. Pass '{new}' instead.", FutureWarning, stacklevel=4
            )
            d[new] = kwargs.pop(old)

    hit = _REMOVED_KWARGS & kwargs.keys()
    if hit:
        raise TypeError(
            f"Removed in HeavyBall 3.0: {', '.join(sorted(hit))}. See docs/heavyball3.md for migration details."
        )

    d.update(kwargs)
    unknown = {k: v for k, v in kwargs.items() if k not in _PASSTHROUGH_KWARGS}
    if unknown:
        utils.warn_once(f"Working with uncaptured keyword arguments: {unknown}")
    return params, d


class ECCConfig:
    __slots__ = ("primary_dtype", "corr_dtype")

    _MODES = {
        "bf16+8": (torch.bfloat16, torch.int8),
        "bf16+16": (torch.bfloat16, torch.int16),
        "fp16+8": (torch.float16, torch.int8),
        "fp16+16": (torch.float16, torch.int16),
    }

    def __init__(self, mode):
        self.primary_dtype, self.corr_dtype = self._MODES[mode]

    @property
    def smax(self):
        return utils._ULPState._SMAX[self.corr_dtype]

    def init_correction(self, correction, fp32, narrow):
        utils._ULPState(correction, self.smax).compute_correction(fp32, narrow)

    @classmethod
    def from_group(cls, group, key="ecc"):
        mode = group.get(key)
        if not mode:
            return None
        return cls(mode)

    def init_state(self, state, key, ref):
        _guard_in_state(
            state, key, lambda: torch.zeros_like(ref, dtype=self.primary_dtype, memory_format=torch.preserve_format)
        )
        _guard_in_state(
            state,
            key + "::ecc",
            lambda: torch.zeros_like(ref, dtype=self.corr_dtype, memory_format=torch.preserve_format),
        )

    @contextlib.contextmanager
    def attached(self, tensors, corrections):
        smax = self.smax
        for t, c in zip(tensors, corrections):
            t._ecc = utils._ULPState(c, smax)
        try:
            yield
        finally:
            for t in tensors:
                t.__dict__.pop("_ecc", None)


def _init_mu_product(state, group, update, grad, param, **kwargs):
    dtype = _storage_dtype(group)
    state["mu_product"] = torch.ones((), dtype=dtype, device=param.device)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn, names)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        ecc = ECCConfig.from_group(group)
        for name in self.names:
            vn = self.val_name(name)
            if ecc is None:
                _zero_guard(state, vn, param, _storage_dtype(group))
            else:
                ecc.init_state(state, vn, param)

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        ecc = ECCConfig.from_group(group)
        if ecc is None:
            return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)

        names = [self.val_name(n) for n in self.names]
        primary_vars = [[st[vn] for st in state] for vn in names]
        with contextlib.ExitStack() as stack:
            for vn, plist in zip(names, primary_vars):
                corrs = [st[vn + "::ecc"] for st in state]
                stack.enter_context(ecc.attached(plist, corrs))
            return self.fn(state, group, update, grad, param, *args, *primary_vars, **kwargs)


class PrecondGradAccumGuard(FunctionTransform):
    def __init__(self, fn):
        super().__init__(fn, ["precond_grad_accum"])
        self.steps_taken_key = None

    def _build_val_names(self):
        super()._build_val_names()
        self.steps_taken_key = f"_{self.fn_name}_steps_taken_{self.transform_idx}"

    def _accum(self, group, state, new):
        group[self.steps_taken_key] = group.get(self.steps_taken_key, 0) + 1
        utils.stochastic_add_(state, new)

    def _reset(self, group, state):
        if group.get(self.steps_taken_key, 0) != 0:
            group[self.steps_taken_key] = 0
            utils.zero_(state)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        if not group.get("precond_grad_accum", False):
            return
        for name in self.names:
            _zero_guard(state, self.val_name(name), param, _storage_dtype(group))

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        base_grad = update if group.get("momentum_into_precond_update", True) else grad
        if not group.get("precond_grad_accum", False):
            return self.fn(state, group, update, grad, param, *args, base_grad, **kwargs)

        (vars,) = vars
        steps_taken = group.get(self.steps_taken_key, 0)
        accum_state = None
        if group["is_preconditioning"]:
            if steps_taken:
                self._accum(group, vars, base_grad)
                utils.stochastic_multiply_(vars, 1 / group[self.steps_taken_key])
                accum_state = vars
            else:
                vars = base_grad
        else:
            self._accum(group, vars, base_grad)
            vars = base_grad
        out = self.fn(state, group, update, grad, param, *args, vars, **kwargs)
        if accum_state is not None:
            self._reset(group, accum_state)
        return out


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn, names)
        self.index = index

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        for name in self.names:
            state[self.val_name(name)] = torch.clone(val)

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class GeneralGuard(FunctionTransform):
    def __init__(self, fn, names, init_fn, skip_first: bool = True):
        super().__init__(fn, names)
        self.init_fn = init_fn
        self.skip_first = skip_first

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        self.init_fn(state, group, update, grad, param, **kwargs)
        for name in self.names:
            state[self.val_name(name)] = state.pop(name, None)
        if self.skip_first:
            return _SKIP

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class NoState(FunctionTransform):
    needs_init = False

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        return self.fn(group, update, grad, param, *args, **kwargs)


class NoStateNoMultiTensor(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        for st in states:
            if "is_initialized" not in st:
                st["is_initialized"] = set()
            st["is_initialized"].add(self.transform_idx)
        updates = []
        skip_update = False
        for a in zip(update, grad, param, *args):
            r = self.fn(group, *a, **kwargs)
            if r is _SKIP:
                skip_update = True
            else:
                updates.append(r)
        if skip_update:
            return _SKIP
        return updates


def _view_preserve_ecc(src, target):
    v = src.view_as(target)
    ecc = getattr(src, "_ecc", None)
    if ecc is not None:
        v._ecc = ecc
    return v


def _squeeze_inner(u: Tensor) -> Tensor:
    inner = tuple(i for i in range(1, u.ndim) if u.shape[i] == 1)
    return u.squeeze(inner) if inner else u


class SqueezeGrad(FunctionTransform):
    needs_init = False

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        original_shapes = [u.shape for u in update]
        update = [_squeeze_inner(u) for u in update]
        grad = [_view_preserve_ecc(x, u) for x, u in zip(grad, update)]
        param = [_view_preserve_ecc(x, u) for x, u in zip(param, update)]
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, (list, tuple)) and isinstance(a[0], Tensor):
                args[i] = [_view_preserve_ecc(x, u) for x, u in zip(a, update)]
        for k, a in kwargs.items():
            if isinstance(a, (list, tuple)) and isinstance(a[0], Tensor):
                kwargs[k] = [_view_preserve_ecc(x, u) for x, u in zip(a, update)]
        out = self.fn(state, group, update, grad, param, *args, **kwargs)
        if out is _SKIP or out is None:
            return out
        return [o.view(s) for o, s in zip(out, original_shapes)]


class TagGuard(FunctionTransform):
    def __init__(self, fn, **tags):
        super().__init__(fn)
        for k, v in tags.items():
            setattr(self, k, v)

    def _init(self, *args, **kwargs):
        pass

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, **kwargs)


class BucketGuard(FunctionTransform):
    """Group same-shape params into a leading-dim slab, run the inner chain once per bucket, unstack."""

    needs_init = False

    @property
    def _bucket_state_key(self):
        return f"__bucket_{self.transform_idx}__"

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        shapes = group.get("_orig_shapes") or {}
        bucket_key = self._bucket_state_key
        buckets: dict = {}
        for i, p in enumerate(param):
            info = shapes.get(id(p))
            sig = (tuple(p.shape), p.dtype, p.device, info.owner if info is not None else None)
            buckets.setdefault(sig, []).append(i)

        out = [None] * len(param)
        skip = False
        for indices in buckets.values():
            views = [param[i] for i in indices]
            grads = [grad[i] for i in indices]
            updates = [update[i] for i in indices]
            n = len(indices)
            if n == 1:
                slab_p, slab_g, slab_u = views[0][None], grads[0][None], updates[0][None]
            else:
                slab_p = torch.stack(views, 0)
                slab_g = torch.stack(grads, 0)
                slab_u = torch.stack(updates, 0)

            eccs = [getattr(v, "_ecc", None) for v in views]
            stacked_corr = None
            if eccs[0] is not None:
                stacked_corr = (
                    eccs[0].correction[None] if n == 1 else torch.stack([e.correction for e in eccs], 0)
                )
                slab_p._ecc = utils._ULPState(stacked_corr, eccs[0].smax)

            bucket_state = states[indices[0]].setdefault(bucket_key, {})
            for i in indices[1:]:
                states[i][bucket_key] = bucket_state

            result = self.fn([bucket_state], group, [slab_u], [slab_g], [slab_p], *args, **kwargs)
            if result is _SKIP:
                skip = True
                if n > 1:
                    for k in range(n):
                        views[k].copy_(slab_p[k])
                    if stacked_corr is not None:
                        for k, e in enumerate(eccs):
                            e.correction.copy_(stacked_corr[k])
                continue

            precond_slab = result[0]
            if n == 1:
                out[indices[0]] = precond_slab[0]
            else:
                for k, i in enumerate(indices):
                    out[i] = precond_slab[k]

        if skip:
            return _SKIP
        return out


class WarmupGuard(FunctionTransform):
    def __init__(self, fn, warmup_fns):
        super().__init__(fn, names=[])
        self.warmup_fns = warmup_fns
        self.warmup_key = None

    def _build_val_names(self):
        super()._build_val_names()
        self.warmup_key = f"_warmup_{self.transform_idx}"

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        warmup_step = min(st.get(self.warmup_key, 0) for st in states)
        if warmup_step < len(self.warmup_fns):
            fn = self.warmup_fns[warmup_step]
            for st, a in zip(states, zip(update, grad, param, *args)):
                fn(st, group, *a, **kwargs)
                st[self.warmup_key] = st.get(self.warmup_key, 0) + 1
            return _SKIP
        for st in states:
            if "is_initialized" not in st:
                st["is_initialized"] = set()
            st["is_initialized"].add(self.transform_idx)
        return self.fn(state, group, update, grad, param, *args, **kwargs)


needs_full_param = functools.partial(TagGuard, needs_full_param=True)

bucket_aware = BucketGuard


def zero_guard(*names):
    return functools.partial(ZeroGuard, names=names)


def copy_guard(index, *names):
    return functools.partial(CopyGuard, index=index, names=names)


def general_guard(*names, init_fn, skip_first: bool = True):
    return functools.partial(GeneralGuard, names=names, init_fn=init_fn, skip_first=skip_first)


def warmup_guard(*warmup_fns):
    return functools.partial(WarmupGuard, warmup_fns=list(warmup_fns))


def no_state(fn):
    return NoState(fn)


def no_state_no_multi_tensor(fn):
    return NoStateNoMultiTensor(fn)


class SkipUpdate(ValueError):
    pass


@zero_guard("mars_old_grad")
@no_state
def mars(group, update, grad, param, mars_old_grad):
    utils.mars_correction(update, mars_old_grad, group["mars_gamma"], utils.get_beta1(group))
    utils.copy_stochastic_list_(grad, update)
    return update


@zero_guard("exp_avg")
@no_state
def exp_avg(group, update, grad, param, exp_avg):
    return utils.scale_by_exp_avg_(exp_avg, update, utils.beta_debias(utils.get_beta1(group), group["step"]))


@copy_guard(2, "init")
@no_state
def weight_decay_to_init(group, update, grad, param, init):
    utils.stochastic_lerp_(param, init, group["weight_decay_to_ema"] * group["lr"])
    return update


def identity(state, group, update, grad, param):
    return update


@no_state
def apply_update(group, update, grad, param):
    utils.update_param_(
        param,
        update,
        group["lr"],
        group["weight_decay"],
        caution=group["caution"],
        cautious_decay=group.get("cautious_weight_decay", False),
        grad=grad,
    )
    return _SKIP


@zero_guard("exp_avg")
@no_state
def weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.weight_decay_to_ema_(
        param,
        exp_avg,
        utils.beta_debias(group["ema_beta"], group["step"]),
        group["weight_decay_to_ema"] * group["lr"],
    )
    return update


@zero_guard("exp_avg")
@no_state
def l1_weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.l1_weight_decay_to_ema_(
        param,
        exp_avg,
        utils.beta_debias(group["ema_beta"], group["step"]),
        group["weight_decay_to_ema"] * group["lr"],
    )
    return update


@zero_guard("exp_avg_sq")
@no_state
def scale_by_exp_avg_sq(group, update, grad, param, exp_avg_sq):
    return utils.scale_by_exp_avg_sq_(
        exp_avg_sq,
        update,
        utils.beta_debias(utils.get_beta2(group), group["step"]),
        group["eps"],
    )


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adam_(
        exp_avg,
        exp_avg_sq,
        update,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],  #
        group["eps"],
    )


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("mu_product", init_fn=_init_mu_product, skip_first=False)
@no_state
def scale_by_nadam(group, update, grad, param, exp_avg, exp_avg_sq, mu_product):
    utils.nadam_(
        param,
        exp_avg,
        exp_avg_sq,
        mu_product,
        update,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["momentum_decay"],
        group["eps"],
        group["weight_decay"],
        group.get("decoupled_weight_decay", False),
    )
    return update


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("mu_product", init_fn=_init_mu_product, skip_first=False)
@no_state
def update_by_nadam(group, update, grad, param, exp_avg, exp_avg_sq, mu_product):
    utils.fused_nadam_(
        param,
        exp_avg,
        exp_avg_sq,
        mu_product,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["momentum_decay"],
        group["weight_decay"],
        group.get("decoupled_weight_decay", False),
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adamc(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["lr"] * group["weight_decay"] / group["max_lr"],
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@zero_guard("exp_avg_fast", "exp_avg_slow", "exp_avg_sq")
@no_state
def scale_by_ademamix(group, update, grad, param, exp_avg_fast, exp_avg_slow, exp_avg_sq):
    return utils.ademamix_(
        exp_avg_fast,
        exp_avg_slow,
        exp_avg_sq,
        update,
        group["betas"],
        group["step"],
        group["eps"],
        group["alpha"],
        group.get("beta3_warmup"),
        group.get("alpha_warmup"),
    )


@zero_guard("exp_avg_fast", "exp_avg_slow", "exp_avg_sq")
@no_state
def update_by_ademamix(group, update, grad, param, exp_avg_fast, exp_avg_slow, exp_avg_sq):
    utils.fused_ademamix_(
        param,
        exp_avg_fast,
        exp_avg_slow,
        exp_avg_sq,
        update,
        grad,
        group["betas"],
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["alpha"],
        group["caution"],
        group.get("cautious_weight_decay", False),
        group.get("beta3_warmup"),
        group.get("alpha_warmup"),
    )
    return _SKIP


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.laprop_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group["step"])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_laprop_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["weight_decay"],
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@needs_full_param
@no_state
def orthogonalize_grad_to_param(group, update, grad, param):
    return utils.orthogonalize_grad_to_param(param, update, group["eps"])


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    # Compute weight_sum once per step, not per param in no-multi_tensor mode.
    if group.get("_sf_step") is not group["step"]:
        weight = abs(group["lr"]) ** group["weight_lr_power"] * group["step"].clamp(min=1) ** group["r"]
        group["weight_sum"] = group.get("weight_sum", 0) + weight
        group["_sf_step"] = group["step"]

    weight_sum = group["weight_sum"]
    weight = abs(group["lr"]) ** group["weight_lr_power"] * group["step"].clamp(min=1) ** group["r"]
    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    update, param, z, grad = utils.list_guard(update, param, z, grad)
    lr, ckp1, beta1 = utils.scalar_guard(group["lr"], ckp1, utils.get_beta1(group), grad[0])
    utils._compilable_schedule_free_(
        param,
        z,
        ckp1,
        update,
        lr,
        beta1,
        group["weight_decay"],
        grad,
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@needs_full_param
@copy_guard(2, "z")
@zero_guard("exp_avg")
@no_state
def update_by_msam(group, update, grad, param, z, exp_avg):
    utils.msam_(
        group["lr"],
        utils.beta_debias(utils.get_beta1(group), group["step"]),
        param,
        z,
        update,
        grad,
        exp_avg,
        group["caution"],
        group["weight_decay"],
        group["sam_step_size"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


def _adopt_warmup_1(state, group, update, grad, param, exp_avg, exp_avg_sq):
    utils.scale_by_exp_avg_sq_([exp_avg_sq], [update], 0, group["eps"])


def _adopt_warmup_2(state, group, update, grad, param, exp_avg, exp_avg_sq):
    u = utils.promote(update)
    easq = utils.promote(exp_avg_sq)
    utils.copy_stochastic_(exp_avg, u / easq.sqrt().clamp_(min=group["eps"]))
    utils.scale_by_exp_avg_sq_(
        [exp_avg_sq], [update], utils.beta_debias(utils.get_beta2(group), group["step"]), group["eps"]
    )


@zero_guard("exp_avg", "exp_avg_sq")
@warmup_guard(_adopt_warmup_1, _adopt_warmup_2)
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adopt_(
        param,
        update,
        grad,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


def _suds_warmup_1(state, group, update, grad, param, exp_avg, exp_avg_sq, fisher_approx):
    utils.copy_stochastic_(fisher_approx, update / update.norm().clamp(min=1e-8))


@needs_full_param
@zero_guard("exp_avg", "exp_avg_sq", "fisher_approx")
@warmup_guard(_suds_warmup_1)
@no_state_no_multi_tensor
def scale_by_suds(group, update, grad, param, exp_avg, exp_avg_sq, fisher_approx):
    precond_update, w = utils.eigvecs_product_rank1(update.flatten(), fisher_approx.flatten().to(update.dtype))
    precond_update = utils.adam_(
        exp_avg,
        exp_avg_sq,
        precond_update.view_as(exp_avg),
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
    )[0]
    precond_update, _ = utils.eigvecs_product_rank1(precond_update.flatten(), fisher_approx.flatten(), w)

    new_approx = utils.oja_update(fisher_approx.flatten().to(update.dtype), update.flatten(), group["precond_lr"])
    new_approx = new_approx.view_as(fisher_approx)
    utils.copy_stochastic_(fisher_approx, new_approx)
    return precond_update


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_unscaled_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    update = utils.unscaled_adam_(
        exp_avg,
        exp_avg_sq,
        update,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
    )
    return update


@zero_guard("exp_avg", "exp_avg_sq")
@warmup_guard(_adopt_warmup_1, _adopt_warmup_2)
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adopt(
        update,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
    )


def _init_psgd_kron(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    tmp = utils.get_temporary(group, param) or {}
    Q = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        tmp.get("hessian_vector"),
        tmp.get("vector"),
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["Q"] = utils.triu_to_line(Q) if group["store_triu_as_line"] else Q
    state["running_lower_bound"] = [torch.zeros((grad.shape[0],), device=q.device, dtype=torch.float64) for q in Q]
    state["step"] = torch.zeros((), device=param.device, dtype=torch.float64)
    if not cached:
        return

    state["Q_cache"] = [torch.empty_like(q) for q in Q]


def _init_psgd_eigen_kron(state, group, update, grad, param, prob: Optional[callable] = None):
    tmp = utils.get_temporary(group, param) or {}
    Q = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        tmp.get("hessian_vector"),
        tmp.get("vector"),
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["running_lower_bound"] = [torch.zeros((grad.shape[0],), device=q.device, dtype=torch.float64) for q in Q]
    state["step"] = torch.zeros((), device=param.device, dtype=torch.float64)

    _update_psgd_precond(
        False,
        None,
        group,
        param,
        update,
        Q,
        state["running_lower_bound"],
        state["step"],
        prob,
        store_triu_as_line=False,
    )
    state["Q"] = utils.triu_to_line(Q) if group["store_triu_as_line"] else Q
    state["Q_basis"] = utils.init_psgd_eigenbasis(Q)


def _init_psgd_pro_kron(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        None,
        None,
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["Q"] = Q
    state["running_lower_bound"] = [torch.zeros((grad.shape[0],), device=q.device, dtype=torch.float64) for q in Q]
    state["step"] = torch.zeros((), device=param.device, dtype=torch.float64)
    if not cached:
        return
    state["Q_cache"] = [torch.empty_like(q) for q in Q]


def _init_psgd_lra(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    tmp = utils.get_temporary(group, param) or {}
    state["U"], state["V"], state["d"] = utils.init_lra(
        grad,
        group["param_count"],
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["rank"],
        tmp.get("hessian_vector"),
        tmp.get("vector"),
        dtype=getattr(torch, group["q_dtype"]),
    )


@needs_full_param
@no_state_no_multi_tensor
def orthogonalize_update(group, update, grad, param, scale_mode: str = "scale"):  # explore scale_mode="graft"
    if update.dim() < 2:
        return update
    original_shape = update.shape
    # doing it this way, as tmp and update are not guaranteed to share memory address or layout
    tmp = update.flatten(1, -1)
    utils.inplace_orthogonal_(tmp, out=tmp, scale_mode=scale_mode)
    return tmp.reshape(original_shape)


@zero_guard("momentum")
@no_state
def nesterov_momentum(group, updates, grads, params, momentum):
    return utils.nesterov_momentum(momentum, updates, utils.get_beta1(group))


@zero_guard("momentum")
@no_state
def nesterov_ema(group, updates, grads, params, momentum):  # equivalent to Grokfast
    return utils.nesterov_ema(momentum, updates, utils.get_beta1(group))


def _store_init_norm(state, group, update, grad, param):
    state["init_norm"] = param.to(_storage_dtype(group)).norm()


@needs_full_param
@general_guard("init_norm", init_fn=_store_init_norm, skip_first=False)
@no_state
def update_by_hyperball(group, update, grad, param, init_norm):
    utils.hyperball_step_(
        param,
        update,
        init_norm,
        group["lr"],
        group["weight_decay"],
        group["caution"],
        grad,
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


def _store_std(state, group, update, grad, param):
    state["init_std"] = torch.std(param.to(_storage_dtype(group)))


@needs_full_param
@general_guard("init_std", init_fn=_store_std, skip_first=False)
@no_state
def mup_approx(group, updates, grads, params, init_std):
    _updates = [(u, i) for u, i in zip(updates, init_std) if u.ndim > 1]
    _updates, _init_std = zip(*_updates)
    utils.stochastic_multiply_(_updates, _init_std)
    return updates


def _init_delta(state, group, update, grad, param, log_space: bool):
    val = group["initial_d"]
    state["delta"] = torch.full((), math.log(val) if log_space else val, dtype=param.dtype, device=param.device)


def _init_full_delta(state, group, update, grad, param, log_space: bool):
    val = group["initial_d"]
    state["delta"] = torch.full_like(param, math.log(val) if log_space else val)


@needs_full_param
@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_delta, log_space=False), skip_first=False)
@no_state
def scale_by_d_adaptation(group, update, grad, param, state, delta):
    utils.d_adaptation(grad, update, state, delta)
    return update


@needs_full_param
@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_delta, log_space=True), skip_first=False)
@no_state
def scale_by_lr_adaptation(group, update, grad, param, state, delta):
    utils.lr_adaptation(grad, update, state, delta, group["lr_lr"])
    return update


@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_full_delta, log_space=True), skip_first=False)
@no_state
def scale_by_pointwise_lr_adaptation(group, update, grad, param, state, delta):
    utils.pointwise_lr_adaptation(grad, update, state, delta, group["lr_lr"])
    return update


@zero_guard("momentum")
@no_state
def heavyball_momentum(group, updates, grads, params, momentum):
    return utils.heavyball_momentum(momentum, updates, utils.get_beta1(group))


def _init_scion_state(state, group, update, grad, param):
    state["scion_state"] = {"initialized": False}


@needs_full_param
@general_guard("scion_state", init_fn=_init_scion_state, skip_first=False)
@no_state
def scion_auto_norm(group, update, grad, param, scion_state):
    scale = group.get("scale", 1.0)
    param_ids = {id(p): i for i, p in enumerate(group["params"])}
    for ctx, p in zip(scion_state, param):
        if not ctx["initialized"]:
            utils.scion_auto_init_param_(p, scale, seed=param_ids.get(id(p), 0))
            ctx["initialized"] = True
    return utils.scion_auto_lmo_(update, scale)


def _init_soap(state, group, update, grad, param):
    utils.init_preconditioner(
        grad, state, group["max_precond_dim"], group["precondition_1d"], group.get("init_factor", 0.0)
    )


def _apply_soap_preconditioner(group, update, Q, GG, *references, use_kl: bool = False, eps=1e-8):
    beta = utils.beta_debias(group["shampoo_beta"], group["step"])
    max_dim, p1d = group["max_precond_dim"], group["precondition_1d"]
    for upd, q, gg, *ref in zip(update, Q, GG, *references):
        g = utils.promote(upd)
        if use_kl:
            utils.update_ggt_kl(g, gg, q, max_dim, p1d, beta, eps)
        else:
            utils.update_ggt(g, gg, max_dim, p1d, beta)
        if group["is_preconditioning"]:
            utils.get_orthogonal_matrix_QR(gg, q, *ref)


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.adam_(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_kl_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.adam_(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg, use_kl=True, eps=group["eps"])
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_kl_shampoo(group, update, grad, param, exp_avg, Q, GG):
    utils.stochastic_lerp_(exp_avg, update, 1 - utils.get_beta1(group))
    precond = [utils.kl_shampoo_precondition(e, q, gg, group["eps"]) for e, q, gg in zip(exp_avg, Q, GG)]
    dampening = group.get("dampening", 0.0)
    accum = [utils.dampen_grad(u, dampening)[1] for u in update] if dampening > 0 else update
    _apply_soap_preconditioner(group, accum, Q, GG, use_kl=True, eps=group["eps"])
    for gg in GG:
        factors = [m for m in gg if isinstance(m, torch.Tensor)]
        if len(factors) >= 2:
            utils.psgd_balance_Q(factors)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("mu_product", init_fn=_init_mu_product, skip_first=False)
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap_nadam(group, update, grad, param, exp_avg, exp_avg_sq, mu_product, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.nadam_(
        grad_projected,
        exp_avg,
        exp_avg_sq,
        mu_product,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["momentum_decay"],
        group["eps"],
        0.0,
        False,
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap_laprop(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.laprop_(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg_fast", "exp_avg_slow", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap_ademamix(group, update, grad, param, exp_avg_fast, exp_avg_slow, exp_avg_sq, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.ademamix_(
        exp_avg_fast,
        exp_avg_slow,
        exp_avg_sq,
        grad_projected,
        group["betas"],
        group["step"] - 1,
        group["eps"],
        group["alpha"],
        group.get("beta3_warmup"),
        group.get("alpha_warmup"),
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg_slow, exp_avg_fast)
    return precond


def _fill_q_cache(Q_cache, Q):
    for i, (c_, q_) in enumerate(zip(Q_cache, Q)):
        if c_ is None:
            Q_cache[i] = c_ = torch.empty_like(q_)
        if q_.ndim == 3:
            torch.matmul(q_.mT, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)


def _update_psgd_precond(
    cached,
    Q_cache,
    group,
    param,
    grad,
    Q,
    running_lower_bound,
    step,
    prob: Optional[callable] = None,
    store_triu_as_line: Optional[bool] = None,
) -> None:
    if prob is None:
        prob = utils.precond_update_prob_schedule()
    if store_triu_as_line is None:
        store_triu_as_line = group["store_triu_as_line"]

    if not group["is_preconditioning"]:
        return

    if (utils.get_temporary(group, param) or {}).get("vector") is None:
        vector, hessian_vector = utils.dampen_grad(grad, group["dampening"])
    else:
        vector, hessian_vector = utils.take_temporary(group, param, "vector", "hessian_vector")

    utils.psgd_update_precond(
        hessian_vector,
        group["precond_lr"],
        Q,
        store_triu_as_line,
        utils.get_beta2(group),
        vector,
        running_lower_bound,
        group["lower_bound_beta"],
        group["precond_update_power_iterations"],
    )
    del vector, hessian_vector

    if isinstance(prob, float):
        float_prob = prob
    else:
        float_prob = prob(group["step"])
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if should_use_cache:
        _fill_q_cache(Q_cache, utils.line_to_triu(Q) if store_triu_as_line else Q)


def _update_psgd_pro_precond(
    cached,
    Q_cache,
    group,
    param,
    grad,
    Q,
    running_lower_bound,
    step,
    prob: Optional[callable] = None,
) -> None:
    if prob is None:
        prob = utils.precond_update_prob_schedule()

    if not group["is_preconditioning"]:
        return

    utils.psgd_pro_update_precond(
        grad,
        group["precond_lr"],
        Q,
        running_lower_bound,
        group["lower_bound_beta"],
        group["precond_update_power_iterations"],
        group["dampening"],
    )

    if isinstance(prob, float):
        float_prob = prob
    else:
        float_prob = prob(group["step"])
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if should_use_cache:
        _fill_q_cache(Q_cache, Q)


def _cached_psgd_precond_grad(group, update, Q, Q_cache, grad):
    kwargs = {"ea": update, "caution": group["caution"], "grad": grad}
    if group.get("is_cached", False) and Q_cache[0] is not None:
        out = utils.precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        out = utils.psgd_precond_grad(preconds=Q, store_triu_as_line=group["store_triu_as_line"], **kwargs)
    group["caution"] = False  # we already cautioned here - shouldn't do it again
    return out


def _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache):
    kwargs = {
        "ea": update,
        "caution": group["caution"],
        "grad": grad,
        "param": param,
        "lr": group["lr"],
        "decay": group["weight_decay"],
        "cautious_decay": group.get("cautious_weight_decay", False),
    }
    if group.get("is_cached", False) and Q_cache[0] is not None:
        utils.fused_precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        utils.fused_psgd_precond_grad(preconds=Q, store_triu_as_line=group["store_triu_as_line"], **kwargs)


def _update_lra(
    group, U: List[Tensor], V: List[Tensor], d: List[Tensor], params: List[Tensor], grads: List[Tensor], delayed: bool
):
    if not group["is_preconditioning"]:
        return utils.multi_flatten((U, 1), (V, 1), (d, 0))

    if (utils.get_temporary(group, params[0]) or {}).get("hessian_vector") is not None:
        vector_hv = [utils.take_temporary(group, p, "vector", "hessian_vector") for p in params]
        vector = utils.flatten([v for v, _ in vector_hv])
        hessian_vector = utils.flatten([hv for _, hv in vector_hv])
    else:
        vector, hessian_vector = utils.dampen_multiple(grads)
    precond_step = group["precond_step"] = group.get("precond_step", -1) + 1
    return utils.update_lra_precond_(
        U, V, d, vector, hessian_vector, group["eps"], group["precond_lr"], delayed, bool(precond_step % 2)
    )


@needs_full_param
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@needs_full_param
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    utils.apply_lra_update(
        param,
        update,
        u,
        v,
        d,
        group["lr"],
        group["weight_decay"],
        group["caution"],
        grad,
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@needs_full_param
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@needs_full_param
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    utils.apply_lra_update(
        param,
        update,
        u,
        v,
        d,
        group["lr"],
        group["weight_decay"],
        group["caution"],
        grad,
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_multi_tensor
def scale_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "Q_basis", "running_lower_bound", "step", init_fn=_init_psgd_eigen_kron, skip_first=True)
@no_state_no_multi_tensor
def scale_by_lather(
    group,
    update,
    grad,
    param,
    update_to_precond,
    exp_avg,
    exp_avg_sq,
    Q,
    Q_basis,
    running_lower_bound: List[Tensor],
    step: Tensor,
    prob: Optional[callable] = None,
):
    projected = utils.project(utils.promote(update), Q_basis, False)
    precond = utils.adam_(
        exp_avg,
        exp_avg_sq,
        projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["eps"],
    )[0]
    precond = utils.project(precond, Q_basis, True)

    if group["is_preconditioning"]:
        _update_psgd_precond(False, None, group, param, update_to_precond, Q, running_lower_bound, step, prob)
        utils.update_psgd_eigenbasis(
            utils.line_to_triu(Q) if group["store_triu_as_line"] else Q,
            Q_basis,
            exp_avg,
        )

    return precond


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_multi_tensor
def scale_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    precond = _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    return precond


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_multi_tensor
def update_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    return _SKIP


@needs_full_param
@no_state
def sign(group, update, grad, param, graft: bool = True):
    return utils.sign_(update, graft)


@no_state
def global_clip(group, update, grad, param, clip_fn: Optional[callable] = None):
    assert clip_fn is not None
    return clip_fn(update)


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_multi_tensor
def update_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    return _SKIP


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_pro_kron, skip_first=False)
@no_state_no_multi_tensor
def scale_by_psgd_pro(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_pro_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", "step", init_fn=_init_psgd_pro_kron, skip_first=False)
@no_state_no_multi_tensor
def update_by_psgd_pro(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_pro_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, step, prob)
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    return _SKIP


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group["step"] ** -group["beta2_scale"]
    betas = group["betas"]
    group["betas"] = (utils.get_beta1(group), beta2, *betas[2:])
    return update


def apply_to_idx(fn, idx):
    name = fn
    if isinstance(fn, str):
        fn = getattr(utils, fn, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown function '{name}'")

    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        return fn(args[idx])

    _fn.__name__ = _fn.__qualname__ = f"apply_{getattr(fn, '__name__', repr(fn))}_to_{idx}"
    return _fn


_FSDP_HEADER_WIDTH = 4
_FSDP_BUCKET_BYTES = 32 << 20
_FSDP_DTYPE_CODES = {
    torch.float64: 0,
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}


class _ShapeInfo:
    __slots__ = ("orig_shape", "offset", "total", "group", "owner", "param_idx")

    def __init__(self, orig_shape, offset=0, total=None, group=None, owner=None, param_idx=None):
        self.orig_shape = orig_shape
        self.offset = offset
        self.total = total if total is not None else math.prod(orig_shape)
        self.group = group
        self.owner = owner
        self.param_idx = param_idx


class _FSDPBucket:
    __slots__ = ("device", "dtype", "send_entries", "send_splits", "recv_entries", "recv_splits")

    def __init__(self, device, dtype, send_entries, send_splits, recv_entries, recv_splits):
        self.device = device
        self.dtype = dtype
        self.send_entries = send_entries
        self.send_splits = send_splits
        self.recv_entries = recv_entries
        self.recv_splits = recv_splits


class _FSDPState:
    __slots__ = ("items", "buckets")

    def __init__(self, items, buckets):
        self.items = items
        self.buckets = buckets


def _dtype_code(dtype):
    if dtype not in _FSDP_DTYPE_CODES:
        raise TypeError(f"Unsupported FSDP shard dtype: {dtype}")
    return _FSDP_DTYPE_CODES[dtype]


def _assign_fsdp_owners(entries, shard_sizes, world_size):
    loads = [0] * world_size
    owners = []
    for i, (p, _, total, _) in enumerate(entries):
        active = shard_sizes[i].nonzero().squeeze(-1).tolist()
        candidates = active or list(range(world_size))
        owner = min(candidates, key=loads.__getitem__)
        loads[owner] += total * p.element_size()
        owners.append(owner)
    return owners


def _detect_orig_shapes(params):
    fsdp_ids = {id(p) for p in params if getattr(p, "_fsdp_flattened", False)}
    if not fsdp_ids:
        return {}
    try:
        from torch.distributed.fsdp._flat_param import FlatParameter
    except ImportError:
        return {}

    import gc

    lookup = {}
    for obj in gc.get_objects():
        if not isinstance(obj, FlatParameter):
            continue
        if not hasattr(obj, "_shard_param_infos") or obj._params is None:
            continue
        for param, spi, shape in zip(obj._params, obj._shard_param_infos, obj._shapes):
            lookup[id(param)] = (tuple(shape), spi)

    # optimizer param order is stable across ranks
    fsdp_entries = [
        (p, s, math.prod(s), spi)
        for p in params
        for s, spi in [lookup.get(id(p), (None, None))]
        if id(p) in fsdp_ids and s is not None
    ]
    result = {}
    ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if ws > 1 and fsdp_entries:
        rank = torch.distributed.get_rank()
        shard_sizes = torch.zeros(len(fsdp_entries), ws, dtype=torch.int64, device=fsdp_entries[0][0].device)
        for i, (p, _, _, spi) in enumerate(fsdp_entries):
            shard_sizes[i, rank] = p.numel() if spi.in_shard else 0
        torch.distributed.all_reduce(shard_sizes)
        owners = _assign_fsdp_owners(fsdp_entries, shard_sizes, ws)
    else:
        owners = [None] * len(fsdp_entries)
    for param_idx, ((p, orig, total, spi), owner) in enumerate(zip(fsdp_entries, owners)):
        offset = 0 if spi.intra_param_start_idx is None else spi.intra_param_start_idx
        result[id(p)] = _ShapeInfo(orig, offset, total, owner=owner, param_idx=param_idx)
    return result


def _exchange_split_sizes(splits, device):
    send = torch.tensor(splits, dtype=torch.int64, device=device)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send)
    return recv.tolist()


def _all_to_all_variable(sendbuf, recv_splits, send_splits):
    recv = sendbuf.new_empty(sum(recv_splits))
    torch.distributed.all_to_all_single(recv, sendbuf, output_split_sizes=recv_splits, input_split_sizes=send_splits)
    return recv


def _fsdp_bucket_schedule(items):
    buckets, current, lookup = [], {}, {}
    for p, info, _ in items:
        key = (p.device, p.dtype)
        idx = current.get(key)
        size = info.total * p.element_size()
        if idx is None or (buckets[idx][2] and buckets[idx][2] + size > _FSDP_BUCKET_BYTES):
            idx = len(buckets)
            buckets.append([p.device, p.dtype, 0])
            current[key] = idx
        buckets[idx][2] += size
        lookup[info.param_idx] = idx
    return [(device, dtype) for device, dtype, _ in buckets], lookup


def _exchange_fsdp_shards(schedule, bucket_lookup, items, tensor_getter, keep_state=False):
    ws = torch.distributed.get_world_size()
    per_bucket = [[] for _ in schedule]
    for p, info, shard in items:
        tensor = tensor_getter(p, shard)
        if tensor is None or tensor.numel() == 0:
            continue
        flat = tensor.reshape(-1)
        bucket_idx = bucket_lookup[info.param_idx]
        device, dtype = schedule[bucket_idx]
        if flat.device != device or flat.dtype != dtype:
            raise RuntimeError(
                f"FSDP bucket mismatch for param {info.param_idx}: expected {(device, dtype)}, got {(flat.device, flat.dtype)}"
            )
        per_bucket[bucket_idx].append((info.owner, info.param_idx, info.offset, flat, shard))

    received, states = {}, []
    for (device, dtype), bucket_entries in zip(schedule, per_bucket):
        by_dst = [[] for _ in range(ws)]
        for entry in bucket_entries:
            by_dst[entry[0]].append(entry)

        send_meta_splits = [len(dst_entries) * _FSDP_HEADER_WIDTH for dst_entries in by_dst]
        send_payload_splits = [sum(flat.numel() for _, _, _, flat, _ in dst_entries) for dst_entries in by_dst]
        recv_meta_splits = _exchange_split_sizes(send_meta_splits, device)
        recv_payload_splits = _exchange_split_sizes(send_payload_splits, device)

        code = _dtype_code(dtype)
        meta = [
            value
            for dst_entries in by_dst
            for _, param_idx, offset, flat, _ in dst_entries
            for value in (param_idx, offset, flat.numel(), code)
        ]
        payload = [flat for dst_entries in by_dst for _, _, _, flat, _ in dst_entries]
        send_meta = (
            torch.tensor(meta, dtype=torch.int64, device=device)
            if meta
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        send_payload = torch.cat(payload) if payload else torch.empty(0, dtype=dtype, device=device)

        recv_meta = _all_to_all_variable(send_meta, recv_meta_splits, send_meta_splits)
        recv_entries = [[] for _ in range(ws)]
        meta_offset = 0
        for src, count in enumerate(recv_meta_splits):
            if count == 0:
                continue
            if count % _FSDP_HEADER_WIDTH:
                raise RuntimeError(f"Malformed FSDP metadata split: {count}")
            rows = recv_meta[meta_offset : meta_offset + count].view(-1, _FSDP_HEADER_WIDTH).cpu().tolist()
            meta_offset += count
            for param_idx, offset, length, got in rows:
                if got != code:
                    raise RuntimeError(f"FSDP dtype mismatch for bucket {dtype}: expected {code}, got {got}")
                recv_entries[src].append((param_idx, offset, length))

        recv_payload = _all_to_all_variable(send_payload, recv_payload_splits, send_payload_splits)
        payload_offset = 0
        for src_entries in recv_entries:
            for param_idx, offset, length in src_entries:
                chunk = recv_payload[payload_offset : payload_offset + length]
                received.setdefault(param_idx, []).append((offset, chunk))
                payload_offset += length
        if payload_offset != recv_payload.numel():
            raise RuntimeError("FSDP payload unpack mismatch")

        if keep_state:
            states.append(_FSDPBucket(device, dtype, by_dst, send_payload_splits, recv_entries, recv_payload_splits))

    return received, states


def _reshape_fsdp_params(items):
    rank = torch.distributed.get_rank()
    schedule, bucket_lookup = _fsdp_bucket_schedule(items)
    params, buckets = _exchange_fsdp_shards(schedule, bucket_lookup, items, lambda _, shard: shard, keep_state=True)
    grads, _ = _exchange_fsdp_shards(schedule, bucket_lookup, items, lambda p, _: p.grad)

    for p, info, shard in items:
        p.grad = None
        if info.owner != rank:
            continue

        pieces = params.get(info.param_idx, ())
        total = sum(chunk.numel() for _, chunk in pieces)
        if total != info.total:
            raise RuntimeError(f"FSDP parameter assembly mismatch for param {info.param_idx}: {total} != {info.total}")

        full = shard.new_empty(info.total)
        for offset, chunk in pieces:
            full[offset : offset + chunk.numel()].copy_(chunk)
        p.data = full.view(info.orig_shape)

        grad_pieces = grads.get(info.param_idx, ())
        if not grad_pieces:
            continue
        grad_total = sum(chunk.numel() for _, chunk in grad_pieces)
        if grad_total != info.total:
            raise RuntimeError(f"FSDP grad assembly mismatch for param {info.param_idx}: {grad_total} != {info.total}")
        grad = full.new_empty(info.total, dtype=grad_pieces[0][1].dtype)
        for offset, chunk in grad_pieces:
            grad[offset : offset + chunk.numel()].copy_(chunk)
        p.grad = grad.view(info.orig_shape)

    return _FSDPState(items, buckets)


def _restore_fsdp_params(state):
    by_param = {info.param_idx: (p, info, shard) for p, info, shard in state.items}
    for bucket in state.buckets:
        payload = []
        for dst, recv_entries in enumerate(bucket.recv_entries):
            for param_idx, offset, length in recv_entries:
                p, info, _ = by_param[param_idx]
                flat = p.data.reshape(-1)
                if flat.numel() != info.total:
                    raise RuntimeError(f"FSDP return path expects full param {param_idx}, got {flat.numel()}")
                payload.append(flat[offset : offset + length])
        send_payload = torch.cat(payload) if payload else torch.empty(0, dtype=bucket.dtype, device=bucket.device)
        recv_payload = _all_to_all_variable(send_payload, bucket.send_splits, bucket.recv_splits)

        payload_offset = 0
        for send_entries in bucket.send_entries:
            for _, _, _, flat, shard in send_entries:
                shard.copy_(recv_payload[payload_offset : payload_offset + flat.numel()])
                payload_offset += flat.numel()
        if payload_offset != recv_payload.numel():
            raise RuntimeError("FSDP return payload unpack mismatch")

    for p, _, shard in state.items:
        p.data = shard
        p.grad = None


def _view_param(p, shape):
    p.data = p.data.view(shape)
    if p.grad is not None:
        p.grad = p.grad.view(shape)


def _reshape_params(params, orig_shapes, gather=True):
    if not orig_shapes:
        return [], []
    dist_ready = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    views, gathers = [], []

    for p in params:
        info = orig_shapes.get(id(p))
        if info is None:
            continue

        if gather and dist_ready and info.owner is not None:
            shard = p.data
            gathers.append((p, info, shard))
            continue

        if p.data.shape == info.orig_shape:
            continue

        orig, numel = info.orig_shape, p.data.numel()
        if numel == info.total:
            target = orig
        elif numel > 0 and len(orig) >= 2:
            inner = math.prod(orig[1:])
            target = (numel // inner, *orig[1:]) if numel % inner == 0 else None
        else:
            continue
        if target is not None:
            flat = p.data.shape
            _view_param(p, target)
            views.append((p, flat))

    if gathers:
        gathers = _reshape_fsdp_params(gathers)

    return views, gathers


def _restore_params(views, gathers):
    if isinstance(gathers, _FSDPState):
        _restore_fsdp_params(gathers)
    else:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for p, info, shard in gathers:
            if rank == info.owner:
                full = p.data.flatten()
            else:
                full = shard.new_empty(info.total)
            torch.distributed.broadcast(full, src=info.owner, group=info.group)
            shard.copy_(full[info.offset : info.offset + shard.numel()])
            p.data = shard
            p.grad = None
    for p, flat in views:
        _view_param(p, flat)


def _inner_chain(state, group, update, grad, param, *fns):
    skip_update = False
    for fn in fns:
        new = fn(state, group, update, grad, param)
        if new is _SKIP:
            skip_update = True
            continue
        if new is None:
            update = None
            break
        update = new
    return update, skip_update


def chain(state: list, group, grad, param, *fns):
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]

    ecc = ECCConfig.from_group(group, key="param_ecc")
    if ecc is None:
        update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
        if skip_update or update is None:
            return
        utils.update_param_(
            param,
            update,
            group["lr"],
            group["weight_decay"],
            caution=group["caution"],
            cautious_decay=group.get("cautious_weight_decay", False),
            grad=grad,
        )
        return

    corrs = [st["param::ecc"] for st in state]
    with ecc.attached(param, corrs):
        update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
        if not skip_update and update is not None:
            utils.update_param_(
                param,
                update,
                group["lr"],
                group["weight_decay"],
                caution=group["caution"],
                cautious_decay=group.get("cautious_weight_decay", False),
                grad=grad,
            )


def _walk_fns(obj):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, FunctionTransform):
            yield cur
            stack.append(cur.fn)
        elif isinstance(cur, functools.partial):
            stack.append(cur.func)
        elif isinstance(cur, Parallel):
            for branch in cur.branches:
                stack.extend(branch)
        elif isinstance(cur, Route):
            for _, fns in cur.routes:
                stack.extend(fns)
            if cur.default is not None:
                stack.extend(cur.default)
        elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
            stack.extend(cur)


def _walk_fns_with_bucket(obj):
    stack = [(obj, False)]
    while stack:
        cur, ub = stack.pop()
        if isinstance(cur, FunctionTransform):
            yield cur, ub
            stack.append((cur.fn, ub or isinstance(cur, BucketGuard)))
        elif isinstance(cur, functools.partial):
            stack.append((cur.func, ub))
        elif isinstance(cur, Parallel):
            for branch in cur.branches:
                stack.extend((b, ub) for b in branch)
        elif isinstance(cur, Route):
            for _, fns in cur.routes:
                stack.extend((f, ub) for f in fns)
            if cur.default is not None:
                stack.extend((f, ub) for f in cur.default)
        elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
            stack.extend((c, ub) for c in cur)


def set_indices(fns: Iterable[callable], retain: bool = True, offset: int = 0):
    if retain and offset:
        raise ValueError("offset cannot be retained")

    if retain:
        offset = max((ft.transform_idx for ft in _walk_fns(fns) if ft.transform_idx is not None), default=-1) + 1

    new_fns = [copy.deepcopy(fn) for fn in fns]
    for ft, under_bucket in _walk_fns_with_bucket(new_fns):
        if not retain or ft.transform_idx is None:
            ft.transform_idx, offset = offset, offset + 1
        ft._under_bucket = under_bucket
        ft._build_val_names()

    return new_fns


class ChainOpt(utils.StatefulOptimizer):
    promote: bool = False
    global_defaults = {
        "caution": False,
        "lr": 1,
        "warmup_steps": 0,
        "weight_decay": 0,
        "eps": 1e-8,
    }

    def __init__(self, params, defaults, *fns):
        orig = defaults.pop("orig_shapes", None)
        self._orig_shapes = (
            {k: _ShapeInfo(v) if isinstance(v, tuple) else v for k, v in orig.items()} if orig is not None else None
        )
        base = self.global_defaults.copy()
        base.update({k: v for k, v in defaults.items() if v is not use_default})
        super().__init__(params, base)
        self.fns = fns
        self._eager_chain = self._run_chain
        if self.compile_step:
            self._run_chain = torch.compile(self._run_chain, fullgraph=True)
        self.register_load_state_dict_post_hook(ChainOpt._restore_ecc_dtypes)
        self._init_param_ecc()

    def _init_param_ecc(self):
        for group in self.param_groups:
            self._init_param_ecc_group(group)

    def _init_param_ecc_group(self, group):
        ecc = ECCConfig.from_group(group, key="param_ecc")
        if ecc is None:
            return
        for p in group["params"]:
            fp32 = None
            if p.dtype != ecc.primary_dtype:
                fp32 = p.data.float()
                p.data = p.data.to(ecc.primary_dtype)
                old_views = self.mapping.pop(p, ())
                for ov in old_views:
                    self.mapping_inverse.pop(utils._tensor_key(ov), None)
            if p not in self.mapping:
                self.mapping[p] = p_views = utils.merge_group(group, p)
                for i, pv in enumerate(p_views):
                    self.mapping_inverse[utils._tensor_key(pv)] = (p, i)
            fp32_views = None if fp32 is None else utils.merge_group(group, fp32)
            for i, pv in enumerate(self.mapping[p]):
                st = self.state_(pv)
                if "param::ecc" in st:
                    continue
                st["param::ecc"] = torch.zeros_like(pv, dtype=ecc.corr_dtype)
                if fp32_views is not None:
                    ecc.init_correction(st["param::ecc"], fp32_views[i], pv.data)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        if not hasattr(self, "mapping"):
            return
        self._init_param_ecc_group(self.param_groups[-1])

    @staticmethod
    def _restore_ecc_dtypes(optimizer, *args):
        for group in optimizer.param_groups:
            ecc = ECCConfig.from_group(group, key="ecc")
            param_ecc = ECCConfig.from_group(group, key="param_ecc")
            if ecc is None and param_ecc is None:
                continue
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                for idx_state in optimizer.state[p].values():
                    if not isinstance(idx_state, dict):
                        continue
                    for k in list(idx_state.keys()):
                        v = idx_state[k]
                        if not isinstance(v, torch.Tensor):
                            continue
                        is_param_key = k == "param" or k.startswith("param::")
                        cfg = param_ecc if is_param_key else ecc
                        if cfg is None:
                            continue
                        if k.endswith("::ecc"):
                            idx_state[k] = v.to(cfg.corr_dtype)
                        elif (k + "::ecc") in idx_state:
                            idx_state[k] = v.to(cfg.primary_dtype)
                if param_ecc is not None and p.dtype != param_ecc.primary_dtype:
                    orig_key = utils._tensor_key(p)
                    p.data = p.data.to(param_ecc.primary_dtype)
                    bf16_key = utils._tensor_key(p)
                    if orig_key in optimizer.mapping_inverse:
                        optimizer.mapping_inverse[bf16_key] = optimizer.mapping_inverse.pop(orig_key)
            if param_ecc is not None:
                optimizer._init_param_ecc_group(group)

    @property
    def fns(self):
        return self._fns

    @fns.setter
    def fns(self, value):
        self._fns = value
        self._set_indices(retain=True)
        self._needs_gather = any(getattr(ft, "needs_full_param", False) for ft in _walk_fns(self._fns))
        self._transform_ids = frozenset(
            ft.transform_idx
            for ft in _walk_fns(self._fns)
            if ft.transform_idx is not None and getattr(ft, "needs_init", True) and not ft._under_bucket
        )

    def _set_indices(self, retain=True):
        self._fns = set_indices(self.fns, retain)

    def _find_val_name(self, name):
        for ft in _walk_fns(self._fns):
            if name in ft._val_names:
                return ft._val_names[name]
        raise KeyError(f"No transform stores '{name}'")

    def _step(self, group):
        if "base_lr" not in group:
            group["base_lr"] = group["lr"]
        if "base_lr" in group and group["base_lr"] != group["lr"]:
            utils.warn_once(
                f"Learning rate changed between steps. This is an experimental feature and "
                f"only supported with multi_tensor=True (currently multi_tensor={group['multi_tensor']})."
            )
            group["base_lr"] = group["lr"]

        if self._orig_shapes is None:
            all_params = [p for g in self.param_groups for p in g["params"]]
            self._orig_shapes = _detect_orig_shapes(all_params)

        views, gathers = _reshape_params(group["params"], self._orig_shapes, self._needs_gather)
        group["_orig_shapes"] = self._orig_shapes
        try:
            self._step_inner(group)
        finally:
            _restore_params(views, gathers)

    def _step_inner(self, group):
        caution = group["caution"]

        vals = list(self.split_p_and_g_in_group(group, should_promote=self.promote))

        if not vals:
            return
        p, g = zip(*vals)

        step = group.get("_group_step")
        if step is None:
            for param in group["params"]:
                param_state = self.state.get(param)
                if not isinstance(param_state, dict):
                    continue
                for idx_state in param_state.values():
                    if isinstance(idx_state, dict) and "step" in idx_state:
                        step = idx_state["step"]
                        break
                if step is not None:
                    break
            else:
                step = 0
        if isinstance(step, torch.Tensor):
            step = step.to(device=p[0].device, dtype=torch.int64)
        else:
            step = utils.scalar_guard(step, p[0])
        group["_group_step"] = group["step"] = step = step + 1
        self.state_(p[0])["step"] = step
        group["prev_lr"] = group["lr"] = group["base_lr"] * step / step.clamp(min=group["warmup_steps"] + 1)
        try:
            if not group["multi_tensor"] or len(p) == 1:
                for param, grad in zip(p, g):
                    self._chain(group, [grad], [param], caution)
            else:
                self._chain(group, g, p, caution)
        finally:
            group["caution"] = caution
            group["lr"] = group["base_lr"]
            group["step"] = None

    def _run_chain(self, state, group, g, p, caution):
        chain(state, group, g, p, *self.fns)

    def _needs_init(self, state):
        ids = self._transform_ids
        if not ids:
            return False
        all_initialized = set()
        for st in state:
            all_initialized.update(st.get("is_initialized", ()))
        return not ids.issubset(all_initialized)

    def _needs_eager(self, group, state):
        if self._needs_init(state):
            return True
        if group.get("is_preconditioning", False):
            return True
        if group.get("ecc") or group.get("param_ecc"):
            return True
        return False

    def _chain(self, group, g, p, caution):
        state = [self.state_(pi) for pi in p]
        fn = self._run_chain
        if self.compile_step and self._needs_eager(group, state):
            fn = self._eager_chain
        fn(state, group, g, p, caution)


str_or_fn = Union[str, callable, None, Literal[use_default]]


def default(a, b):
    return b if a is use_default else a


# not supported: update_by_schedule_free, scale_by_soap, scale_by_exp_avg_sq
_scale_to_update_map = {
    scale_by_delayed_psgd.get_fn(): update_by_delayed_psgd,  #
    scale_by_psgd.get_fn(): update_by_psgd,  #
    scale_by_psgd_lra.get_fn(): update_by_psgd_lra,  #
    scale_by_delayed_psgd_lra.get_fn(): update_by_delayed_psgd_lra,  #
    scale_by_adam.get_fn(): update_by_adam,  #
    scale_by_nadam.get_fn(): update_by_nadam,  #
    scale_by_laprop.get_fn(): update_by_laprop,  #
    scale_by_adopt.get_fn(): update_by_adopt,  #
    scale_by_ademamix.get_fn(): update_by_ademamix,  #
    scale_by_psgd_pro.get_fn(): update_by_psgd_pro,  #
}
_scale_to_update_map_inv = {
    update_by_delayed_psgd.get_fn(): scale_by_delayed_psgd,  #
    update_by_psgd.get_fn(): scale_by_psgd,  #
    update_by_psgd_lra.get_fn(): scale_by_psgd_lra,  #
    update_by_delayed_psgd_lra.get_fn(): scale_by_delayed_psgd_lra,  #
    update_by_adam.get_fn(): scale_by_adam,  #
    update_by_nadam.get_fn(): scale_by_nadam,  #
    update_by_laprop.get_fn(): scale_by_laprop,  #
    update_by_adopt.get_fn(): scale_by_adopt,  #
    update_by_ademamix.get_fn(): scale_by_ademamix,  #
    update_by_psgd_pro.get_fn(): scale_by_psgd_pro,  #
}


class BaseOpt(ChainOpt):
    """
    Base Optimizer

    compile_step: bool = False
    Whether to torch.compile the optimizer step (fullgraph=True).
    Initialization runs eagerly on the first step; subsequent steps are compiled.

    promote: bool = False
    Whether to promote the gradients to fp32 before applying the optimizer.

    gradient_clipping: str_or_fn = None
    Clipping function applied to incoming gradients before any other transforms.

    update_clipping: str_or_fn = None
    Clipping function applied to outgoing updates. Disables fused updates.
    """

    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False
    auto_fuse: bool = True

    def __init__(
        self,
        params,
        defaults,
        gradient_clipping: str_or_fn = None,
        update_clipping: str_or_fn = None,
        palm: bool = use_default,
        fns: Iterable[callable] = (),
        compile_step: bool = use_default,
        promote: bool = use_default,
    ):
        if not fns:
            raise ValueError("No functions provided. If that's on purpose (SGD-like), use `identity`")

        args, kwargs = None, None
        fns = tuple(fns)
        fn = fns[-1]
        if isinstance(fn, functools.partial):
            fn, args, kwargs = fn.func, fn.args, fn.keywords
        if isinstance(fn, FunctionTransform):
            fn = fn.get_fn()

        if default(update_clipping, self.update_clipping) is None:
            if self.auto_fuse:
                if fn in _scale_to_update_map:
                    fn = _scale_to_update_map[fn]
                    if args is not None:
                        fn = functools.partial(fn, *args, **kwargs)
                    fns = tuple(fns)[:-1] + (fn,)
        elif fn in _scale_to_update_map_inv:
            if not self.auto_fuse:
                raise ValueError(
                    "update_clipping is currently not compatible with update_by_* functions. "
                    "Manually select scale_by_* functions or set auto_fuse=True."
                )
            fn = _scale_to_update_map_inv[fn]
            if args is not None:
                fn = functools.partial(fn, *args, **kwargs)
            fns = tuple(fns)[:-1] + (fn,)

        self.compile_step = default(default(compile_step, defaults.pop("compile_step", use_default)), self.compile_step)
        self.promote = default(default(promote, defaults.pop("promote", use_default)), self.promote)
        if default(palm, self.palm):
            fns = (palm_beta2,) + fns
        if default(gradient_clipping, self.gradient_clipping) is not None:
            fns = (apply_to_idx(gradient_clipping, 2),) + fns
        if defaults.get("mars", False):
            fns = (mars,) + fns
        if default(update_clipping, self.update_clipping) is not None:
            fns = fns + (apply_to_idx(update_clipping, 2),)

        super().__init__(params, defaults, *fns)


class ScheduleFree(BaseOpt):
    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        z_key = self._find_val_name("z")
        for group in self.param_groups:
            train_mode = group.get("train_mode", True)
            if train_mode == mode:
                continue
            group["train_mode"] = mode
            beta1 = utils.get_beta1(group)
            if beta1 <= 0:
                continue
            weight = 1 - beta1 if mode else 1 - 1 / beta1
            for p in group["params"]:
                state = self.state_(p)
                if z_key in state:
                    z = utils.promote(state[z_key])
                    p32 = utils.promote(p.data)
                    p32.lerp_(end=z, weight=weight)
                    utils.copy_stochastic_(p.data, p32)
        return self


class MSAM(BaseOpt):
    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        z_key = self._find_val_name("z")
        for group in self.param_groups:
            train_mode = group.get("train_mode", True)
            if train_mode == mode:
                continue
            group["train_mode"] = mode
            for p in group["params"]:
                state = self.state_(p)
                if z_key in state:
                    p_copy = p.data.clone()
                    utils.copy_stochastic_(p.data, state[z_key])
                    utils.copy_stochastic_(state[z_key], p_copy)
        return self
