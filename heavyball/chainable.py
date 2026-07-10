import contextlib
import copy
import functools
import math
from collections.abc import Iterable as _Iterable
from numbers import Real
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


def _normalize_chain(fns):
    if fns is None:
        return None
    return fns if isinstance(fns, (list, tuple)) else (fns,)


def _terminal_transform(fn):
    if isinstance(fn, functools.partial):
        fn = fn.func
    if isinstance(fn, FunctionTransform):
        fn = fn.get_fn()
    name = getattr(fn, "__name__", "")
    return name == "apply_update" or name.startswith("update_by_")


def _terminal_chain(fns):
    terminals = [i for i, fn in enumerate(fns or ()) if _terminal_transform(fn)]
    if terminals and terminals != [len(fns) - 1]:
        raise ValueError("parameter update must be the final transform")
    return bool(terminals)


class Parallel:
    def __init__(self, branches: List[List[callable]], merge_fn: callable):
        if any(_terminal_chain(branch) for branch in branches):
            raise ValueError("Parallel branches must return updates")
        self.branches = branches
        self.merge_fn = merge_fn

    def __deepcopy__(self, memo):
        return type(self)([[copy.deepcopy(fn) for fn in branch] for branch in self.branches], self.merge_fn)

    def __call__(self, state, group, update, grad, param):
        results = []
        for branch in self.branches:
            branch_update = [torch.clone(u, memory_format=torch.preserve_format) for u in update]
            result, skip = _inner_chain(state, group, branch_update, grad, param, *branch)
            results.append((result, skip))
        terminated = [result is None for result, _ in results]
        if any(terminated) and not all(terminated):
            raise ValueError("Parallel branches must terminate uniformly")
        if all(terminated):
            return None
        skipped = [skip for _, skip in results]
        if any(skipped) and not all(skipped):
            raise ValueError("Parallel branches must skip uniformly")
        if all(skipped):
            return _SKIP
        return self.merge_fn([result for result, _ in results])


class Route:
    def __init__(self, *routes, default=None):
        self.routes = [(pred, _normalize_chain(fns)) for pred, fns in routes]
        self.default = _normalize_chain(default)
        chains = [fns for _, fns in self.routes] + [self.default]
        terminal = [_terminal_chain(fns) for fns in chains]
        if any(terminal) and not all(terminal):
            raise ValueError("Route branches must terminate uniformly")
        if all(terminal) and any(
            getattr(ft, "skip_first", False) or hasattr(ft, "warmup_fns")
            for fns in chains
            for ft in _walk_fns(fns)
        ):
            raise ValueError("terminal Route branches cannot skip initialization")

    def __deepcopy__(self, memo):
        def clone(fns):
            return None if fns is None else tuple(copy.deepcopy(fn) for fn in fns)

        routes = ((pred, clone(fns)) for pred, fns in self.routes)
        return type(self)(*routes, default=clone(self.default))

    def _route_index(self, param):
        return next((i for i, (pred, _) in enumerate(self.routes) if pred(param)), len(self.routes))

    def _select(self, param):
        index = self._route_index(param)
        return self.default if index == len(self.routes) else self.routes[index][1]

    def __call__(self, state, group, update, grad, param):
        buckets = {}
        for i, p in enumerate(param):
            buckets.setdefault(self._route_index(p), []).append(i)

        def _sel(lst, idx):
            return [lst[i] for i in idx]

        results = []

        all_chains = [(buckets.get(j), fns) for j, (_, fns) in enumerate(self.routes)]
        if default_idx := buckets.get(len(self.routes)):
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

        terminated = [result is None for result, _, _ in results]
        if any(terminated) and not all(terminated):
            raise ValueError("Route branches must terminate uniformly")
        if all(terminated):
            return None
        skipped = [skip for _, skip, _ in results]
        if any(skipped) and not all(skipped):
            raise ValueError("Route branches must skip uniformly")
        if all(skipped):
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


def _build_defaults(locals_dict):
    d = {key: value for key, value in locals_dict.items() if not key.startswith("_") and key != "self"}
    params = d.pop("params")
    kwargs = d.pop("kwargs")
    d.update(kwargs)
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
        try:
            self.primary_dtype, self.corr_dtype = self._MODES[mode]
        except KeyError as error:
            raise ValueError(f"Unknown ECC mode {mode!r}; expected one of {sorted(self._MODES)}") from error

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
    state["mu_product"] = torch.ones(1, dtype=dtype, device=param.device)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn, names)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        ecc = group.get("_ecc_config")
        for name in self.names:
            vn = self.val_name(name)
            wide_square = "exp_avg_sq" in name and (
                _storage_dtype(group) == torch.float16 or ecc is not None and ecc.primary_dtype == torch.float16
            )
            if ecc is None or wide_square:
                dtype = torch.float32 if wide_square else _storage_dtype(group)
                _zero_guard(state, vn, param, dtype)
            else:
                ecc.init_state(state, vn, param)

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        ecc = group.get("_ecc_config")
        if ecc is None:
            return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)

        names = [self.val_name(n) for n in self.names]
        primary_vars = [[st[vn] for st in state] for vn in names]
        with contextlib.ExitStack() as stack:
            for vn, plist in zip(names, primary_vars):
                if all(vn + "::ecc" in st for st in state):
                    corrs = [st[vn + "::ecc"] for st in state]
                    stack.enter_context(ecc.attached(plist, corrs))
            return self.fn(state, group, update, grad, param, *args, *primary_vars, **kwargs)


class PrecondGradAccumGuard(FunctionTransform):
    def __init__(self, fn):
        super().__init__(fn, ["precond_grad_accum", "precond_grad_accum_steps"])

    @staticmethod
    def _accum(state, steps, new):
        out = []
        for stored, count, value in zip(state, steps, new):
            count.add_(1)
            mean, value = utils.promote(stored), utils.promote(value)
            dtype = torch.promote_types(mean.dtype, value.dtype)
            mean, value = mean.to(dtype), value.to(dtype)
            shape = (*count.shape, *([1] * (mean.ndim - count.ndim)))
            inv_count = count.to(dtype).reciprocal().view(shape)
            value = mean * (1 - inv_count) + value * inv_count
            utils.copy_stochastic_(stored, value)
            out.append(value)
        return out

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        if not group.get("precond_grad_accum", False):
            return
        dtype = _storage_dtype(group)
        _zero_guard(state, self.val_name("precond_grad_accum"), param, torch.float32 if dtype == torch.float16 else dtype)
        state[self.val_name("precond_grad_accum_steps")] = torch.zeros((), dtype=torch.int64, device=param.device)

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        base_grad = update if group.get("momentum_into_precond_update", True) else grad
        if not group.get("precond_grad_accum", False):
            return self.fn(state, group, update, grad, param, *args, base_grad, **kwargs)

        accum, steps = vars
        if group["is_preconditioning"]:
            mean = self._accum(accum, steps, base_grad)
            out = self.fn(state, group, update, grad, param, *args, mean, **kwargs)
            utils.zero_([*accum, *steps])
            return out
        else:
            self._accum(accum, steps, base_grad)
        return self.fn(state, group, update, grad, param, *args, base_grad, **kwargs)


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn, names)
        self.index = index

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        source = utils.promote(val)
        ecc = group.get("_ecc_config")
        for name in self.names:
            key = self.val_name(name)
            if ecc is None:
                target = state[key] = torch.empty_like(val, dtype=_storage_dtype(group))
                utils.copy_stochastic_(target, source)
            else:
                ecc.init_state(state, key, val)
                with ecc.attached([state[key]], [state[key + "::ecc"]]):
                    utils.copy_stochastic_(state[key], source)

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
        v._ecc = utils._ULPState(ecc.correction.view_as(v), ecc.smax)
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


def _stack_value(vals):
    first = next((v for v in vals if v is not None), None)
    if first is None:
        return None
    if isinstance(first, Tensor):
        ts = [v if isinstance(v, Tensor) else torch.zeros_like(first) for v in vals]
        return torch.cat([t.unsqueeze(0) if t.ndim == 0 else t for t in ts], 0)
    if isinstance(first, tuple):
        return tuple(_stack_value([v[i] if isinstance(v, tuple) else None for v in vals]) for i in range(len(first)))
    if isinstance(first, list):
        return [
            _stack_value([v[i] if isinstance(v, list) and i < len(v) else None for v in vals])
            for i in range(len(first))
        ]
    if isinstance(first, set):
        merged = set()
        for v in vals:
            if isinstance(v, set):
                merged |= v
        return merged
    return first


def _unstack_value(slab_val, i, n):
    if isinstance(slab_val, Tensor):
        if slab_val.ndim >= 1 and slab_val.shape[0] == n:
            return slab_val[i : i + 1].clone()
        return slab_val.clone()
    if isinstance(slab_val, tuple):
        return tuple(_unstack_value(elem, i, n) for elem in slab_val)
    if isinstance(slab_val, list):
        return [_unstack_value(elem, i, n) for elem in slab_val]
    if isinstance(slab_val, set):
        return slab_val.copy()
    return slab_val


class BucketGuard(FunctionTransform):
    needs_init = False

    def _build_val_names(self):
        super()._build_val_names()
        self.__dict__.pop("_chain_keys", None)
        self.__dict__.pop("_init_guards", None)

    @functools.cached_property
    def _chain_keys(self):
        keys = set()
        for ft in _walk_fns(self.fn):
            keys.update(ft._val_names.values())
        return keys | {f"{vn}::ecc" for vn in keys}

    @functools.cached_property
    def _init_guards(self):
        return tuple(ft for ft in _walk_fns(self.fn) if isinstance(ft, (GeneralGuard, WarmupGuard)))

    def _init_mode(self, state):
        initialized = state.get("is_initialized", ())
        singleton = False
        for ft in self._init_guards:
            if isinstance(ft, GeneralGuard) and ft.transform_idx not in initialized:
                if ft.skip_first:
                    return _SKIP
                singleton = True
            elif isinstance(ft, WarmupGuard) and state.get(ft.warmup_key, 0) < len(ft.warmup_fns):
                return _SKIP
        return singleton

    def _run_subgroup(self, subgroup, states, group, update, grad, param, *args, **kwargs):
        n = len(subgroup)
        member_states = [states[i] for i in subgroup]
        views = [param[i] for i in subgroup]
        grads = [grad[i] for i in subgroup]
        updates = [update[i] for i in subgroup]
        eccs = [getattr(v, "_ecc", None) for v in views]
        corrs = [e.correction for e in eccs] if eccs[0] is not None else None

        slab_p = views[0][None] if n == 1 else torch.stack(views, 0)
        slab_g = grads[0][None] if n == 1 else torch.stack(grads, 0)
        slab_u = updates[0][None] if n == 1 else torch.stack(updates, 0)
        if corrs is not None:
            corr = corrs[0][None] if n == 1 else torch.stack(corrs, 0)
            slab_p._ecc = utils._ULPState(corr, eccs[0].smax)

        slab_state = {
            k: v for k in self._chain_keys if (v := _stack_value([m.get(k) for m in member_states])) is not None
        }
        slab_state["is_initialized"] = set().union(*[m.get("is_initialized") or () for m in member_states])
        result = self.fn([slab_state], group, [slab_u], [slab_g], [slab_p], *args, **kwargs)

        for i, member in enumerate(member_states):
            for key, value in slab_state.items():
                member[key] = _unstack_value(value, i, n)

        if result is _SKIP and n > 1:
            for i, view in enumerate(views):
                view.copy_(slab_p[i])
            if corrs is not None:
                for i, ecc in enumerate(eccs):
                    ecc.correction.copy_(slab_p._ecc.correction[i])
        return result

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        shapes = group.get("_orig_shapes") or {}
        caution_grad = [None] * len(param) if group.get("caution", False) else None

        init_modes = [self._init_mode(member) for member in states]
        initializing = [i for i, mode in enumerate(init_modes) if mode is _SKIP]
        if initializing:
            for i in initializing:
                self._run_subgroup([i], states, group, update, grad, param, *args, **kwargs)
            return _SKIP

        if any(init_modes):
            results = [
                self._run_subgroup([i], states, group, update, grad, param, *args, **kwargs)
                for i in range(len(param))
            ]
            skips = [result is _SKIP for result in results]
            if all(skips):
                return _SKIP
            if any(skips):
                raise ValueError("Bucket members must uniformly skip during initialization")
            return [result[0][0] for result in results]

        buckets = {}
        for i, p in enumerate(param):
            info = shapes.get(id(p))
            sig = (
                tuple(p.shape),
                p.dtype,
                p.device,
                info.owner if info is not None else None,
            )
            buckets.setdefault(sig, []).append(i)

        out = [None] * len(param)
        has_output = False
        for subgroup in buckets.values():
            result = self._run_subgroup(subgroup, states, group, update, grad, param, *args, **kwargs)
            source = group.pop("_caution_grad", None)
            if caution_grad is not None and source is not None:
                for k, i in enumerate(subgroup):
                    caution_grad[i] = source[0][k]
            if result is _SKIP:
                continue

            has_output = True
            precond_slab = result[0]
            for k, i in enumerate(subgroup):
                out[i] = precond_slab[k]

        if not has_output:
            return _SKIP
        if caution_grad is not None and any(g is not None for g in caution_grad):
            group["_caution_grad"] = [g if c is None else c for g, c in zip(grad, caution_grad)]
        return [torch.zeros_like(u) if o is None else o for o, u in zip(out, update)]


class WarmupGuard(FunctionTransform):
    def __init__(self, fn, warmup_fns):
        super().__init__(fn, names=["warmup"])
        self.warmup_fns = warmup_fns

    def _build_val_names(self):
        self._val_names = {"warmup": f"_warmup_{self.transform_idx}"}
        self.warmup_key = self._val_names["warmup"]

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = state if isinstance(state, list) else [state(p) for p in param]
        warmup_steps = [st.get(self.warmup_key, 0) for st in states]
        if any(step < len(self.warmup_fns) for step in warmup_steps):
            for st, a in zip(states, zip(update, grad, param, *args)):
                step = st.get(self.warmup_key, 0)
                if step < len(self.warmup_fns):
                    self.warmup_fns[step](st, group, *a, **kwargs)
                    st[self.warmup_key] = step + 1
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


@zero_guard("mars_old_grad")
@no_state
def mars(group, update, grad, param, mars_old_grad):
    update = utils.mars_correction(update, mars_old_grad, utils.get_beta1(group), group["mars_gamma"])
    utils.copy_stochastic_list_(grad, update)
    grad[:] = update
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
    coupled = group["weight_decay"] != 0 and not group.get("decoupled_weight_decay", False)
    if coupled and group["caution"]:
        group["_caution_grad"] = [
            utils.promote(u) + utils.promote(p) * group["weight_decay"] for u, p in zip(update, param)
        ]
    return utils.nadam_(
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
    max_lr = group["max_lr"]
    decay = group["lr"] * group["weight_decay"] / max_lr
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
        decay,
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
    return utils.laprop_(
        exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group["step"], group["eps"]
    )


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
        group["eps"],
    )
    return _SKIP


@needs_full_param
@no_state
def orthogonalize_grad_to_param(group, update, grad, param):
    return utils.orthogonalize_grad_to_param(param, update, group["eps"])


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    update, param, z, grad = utils.list_guard(update, param, z, grad)
    lr, ckp1, beta1 = utils.scalar_guard(group["lr"], group["_sf_ckp1"], utils.get_beta1(group), grad[0])
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


@zero_guard("exp_avg", "exp_avg_sq")
@warmup_guard(_adopt_warmup_1)
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
        group["step"] - 1,
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
        group.get("cautious_weight_decay", False),
    )
    return _SKIP


def _suds_warmup_1(state, group, update, grad, param, exp_avg, exp_avg_sq, fisher_approx):
    utils.copy_stochastic_(fisher_approx, utils.stable_l2_normalize(update, eps=1e-8))


@needs_full_param
@zero_guard("exp_avg", "exp_avg_sq", "fisher_approx")
@warmup_guard(_suds_warmup_1)
@no_state_no_multi_tensor
def scale_by_suds(group, update, grad, param, exp_avg, exp_avg_sq, fisher_approx):
    update, fisher = utils.promote(update), utils.promote(fisher_approx)
    fisher = torch.where(fisher.abs().amax() == 0, utils.stable_l2_normalize(update, eps=1e-8), fisher)
    precond_update, w = utils.eigvecs_product_rank1(update.flatten(), fisher.flatten())
    precond_update = utils.adam_(
        exp_avg,
        exp_avg_sq,
        precond_update.view_as(exp_avg),
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )[0]
    precond_update, _ = utils.eigvecs_product_rank1(precond_update.flatten(), fisher.flatten(), w)

    new_approx = utils.oja_update(fisher.flatten(), update.flatten(), group["precond_lr"])
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
        group["eps"],
    )
    return update


@zero_guard("exp_avg", "exp_avg_sq")
@warmup_guard(_adopt_warmup_1)
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adopt(
        update,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
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
    if not cached:
        return

    state["Q_cache"] = [torch.empty_like(q, dtype=utils.promote(q.dtype)) for q in Q]


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

    _update_psgd_precond(
        False,
        None,
        group,
        param,
        update,
        Q,
        state["running_lower_bound"],
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
    if not cached:
        return
    state["Q_cache"] = [torch.empty_like(q, dtype=utils.promote(q.dtype)) for q in Q]


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
    scale, norm = utils.stable_l2_components(param)
    state["init_norm"] = torch.stack((scale, norm))


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
    state["init_std"] = torch.std(utils.promote(param), correction=0).to(_storage_dtype(group))


@needs_full_param
@general_guard("init_std", init_fn=_store_std, skip_first=False)
@no_state
def mup_approx(group, updates, grads, params, init_std):
    _updates = [(u, i) for u, i in zip(updates, init_std) if u.ndim > 1]
    if not _updates:
        return updates
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


def _init_scion_param(state, group, update, grad, param):
    utils.scion_auto_init_param_(param, group.get("scale", 1.0), seed=group["_param_indices"][id(param)])


@needs_full_param
@general_guard(init_fn=_init_scion_param, skip_first=False)
@no_state
def scion_auto_norm(group, update, grad, param):
    scale = group.get("scale", 1.0)
    return utils.scion_auto_lmo_(update, scale, group["eps"])


def _init_soap(state, group, update, grad, param):
    utils.init_preconditioner(
        grad,
        state,
        group["max_precond_dim"],
        group["precondition_1d"],
        group.get("init_factor", 0.0),
        state_dtype=_storage_dtype(group),
    )


def _apply_soap_preconditioner(
    group, update, Q, GG, *exp_avgs, use_kl: bool = False, eps=1e-8, exp_avg_sq=None, heavy: bool = False
):
    beta = utils.beta_debias(group["shampoo_beta"], group["step"])
    max_dim, p1d = group["max_precond_dim"], group["precondition_1d"]
    eas = exp_avg_sq or [None] * len(update)
    for upd, q, gg, ea_sq, *ref in zip(update, Q, GG, eas, *exp_avgs):
        g = utils.promote(upd)
        if use_kl:
            utils.update_ggt_kl(g, gg, q, beta, eps, heavy=heavy)
        else:
            utils.update_ggt(g, gg, max_dim, p1d, beta)
        if group["is_preconditioning"]:
            utils.get_orthogonal_matrix_QR(gg, q, *ref, exp_avg_sq=ea_sq if heavy else None, heavy=heavy)


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
    beta1 = utils.beta_debias(utils.get_beta1(group), group["step"] - 1)
    ema = utils._lerp(exp_avg, update, beta1)
    precond = [utils.kl_shampoo_precondition(e, q, gg, group["eps"]) for e, q, gg in zip(ema, Q, GG)]
    _apply_soap_preconditioner(group, update, Q, GG, use_kl=True, eps=group["eps"])
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
    coupled = group["weight_decay"] != 0 and not group["decoupled_weight_decay"]
    source = (
        [utils.promote(u) + utils.promote(p) * group["weight_decay"] for u, p in zip(update, param)]
        if coupled
        else update
    )
    if coupled and group["caution"]:
        group["_caution_grad"] = source
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(source, Q)]
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
    _apply_soap_preconditioner(group, source, Q, GG, exp_avg)
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
        group["eps"],
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


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_heavy_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
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
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg, exp_avg_sq=exp_avg_sq, heavy=True)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_heavy_kl_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
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
    _apply_soap_preconditioner(
        group, update, Q, GG, exp_avg, use_kl=True, eps=group["eps"], exp_avg_sq=exp_avg_sq, heavy=True
    )
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_heavy_kl_shampoo(group, update, grad, param, exp_avg, Q, GG):
    beta1 = utils.beta_debias(utils.get_beta1(group), group["step"] - 1)
    ema = utils._lerp(exp_avg, update, beta1)
    precond = [utils.kl_shampoo_precondition(e, q, gg, group["eps"]) for e, q, gg in zip(ema, Q, GG)]
    _apply_soap_preconditioner(group, update, Q, GG, use_kl=True, eps=group["eps"], heavy=True)
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
def scale_by_heavy_soap_nadam(group, update, grad, param, exp_avg, exp_avg_sq, mu_product, Q, GG):
    coupled = group["weight_decay"] != 0 and not group["decoupled_weight_decay"]
    source = (
        [utils.promote(u) + utils.promote(p) * group["weight_decay"] for u, p in zip(update, param)]
        if coupled
        else update
    )
    if coupled and group["caution"]:
        group["_caution_grad"] = source
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(source, Q)]
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
    _apply_soap_preconditioner(group, source, Q, GG, exp_avg, exp_avg_sq=exp_avg_sq, heavy=True)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_heavy_soap_laprop(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    precond = utils.laprop_(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]
    _apply_soap_preconditioner(group, update, Q, GG, exp_avg, exp_avg_sq=exp_avg_sq, heavy=True)
    return precond


@needs_full_param
@bucket_aware
@zero_guard("exp_avg_fast", "exp_avg_slow", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_heavy_soap_ademamix(group, update, grad, param, exp_avg_fast, exp_avg_slow, exp_avg_sq, Q, GG):
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
    _apply_soap_preconditioner(
        group,
        update,
        Q,
        GG,
        exp_avg_slow,
        exp_avg_fast,
        exp_avg_sq=exp_avg_sq,
        heavy=True,
    )
    return precond


def _fill_q_cache(Q_cache, Q):
    for i, (c_, q_) in enumerate(zip(Q_cache, Q)):
        q_ = utils.promote(q_)
        if c_ is None or c_.dtype != q_.dtype:
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
    prob: Optional[callable] = None,
    store_triu_as_line: Optional[bool] = None,
) -> None:
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
        vector,
        running_lower_bound,
        group["lower_bound_beta"],
    )
    del vector, hessian_vector

    float_prob = (
        group.get("_precond_prob", 1.0) if prob is None else prob(group["step_count"]) if callable(prob) else prob
    )
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if should_use_cache:
        _fill_q_cache(Q_cache, utils.line_to_triu(Q) if store_triu_as_line else Q)


def _update_psgd_pro_precond(
    cached,
    Q_cache,
    group,
    grad,
    Q,
    running_lower_bound,
    prob: Optional[callable] = None,
) -> None:
    if not group["is_preconditioning"]:
        return

    utils.psgd_pro_update_precond(
        grad,
        group["precond_lr"],
        Q,
        running_lower_bound,
        group["lower_bound_beta"],
        group["dampening"],
    )

    float_prob = (
        group.get("_precond_prob", 1.0) if prob is None else prob(group["step_count"]) if callable(prob) else prob
    )
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if should_use_cache:
        _fill_q_cache(Q_cache, Q)


def _cached_psgd_precond_grad(group, update, Q, Q_cache, grad):
    sqrt = group.get("sqrt", False)
    kwargs = {"ea": update, "caution": False, "grad": grad}
    if not sqrt and group.get("is_cached", False) and Q_cache[0] is not None:
        return utils.precond_grad_cached_(cached_q=Q_cache, **kwargs)
    return utils.psgd_precond_grad(preconds=Q, store_triu_as_line=group["store_triu_as_line"], sqrt=sqrt, **kwargs)


def _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache):
    sqrt = group.get("sqrt", False)
    kwargs = {
        "ea": update,
        "caution": group["caution"],
        "grad": grad,
        "param": param,
        "lr": group["lr"],
        "decay": group["weight_decay"],
        "cautious_decay": group.get("cautious_weight_decay", False),
    }
    if not sqrt and group.get("is_cached", False) and Q_cache[0] is not None:
        utils.fused_precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        utils.fused_psgd_precond_grad(preconds=Q, store_triu_as_line=group["store_triu_as_line"], sqrt=sqrt, **kwargs)


def _update_lra(
    group, U: List[Tensor], V: List[Tensor], d: List[Tensor], params: List[Tensor], grads: List[Tensor], delayed: bool
):
    if not group["is_preconditioning"]:
        return utils._flatten_lra(U, V, d)

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
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "Q_basis", "running_lower_bound", init_fn=_init_psgd_eigen_kron, skip_first=True)
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
    prob: Optional[callable] = None,
):
    projected = utils.project(utils.promote(update), Q_basis, False)
    precond = utils.adam_(
        exp_avg,
        exp_avg_sq,
        projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )[0]
    precond = utils.project(precond, Q_basis, True)

    if group["is_preconditioning"]:
        _update_psgd_precond(False, None, group, param, update_to_precond, Q, running_lower_bound, prob)
        utils.update_psgd_eigenbasis(
            utils.line_to_triu(Q) if group["store_triu_as_line"] else Q,
            Q_basis,
            exp_avg,
            exp_avg_sq=exp_avg_sq,
        )

    return precond


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    precond = _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, prob)
    return precond


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, prob)
    _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache)
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
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, running_lower_bound, prob)
    return _SKIP


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_pro_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_pro_precond(cached, Q_cache, group, update_to_precond, Q, running_lower_bound, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@needs_full_param
@bucket_aware
@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "running_lower_bound", init_fn=_init_psgd_pro_kron, skip_first=False)
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
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_pro_precond(cached, Q_cache, group, update_to_precond, Q, running_lower_bound, prob)
    _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache)
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
    elif not callable(fn):
        raise ValueError(f"Expected a callable or function name, got {fn!r}")

    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        args[idx] = utils.promote(args[idx])
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
    if fsdp_ids - result.keys():
        utils.warn_once(
            "FSDP parameters detected but original shapes could not be recovered. "
            "Shape-aware optimizers (SOAP, Muon, PSGD, Scion) will fall back to per-element updates. "
            "Pass use_orig_params=True to FSDP to enable shape recovery."
        )
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
        flat = flat.to(device=device, dtype=dtype)
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
    grad = list(grad)
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]
    decay = group["weight_decay"] if group.get("decoupled_weight_decay", True) else 0

    ecc = group.get("_param_ecc_config")
    if ecc is None:
        update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
        if skip_update or update is None:
            group.pop("_caution_grad", None)
            return
        caution_grad = group.pop("_caution_grad", grad)
        utils.update_param_(
            param,
            update,
            group["lr"],
            decay,
            caution=group["caution"],
            cautious_decay=group.get("cautious_weight_decay", False),
            grad=caution_grad,
        )
        return

    corrs = [st["param::ecc"].view_as(pi) for st, pi in zip(state, param)]
    with ecc.attached(param, corrs):
        update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
        if not skip_update and update is not None:
            caution_grad = group.pop("_caution_grad", grad)
            utils.update_param_(
                param,
                update,
                group["lr"],
                decay,
                caution=group["caution"],
                cautious_decay=group.get("cautious_weight_decay", False),
                grad=caution_grad,
            )
        else:
            group.pop("_caution_grad", None)


def _contains_route(obj):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Route):
            return True
        if isinstance(cur, FunctionTransform):
            stack.append(cur.fn)
        elif isinstance(cur, functools.partial):
            stack.append(cur.func)
        elif isinstance(cur, Parallel):
            stack.extend(cur.branches)
        elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
            stack.extend(cur)
    return False


def _required_transform_ids(obj, param):
    required = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, FunctionTransform):
            if cur.transform_idx is not None and getattr(cur, "needs_init", True) and not cur._under_bucket:
                required.add(cur.transform_idx)
            stack.append(cur.fn)
        elif isinstance(cur, functools.partial):
            stack.append(cur.func)
        elif isinstance(cur, Parallel):
            stack.extend(cur.branches)
        elif isinstance(cur, Route):
            selected = cur._select(param)
            if selected is not None:
                stack.extend(selected)
        elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
            stack.extend(cur)
    return required


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
                if fns is not None:
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
                if fns is not None:
                    stack.extend((f, ub) for f in fns)
            if cur.default is not None:
                stack.extend((f, ub) for f in cur.default)
        elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
            stack.extend((c, ub) for c in cur)


def set_indices(fns: Iterable[callable], retain: bool = True, offset: int = 0):
    fns = list(fns)
    if retain and offset:
        raise ValueError("offset cannot be retained")

    if retain:
        offset = max((ft.transform_idx for ft in _walk_fns(fns) if ft.transform_idx is not None), default=-1) + 1

    new_fns = [copy.deepcopy(fn) for fn in fns]
    seen = set()
    for ft, under_bucket in _walk_fns_with_bucket(new_fns):
        if not retain or ft.transform_idx is None or ft.transform_idx in seen:
            ft.transform_idx, offset = offset, offset + 1
        seen.add(ft.transform_idx)
        ft._under_bucket = under_bucket
        ft._build_val_names()

    return new_fns


class ChainOpt(utils.StatefulOptimizer):
    promote: bool = False
    _INSTANCE_ATTRS = utils.StatefulOptimizer._INSTANCE_ATTRS + ("promote",)
    global_defaults = {
        "caution": False,
        "lr": 1,
        "warmup_steps": 0,
        "weight_decay": 0,
        "eps": 1e-8,
    }

    def __init__(self, params, defaults, *fns):
        orig = defaults.pop("orig_shapes", None)
        base = self.global_defaults.copy()
        base.update({k: v for k, v in defaults.items() if v is not use_default})
        use_ema = base.pop("use_ema", False)
        ema_decay = base.pop("ema_decay", None)
        super().__init__(params, base, use_ema=use_ema)
        if ema_decay is not None:
            self.ema_decay = ema_decay
        self._orig_shapes = self._resolve_orig_shapes(orig)
        self._refresh_param_indices()
        self._refresh_ecc_configs()
        self.fns = fns
        self._eager_chain = self._run_chain
        if self.compile_step:
            self._run_chain = utils.fusions.compile(self._run_chain, fullgraph=True)
        self.register_load_state_dict_post_hook(ChainOpt._restore_state_dtypes)
        self._init_param_ecc()

    def _resolve_orig_shapes(self, orig):
        all_params = [p for g in self.param_groups for p in g["params"]]
        detected = _detect_orig_shapes(all_params)
        if orig is None:
            return detected or None
        user = {k: _ShapeInfo(v) if isinstance(v, tuple) else v for k, v in orig.items()}
        if detected:
            utils.warn_once(
                "orig_shapes was passed but FSDP was detected. "
                "Ignoring orig_shapes in favor of auto-detection for correct gather/scatter."
            )
        return {**user, **detected}

    def state_dict(self):
        sd = super().state_dict()
        for g in sd["param_groups"]:
            for k in [k for k in g if k.startswith("_")]:
                del g[k]
        return sd

    def _init_param_ecc(self):
        for group in self.param_groups:
            self._init_param_ecc_group(group)

    def _init_param_ecc_group(self, group):
        ecc = group["_param_ecc_config"]
        if ecc is None:
            return
        for p in group["params"]:
            fp32 = None
            if p.dtype != ecc.primary_dtype:
                fp32 = p.data.float()
                p.data = p.data.to(ecc.primary_dtype)
            p_views = self._set_views(p, group)
            fp32_views = None if fp32 is None else utils.merge_group(group, fp32)
            for i, pv in enumerate(p_views):
                st = self.state_(pv)
                if "param::ecc" in st:
                    continue
                st["param::ecc"] = torch.zeros_like(pv, dtype=ecc.corr_dtype)
                if fp32_views is not None:
                    ecc.init_correction(st["param::ecc"], fp32_views[i], pv.data)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        if hasattr(self, "mapping"):
            self._refresh_param_indices()
            self._refresh_ecc_configs()
            self._init_param_ecc_group(self.param_groups[-1])

    def _refresh_param_indices(self):
        indices = {id(p): i for i, p in enumerate(p for group in self.param_groups for p in group["params"])}
        for group in self.param_groups:
            group["_param_indices"] = {id(p): indices[id(p)] for p in group["params"]}

    def _refresh_ecc_configs(self):
        for group in self.param_groups:
            group["_ecc_config"] = ECCConfig.from_group(group)
            group["_param_ecc_config"] = ECCConfig.from_group(group, key="param_ecc")

    @staticmethod
    def _restore_state_dtypes(optimizer, *args):
        def restore(value, source):
            if isinstance(value, Tensor) and isinstance(source, Tensor):
                return source.to(device=value.device).clone(memory_format=torch.preserve_format)
            if isinstance(value, dict) and isinstance(source, dict):
                for key, item in source.items():
                    value[key] = restore(value[key], item)
            elif isinstance(value, list) and isinstance(source, list):
                value[:] = [restore(v, s) for v, s in zip(value, source, strict=True)]
            elif isinstance(value, tuple) and isinstance(source, tuple):
                return tuple(restore(v, s) for v, s in zip(value, source, strict=True))
            return value

        for p, dtype in optimizer._loaded_param_dtypes.items():
            if p.dtype != dtype:
                p.data = p.data.to(dtype)
        del optimizer._loaded_param_dtypes
        optimizer.mapping.clear()
        optimizer.mapping_inverse.clear()

        values = optimizer._loaded_state_values
        for p, source in values.items():
            if p in optimizer.state:
                optimizer.state[p] = restore(optimizer.state[p], source)
        del optimizer._loaded_state_values
        if isinstance(optimizer, BaseOpt):
            optimizer._rewire_fns()
        optimizer._refresh_param_indices()
        optimizer._refresh_ecc_configs()

    @property
    def fns(self):
        return self._fns

    @fns.setter
    def fns(self, value):
        self._fns = value
        self._set_indices(retain=True)
        self._route_aware_init = _contains_route(self._fns)
        self._needs_gather = any(getattr(ft, "needs_full_param", False) for ft in _walk_fns(self._fns))
        sf_fn = update_by_schedule_free.get_fn()
        self._uses_schedule_free = any(ft.get_fn() is sf_fn for ft in _walk_fns(self._fns))
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
        group["base_lr"] = group["lr"]

        views, gathers = _reshape_params(group["params"], self._orig_shapes, self._needs_gather)
        changed = [p for p, _ in views]
        if isinstance(gathers, _FSDPState):
            changed.extend(p for p, _, _ in gathers.items)
        for p in changed:
            self._clear_views(p)
        group["_orig_shapes"] = self._orig_shapes
        try:
            self._step_inner(group)
        finally:
            _restore_params(views, gathers)
            for p in changed:
                self._clear_views(p)

    def _step_inner(self, group):
        vals = list(self.split_p_and_g_in_group(group, should_promote=self.promote))
        if not vals:
            return

        group["step_count"] = group.get("step_count", 0) + 1
        warmup = group["warmup_steps"] + 1
        if isinstance(group["base_lr"], Real):
            lr = group["base_lr"] * group["step_count"] / max(group["step_count"], warmup)
            group["lr"] = utils.scalar_guard(float(lr), vals[0][0]) if group["warmup_steps"] else float(lr)
            if self._uses_schedule_free:
                weight = abs(lr) ** group["weight_lr_power"]
                if group["r"] != 0:
                    weight *= group["step_count"] ** group["r"]
                weight_sum = group.get("weight_sum", 0.0) + weight
                group["weight_sum"] = weight_sum
                group["_sf_ckp1"] = utils.scalar_guard(0 if weight_sum == 0 else weight / weight_sum, vals[0][0])
        else:
            global_step_t = utils.scalar_guard(float(group["step_count"]), vals[0][0])
            base_lr = utils.scalar_guard(group["base_lr"], vals[0][0])
            group["lr"] = base_lr * global_step_t / global_step_t.clamp(min=warmup)
            if self._uses_schedule_free:
                weight = group["lr"].abs().pow(group["weight_lr_power"])
                if group["r"] != 0:
                    weight = weight * global_step_t.pow(group["r"])
                weight_sum = utils.scalar_guard(group.get("weight_sum", 0.0), weight) + weight
                group["weight_sum"] = weight_sum
                group["_sf_ckp1"] = torch.where(weight_sum != 0, weight / weight_sum, torch.zeros_like(weight_sum))

        buckets = {}
        for param, grad in vals:
            state = self.state_(param)
            step_count = state.get("step_count", 0) + 1
            state["step_count"] = step_count
            key = step_count, param.device, utils.promote(param.dtype), param.dtype == torch.float64 and param.ndim == 0
            buckets.setdefault(key, []).append((param, grad, state))

        try:
            for (step_count, _, _, _), items in buckets.items():
                p, g, states = zip(*items)
                states = list(states)
                step = utils.scalar_guard(step_count, p[0])
                group["step"] = step
                if not group["multi_tensor"] or len(p) == 1:
                    for param, grad, state in zip(p, g, states):
                        self._chain(group, [grad], [param], [state])
                else:
                    self._chain(group, g, p, states)
        finally:
            group.pop("_caution_grad", None)
            group["lr"] = group["base_lr"]
            group["step"] = None

    def _run_chain(self, state, group, g, p):
        chain(state, group, g, p, *self.fns)

    def _required_ids(self, param):
        return _required_transform_ids(self.fns, param)

    def _needs_init(self, state, param=None):
        ids = self._transform_ids
        if not ids:
            return False
        if self._route_aware_init:
            if param is None:
                initialized = set().union(*(st.get("is_initialized", ()) for st in state))
                return not ids.issubset(initialized)
            return any(
                not self._required_ids(p).issubset(st.get("is_initialized", ()))
                for st, p in zip(state, param)
            )
        return any(not ids.issubset(st.get("is_initialized", ())) for st in state)

    def _needs_eager(self, group, state, param):
        if self._needs_init(state, param):
            return True
        if group.get("is_preconditioning", False):
            return True
        if group.get("ecc") or group.get("param_ecc"):
            return True
        return False

    def _chain(self, group, g, p, state):
        if p[0].dtype == torch.float64 and p[0].ndim == 0:
            with utils.force_eager():
                self._eager_chain(state, group, g, p)
            return
        fn = self._run_chain
        if self.compile_step and self._needs_eager(group, state, p):
            fn = self._eager_chain
        fn(state, group, g, p)


str_or_fn = Union[str, callable, None, Literal[use_default]]


def default(a, b):
    return b if a is use_default else a


# not supported: update_by_schedule_free, scale_by_soap, scale_by_exp_avg_sq
_FUSION_PAIRS = (
    (scale_by_delayed_psgd, update_by_delayed_psgd),
    (scale_by_psgd, update_by_psgd),
    (scale_by_psgd_lra, update_by_psgd_lra),
    (scale_by_delayed_psgd_lra, update_by_delayed_psgd_lra),
    (scale_by_adam, update_by_adam),
    (scale_by_nadam, update_by_nadam),
    (scale_by_laprop, update_by_laprop),
    (scale_by_adopt, update_by_adopt),
    (scale_by_ademamix, update_by_ademamix),
    (scale_by_psgd_pro, update_by_psgd_pro),
)
_scale_to_update_map = {s.get_fn(): u for s, u in _FUSION_PAIRS}
_scale_to_update_map_inv = {u.get_fn(): s for s, u in _FUSION_PAIRS}


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
    _TOPOLOGY_KEYS = ("palm", "gradient_clipping", "update_clipping", "mars")

    def _core_fns_from_group(self, group):
        return self._core_fns

    def _checkpoint_topology(self):
        return {key: ("custom",) if callable(value) else value for key, value in self._topology.items()}

    def _restore_topology(self):
        for key, value in self._loaded_topology.items():
            if value != ("custom",):
                self._topology[key] = value
        self._loaded_topology = None

    @staticmethod
    def _checkpoint_option(value):
        if callable(value) and getattr(value, "__module__", None) == utils.__name__:
            return value.__name__
        return value

    def _wire_fns(self, fns, group):
        fns = tuple(fns)
        fn = fns[-1]
        args = kwargs = None
        if isinstance(fn, functools.partial):
            fn, args, kwargs = fn.func, fn.args, fn.keywords
        if isinstance(fn, FunctionTransform):
            fn = fn.get_fn()

        update_clipping = group["update_clipping"]
        if update_clipping is None and self.auto_fuse and fn in _scale_to_update_map:
            fn = _scale_to_update_map[fn]
            if args is not None:
                fn = functools.partial(fn, *args, **kwargs)
            fns = fns[:-1] + (fn,)
        elif update_clipping is not None and fn in _scale_to_update_map_inv:
            fn = _scale_to_update_map_inv[fn]
            if args is not None:
                fn = functools.partial(fn, *args, **kwargs)
            fns = fns[:-1] + (fn,)

        if group["palm"]:
            fns = (palm_beta2,) + fns
        if group["gradient_clipping"] is not None:
            fns = (apply_to_idx(group["gradient_clipping"], 2),) + fns
        if group.get("mars", False):
            fns = (mars,) + fns
        if update_clipping is not None:
            fns += (apply_to_idx(update_clipping, 2),)
        return fns

    def _rewire_fns(self):
        self._restore_topology()
        self.fns = self._wire_fns(self._core_fns_from_group(self._topology), self._topology)

    def add_param_group(self, param_group):
        for key, expected in self._topology.items():
            if key not in param_group:
                continue
            value = param_group.pop(key)
            value = expected if value is use_default else self._checkpoint_option(value)
            if value != expected:
                raise ValueError(f"{key} is optimizer-wide")
        super().add_param_group(param_group)

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

        self.compile_step = default(default(compile_step, defaults.pop("compile_step", use_default)), self.compile_step)
        self.promote = default(default(promote, defaults.pop("promote", use_default)), self.promote)
        defaults["palm"] = default(palm, self.palm)
        defaults["gradient_clipping"] = self._checkpoint_option(default(gradient_clipping, self.gradient_clipping))
        defaults["update_clipping"] = self._checkpoint_option(default(update_clipping, self.update_clipping))
        self._topology = {key: defaults[key] for key in self._TOPOLOGY_KEYS if key in defaults}
        for key in self._topology:
            defaults.pop(key)
        self._core_fns = tuple(fns)
        super().__init__(params, defaults, *self._wire_fns(self._core_fns, self._topology))


class ScheduleFree(BaseOpt):
    def eval(self):
        return self.train(False)

    @torch.no_grad()
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
                for pv in self._set_views(p, group):
                    state = self.state_(pv)
                    if z_key not in state:
                        continue
                    z = state[z_key]
                    with contextlib.ExitStack() as stack:
                        param_ecc = group.get("_param_ecc_config")
                        if param_ecc is not None:
                            stack.enter_context(param_ecc.attached([pv], [state["param::ecc"].view_as(pv)]))
                        state_ecc = group.get("_ecc_config")
                        if state_ecc is not None:
                            stack.enter_context(state_ecc.attached([z], [state[z_key + "::ecc"]]))
                        p32, z32 = utils.promote(pv), utils.promote(z)
                        utils.copy_stochastic_(pv, p32 + (z32 - p32) * weight)
        return self


class MSAM(BaseOpt):
    def eval(self):
        return self.train(False)

    @torch.no_grad()
    def train(self, mode: bool = True):
        z_key = self._find_val_name("z")
        for group in self.param_groups:
            train_mode = group.get("train_mode", True)
            if train_mode == mode:
                continue
            group["train_mode"] = mode
            for p in group["params"]:
                for pv in self._set_views(p, group):
                    state = self.state_(pv)
                    if z_key not in state:
                        continue
                    z = state[z_key]
                    with contextlib.ExitStack() as stack:
                        param_ecc = group.get("_param_ecc_config")
                        if param_ecc is not None:
                            stack.enter_context(param_ecc.attached([pv], [state["param::ecc"].view_as(pv)]))
                        state_ecc = group.get("_ecc_config")
                        if state_ecc is not None:
                            stack.enter_context(state_ecc.attached([z], [state[z_key + "::ecc"]]))
                        value = utils.promote(pv).clone()
                        utils.copy_stochastic_(pv, utils.promote(z))
                        utils.copy_stochastic_(z, value)
        return self
