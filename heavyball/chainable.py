import copy
import functools
import math
import random
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


def _inplace_guard_(state, key, template_fn):
    key_not_in_state = not _key_in_state(state, key)
    if key_not_in_state:
        template_fn()
    return key_not_in_state


def _guard_in_state(state, key, template_fn):
    if not _key_in_state(state, key):
        state[key] = template_fn()
    return state[key]


class FunctionTransform:
    def __init__(self, fn, names: list[str] | None = None):
        if names is None:
            names = []
        self.fn = fn
        self.fn_name = self.get_fn().__name__
        self.transform_idx = None
        self.is_initialized = False
        self.names = names

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        raise NotImplementedError

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = [state(p) for p in param]
        skip_update = False
        for st, a in zip(states, zip(update, grad, param, *args)):
            if self.transform_idx not in st.get("is_initialized", set()):
                try:
                    self._init(st, group, *a, **kwargs)
                except SkipUpdate:
                    skip_update = True
                finally:
                    if "is_initialized" not in st:
                        st["is_initialized"] = set()
                    st["is_initialized"].add(self.transform_idx)
        if skip_update:
            raise SkipUpdate from None
        vars = [[st.get(self.val_name(name), None) for st in states] for name in self.names]
        return self._call(state, group, update, grad, param, vars, *args, **kwargs)

    def get_fn(self):
        if utils.hasattr_none(self.fn, "get_fn"):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        assert self.transform_idx is not None
        return f"{self.fn_name}_{name}_{self.transform_idx}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fn}, transform_idx={self.transform_idx})"


class Branch:
    def __init__(self, branches: List[List[callable]], merge_fn: callable):
        self.branches = branches
        self.merge_fn = merge_fn

    def __call__(self, state, group, update, grad, param):
        outputs = []
        for branch in self.branches:
            branch_update = [torch.clone(u, memory_format=torch.preserve_format) for u in update]
            branch_update, skip_update = _inner_chain(state, group, branch_update, grad, param, *branch)
            if skip_update:
                raise ValueError("Branches should not skip updates")
            outputs.append(branch_update)
        return self.merge_fn(outputs)


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key, lambda: torch.zeros_like(ref, dtype=dtype, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get("storage_dtype", "float32")
    return getattr(torch, dtype)


def _init_mu_product(state, group, update, grad, param, **kwargs):
    dtype = _storage_dtype(group)
    state["mu_product"] = torch.ones((), dtype=dtype, device=param.device)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn, names)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        for name in self.names:
            _zero_guard(state, self.val_name(name), param, _storage_dtype(group))

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class PrecondGradAccumGuard(FunctionTransform):
    def __init__(self, fn):
        super().__init__(fn, ["precond_grad_accum"])
        self.steps_taken = 0
        self.pass_through = None

    def _accum(self, state, new):
        self.steps_taken += 1
        utils.stochastic_add_(state, new)

    def _reset(self, state):
        if self.steps_taken != 0:
            self.steps_taken = 0
            utils.zero_(state)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        if self.pass_through is None:
            self.pass_through = not group.get("precond_grad_accum", False)
        if self.pass_through is False:
            for name in self.names:
                _zero_guard(state, self.val_name(name), param, _storage_dtype(group))

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        base_grad = update if group.get("momentum_into_precond_update", True) else grad
        if self.pass_through:
            return self.fn(state, group, update, grad, param, *args, base_grad, **kwargs)

        (vars,) = vars
        if group["is_preconditioning"]:
            if self.steps_taken:
                self._accum(vars, base_grad)
                utils.stochastic_multiply_(vars, 1 / self.steps_taken)
            else:
                vars = base_grad
        else:
            self._accum(vars, base_grad)
            vars = base_grad
        try:
            out = self.fn(state, group, update, grad, param, *args, vars, **kwargs)
        finally:
            if group["is_preconditioning"]:
                self._reset(vars)

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
        self.named_to_anonymous = None
        self.anonymous_to_named = None

    def _map(self, state_fn, param, mapping):
        for p in param:
            state = state_fn(p)
            for name, mapped in mapping.items():
                if mapped in state:
                    raise ValueError(f"Name {name} already mapped to {mapped}")
                if name in state:
                    state[mapped] = state.pop(name)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        self.init_fn(state, group, update, grad, param, **kwargs)
        for name in self.names:
            state[self.val_name(name)] = state.pop(name, None)
        if self.skip_first:
            raise SkipUpdate from None

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class NoState(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        return self.fn(group, update, grad, param, *args, **kwargs)


class NoStateNoForeach(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        updates = []
        skip_update = False
        for a in zip(update, grad, param, *args):
            try:
                updates.append(self.fn(group, *a, **kwargs))
            except SkipUpdate:
                skip_update = True
                pass
        if skip_update:
            raise SkipUpdate from None
        return updates


class SqueezeGrad(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        original_shapes = [u.shape for u in update]
        update = [u.squeeze() if u.numel() > 1 else u.view(-1) for u in update]
        grad = [x.view_as(u) for x, u in zip(grad, update)]
        param = [x.view_as(u) for x, u in zip(param, update)]
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, (list, tuple)) and isinstance(a[0], Tensor):
                args[i] = [x.view_as(u) for x, u in zip(a, update)]
        for k, a in kwargs.items():
            if isinstance(a, (list, tuple)) and isinstance(a[0], Tensor):
                kwargs[k] = [x.view_as(u) for x, u in zip(a, update)]
        out = self.fn(state, group, update, grad, param, *args, **kwargs)
        return [o.view(s) for o, s in zip(out, original_shapes)]


def zero_guard(*names):
    return functools.partial(ZeroGuard, names=names)


def copy_guard(index, *names):
    return functools.partial(CopyGuard, index=index, names=names)


def general_guard(*names, init_fn, skip_first: bool = True):
    return functools.partial(GeneralGuard, names=names, init_fn=init_fn, skip_first=skip_first)


def no_state(fn):
    return NoState(fn)


def no_state_no_foreach(fn):
    return NoStateNoForeach(fn)


class SkipUpdate(ValueError):
    pass


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


@no_state
def cautious_weight_decay(group, update, grad, param):
    utils.cautious_weight_decay_(param, update, group["cautious_weight_decay"] * group["lr"])
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
        exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group["step"]), group["eps"]
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
    )
    raise SkipUpdate from None


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
    )
    raise SkipUpdate from None


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
    )
    raise SkipUpdate from None


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
        group.get("beta3_warmup"),
        group.get("alpha_warmup"),
    )
    raise SkipUpdate from None


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
    )
    raise SkipUpdate from None


@no_state
def orthogonalize_grad_to_param(group, update, grad, param):
    return utils.orthogonalize_grad_to_param(param, update, group["eps"])


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    group["weight_sum"] = utils.schedule_free_(
        group["lr"],
        group["weight_lr_power"],
        group.get("weight_sum", 0),
        utils.get_beta1(group),
        param,
        z,
        update,
        grad,
        group["caution"],
        group["r"],
        group["step"],
        group["weight_decay"],
    )
    raise SkipUpdate from None


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
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate from None

    if group["step"] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group["eps"])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(
            exp_avg_sq,
            update,
            utils.beta_debias(utils.get_beta2(group), group["step"]),
            group["eps"],
        )
        raise SkipUpdate from None

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
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq", "fisher_approx")
@no_state_no_foreach
def scale_by_suds(group, update, grad, param, exp_avg, exp_avg_sq, fisher_approx):
    if group["step"] == 1:
        utils.copy_stochastic_(fisher_approx, update / update.norm().clamp(min=1e-8))
        raise SkipUpdate from None

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
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate from None

    if group["step"] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group["eps"])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(
            exp_avg_sq,
            update,
            utils.beta_debias(utils.get_beta2(group), group["step"]),
            group["eps"],
        )
        raise SkipUpdate from None

    return utils.adopt(
        update,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
    )


def _init_psgd_kron(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        getattr(param, "hessian_vector", None),
        getattr(param, "vector", None),
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["Q"] = utils.triu_to_line(Q) if group["store_triu_as_line"] else Q
    state["running_lower_bound"] = [torch.zeros((1,), device=q.device, dtype=torch.float64) for q in Q]
    state["step"] = torch.zeros((), device=param.device, dtype=torch.float64)  # torch casts int to float in ckpt load
    if group["adaptive"]:
        state["velocity"] = [torch.zeros((), device=q.device, dtype=q.dtype) for q in Q]
    if not cached:
        return

    state["Q_cache"] = [torch.empty_like(q) for q in Q]


def _init_psgd_lra(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    state["U"], state["V"], state["d"] = utils.init_lra(
        grad,
        group["param_count"],
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["rank"],
        getattr(param, "hessian_vector", None),
        getattr(param, "vector", None),
        dtype=getattr(torch, group["q_dtype"]),
    )


def precond_schedule(group, prob: Union[callable, float, None] = None, name: str = "cumulative_prob"):
    step = group["step"]
    if "precondition_frequency" in group:
        return step > 0 and step % group["precondition_frequency"] == 0
    if isinstance(step, torch.Tensor):
        utils.warn_once("Preconditioner schedule is not supported with torch.Tensor step.")
        rng = random.Random(0x172381)
    else:
        rng = random.Random(0x172381 ^ step)
    if "precond_scheduler" in group:
        return utils.precond_schedule(step, group["precond_scheduler"], rng)
    if prob is not None:
        return utils.psgd_should_update(group, prob, rng, name=name)
    raise ValueError("No preconditioner update schedule specified.")


@no_state_no_foreach
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


def _store_std(state, group, update, grad, param):
    state["init_std"] = torch.std(param)


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


@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_delta, log_space=False), skip_first=False)
@no_state
def scale_by_d_adaptation(group, update, grad, param, state, delta):
    utils.d_adaptation(grad, update, state, delta)
    return update


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


@general_guard("scion_state", init_fn=_init_scion_state, skip_first=False)
@no_state
def scion_auto_norm(group, update, grad, param, scion_state):
    scale = group.get("scale", 1.0)
    for ctx, p in zip(scion_state, param):
        if not ctx["initialized"]:
            utils.scion_auto_init_param_(p, scale)
            ctx["initialized"] = True
    return utils.scion_auto_lmo_(update, scale)


def _init_soap(state, group, update, grad, param):
    utils.init_preconditioner(grad, state, group["max_precond_dim"], group["precondition_1d"])


def _apply_soap_preconditioner(group, update, Q, GG, *references):
    for upd, q, gg, *ref in zip(update, Q, GG, *references):
        utils.update_preconditioner(
            utils.promote(upd),
            q,
            gg,
            ref,
            group["max_precond_dim"],
            group["precondition_1d"],
            utils.beta_debias(group["shampoo_beta"], group["step"]),
            group["is_preconditioning"],
        )


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


def _update_psgd_precond(
    cached, Q_cache, group, param, grad, Q, velocity, running_lower_bound, step, prob: Optional[callable] = None
) -> Optional[Tensor]:
    if prob is None:
        prob = utils.precond_update_prob_schedule()

    if not group["is_preconditioning"]:
        return

    if utils.hasattr_none(param, "vector"):
        vector, hessian_vector = param.vector, param.hessian_vector
        del param.vector
        del param.hessian_vector
    elif group["inverse_free"]:
        vector, hessian_vector = None, grad
    else:
        vector, hessian_vector = utils.dampen_grad(grad, group["dampening"])

    precond = utils.psgd_update_precond(
        hessian_vector,
        group["precond_lr"],
        Q,
        group["store_triu_as_line"],
        velocity,
        utils.get_beta2(group),
        group["ortho_method"],
        vector,
        running_lower_bound,
        group["lower_bound_beta"],
        group["precond_update_power_iterations"],
    )
    del vector, hessian_vector

    if isinstance(prob, float):
        float_prob = prob
    else:
        prob_step = group.get(f"cumulative_prob_{id(Q)}_prob_step", 1)
        float_prob = prob(prob_step)
        group[f"cumulative_prob_{id(Q)}_prob_step"] = prob_step + 1
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if precond is not None:
        return precond
    if not should_use_cache or not cached:
        return None  # caching adds extra ops and is not worth the overhead when we precondition at every step

    for c_, q_ in zip(Q_cache, utils.line_to_triu(Q, group["inverse_free"]) if group["store_triu_as_line"] else Q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return None


def _cached_psgd_precond_grad(group, update, Q, Q_cache, grad):
    kwargs = {"ea": update, "caution": group["caution"], "grad": grad}
    if group.get("is_cached", False):
        out = utils.precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        out = utils.psgd_precond_grad(
            preconds=Q, store_triu_as_line=group["store_triu_as_line"], symmetric_output=group["inverse_free"], **kwargs
        )
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
    }
    if group.get("is_cached", False):
        utils.fused_precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        utils.fused_psgd_precond_grad(
            preconds=Q, store_triu_as_line=group["store_triu_as_line"], symmetric_output=group["inverse_free"], **kwargs
        )


def _update_lra(
    group, U: List[Tensor], V: List[Tensor], d: List[Tensor], params: List[Tensor], grads: List[Tensor], delayed: bool
):
    if not group["is_preconditioning"]:
        return utils.multi_flatten((U, 1), (V, 1), (d, 0))

    if utils.hasattr_none(params[0], "hessian_vector"):
        vector = utils.flatten([p.vector for p in params])
        hessian_vector = utils.flatten([p.hessian_vector for p in params])
        for p in params:
            del p.vector
            del p.hessian_vector
    else:
        vector, hessian_vector = utils.dampen_multiple(grads)
    precond_step = group["precond_step"] = group.get("precond_step", -1) + 1
    return utils.update_lra_precond_(
        U, V, d, vector, hessian_vector, group["eps"], group["precond_lr"], delayed, bool(precond_step % 2)
    )


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    utils.apply_lra_update(param, update, u, v, d, group["lr"], group["weight_decay"], group["caution"], grad)
    raise SkipUpdate from None


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    utils.apply_lra_update(param, update, u, v, d, group["lr"], group["weight_decay"], group["caution"], grad)
    raise SkipUpdate from None


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def scale_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, step, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def scale_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    if group.get("inverse_free", False):
        precond = None
    else:
        precond = _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)
    new = _update_psgd_precond(
        cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, step, prob
    )
    return new if precond is None else precond


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def update_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, step, prob)
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    raise SkipUpdate from None


@no_state
def sign(group, update, grad, param, graft: bool = True):
    return utils.sign_(update, graft)


@no_state
def global_clip(group, update, grad, param, clip_fn: Optional[callable] = None):
    assert clip_fn is not None
    return clip_fn(update)


@SqueezeGrad
@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", "step", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def update_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    step: Tensor,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, step, prob)
    raise SkipUpdate from None


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group["step"] ** -group["beta2_scale"]
    group["betas"] = (utils.get_beta1(group), beta2)
    return update


def apply_to_idx(fn, idx):
    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        return fn(args[idx])

    return _fn


def _inner_chain(state, group, update, grad, param, *fns):
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
        if update is None:
            break
    return update, skip_update


def chain(state: Union[callable, dict], group, grad, param, *fns):
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]
    update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
    if not skip_update and update is not None:
        utils.update_param_(param, update, group["lr"], group["weight_decay"], caution=group["caution"], grad=grad)


def set_indices(fns: Iterable[callable], retain: bool = True, offset: int = 0):
    if retain and offset:
        raise ValueError("offset cannot be retained")

    def _walk(obj):
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, FunctionTransform):
                yield cur
                stack.append(cur.fn)
            elif isinstance(cur, functools.partial):
                stack.append(cur.func)
            elif isinstance(cur, Branch):
                for branch in cur.branches:
                    stack.extend(branch)
            elif isinstance(cur, _Iterable) and not isinstance(cur, (str, bytes, bytearray)):
                stack.extend(cur)

    if retain:
        offset = max((ft.transform_idx for ft in _walk(fns) if ft.transform_idx is not None), default=-1) + 1

    new_fns = [copy.deepcopy(fn) for fn in fns]
    for ft in _walk(new_fns):
        if not retain or ft.transform_idx is None:
            ft.transform_idx, offset = offset, offset + 1

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

    def __init__(self, params, defaults, foreach: bool, *fns):
        base = self.global_defaults.copy()
        base.update({k: v for k, v in defaults.items() if v is not use_default})
        super().__init__(params, base, foreach)
        self.fns = fns

    @property
    def fns(self):
        return self._fns

    @fns.setter
    def fns(self, value):
        self._fns = value
        self._set_indices(retain=True)

    def _set_indices(self, retain=True):
        self._fns = set_indices(self.fns, retain)

    def _step(self, group):
        if "base_lr" not in group:
            group["base_lr"] = group["lr"]
        if "prev_lr" in group and group["prev_lr"] != group["lr"]:
            utils.warn_once(
                f"Learning rate changed between steps. This is an experimental feature and "
                f"only supported with foreach=True (currently foreach={group['foreach']})."
            )
            group["base_lr"] = group["lr"]

        caution = group["caution"]

        vals = list(self.split_p_and_g_in_group(group, should_promote=self.promote, beta1=utils.get_beta1(group)))

        if not vals:
            return
        p, g = zip(*vals)

        for param in p:
            state = self.state_(param)
            if "step" in state:
                step = state["step"]
            elif self.compile_step:
                step = utils.scalar_guard(0, param)
            else:
                step = 0
            break

        group["step"] = state["step"] = step = step + 1
        group["prev_lr"] = group["lr"] = group["base_lr"] * step / max(step, group["warmup_steps"] + 1)

        if not group["foreach"] or len(p) == 1:
            for param, grad in zip(p, g):
                chain(self.state_, group, [grad], [param], *self.fns)
                group["caution"] = caution
        else:
            chain(self.state_, group, g, p, *self.fns)

        group["caution"] = caution
        group["lr"] = group["prev_lr"]
        group["step"] = None


str_or_fn = Union[str, callable, None, Literal[use_default]]


def _get_clip_fn(name: str_or_fn, default_val: str_or_fn):
    name = default(name, default_val)
    if callable(name):
        return name
    elif name not in (
        "l2_clip_",
        "rmsnorm_clip_",
        "trust_region_clip_",
        "a_law_compress",
        "mu_law_compress",
    ):
        raise ValueError(f"Clipping function {name} not found")
    return getattr(utils, name)


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
}


class BaseOpt(ChainOpt):
    """
    Base Optimizer

    compile_step: bool = False
    Whether to change some internals to try to make the optimizer compilable
    This does not compile the step by itself and breaks some optimizers loudly (e.g. SOAP)

    promote: bool = False
    Whether to promote the gradients to fp32 before applying the optimizer
    Improves update quality for low-precision parameters, but increases costs
    Compiling the optimizer step would reduce memory and compute. Alternatively, `foreach=False` decreases memory at the cost of runtime

    gradient_clipping: str_or_fn = None
    The function to use for clipping the incoming gradients, before any other transformations.
    This is syntactic sugar, equivalent to manually passing the function as the first element of the optimizer chain.

    update_clipping: str_or_fn = None
    The function to use for clipping the outgoing updates before applying them, after all other transformations.
    This will turn off fused updates.
    This is syntactic sugar, equivalent to manually passing the function as the last element of the optimizer chain.

    """

    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False
    auto_fuse: bool = True

    def __init__(
        self,
        params,
        defaults,
        foreach: bool = True,
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

        self.compile_step = default(compile_step, self.compile_step)
        self.promote = default(promote, self.promote)
        if default(palm, self.palm):
            fns = (palm_beta2,) + fns
        if default(gradient_clipping, self.gradient_clipping) is not None:
            fns = (apply_to_idx(gradient_clipping, 2),) + fns
        if default(update_clipping, self.update_clipping) is not None:
            fns = fns + (apply_to_idx(update_clipping, 2),)

        super().__init__(params, defaults, foreach, *fns)


class ScheduleFree(BaseOpt):
    def eval(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and not train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        # Set p.data to x
                        z = utils.promote(state["z"])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        utils.copy_stochastic_(p.data, p32)

    def train(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        z = utils.promote(state["z"])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        utils.copy_stochastic_(p.data, p32)


class MSAM(BaseOpt):
    def eval(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            if not train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        p_copy = p.data.clone()
                        utils.copy_stochastic_(p.data, state["z"])
                        utils.copy_stochastic_(state["z"], p_copy)

    def train(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            if train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        p_copy = p.data.clone()
                        utils.copy_stochastic_(p.data, state["z"])
                        utils.copy_stochastic_(state["z"], p_copy)
