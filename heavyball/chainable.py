import functools
import random
from typing import List, Literal, Optional, Union

import torch

from . import utils

balance_probability: float = 0.01


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
    def __init__(self, fn):
        self.fn = fn
        self.fn_name = self.get_fn().__name__

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        raise NotImplementedError

    def get_fn(self):
        if hasattr(self.fn, "get_fn"):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        return f"{self.fn_name}_{name}"


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key, lambda: torch.zeros_like(ref, dtype=dtype, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get("storage_dtype", "float32")
    return getattr(torch, dtype)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn)
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        vars = [
            [_zero_guard(state(p), self.val_name(name), p, _storage_dtype(group)) for p in param]  #
            for name in self.names
        ]
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn)
        self.index = index
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        vars = [
            [_guard_in_state(state(p), self.val_name(name), lambda: torch.clone(v)) for p, v in zip(param, val)]  #
            for name in self.names
        ]
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class GeneralGuard(FunctionTransform):  # We can't guard against reuse in the general case
    def __init__(self, fn, names, init_fn, skip_first: bool = True):
        super().__init__(fn)
        self.names = names
        self.init_fn = init_fn
        self.skip_first = skip_first

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        vars = []
        skip_update = False
        # Note: The logic here assumes fn operates per-parameter due to loop structure
        for p, g, u in zip(param, grad, update):
            st = state(p)
            # Initialize state if keys are missing for the current parameter p
            skip_update |= _inplace_guard_(st, self.names, lambda: self.init_fn(st, group, u, g, p, **kwargs))
            # Collect state variables for the current parameter p
            vars.append([st[name] if isinstance(name, str) else st.get(name[0], name[1]) for name in self.names])

        if skip_update and self.skip_first:
            raise SkipUpdate

        # Call the wrapped function with collected state variables (transposed)
        # The original function fn will receive state variables corresponding to each parameter
        # E.g., if names = ["U", "V"], it receives (*args, all_U_values, all_V_values, **kwargs)
        return self.fn(state, group, update, grad, param, *args, *zip(*vars), **kwargs)


class NoState(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        return self.fn(group, update, grad, param, *args, **kwargs)


class NoStateNoForeach(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        updates = []
        skip_update = False
        # Iterate through individual parameters/gradients/updates
        for idx, single_param in enumerate(param):
            single_update = update[idx]
            single_grad = grad[idx]
            single_args = [a[idx] for a in args]  # Assuming args are lists aligned with params
            try:
                # Pass the state dictionary specific to this parameter
                param_state = state(single_param)
                # Call the function with individual items + unpacked state vars
                result = self.fn(group, single_update, single_grad, single_param, *single_args, **param_state, **kwargs)
                updates.append(result)
            except SkipUpdate:
                skip_update = True
                # If fused, the update was already done. No need to append anything.
                pass
        if skip_update and not updates:  # If all updates were skipped (likely fused)
            raise SkipUpdate
        elif skip_update and updates:  # Should not happen if fused correctly
            raise RuntimeError("Mixed skipped and non-skipped updates in no_state_no_foreach")
        return updates


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


@zero_guard("exp_avg")
@no_state
def weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.weight_decay_to_ema_(
        exp_avg,
        update,
        utils.beta_debias(group["ema_beta"], group["step"]),
        group["weight_decay_to_ema"] * group["lr"],
    )
    return update


@zero_guard("exp_avg")
@no_state
def l1_weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.l1_weight_decay_to_ema_(
        exp_avg,
        update,
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
    raise SkipUpdate


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
    raise SkipUpdate


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
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate

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
        raise SkipUpdate

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
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate

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
        raise SkipUpdate

    return utils.adopt(
        update,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
    )


def _init_soap(state, group, update, grad, param, inner: str = ""):
    utils.init_preconditioner(grad, state, group["max_precond_dim"], group["precondition_1d"])


def _init_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q, state["exprs"] = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        getattr(param, "hessian_vector", None),
        getattr(param, "vector", None),
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["Q"] = utils.triu_to_line(Q) if group["store_triu_as_line"] else Q

    if not cached:
        return

    state["Q_cache"] = [torch.empty_like(q) for q in Q]

    expr = [f"{c.upper()}{c}" if q_.ndim == 2 else c for c, q_ in zip(utils.einsum_base, Q)]
    expr = ",".join(expr)
    grad_expr = "".join(c for c, _ in zip(utils.einsum_base, grad.shape))
    out_expr = "".join(c.upper() if c.upper() in expr else c for c in grad_expr)
    expr = f"{expr},{grad_expr}->{out_expr}"

    state["cache_expr"] = expr


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
    if update.dim() == 1:
        return update
    original_shape = update.shape
    # doing it this way, as tmp and update are not guaranteed to share memory address or layout
    tmp = update.flatten(1, -1)
    utils.inplace_orthogonal_(tmp, utils.zeroth_power_mode, tmp, scale_mode)
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
    state["init_std"] = torch.std(grad, dim=0)


@general_guard("init_std", init_fn=_store_std)
@no_state
def mup_approx(group, updates, grads, params, init_std):
    _updates = [(u, i) for u, i in zip(updates, init_std) if u.ndim > 1]
    _updates, _init_std = zip(*_updates)
    utils.stochastic_multiply_(_updates, _init_std)
    return updates


@zero_guard("momentum")
@no_state
def heavyball_momentum(group, updates, grads, params, momentum):
    return utils.heavyball_momentum(momentum, updates, utils.get_beta1(group))


_optim_fns = {"adam": utils.adam_, "laprop": utils.laprop_}


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG, inner: str = "adam"):
    update = utils.promote(update)  # Promote to highest precision if needed

    grad_projected = [utils.project(u, q, False) for u, q in zip(update, Q)]
    fn = _optim_fns[inner]
    precond = fn(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]

    for u, q, gg, ea in zip(update, Q, GG, exp_avg):
        utils.update_preconditioner(
            u,
            q,
            gg,
            ea,
            group["max_precond_dim"],
            group["precondition_1d"],
            utils.beta_debias(group["shampoo_beta"], group["step"]),
            group["is_preconditioning"],
        )
    return precond


def _update_psgd_precond(cached, Q_cache, group, param, grad, Q_mat, Q, exprs, prob: Optional[callable] = None):
    if prob is None:
        prob = utils.precond_update_prob_schedule()

    if not group["is_preconditioning"]:
        return Q_mat

    utils.psgd_update_precond(
        Q_mat,
        exprs,
        getattr(param, "hessian_vector", grad),
        group["precond_lr"],
        Q,
        group["store_triu_as_line"],
        getattr(param, "vector", None),
    )
    if hasattr(param, "vector"):
        del param.vector
        del param.hessian_vector

    if grad.dim() > 1 and precond_schedule(group, balance_probability, f"balance_prob_{id(Q)}"):
        if group["store_triu_as_line"]:
            utils.psgd_balance_Q([q_ for _, q_ in Q])
        else:
            utils.psgd_balance_Q(Q)

    if isinstance(prob, float):
        float_prob = prob
    else:
        float_prob = prob(group.get(f"cumulative_prob_{id(Q)}_prob_step", 1))
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if should_use_cache:  # caching adds extra ops and is not worth the overhead when we precondition at every step
        return _update_psgd_cache(cached, Q_cache, Q_mat)
    return Q_mat


def _update_psgd_cache(cached, Q_cache, q):
    if not cached:
        return q

    for c_, q_ in zip(Q_cache, q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return Q_cache


def _cached_psgd_precond_grad(group, cache_expr, exprs, update, Q_mat, Q_cache, grad):
    if group.get("is_cached", False):
        out = utils.precond_grad_cached_(cache_expr, update, *Q_cache, caution=group["caution"], grad=grad)
    out = utils.psgd_precond_grad(exprs[-1], update, *Q_mat, caution=group["caution"], grad=grad)
    group["caution"] = False  # we already cautioned here - shouldn't do it again
    return out


def _fused_cached_psgd_precond_grad(group, grad, param, cache_expr, exprs, update, Q_mat, Q_cache):
    if group.get("is_cached", False):
        utils.fused_precond_grad_cached_(
            cache_expr,
            update,
            param,
            group["lr"],
            grad,
            group["weight_decay"],
            group["caution"],
            *Q_cache,
        )
    else:
        utils.fused_psgd_precond_grad(
            exprs[-1],
            update,
            param,
            group["lr"],
            grad,
            group["weight_decay"],
            group["caution"],
            *Q_mat,
        )


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def scale_by_psgd(
    group,
    update,
    grad,
    param,
    Q,
    exprs,
    Q_cache,
    cache_expr: str,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    update = update.to(memory_format=torch.contiguous_format)
    Q_mat = utils.line_to_triu(Q) if group["store_triu_as_line"] else Q
    Q_mat = _update_psgd_precond(
        cached,
        Q_cache,
        group,
        param,
        update if group["momentum_into_precond_update"] else grad,
        Q_mat,
        Q,
        exprs,
        prob,
    )
    return _cached_psgd_precond_grad(group, cache_expr, exprs, update, Q_mat, Q_cache, grad)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def scale_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    Q,
    exprs,
    Q_cache,
    cache_expr: str,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    Q_mat = utils.line_to_triu(Q) if group["store_triu_as_line"] else Q
    precond = _cached_psgd_precond_grad(group, cache_expr, exprs, update, Q_mat, Q_cache, grad)
    _ = _update_psgd_precond(
        cached,
        Q_cache,
        group,
        param,
        update if group["momentum_into_precond_update"] else grad,
        Q_mat,
        Q,
        exprs,
        prob,
    )
    return precond


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def update_by_psgd(
    group,
    update,
    grad,
    param,
    Q,
    exprs,
    Q_cache,
    cache_expr: str,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    Q_mat = utils.line_to_triu(Q) if group["store_triu_as_line"] else Q
    Q_mat = _update_psgd_precond(
        cached,
        Q_cache,
        group,
        param,
        update if group["momentum_into_precond_update"] else grad,
        Q_mat,
        Q,
        exprs,
        prob,
    )
    _fused_cached_psgd_precond_grad(group, update, param, cache_expr, exprs, update, Q_mat, Q_cache)
    raise SkipUpdate


@no_state
def sign(group, update, grad, param, graft: bool = True):
    return utils.sign_(update, graft)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def update_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    Q,
    exprs,
    Q_cache,
    cache_expr: str,
    cached: bool = False,
    prob: Optional[callable] = None,
):
    Q_mat = utils.line_to_triu(Q) if group["store_triu_as_line"] else Q
    _fused_cached_psgd_precond_grad(group, update, param, cache_expr, exprs, update, Q_mat, Q_cache)
    _ = _update_psgd_precond(
        cached,
        Q_cache,
        group,
        param,
        update if group["momentum_into_precond_update"] else grad,
        Q_mat,
        Q,
        exprs,
        prob,
    )
    raise SkipUpdate


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
        # Apply weight decay *before* the final update if not decoupled
        if not group.get("decoupled_weight_decay", False) and group["weight_decay"] != 0:
            [p.add_(p.data, alpha=group["weight_decay"]) for p in param]  # WD = WD + P

        utils.update_param_(
            param, update, group["lr"], 0.0, caution=group["caution"], grad=grad
        )  # WD handled above or inside fused step


def create_branch(branches: List[List[callable]], merge_fn: callable):
    def _branch(state, group, update, grad, param):
        outputs = []
        for branch in branches:
            branch_update = [torch.clone(u, memory_format=torch.preserve_format) for u in update]
            branch_update, skip_update = _inner_chain(state, group, branch_update, grad, param, *branch)
            if skip_update:
                raise ValueError("Branches should not skip updates")
            outputs.append(branch_update)
        return merge_fn(outputs)

    return _branch


class ChainOpt(utils.StatefulOptimizer):
    promote: bool = False

    def __init__(self, params, defaults, foreach: bool, *fns):
        super().__init__(params, defaults, foreach)
        self.fns = tuple(fns)

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

        # Determine current step
        state_ref = self.state_(p[0])
        step = state_ref.get("step", 0)
        step += 1
        # Update step in group and state for all params
        group["step"] = step
        for param in p:
            self.state_(param)["step"] = step

        # Warmup LR
        group["prev_lr"] = group["lr"] = group["base_lr"] * min(1.0, step / max(1, group["warmup_steps"]))

        # Apply optimizer chain
        if not group["foreach"] or len(p) == 1:
            for param, grad in zip(p, g):
                # Pass state_ function, not the dict directly
                chain(self.state_, group, [grad], [param], *self.fns)
        else:
            # Pass state_ function, not the dict directly
            chain(self.state_, group, g, p, *self.fns)

        group["caution"] = caution
        # group["lr"] = group["prev_lr"] # LR is handled within chain for fused optimizers now
        # group["step"] = None # Step needs to persist in state


use_default = object()
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


# Map scaling functions to their fused update equivalents
_scale_to_update_map = {
    scale_by_psgd.get_fn(): update_by_psgd,
    scale_by_delayed_psgd.get_fn(): update_by_delayed_psgd,
    scale_by_adam.get_fn(): update_by_adam,
    scale_by_laprop.get_fn(): update_by_laprop,
    scale_by_adopt.get_fn(): update_by_adopt,
    # scale_by_psgd_lra is implicitly fused, map it to itself (or a dummy update func if needed)
    # scale_by_psgd_lra.get_fn(): update_by_psgd_lra # Placeholder if we create a separate update func
}
_scale_to_update_map_inv = {v: k for k, v in _scale_to_update_map.items()}


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
    This will turn off
    This is syntactic sugar, equivalent to manually passing the function as the last element of the optimizer chain.

    """

    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False
    auto_fuse: bool = True
    compile_step: bool = False  # Default value for compile_step

    def __init__(
        self,
        params,
        defaults,
        foreach: bool,
        gradient_clipping: str_or_fn,
        update_clipping: str_or_fn,
        palm: bool = use_default,
        *fns,
        compile_step: bool = use_default,
        promote: bool = use_default,
    ):
        if not fns:
            raise ValueError("No functions provided. If that's on purpose (SGD-like), use `identity`")

        args, kwargs = None, None
        fn = fns[-1]
        if isinstance(fn, functools.partial):
            fn, args, kwargs = fn.func, fn.args, fn.keywords
        if isinstance(fn, FunctionTransform):
            fn = fn.get_fn()

        # Check if the last function is a fused update function
        is_fused = fn in _scale_to_update_map_inv

        if default(update_clipping, self.update_clipping) is None:
            # Try to fuse if auto_fuse is enabled and the last function is a scale function
            if self.auto_fuse and fn in _scale_to_update_map:
                fn = _scale_to_update_map[fn]
                if args is not None:
                    fn = functools.partial(fn, *args, **kwargs)
                fns = tuple(fns)[:-1] + (fn,)
                is_fused = True  # Mark as fused
        elif fn in _scale_to_update_map_inv:
            # If update clipping exists, and the last fn is an update_by_*, unfuse it
            if not self.auto_fuse:
                raise ValueError(
                    "update_clipping is currently not compatible with update_by_* functions. "
                    "Manually select scale_by_* functions or set auto_fuse=True."
                )
            fn = _scale_to_update_map_inv[fn]
            if args is not None:
                fn = functools.partial(fn, *args, **kwargs)
            fns = tuple(fns)[:-1] + (fn,)
            is_fused = False  # Mark as not fused

        # Handle LR and Weight Decay based on whether the step is fused
        defaults["decoupled_weight_decay"] = defaults.get("decoupled_weight_decay", False)
        if is_fused:
            # If fused, the chainable function handles LR and possibly WD.
            # Set optimizer's lr=1.0 and wd=0.0 to avoid double application.
            defaults["_original_lr"] = defaults.get("lr", 1.0)  # Store original LR if needed inside chainable
            defaults["_original_wd"] = defaults.get("weight_decay", 0.0)  # Store original WD
            defaults["lr"] = 1.0
            if not defaults["decoupled_weight_decay"]:
                defaults["weight_decay"] = 0.0  # Non-decoupled WD handled in chainable
        # else: LR and WD applied by the base ChainOpt.chain function

        self.compile_step = default(compile_step, self.compile_step)
        self.promote = default(promote, self.promote)
        if default(palm, self.palm):
            fns = (palm_beta2,) + fns
        if default(gradient_clipping, self.gradient_clipping) is not None:
            grad_clip_func = _get_clip_fn(gradient_clipping, self.gradient_clipping)
            fns = (apply_to_idx(grad_clip_func, 2),) + fns  # Apply to grad (index 2)
        if default(update_clipping, self.update_clipping) is not None:
            update_clip_func = _get_clip_fn(update_clipping, self.update_clipping)
            fns = fns + (apply_to_idx(update_clip_func, 2),)  # Apply to update (index 2)

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


def _init_psgd_lra(state, group, update, grad, param):
    num_params = param.numel()
    rank = group["rank_of_approximation"]
    rank = min(rank, num_params)
    dtype = _storage_dtype(group)  # Use storage dtype
    device = param.device

    # Add +10 to rank denominator for numerical stability, as in original PSGD
    u_init_std = (num_params * (rank + 10)) ** -0.5
    state["U"] = torch.randn(num_params, rank, dtype=dtype, device=device) * u_init_std
    state["V"] = torch.randn(num_params, rank, dtype=dtype, device=device) * u_init_std

    # Initialize 'd' based on group setting, or None if dynamic init is needed
    if group["preconditioner_init_scale"] is not None:
        state["d"] = torch.ones(num_params, 1, dtype=dtype, device=device) * group["preconditioner_init_scale"]
    else:
        state["d"] = None  # Will be initialized on first update

    state["m"] = None  # Momentum buffer


@general_guard("U", "V", "d", "m", init_fn=_init_psgd_lra, skip_first=False)
@no_state_no_foreach  # This decorator means the function processes one param at a time
def scale_by_psgd_lra(group, update, grad, param, U, V, d, m):
    """
    Performs a single PSGD LRA step for one parameter.
    Includes preconditioner update and parameter update (fused).
    """
    num_params = param.numel()
    # Ensure rank is valid for this parameter
    rank = min(group["rank_of_approximation"], num_params)
    if U.shape[1] != rank:  # Adjust U,V shape if rank was capped
        u_init_std = (num_params * (rank + 10)) ** -0.5
        U = torch.randn(num_params, rank, dtype=U.dtype, device=U.device) * u_init_std
        V = torch.randn(num_params, rank, dtype=V.dtype, device=V.device) * u_init_std

    grad_flat = grad.detach().flatten().view(-1, 1)  # Use detach()

    # --- Momentum ---
    if group["momentum"] > 0:
        beta1 = group["momentum"]
        if m is None:
            # Initialize momentum buffer m if it's the first step or momentum was turned on
            m = torch.zeros_like(grad_flat)  # Initialize with zeros is safer
            m.add_(grad_flat, alpha=1 - beta1)
        else:
            m.mul_(beta1).add_(grad_flat, alpha=1 - beta1)
        current_update = m  # Use momentum buffer as the effective gradient/update
    else:
        m = None  # Clear momentum buffer if momentum is zero
        current_update = grad_flat

    tiny = torch.finfo(grad.dtype).tiny

    # --- Preconditioner Update ---
    if (
        precond_schedule(group, group["preconditioner_update_probability"], name=f"psgd_lra_prob_{id(param)}")
        or d is None
    ):
        if group["preconditioner_type"] == "Newton":
            # Check if HVP info is available (calculated by StatefulOptimizer._handle_closure)
            vs_flat = getattr(param, "vector", None)
            Hvs_flat = getattr(param, "hessian_vector", None)

            if vs_flat is None or Hvs_flat is None:
                # Fallback or error if HVP is needed but not provided
                if group["exact_hessian_vector_product"]:
                    raise RuntimeError(
                        "PSGD LRA (Newton) requires HVP. Ensure closure is provided and hessian_approx=True in optimizer."
                    )
                else:  # Approximate using damped gradient if exact HVP failed
                    vs_flat, Hvs_flat = utils.damped_pair_vg(current_update)
            else:
                vs_flat = vs_flat.detach().flatten().view(-1, 1)
                Hvs_flat = Hvs_flat.detach().flatten().view(-1, 1)
                # Apply L2 regularization if specified
                if group["l2_regularization"] > 0:
                    Hvs_flat.add_(vs_flat, alpha=group["l2_regularization"])

            # Initialize 'd' dynamically if needed
            if d is None:
                d = (
                    (torch.mean(vs_flat * vs_flat)) ** (1 / 4)
                    * (torch.mean(Hvs_flat**4)) ** (-1 / 8)
                    * torch.ones(num_params, 1, dtype=U.dtype, device=U.device)
                )

            # Update U, V, d using HVP
            utils.psgd_lra_update_precond_(
                U, V, d, vs_flat, Hvs_flat, group["lr_preconditioner"], group["step_normalizer"], tiny
            )

        else:  # Whitening type
            # Initialize 'd' dynamically if needed
            if d is None:
                # Use gradient statistics for initialization
                grad_abs_mean_pow4 = torch.mean((torch.abs(grad_flat)) ** 4)
                # Handle potential grad_abs_mean_pow4 being zero
                init_scale_d = grad_abs_mean_pow4 ** (-1 / 8) if grad_abs_mean_pow4 > tiny else 1.0
                d = init_scale_d * torch.ones(num_params, 1, dtype=U.dtype, device=U.device)

            # Update U, V, d using gradient (damped pair)
            vs_flat, Hvs_flat = utils.damped_pair_vg(current_update)  # Hvs_flat is actually the damped grad here
            utils.psgd_lra_update_precond_(
                U, V, d, vs_flat, Hvs_flat, group["lr_preconditioner"], group["step_normalizer"], tiny
            )

    # --- Preconditioned Gradient ---
    # Apply preconditioner Q = (I + UV')*diag(d) -> P = Q'Q
    pre_grad = utils.precond_grad_psgd_lra_(U, V, d, current_update)

    # Effective learning rate for parameters
    lr = group["lr_params"]

    # --- Parameter Update ---
    delta = lr * pre_grad

    # Apply decoupled weight decay if enabled
    if group["decoupled_weight_decay"] and group["weight_decay"] > 0:
        param.data.add_(param.data, alpha=-group["weight_decay"] * group["lr_params"])

    param.data.sub_(delta.view_as(param.data))
    raise SkipUpdate  # Indicate that the update was applied internally


# =====================================
