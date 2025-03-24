import functools
import random
from typing import Literal, Optional, Union

import torch

from . import utils
from .utils import decorator_knowngood, modify_closure

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
        if hasattr(self.fn, 'get_fn'):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        return f"{self.fn_name}_{name}"


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key, lambda: torch.zeros_like(ref, dtype=dtype, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get('storage_dtype', "float32")
    return getattr(torch, dtype)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn)
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        vars = [[_zero_guard(state(p), self.val_name(name), p, _storage_dtype(group)) for p in param]  #
                for name in self.names]
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn)
        self.index = index
        self.names = names

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        vars = [[_guard_in_state(state(p), self.val_name(name), lambda: torch.clone(v)) for p, v in zip(param, val)]  #
                for name in self.names]
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
        for p, g, u in zip(param, grad, update):
            st = state(p)
            skip_update |= _inplace_guard_(st, self.names, lambda: self.init_fn(st, group, u, g, p, **kwargs))
            vars.append([st[name] if isinstance(name, str) else st.get(name[0], name[1]) for name in self.names])
        if skip_update and self.skip_first:
            raise SkipUpdate
        return self.fn(state, group, update, grad, param, *args, *zip(*vars), **kwargs)


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
            raise SkipUpdate
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


@zero_guard('exp_avg')
@no_state
def weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.weight_decay_to_ema_(exp_avg, update, utils.beta_debias(group['ema_beta'], group['step']),
                               group['weight_decay_to_ema'] * group['lr'])
    return update


@zero_guard('exp_avg')
@no_state
def l1_weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.l1_weight_decay_to_ema_(exp_avg, update, utils.beta_debias(group['ema_beta'], group['step']),
                                  group['weight_decay_to_ema'] * group['lr'])
    return update


@zero_guard("exp_avg_sq")
@no_state
def scale_by_exp_avg_sq(group, update, grad, param, exp_avg_sq):
    return utils.scale_by_exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group["step"]),
                                      group['eps'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adam_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group['step'],  #
                       group['eps'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(param, exp_avg, exp_avg_sq, update, grad, utils.get_beta1(group), utils.get_beta2(group),
                      group['step'], group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.laprop_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group['step'])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_laprop_(param, exp_avg, exp_avg_sq, update, grad, utils.get_beta1(group), utils.get_beta2(group),
                        group['step'], group['lr'], group['weight_decay'], group['caution'])
    raise SkipUpdate


def _init_hessian_estimator(state, group, update, grad, param, **kwargs):
    """Initialize Sophia's Hessian estimator state."""
    state['hessian_step'] = 0
    state['next_hessian_update'] = 1  # Update on first step


@no_state_no_foreach
def estimate_hessian_h(group, update, grad, param):
    """Estimate diagonal Hessian using Hutchinson's method."""
    state = getattr(param, 'optimizer_state', {})
    k = group.get('sophia_update_freq', 10)
    beta2 = utils.get_beta2(group)

    if 'hessian_step' not in state:
        _init_hessian_estimator(state, group, update, grad, param)

    state['hessian_step'] += 1

    # Check if it's time to update the Hessian estimate
    if state['hessian_step'] >= state['next_hessian_update']:
        state['next_hessian_update'] = state['hessian_step'] + k

        # Generate random vector for Hutchinson's method
        v = torch.randn_like(grad)

        # Compute Hessian-vector product
        with torch.enable_grad():
            loss = modify_closure(lambda: torch.sum(grad * v))
            g = torch.autograd.grad(loss, param, create_graph=True)[0]
            h = v * g  # Element-wise product gives diagonal estimate

        # Update EMA of diagonal Hessian
        if 'diag_hessian' not in state:
            state['diag_hessian'] = h.abs().clone()
        else:
            state['diag_hessian'].mul_(beta2).add_(h.abs(), alpha=1-beta2)

        # Ensure positive values for numerical stability
        state['diag_hessian'].clamp_(min=1e-6)

    param.optimizer_state = state
    return update


def _hutchinson_hessian(state, param, grad):
    """Helper function to compute Hutchinson's diagonal Hessian estimate."""

    return grad * grad



@general_guard(["diag_hessian", "hessian_step", "next_hessian_update"], init_fn=_init_hessian_estimator, skip_first=False)
@no_state
def estimate_hessian_g(state, group, update, grad, param):
    """Estimate diagonal Hessian using Gauss-Newton-Bartlett method."""
    k = group.get('sophia_update_freq', 10)
    beta2 = utils.get_beta2(group)
    state['hessian_step'] += 1

    # Check if it's time to update the Hessian estimate
    if state['hessian_step'] >= state['next_hessian_update']:
        state['next_hessian_update'] = state['hessian_step'] + k

        # For GNB we can use a simplified approximation based on squared gradients
        h = grad * grad  # Simplified GNB estimator

        # Update EMA of diagonal Hessian
        if 'diag_hessian' in state:
            state['diag_hessian'].mul_(beta2).add_(h.abs(), alpha=1-beta2)
        else:
            state['diag_hessian'] = h.abs().clone()

        # Ensure positive values for numerical stability
        state['diag_hessian'].clamp_(min=1e-6)

    return update


@zero_guard("exp_avg", "diag_hessian")
@no_state
def scale_by_sophia(group, update, grad, param, exp_avg, diag_hessian):
    """
    Sophia optimizer update rule. Divides the EMA of gradients by the EMA of the diagonal Hessian,
    and then applies element-wise clipping.
    """
    # Update momentum using beta1
    beta1 = utils.get_beta1(group)
    utils.scale_by_exp_avg_(exp_avg, update, utils.beta_debias(beta1, group["step"]))

    # Get clipping threshold
    gamma = group.get('sophia_gamma', 0.01)
    eps = group.get('eps', 1e-12)

    # For each parameter
    for u, m, h in zip(update, exp_avg, diag_hessian):
        # Computation for each parameters : mt / max(γ * ht, ε)
        m32 = utils.promote(m)
        h32 = utils.promote(h)

        # Compute denominator with safeguard
        denom = (gamma * h32).clamp(min=eps)

        # Scale and clip the update
        scaled = m32 / denom
        scaled.clamp_(min=-1.0, max=1.0)

        # Copy back to update
        utils.copy_stochastic_(u, scaled)

    return update

def _init_sophia_state(state, group, update, grad, param):
    """Initialize Sophia's Hessian state."""
    state['hessian_step'] = 0
    state['next_hessian_update'] = 1

    # Initialize diagonal Hessian with squared gradient (simple approximation)
    # Both SophiaH and SophiaG will start with this approximation
    state['diag_hessian'] = (grad.detach() * grad.detach()).clone()

def _init_adalomo(state, group, update, grad, param):
    """
    Initialize AdaLomo optimizer state.

    Sets up row sums (r_t) and column sums (c_t) matrices for the factorized second moment estimation.

    Args:
        state: Optimizer state dictionary for the parameter
        group: Parameter group dictionary
        update: Parameter update/gradient
        grad: Raw gradient
        param: Parameter tensor
    """
    # Original shape for reshaping if needed
    original_shape = update.shape

    # Initialize row sum matrix (r_t)
    if len(original_shape) <= 1:
        # For scalar or 1D tensors, r_t is just the squared gradient
        state['r_t'] = torch.zeros_like(update)
    else:
        # For multi-dimensional tensors, r_t tracks row sums
        # Shape is [original_shape[0], 1]
        state['r_t'] = torch.zeros([original_shape[0], 1],
                                   device=update.device,
                                   dtype=update.dtype)

    # Initialize column sum matrix (c_t)
    if len(original_shape) <= 1:
        # For 1D tensors, c_t is not needed (will match r_t)
        state['c_t'] = state['r_t']
    else:
        # For multi-dimensional tensors, c_t tracks column sums
        # Reshape for matrix operations - remaining dimensions flattened
        flattened_size = original_shape[1:].numel()
        # Shape is [1, flattened_size]
        state['c_t'] = torch.zeros([1, flattened_size],
                                   device=update.device,
                                   dtype=update.dtype)

def _adalomo_factorized_v_t(update, state, beta, eps):
    """
    Computes v_t using AdaLomo's factorized second moment estimation.
    This function is shared by both scale_by_adalomo and update_by_adalomo.
    """
    original_shape = update.shape

    if len(original_shape) <= 1:
        # Scalar or 1D case
        state['r_t'].mul_(beta).add_(update * update, alpha=1 - beta)
        return state['r_t']
    else:
        # Multi-dimensional case
        g_sq = update * update
        g_reshaped = g_sq.reshape(original_shape[0], -1)

        row_sum = g_reshaped.sum(dim=1, keepdim=True)
        state['r_t'].mul_(beta).add_(row_sum, alpha=1 - beta)

        col_sum = g_reshaped.sum(dim=0, keepdim=True)
        state['c_t'].mul_(beta).add_(col_sum, alpha=1 - beta)

        r_sum = state['r_t'].sum() + eps
        v_t = (state['r_t'] / r_sum) * state['c_t']
        return v_t.reshape(original_shape)

@general_guard("r_t", "c_t", init_fn=_init_adalomo)
@no_state_no_foreach
def update_by_adalomo(group, update, grad, param, r_t, c_t):
    """
    Fused AdaLomo update with parameter modification.
    """
    beta = group.get('beta', 0.99)
    eps = group.get('eps', 1e-8)
    lr = group.get('lr', 0.001)
    weight_decay = group.get('weight_decay', 0)

    state = {'r_t': r_t, 'c_t': c_t}
    v_t = _adalomo_factorized_v_t(update, state, beta, eps)

    # Compute adaptive learning rate update
    denom = (v_t + eps).sqrt()
    u_t = update / denom
    u_norm = utils.group_update_norm_([u_t], [param], eps)[0]

    # Apply weight decay and update parameters
    utils.update_param_([param], [u_norm], lr, weight_decay, caution=group.get('caution', False), grad=[grad])

    return update


@general_guard("r_t", "c_t", init_fn=_init_adalomo)
@no_state_no_foreach
def scale_by_adalomo(group, update, grad, param, r_t, c_t):
    """
    Scale gradients using AdaLomo's adaptive learning rate.
    Uses factorized second moment estimation.
    """
    beta = group.get('beta', 0.99)
    eps = group.get('eps', 1e-8)

    state = {'r_t': r_t, 'c_t': c_t}
    v_t = _adalomo_factorized_v_t(update, state, beta, eps)

    # Compute adaptive update
    denom = (v_t + eps).sqrt()
    u_t = update / denom
    utils.copy_stochastic_(update, u_t)  # Ensure safe copy

    return update


@general_guard("hessian_step", "next_hessian_update", init_fn=_init_sophia_state)
@no_state
def update_sophia_hessian(group, update, grad, param, diag_hessian, hessian_step, next_hessian_update):
    """
    Update the diagonal Hessian estimate used in Sophia optimizer.
    Both SophiaH and SophiaG use squared gradients to avoid autograd issues.
    """
    k = group.get('sophia_update_freq', 10)
    beta2 = utils.get_beta2(group)

    # Revise status value directory directly if hessian_step is tuple
    if isinstance(hessian_step, tuple):
        state = param.optimizer_state if hasattr(param, 'optimizer_state') else {}
        state['hessian_step'] = state.get('hessian_step', 0) + 1

        # Set next update time
        if state['hessian_step'] >= state.get('next_hessian_update', 1):
            state['next_hessian_update'] = state['hessian_step'] + k

            # Hessian approximation using squared gradient
            h = (grad.detach() * grad.detach())
            # Update EMA of diagonal Hessian
            if 'diag_hessian' in state:
                state['diag_hessian'].mul_(beta2).add_(h, alpha=1-beta2)
            else:
                state['diag_hessian'] = h.clone()

            # Set minimum value for numerical stability
            state['diag_hessian'].clamp_(min=1e-6)
    else:
        hessian_step[0] += 1

        # Check if it's time to update the Hessian estimate
        if hessian_step[0] >= next_hessian_update[0]:
            next_hessian_update[0] = hessian_step[0] + k

            # Hessian approximation using squared gradient
            h = (grad.detach() * grad.detach())

            # Update Hessian EMA
            diag_hessian.mul_(beta2).add_(h, alpha=1-beta2)

            # Set minimum value for numerical stability
            diag_hessian.clamp_(min=1e-6)

    return update

@zero_guard("exp_avg", "diag_hessian")
@no_state
@decorator_knowngood
def update_by_sophia(group, update, grad, param, exp_avg, diag_hessian):
    """
    Fused implementation of Sophia update that directly updates parameters.
    """
    # Get clipping threshold
    gamma = group.get('sophia_gamma', 0.01)

    # Update momentum using beta1
    beta1 = utils.get_beta1(group)
    utils.scale_by_exp_avg_(exp_avg, update, utils.beta_debias(beta1, group["step"]))

    # Calculate scaled update: mt / max(γ * ht, ε)
    eps = group.get('eps', 1e-12)

    # Apply weight decay and learning rate
    lr = group['lr']
    weight_decay = group['weight_decay']

    for p, u, m, h in zip(param, update, exp_avg, diag_hessian):
        if weight_decay != 0:
            p_data = utils.promote(p.data)
            p_data.mul_(1 - lr * weight_decay)
            utils.copy_stochastic_(p.data, p_data)

        m32 = utils.promote(m)
        h32 = utils.promote(h)

        # Compute denominator with safeguard
        denom = (gamma * h32).clamp(min=eps)

        # Calculate update with clipping
        scaled = m32 / denom
        scaled.clamp_(min=-1.0, max=1.0)

        # Apply update
        utils.copy_stochastic_(u, scaled)

    return update


@no_state
def orthogonalize_grad_to_param(group, update, grad, param):
    return utils.orthogonalize_grad_to_param(param, update, group['eps'])


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    group['weight_sum'] = utils.schedule_free_(group['lr'], group['weight_lr_power'], group.get('weight_sum', 0),
                                               utils.get_beta1(group), param, z, update, grad, group['caution'],
                                               group['r'], group['step'], group['weight_decay'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group['step'] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group['eps'])
        raise SkipUpdate

    if group['step'] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group['eps'])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group['step']),
                                   group['eps'])
        raise SkipUpdate

    utils.fused_adopt_(param, update, grad, exp_avg_sq, exp_avg, utils.get_beta1(group), utils.get_beta2(group),
                       group['step'] - 2, group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group['step'] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group['eps'])
        raise SkipUpdate

    if group['step'] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group['eps'])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group['step']),
                                   group['eps'])
        raise SkipUpdate

    return utils.adopt(update, exp_avg_sq, exp_avg, utils.get_beta1(group), utils.get_beta2(group), group['step'] - 2)


def _init_soap(state, group, update, grad, param, inner: str = ''):
    utils.init_preconditioner(grad, state, group['max_precond_dim'], group['precondition_1d'])

def _init_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q, state["exprs"] = utils.init_Q_exprs(grad, group['precond_init_scale'], group['max_size_triangular'],
                                           group['min_ndim_triangular'], group['memory_save_mode'],
                                           dtype=getattr(torch, group['q_dtype']))
    state["Q"] = utils.triu_to_line(Q) if group['store_triu_as_line'] else Q

    if not cached:
        return

    state['Q_cache'] = [torch.empty_like(q) for q in Q]

    expr = [f'{c.upper()}{c}' if q_.ndim == 2 else c for c, q_ in zip(utils.einsum_base, Q)]
    expr = ','.join(expr)
    grad_expr = ''.join(c for c, _ in zip(utils.einsum_base, grad.shape))
    out_expr = ''.join(c.upper() if c.upper() in expr else c for c in grad_expr)
    expr = f'{expr},{grad_expr}->{out_expr}'

    state['cache_expr'] = expr


def precond_schedule(group, prob: Union[callable, float, None] = None, name: str = 'cumulative_prob'):
    step = group['step']
    if 'precondition_frequency' in group:
        return step > 0 and step % group['precondition_frequency'] == 0
    if isinstance(step, torch.Tensor):
        utils.warn_once("Preconditioner schedule is not supported with torch.Tensor step.")
        rng = random.Random(0x172381)
    else:
        rng = random.Random(0x172381 ^ step)
    if 'precond_scheduler' in group:
        return utils.precond_schedule(step, group['precond_scheduler'], rng)
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


@zero_guard('momentum')
@no_state
def nesterov_ema(group, updates, grads, params, momentum):  # equivalent to Grokfast
    return utils.nesterov_ema(momentum, updates, utils.get_beta1(group))


def _store_std(state, group, update, grad, param):
    state['init_std'] = torch.std(grad, dim=0)


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


_optim_fns = {'adam': utils.adam_, 'laprop': utils.laprop_}


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG, inner: str = 'adam'):
    update = utils.promote(update)  # Promote to highest precision if needed

    grad_projected = [utils.project(u, q, False) for u, q in zip(update, Q)]
    fn = _optim_fns[inner]
    precond = fn(exp_avg, exp_avg_sq, grad_projected, utils.get_beta1(group), utils.get_beta2(group), group['step'] - 1,
                 group['eps'])
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]

    for u, q, gg, ea in zip(update, Q, GG, exp_avg):
        utils.update_preconditioner(u, q, gg, ea, group['max_precond_dim'], group['precondition_1d'],
                                    utils.beta_debias(group['shampoo_beta'], group['step']),
                                    group['is_preconditioning'])
    return precond


def _update_psgd_precond(cached, Q_cache, group, param, grad, Q_mat, Q, exprs, prob: Optional[callable] = None):
    if prob is None:
        prob = utils.precond_update_prob_schedule()

    if not group['is_preconditioning']:
        return Q_mat

    utils.psgd_update_precond(Q_mat, exprs, getattr(param, 'hessian_vector', grad), group['precond_lr'], Q,
                              group['store_triu_as_line'], getattr(param, 'vector', None))
    if hasattr(param, 'vector'):
        del param.vector
        del param.hessian_vector

    if grad.dim() > 1 and precond_schedule(group, balance_probability, f"balance_prob_{id(Q)}"):
        if group['store_triu_as_line']:
            utils.psgd_balance_Q([q_ for _, q_ in Q])
        else:
            utils.psgd_balance_Q(Q)

    if isinstance(prob, float):
        float_prob = prob
    else:
        float_prob = prob(group.get(f'cumulative_prob_{id(Q)}_prob_step', 1))
    group['is_cached'] = should_use_cache = cached and float_prob < 0.5

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
    if group.get('is_cached', False):
        out = utils.precond_grad_cached_(cache_expr, update, *Q_cache, caution=group['caution'], grad=grad)
    out = utils.psgd_precond_grad(exprs[-1], update, *Q_mat, caution=group['caution'], grad=grad)
    group['caution'] = False  # we already cautioned here - shouldn't do it again
    return out


def _fused_cached_psgd_precond_grad(group, grad, param, cache_expr, exprs, update, Q_mat, Q_cache):
    if group.get('is_cached', False):
        utils.fused_precond_grad_cached_(cache_expr, update, param, group['lr'], grad, group['weight_decay'],
                                         group['caution'], *Q_cache)
    else:
        utils.fused_psgd_precond_grad(exprs[-1], update, param, group['lr'], grad, group['weight_decay'],
                                      group['caution'], *Q_mat)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def scale_by_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                  prob: Optional[callable] = None):
    update = update.to(memory_format=torch.contiguous_format)
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    Q_mat = _update_psgd_precond(cached, Q_cache, group, param,
                                 update if group['momentum_into_precond_update'] else grad, Q_mat, Q, exprs, prob)
    return _cached_psgd_precond_grad(group, cache_expr, exprs, update, Q_mat, Q_cache, grad)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def scale_by_delayed_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                          prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    precond = _cached_psgd_precond_grad(group, cache_expr, exprs, update, Q_mat, Q_cache, grad)
    _ = _update_psgd_precond(cached, Q_cache, group, param, update if group['momentum_into_precond_update'] else grad,
                             Q_mat, Q, exprs, prob)
    return precond


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def update_by_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                   prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    Q_mat = _update_psgd_precond(cached, Q_cache, group, param,
                                 update if group['momentum_into_precond_update'] else grad, Q_mat, Q, exprs, prob)
    _fused_cached_psgd_precond_grad(group, update, param, cache_expr, exprs, update, Q_mat, Q_cache)
    raise SkipUpdate


@no_state
def sign(group, update, grad, param, graft: bool = True):
    return utils.sign_(update, graft)


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd, skip_first=False)
@no_state_no_foreach
def update_by_delayed_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                           prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    _fused_cached_psgd_precond_grad(group, update, param, cache_expr, exprs, update, Q_mat, Q_cache)
    _ = _update_psgd_precond(cached, Q_cache, group, param, update if group['momentum_into_precond_update'] else grad,
                             Q_mat, Q, exprs, prob)
    raise SkipUpdate


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group['step'] ** -group['beta2_scale']
    group['betas'] = (utils.get_beta1(group), beta2)
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
        utils.update_param_(param, update, group['lr'], group['weight_decay'], caution=group['caution'], grad=grad)


def create_branch(branches: list[list[callable]], merge_fn: callable):
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
        if 'base_lr' not in group:
            group['base_lr'] = group['lr']
        if 'prev_lr' in group and group['prev_lr'] != group['lr']:
            utils.warn_once(f'Learning rate changed between steps. This is an experimental feature and '
                            f'only supported with foreach=True (currently foreach={group["foreach"]}).')
            group['base_lr'] = group['lr']

        caution = group['caution']

        vals = list(self.split_p_and_g_in_group(group, should_promote=self.promote, beta1=utils.get_beta1(group)))

        if not vals:
            return
        p, g = zip(*vals)

        for param in p:
            state = self.state_(param)
            if 'step' in state:
                step = state['step']
            elif self.compile_step:
                step = utils.scalar_guard(0, param)
            else:
                step = 0
            break

        group['step'] = state['step'] = step = step + 1
        group['prev_lr'] = group['lr'] = group['base_lr'] * step / max(step, group['warmup_steps'] + 1)

        if not group['foreach'] or len(p) == 1:
            for param, grad in zip(p, g):
                chain(self.state_, group, [grad], [param], *self.fns)
        else:
            chain(self.state_, group, g, p, *self.fns)

        group['caution'] = caution
        group['lr'] = group['prev_lr']
        group['step'] = None


use_default = object()
str_or_fn = Union[str, callable, None, Literal[use_default]]


def _get_clip_fn(name: str_or_fn, default_val: str_or_fn):
    name = default(name, default_val)
    if callable(name):
        return name
    elif name not in ('l2_clip_', 'rmsnorm_clip_', 'trust_region_clip_', 'a_law_compress', 'mu_law_compress'):
        raise ValueError(f"Clipping function {name} not found")
    return getattr(utils, name)


def default(a, b):
    return b if a is use_default else a


# not supported: update_by_schedule_free, scale_by_soap, scale_by_exp_avg_sq
_scale_to_update_map = {scale_by_delayed_psgd.get_fn(): update_by_delayed_psgd,  #
                        scale_by_psgd.get_fn(): update_by_psgd,  #
                        scale_by_adam.get_fn(): update_by_adam,  #
                        scale_by_laprop.get_fn(): update_by_laprop,  #
                        scale_by_adopt.get_fn(): update_by_adopt}
_scale_to_update_map_inv = {update_by_delayed_psgd.get_fn(): scale_by_delayed_psgd,  #
                            update_by_psgd.get_fn(): scale_by_psgd,  #
                            update_by_adam.get_fn(): scale_by_adam,  #
                            update_by_laprop.get_fn(): scale_by_laprop,  #
                            update_by_adopt.get_fn(): scale_by_adopt}


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

    def __init__(self, params, defaults, foreach: bool, gradient_clipping: str_or_fn, update_clipping: str_or_fn,
                 palm: bool = use_default, *fns, compile_step: bool = use_default, promote: bool = use_default):
        if not fns:
            raise ValueError("No functions provided. If that's on purpose (SGD-like), use `identity`")

        args, kwargs = None, None
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
                raise ValueError("update_clipping is currently not compatible with update_by_* functions. "
                                 "Manually select scale_by_* functions or set auto_fuse=True.")
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
            group['train_mode'] = train_mode = not group.get('train_mode')
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        # Set p.data to x
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        utils.copy_stochastic_(p.data, p32)

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = train_mode = not group.get('train_mode')
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        utils.copy_stochastic_(p.data, p32)
