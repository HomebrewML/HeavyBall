import contextlib
import contextvars
import enum
import functools
import gc
import itertools
import math
import re
import string
import warnings
from collections.abc import Callable
from numbers import Real

import numpy as np
import torch
from torch import Tensor
from torch.backends import cudnn, opt_einsum
from torch.nn import functional as F
from torch.utils._pytree import tree_map

from . import fusions

compile_mode = "max-autotune-no-cudagraphs"
dynamic = False
compile_mode_recommended_to_none = None
zeroth_power_mode = "newtonschulz"
_cudnn_double_backward_pattern = re.compile(
    r"the derivative for .* is not implemented\. Double backwards .* To run double backwards"
)
_torch_compile_double_backward_pattern = re.compile(r"compile.*does not currently support double backward")
_fd_error = (
    "You can accelerate startup by globally enabling finite_differences first "
    "(via opt.finite_differences=True or by subclassing it)\n"
    "Original Error: "
)
default_division_backend = "eps_clamp"
atan2_scale = 16.0
dither_steps = 1
_force_eager = contextvars.ContextVar("heavyball_force_eager", default=False)


@contextlib.contextmanager
def force_eager():
    token = _force_eager.set(True)
    try:
        yield
    finally:
        _force_eager.reset(token)


def _strictly_aligned(a: Tensor, b: Tensor) -> Tensor:
    return ((a > 0) & (b > 0)) | ((a < 0) & (b < 0))


def _add_weight_decay(update: Tensor, param: Tensor, decay: float | Tensor, cautious: bool) -> Tensor:
    decay = decay * _strictly_aligned(param, update).to(param.dtype) if cautious else decay
    if isinstance(decay, Tensor):
        return torch.where(decay != 0, update + param * decay, update)
    return update + param * decay


def stable_l2_components(x: Tensor, dim=None, keepdim=False):
    x = promote(x)
    if x.numel() == 0:
        zero = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
        return zero, torch.ones_like(zero)
    broadcast_scale = x.abs().amax(dim=dim, keepdim=keepdim or dim is not None)
    scale = broadcast_scale if keepdim or dim is None else broadcast_scale.squeeze(dim)
    safe = torch.where(broadcast_scale != 0, broadcast_scale, 1)
    return scale, torch.linalg.vector_norm(x / safe, dim=dim, keepdim=keepdim)


def stable_l2_norm(x: Tensor, dim=None, keepdim=False):
    scale, norm = stable_l2_components(x, dim=dim, keepdim=keepdim)
    return norm * scale


def stable_l2_normalize(x: Tensor, dim=None, eps=0):
    x = promote(x)
    if x.numel() == 0:
        return x
    eps = torch.as_tensor(eps, device=x.device, dtype=x.real.dtype).reshape(())
    scale = x.abs().amax(dim=dim, keepdim=True)
    safe = torch.where(scale != 0, scale, 1)
    scaled = x / safe
    norm = torch.linalg.vector_norm(scaled, dim=dim, keepdim=True)
    unit = scaled / torch.where(norm != 0, norm, torch.ones_like(norm))
    direct = x / torch.where(eps != 0, eps, torch.ones_like(eps))
    return torch.where(norm * scale >= eps, unit, direct)


def stable_l2_norm_list(xs):
    values = [promote(x) for x in xs if x.numel()]
    if not values:
        ref = xs[0] if xs else torch.empty(0)
        return torch.zeros((), device=ref.device, dtype=promote(ref.real.dtype))
    scale = torch.stack([x.abs().amax() for x in values]).amax()
    safe = torch.where(scale != 0, scale, 1)
    return torch.stack([(x / safe).abs().square().sum() for x in values]).sum().sqrt() * scale


class ZerothPowerMode(enum.Enum):
    newtonschulz = "newtonschulz"
    svd = "svd"
    thinky_polar_express = "thinky_polar_express"


class OrthoScaleMode(enum.Enum):
    none = "none"
    scale = "scale"
    graft = "graft"


class DivisionBackend(enum.Enum):
    eps_add = "eps_add"
    eps_clamp = "eps_clamp"
    atan2 = "atan2"
    nan_to_0 = "nan_to_0"


DivisionBackendLike = DivisionBackend | str | None


def _normalize_division_backend(backend: DivisionBackendLike) -> DivisionBackend:
    if backend is None:
        return DivisionBackend(default_division_backend)
    if isinstance(backend, DivisionBackend):
        return backend
    try:
        return DivisionBackend(backend)
    except ValueError as error:
        raise ValueError(f"Unknown division backend '{backend}'") from error


def _has_scalar_double(value):
    if isinstance(value, Tensor):
        return value.dtype == torch.float64 and value.ndim == 0
    if isinstance(value, dict):
        return any(_has_scalar_double(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_scalar_double(item) for item in value)
    return False


def decorator(func):
    compiled = {}

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if (
            is_compiling()
            or _force_eager.get()
            or compile_mode_recommended_to_none is None
            or _has_scalar_double((args, kwargs))
        ):
            return func(*args, **kwargs)
        key = compile_mode_recommended_to_none, dynamic
        if key not in compiled:
            compiled[key] = fusions.compile(
                func, fullgraph=True, dynamic=dynamic, mode=compile_mode_recommended_to_none
            )
        return compiled[key](*args, **kwargs)

    return _fn


def decorator_knowngood(func: Callable, fullgraph: bool = True):
    compiled = {}

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or _force_eager.get() or compile_mode is None or _has_scalar_double((args, kwargs)):
            return func(*args, **kwargs)
        key = compile_mode, dynamic
        if key not in compiled:
            compiled[key] = fusions.compile(func, fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)
        return compiled[key](*args, **kwargs)

    return _fn


def decorator_no_fullgraph(func: Callable):
    return decorator_knowngood(func, fullgraph=False)


einsum_base = string.ascii_lowercase

no_compile_qr = torch.compiler.disable(torch.linalg.qr)
no_compile_eigh = torch.compiler.disable(torch.linalg.eigh)
no_compile_svd = torch.compiler.disable(torch.linalg.svd)
no_compile_solve_triangular = torch.compiler.disable(torch.linalg.solve_triangular)


def compiled_einsum(expr, *args):
    """
    this is necessary to avoid the slowdown introduced by uncompiled einsum
    uncompiled einsum is twice as slow if we add three 1-sized dimensions
    for more, see https://gist.github.com/ClashLuke/a9530f1b9ba4e525369e2dba48528957
    """
    if is_compiling() or _force_eager.get() or compile_mode is None:
        return torch.einsum(expr, *args)
    return _compiled_einsum(expr, compile_mode, dynamic)(*args)


@functools.lru_cache(maxsize=None)
def _compiled_einsum(expr, mode, dynamic):
    def einsum(*operands):
        return torch.einsum(expr, *operands)

    return fusions.compile(einsum, fullgraph=True, dynamic=dynamic, mode=mode)


@decorator_knowngood
def _compilable_schedule_free_(
    p: list[Tensor],
    z: list[Tensor],
    ckp1: Tensor,
    update: list[Tensor],
    lr: Tensor,
    beta1: Tensor,
    decay: float,
    grad: list[Tensor],
    caution,
    cautious_decay: bool,
):
    for op, oz, u_, g_ in zip(p, z, update, grad):
        u_ = u_.view_as(op)
        p_, z_, u_ = map(promote, (op, oz, u_))
        dtype = functools.reduce(torch.promote_types, (p_.dtype, z_.dtype, u_.dtype))
        p_, z_, u_ = p_.to(dtype), z_.to(dtype), u_.to(dtype)
        if isinstance(decay, Tensor) or decay != 0:
            u_ = _add_weight_decay(u_, p_, decay, cautious_decay)
        if caution:
            u_ = _compilable_cautioning(g_, u_)
        p_ = p_.lerp(z_, ckp1)
        p_ = p_ + u_ * (lr * (beta1 * (1 - ckp1)) - lr)
        z_ = z_ + u_ * -lr
        copy_stochastic_(op, p_)
        copy_stochastic_(oz, z_)


def schedule_free_(
    lr: float,
    weight_lr_power: float,
    weight_sum: float,
    beta1: float,
    parameters: list[Tensor],
    z: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    caution: bool = False,
    r: float = 0.0,
    step: int = 0,
    decay: float = 0.0,
    cautious_decay: bool = False,
):
    update, parameters, z, grad = list_guard(update, parameters, z, grad)
    if not parameters:
        return weight_sum
    weight = abs(lr) ** weight_lr_power * max(step, 1) ** r
    weight_sum = weight_sum + weight

    if isinstance(weight_sum, Tensor):
        ckp1 = torch.where(weight_sum != 0, weight / weight_sum, torch.zeros_like(weight_sum))
    else:
        ckp1 = 0 if weight_sum == 0 else weight / weight_sum

    lr, ckp1, beta1 = scalar_guard(lr, ckp1, beta1, grad[0])
    _compilable_schedule_free_(parameters, z, ckp1, update, lr, beta1, decay, grad, caution, cautious_decay)
    return weight_sum


@decorator_knowngood
def _compilable_msam(
    lr: Tensor,
    beta1: Tensor,
    param: list[Tensor],
    z: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    exp_avg: list[Tensor],
    caution: bool,
    cautious_decay: bool,
    decay: Tensor,
    sam_step_size: Tensor,
):
    exp_avg32 = _lerp(exp_avg, update, beta1)
    for u_, g_, z_, p_ in zip(exp_avg32, grad, z, param):
        u_ = u_.view_as(z_)
        z32_ = promote(z_)
        if caution:
            u_ = _compilable_cautioning(g_, u_)
        d = decay * _strictly_aligned(z32_, u_).to(z32_.dtype) if cautious_decay else decay
        z32_ = z32_ * (1 - d * lr) + u_ * -lr
        copy_stochastic_(z_, z32_)
        copy_stochastic_(p_, z32_ - stable_l2_normalize(u_, eps=1e-8) * sam_step_size)


def msam_(
    lr: float,
    beta1: float,
    param: list[Tensor],
    z: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    exp_avg: list[Tensor],
    caution: bool,
    weight_decay: float,
    sam_step_size: float,
    cautious_decay: bool = False,
):
    param, z, update, grad, exp_avg = list_guard(param, z, update, grad, exp_avg)
    if not param:
        return
    lr, beta1, weight_decay, sam_step_size = scalar_guard(lr, beta1, weight_decay, sam_step_size, exp_avg[0])
    _compilable_msam(lr, beta1, param, z, update, grad, exp_avg, caution, cautious_decay, weight_decay, sam_step_size)


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done

    By @francois-rozet (commit: 68cde41eaf7e73b4c46eacb6a944865dcc081f1d), re-commited due to faulty merge
    """
    if grad.ndim == 0:
        return grad

    new_shape = []
    cum_size = 1

    for s in grad.shape[1:][::-1]:
        temp_size = cum_size * s
        if temp_size > max_precond_dim:
            if cum_size > 1:
                new_shape.append(cum_size)
                cum_size = s
            else:
                new_shape.append(s)
                cum_size = 1
        else:
            cum_size = temp_size

    if cum_size > 1:
        new_shape.append(cum_size)

    new_shape = [grad.shape[0], *new_shape[::-1]]
    new_grad = grad.reshape(new_shape)
    if not split:
        return new_grad

    grads = [new_grad]
    for i, sh in reversed(list(enumerate(new_shape[:]))):
        if sh == 1:
            grads = [g.squeeze(dim=i) for g in grads]
            continue
        if sh <= max_precond_dim:
            continue
        grads = [a for g in grads for a in g.split(max_precond_dim, dim=i)]
    if len(grads) == 1:
        return new_grad
    new_grads = []
    for g in grads:
        append_or_extend(new_grads, dim_merger(g, max_precond_dim, split))
    return new_grads


def linear_warmup_scheduler(step: int, alpha_end: float, alpha_start: float = 0.0, warmup: int | None = None) -> float:
    if warmup is None or warmup <= 0:
        return alpha_end
    if isinstance(step, Tensor):
        a = (step / warmup).clamp(min=0, max=1)
        return alpha_start + a * (alpha_end - alpha_start)
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(
    step: int, beta_end: float, beta_start: float, warmup: int | None = None, eps: float = 1e-8
) -> float:
    if warmup is None or warmup <= 0:
        return beta_end

    def half_life(beta: float) -> float:
        if isinstance(beta, Tensor):
            return math.log(0.5) / torch.log(beta + eps) - 1
        return math.log(0.5) / math.log(beta + eps) - 1

    def inv_half_life(t: float) -> float:
        if isinstance(t, Tensor):
            return torch.exp2(-1.0 / (t + 1.0))
        return math.pow(0.5, 1.0 / (t + 1.0))

    if isinstance(step, Tensor):
        a = (step / warmup).clamp(min=0, max=1)
        target = half_life(beta_start) + a * (half_life(beta_end) - half_life(beta_start))
        return inv_half_life(target).clamp(min=0.0, max=1.0 - eps)
    if step < warmup:
        a = step / float(warmup)
        target = (1.0 - a) * half_life(beta_start) + a * half_life(beta_end)
        beta = inv_half_life(target)
        return min(max(beta, 0.0), 1.0 - eps)
    return beta_end


def _compute_ademamix_hparams(
    betas: tuple[float, float, float],
    step: int,
    alpha: float,
    beta3_warmup: int | None,
    alpha_warmup: int | None,
) -> tuple[float, float, float, float]:
    beta1, beta2, beta3_final = betas
    alpha_eff = linear_warmup_scheduler(step, alpha_end=alpha, alpha_start=0.0, warmup=alpha_warmup)
    beta3_eff = linear_hl_warmup_scheduler(step, beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
    return beta1, beta2, beta3_eff, alpha_eff


def beta_debias(beta, step):
    if isinstance(beta, Tensor) or isinstance(step, Tensor):
        if isinstance(beta, Tensor):
            beta_t = beta if beta.is_floating_point() else beta.double()
        else:
            dtype = step.dtype if step.is_floating_point() else torch.float64
            beta_t = torch.as_tensor(beta, device=step.device, dtype=dtype)
        step_t = torch.as_tensor(step, device=beta_t.device, dtype=beta_t.dtype)
        log_beta = beta_t.log()
        out = beta_t * (-torch.expm1((step_t - 1) * log_beta)) / (-torch.expm1(step_t * log_beta))
        out = torch.where(beta_t == 0, torch.zeros_like(out), out)
        out = torch.where(beta_t == 1, (step_t - 1) / step_t.clamp_min(1), out)
        return torch.where(step_t <= 1, torch.zeros_like(out), out)
    if step <= 1:
        return 0.0
    if beta == 0:
        return 0.0
    if beta == 1:
        return (step - 1) / step
    log_beta = math.log(beta)
    return beta * -math.expm1((step - 1) * log_beta) / -math.expm1(step * log_beta)


def _nadam_moments(beta1: Tensor, step: Tensor, momentum_decay: float) -> tuple[Tensor, Tensor]:
    md = torch.as_tensor(momentum_decay, dtype=beta1.dtype, device=beta1.device)
    base = torch.tensor(0.96, dtype=beta1.dtype, device=beta1.device)
    step_f = step.to(beta1.dtype)
    mu = beta1 * (1 - 0.5 * torch.pow(base, step_f * md))
    mu_next = beta1 * (1 - 0.5 * torch.pow(base, (step_f + 1) * md))
    return mu, mu_next


def _nadam_prepare_weight_decay(
    update: list[Tensor],
    param: list[Tensor],
    weight_decay: float | Tensor,
    decoupled: bool,
) -> float | Tensor:
    if decoupled:
        return weight_decay
    if not isinstance(weight_decay, Tensor) and weight_decay == 0:
        return 0.0
    for u_, p_ in zip(update, param):
        u_.copy_(_add_weight_decay(u_, p_, weight_decay, False))
    return weight_decay.new_zeros(()) if isinstance(weight_decay, Tensor) else 0.0


def _nadam_compute_update(
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    mu_product: list[Tensor],
    update: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
    mu: Tensor,
    mu_next: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    exp_avg32 = _lerp(exp_avg, update, beta1)
    beta2_corr = beta_debias(beta2, step)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, update, beta2_corr, eps, [None])

    mu_product32 = [promote(mp_) * mu for mp_ in mu_product]
    mu_t, mu_next_t = scalar_guard(mu, mu_next, mu_product32[0])
    one = mu_t.new_ones(())
    grad_scale = one - mu_t

    out = []
    for u_, e_, d_, mp_ in zip(update, exp_avg32, denom, mu_product32):
        if u_.ndim:
            mp_ = mp_.reshape(-1, *([1] * (u_.ndim - 1)))
        else:
            mp_ = mp_.squeeze()
        gw = grad_scale / (one - mp_)
        ew = mu_next_t / (one - mp_ * mu_next_t)
        out.append(u_ / d_ * gw + e_ / d_ * ew)
    return out, mu_product32


def eps_sqrt(item, eps):
    return item.sqrt().clamp(min=eps)


@decorator_knowngood
def _compilable_exp_avg_sq_(
    state: list[Tensor],
    grad: list[Tensor],
    beta2: Tensor,
    eps: Tensor,
    out: None | list[None | Tensor],
):
    work = []
    for g, s in zip(grad, state):
        g = promote(g)
        work.append(g.to(torch.promote_types(g.dtype, promote(s.dtype))))
    s32 = _lerp(state, [g * g for g in work], beta2)

    denom = [eps_sqrt(d, eps) for d in s32]

    if out is None or out[0] is None:
        return denom

    copy_stochastic_list_(out, denom)
    return out


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    state, grad = list_guard(state, grad)
    if not state:
        return []
    out = list_guard(out)
    beta2, eps = scalar_guard(beta2, eps, state[0])
    return _compilable_exp_avg_sq_(state, grad, beta2, eps, out)


@decorator_knowngood
def _compilable_scale_by_exp_avg_sq_(state: list[Tensor], grad: list[Tensor], beta2: Tensor, eps: Tensor):
    g32 = promote(grad)
    denom = _compilable_exp_avg_sq_(state, g32, beta2, eps, [None])
    return [g_ / d_ for g_, d_ in zip(g32, denom)]


def scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps):
    grad, exp_avg_sq = list_guard(grad, exp_avg_sq)
    if not grad:
        return grad
    beta2, eps = scalar_guard(beta2, eps, grad[0])
    return _compilable_scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps)


def scale_by_exp_avg_(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad
    beta = scalar_guard(beta, state[0])
    return _lerp(state, grad, beta)


@decorator_knowngood
def _compilable_agc_(parameters: list[Tensor], gradients: list[Tensor], clip_val: float, minimum: float, eps: float):
    clip_val = torch.as_tensor(
        clip_val, device=parameters[0].device, dtype=promote(parameters[0]).real.dtype
    ).clamp_min(0)
    for param, grad in zip(parameters, gradients):
        p32, g32 = promote(param), promote(grad)
        p_scale, p_norm = stable_l2_components(p32)
        g_scale, g_norm = stable_l2_components(g32)
        common = torch.stack(
            [
                p_scale,
                g_scale,
                torch.as_tensor(minimum, device=p32.device, dtype=p_scale.dtype),
                torch.as_tensor(eps, device=p32.device, dtype=p_scale.dtype),
            ]
        ).amax()
        safe = common.clamp_min(torch.finfo(common.dtype).tiny)
        p_scaled = (p_scale / safe * p_norm).clamp_min(minimum / safe)
        g_scaled = g_scale / safe * g_norm
        limit = p_scaled * clip_val
        should_clip = g_scaled > limit
        direction = stable_l2_normalize(g32)
        from_param = direction * (p_norm * clip_val) * p_scale
        from_minimum = direction * (minimum * clip_val)
        param_dominates = (p_scale != 0) & (p_norm >= minimum / p_scale)
        clipped = torch.where(param_dominates, from_param, from_minimum)
        clipped = torch.where(g_scaled < eps / safe, g32 * (limit / (eps / safe)), clipped)
        copy_stochastic_(grad, torch.where(should_clip, clipped, g32))


def adaptive_gradient_clipping_(
    parameters: list[Tensor], gradients: list[Tensor], clip_val: float, minimum: float = 1e-3, eps: float = 1e-8
):
    parameters, gradients = list_guard(parameters, gradients)
    if not parameters:
        return gradients
    clip_val, minimum, eps = scalar_guard(clip_val, minimum, eps, parameters[0])
    _compilable_agc_(parameters, gradients, clip_val, minimum, eps)
    return gradients


def is_compiling():
    return torch.compiler.is_compiling()


def set_(dst: Tensor, src: Tensor):
    dst.copy_(src)


def capture_param_shapes(params):
    """Capture param shapes before sharding for non-FSDP parallelism backends.

    Pass as ``orig_shapes=`` to any optimizer. Not needed under FSDP, which
    auto-detects shapes when ``use_orig_params=True``.
    """
    if hasattr(params, "parameters"):
        params = params.parameters()
    return {id(p): tuple(p.shape) for p in params}


def clean():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _ignore_warning(msg):
    warnings.filterwarnings("ignore", f".*{re.escape(msg)}.*")


def set_torch(benchmark_limit: int = 32, einsum_strategy: str = "auto-hq"):
    import opt_einsum as _opt_einsum

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.benchmark_limit = benchmark_limit
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
    opt_einsum.set_flags(True)
    if einsum_strategy == "heavyball":
        opt_einsum.strategy = "auto-hq"
        choices = _opt_einsum.paths._AUTO_HQ_CHOICES
        for max_val, fn in ((20, _opt_einsum.paths.dynamic_programming), (64, 512), (128, 256)):
            if isinstance(fn, int):
                fn = functools.partial(_opt_einsum.path_random.random_greedy, max_repeats=fn)
            for i in range(max(choices.keys()), max_val):
                if i not in choices:
                    choices[i] = fn
    else:
        opt_einsum.strategy = einsum_strategy

    # Torch calls these for 2nd-order optimization in HeavyBall, but they are explicitly handled.
    _ignore_warning(
        "Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak"
    )
    _ignore_warning(
        "We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak"
    )
    _ignore_warning(
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead."
    )


def _stable_matrix_normalize(G, eps):
    return stable_l2_normalize(G, dim=(-2, -1), eps=eps)


@decorator_knowngood
def zeropower_via_newtonschulz5(G, eps=1e-7):
    # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim >= 2
    dtype = G.dtype
    G = _stable_matrix_normalize(G, eps)
    x = G if G.dtype == torch.float64 else G.to(torch.bfloat16)
    if G.size(-2) > G.size(-1):
        x = x.mT

    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if G.size(-2) > G.size(-1):
        x = x.mT
    return x.to(dtype)


###### START
# Based on https://arxiv.org/pdf/2505.16932v3
# and https://github.com/NoahAmsel/PolarExpress/blob/5454910920ca8c65afda28820cdf9e49b9436ed0/polar_express.py#L69-L82
# and https://github.com/thinking-machines-lab/manifolds/blob/89dcae50f01af59f1e0570289474da3a2ecaa60b/src/msign.py#L47
#
# under the MIT License

# Coefficients are from https://arxiv.org/pdf/2505.16932v3
ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# safety factor for numerical stability (but exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


def msign(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign function:
    https://arxiv.org/abs/2505.16932
    """
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    dtype = G.dtype
    G = _stable_matrix_normalize(G, eps)
    x = G if G.dtype == torch.float64 else G.to(torch.bfloat16)
    if should_transpose:
        x = x.mT

    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        # goal is to compute x = a x + b S x + c S^2 x
        # we can break this up into: x = (a I + (b I + c S) S) x
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    return x.to(dtype)


###### END


def _scion_bias_rms_direction(x: Tensor, eps: float = 1e-8) -> Tensor:
    if x.ndim == 0:
        return x / x.abs().clamp(min=eps)
    return stable_l2_normalize(x, dim=0, eps=eps * math.sqrt(x.shape[0])) * math.sqrt(x.shape[0])


def _scion_spectral_direction(x: Tensor) -> Tensor:
    flat = x.reshape(x.shape[0], -1)
    flat = inplace_orthogonal_(flat)
    normalized = flat.reshape_as(x)
    in_dim = max(flat.shape[1], 1)
    scale = math.sqrt(x.shape[0] / in_dim)
    return normalized * scale


def _scion_spectral_conv_direction(x: Tensor) -> Tensor:
    flat = x.reshape(x.shape[0], -1)
    flat = inplace_orthogonal_(flat)
    normalized = flat.reshape_as(x)
    out_channels, in_channels = x.shape[:2]
    spatial = math.prod(x.shape[2:]) if x.ndim > 2 else 1
    scale = math.sqrt(out_channels / max(in_channels, 1)) / max(spatial, 1)
    return normalized * scale


@decorator_knowngood
def _compilable_scion_lmo_(update: list[Tensor] | Tensor, scale: Tensor, eps: Tensor):
    for tensor in update:
        promoted = promote(tensor)
        if promoted.ndim >= 3:
            direction = _scion_spectral_conv_direction(promoted)
        elif promoted.ndim == 2:
            direction = _scion_spectral_direction(promoted)
        else:
            direction = _scion_bias_rms_direction(promoted, eps)

        scale_value = scale.to(dtype=direction.dtype, device=direction.device)
        direction = direction * scale_value
        copy_stochastic_(tensor, direction)


def scion_auto_lmo_(update: list[Tensor] | Tensor, scale: float | Tensor, eps: float = 1e-8):
    update = list_guard(update)
    if not update:
        return update

    scale_tensor, eps_tensor = scalar_guard(scale, eps, update[0])
    _compilable_scion_lmo_(update, scale_tensor, eps_tensor)
    return update


def scion_auto_init_param_(param: Tensor, scale: float | Tensor, seed: int = 0):
    scale_tensor = scalar_guard(scale, param)
    promoted = promote(param)

    gen = torch.Generator(device=param.device)
    gen.manual_seed(seed)

    if param.ndim >= 2:
        init_fp64 = promoted.clone().double()
        for idx in itertools.product(*(range(s) for s in init_fp64.shape[2:])):
            torch.nn.init.orthogonal_(init_fp64[(slice(None), slice(None), *idx)], generator=gen)
        fan_out, fan_in = init_fp64.shape[:2]
        spatial = math.prod(init_fp64.shape[2:])
        init_fp64.mul_(math.sqrt(fan_out / max(fan_in, 1)) / max(spatial, 1))
        init = init_fp64.to(dtype=promoted.dtype)
    else:
        init = promoted.clone()
        torch.nn.init.zeros_(init)

    init.mul_(scale_tensor.to(dtype=init.dtype, device=init.device))
    ecc = getattr(param, "_ecc", None)
    if ecc is None:
        set_(param, _stochastic_round(param, init, gen))
    else:
        ecc.encode(init, param, gen)


@decorator_knowngood
def _compilable_heavyball_momentum_(state, grad, beta):
    out = []
    for s_, g_ in zip(state, grad):
        v = promote(s_) * beta + promote(g_)
        copy_stochastic_(s_, v)
        out.append(v)
    return out


@decorator_knowngood
def _compilable_nesterov_momentum_(state, grad, beta):
    out = []
    for s_, g_ in zip(state, grad):
        v = promote(s_) * beta + promote(g_)
        u = promote(g_) + v * beta
        copy_stochastic_(s_, v)
        out.append(u)
    return out


def heavyball_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad
    beta = scalar_guard(beta, state[0])
    return _compilable_heavyball_momentum_(state, grad, beta)


def nesterov_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad
    beta = scalar_guard(beta, state[0])
    return _compilable_nesterov_momentum_(state, grad, beta)


@decorator_knowngood
def _compilable_nesterov_ema_(state, grad, beta):
    ema32 = _lerp(state, grad, beta)
    return [promote(g) + ema for g, ema in zip(grad, ema32)]


def nesterov_ema(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad
    beta = scalar_guard(beta, state[0])
    return _compilable_nesterov_ema_(state, grad, beta)


@decorator_knowngood
def _compilable_grafting(magnitude, direction):
    magnitude, direction = promote(magnitude), promote(direction)
    scale, norm = stable_l2_components(magnitude)
    return stable_l2_normalize(direction, eps=1e-6) * norm * scale


@decorator_no_fullgraph
def _compilable_orthogonal_(x: Tensor, mode: str | ZerothPowerMode, out: Tensor | None, scale_mode: str):
    if not isinstance(mode, ZerothPowerMode):
        mode = ZerothPowerMode(mode)
    if not isinstance(scale_mode, OrthoScaleMode):
        scale_mode = OrthoScaleMode(scale_mode)
    if mode == ZerothPowerMode.newtonschulz:
        y = zeropower_via_newtonschulz5(x)
    elif mode == ZerothPowerMode.thinky_polar_express:
        y = msign(x, 10)
    elif mode == ZerothPowerMode.svd:
        u, _s, vt = no_compile_svd(promote(x), full_matrices=False)
        y = u @ vt
    else:
        raise NotImplementedError(f"Unknown zeroth_power_mode: {mode}")
    if scale_mode == OrthoScaleMode.none:
        pass
    elif scale_mode == OrthoScaleMode.scale:
        y *= max(1, x.size(-2) / x.size(-1)) ** 0.5
    elif scale_mode == OrthoScaleMode.graft:
        y = _compilable_grafting(x, y)
    else:
        raise NotImplementedError(f"Unknown scale_mode: {scale_mode}")
    if out is None:
        return y

    copy_stochastic_(out, y)


def inplace_orthogonal_(x: Tensor, mode: str | None = None, out: Tensor | None = None, scale_mode: str = "none"):
    return _compilable_orthogonal_(x, mode or zeroth_power_mode, out, scale_mode)


@decorator_no_fullgraph
def get_orthogonal_matrix_QR(
    GG: list[Tensor],
    Q: list[Tensor],
    *exp_avg: Tensor,
    exp_avg_sq: Tensor = None,
    eigenvalues: list[Tensor | None] = (),
    heavy: bool = False,
):
    if isinstance(Q, list) and not Q:
        return

    ref = exp_avg[0] if exp_avg else exp_avg_sq
    if ref is not None and ref.dim() <= 1:
        Q.clear()
        return

    if ref is not None and ref.dim() - 1 != len(Q):
        raise ValueError(f"ref dim {ref.dim()} (excluding bucket axis) does not match Q length {len(Q)}")

    old_qs = [None if m is None else promote(q).clone() for m, q in zip(GG, Q)]
    new_qs, orders = [], []
    for m, q_old in zip(GG, old_qs):
        if m is None:
            new_qs.append(None)
            orders.append(None)
            continue
        m = promote(m.data)
        if heavy:
            oriented = no_compile_qr(m @ q_old).Q
            eig = compiled_einsum("...ij,...ij->...j", oriented, m @ oriented)
            order = torch.argsort(eig, descending=True)
            new_qs.append(oriented.gather(-1, order.unsqueeze(-2).expand_as(oriented)))
            orders.append(order)
            continue
        tmp = m @ q_old
        eig = compiled_einsum("...ij,...ij->...j", q_old, tmp)
        idx = torch.argsort(eig, descending=True).unsqueeze(-2).expand_as(tmp)
        tmp.scatter_(-1, idx, no_compile_qr(tmp.gather(-1, idx)).Q)
        new_qs.append(tmp)
        orders.append(None)

    for q, q_new in zip(Q, new_qs):
        if q_new is not None:
            copy_stochastic_(q, q_new)
    for value, order in zip(eigenvalues, orders):
        if order is not None:
            copy_stochastic_(value, promote(value).gather(-1, order))
    new_qs = [None if old is None else q for old, q in zip(old_qs, Q)]

    if ref is None:
        return

    assert ref.dim() < 14, "ref.ndim must be less than 14"
    if exp_avg:
        _transform_projected_state(old_qs, new_qs, *exp_avg)

    if heavy and exp_avg_sq is not None:
        _transform_projected_square_state(old_qs, new_qs, exp_avg_sq)


def _transform_projected_state(old_qs: list[Tensor | None], new_qs: list[Tensor | None], *states: Tensor):
    if not states:
        return

    ref = states[0]
    if ref is None or ref.dim() <= 1:
        return

    assert ref.dim() < 14, "ref.ndim must be less than 14"
    param_dim = ref.dim() - 1
    in_str = einsum_base[:param_dim]
    out_str = einsum_base[param_dim : 2 * param_dim]

    old_basis = ",".join([f"...{o}{i}" for q, i, o in zip(old_qs, in_str, in_str.upper()) if q is not None])
    if not old_basis:
        return

    new_basis = ",".join([f"...{i}{o}" for q, i, o in zip(new_qs, in_str.upper(), out_str) if q is not None])
    out_str = "".join([o if o in new_basis else i for i, o in zip(in_str, out_str)])
    subscripts = f"...{in_str},{old_basis},{new_basis}->...{out_str}"
    old_basis = [promote(q) for q in old_qs if q is not None]
    new_basis = [promote(q) for q in new_qs if q is not None]

    for state in states:
        new = casted_einsum(subscripts, promote(state), *old_basis, *new_basis)
        copy_stochastic_(state, new)


def _transform_projected_square_state(old_qs: list[Tensor | None], new_qs: list[Tensor | None], *states: Tensor):
    if not states or states[0] is None or states[0].dim() <= 1:
        return

    param_dim = states[0].dim() - 1
    in_str = einsum_base[:param_dim]
    out_str = einsum_base[param_dim : 2 * param_dim]
    dtype = promote(states[0].dtype)
    for basis in (*old_qs, *new_qs):
        if basis is not None:
            dtype = torch.promote_types(dtype, promote(basis.dtype))
    terms, transitions = [], []
    for old, new, i, o in zip(old_qs, new_qs, in_str, out_str):
        if old is None:
            continue
        terms.append(f"...{i}{o}")
        transitions.append(casted_einsum("...ji,...jk->...ik", promote(old).to(dtype), promote(new).to(dtype)).square())
    if not terms:
        return
    output = "".join(o if f"...{i}{o}" in terms else i for i, o in zip(in_str, out_str))
    subscripts = f"...{in_str},{','.join(terms)}->...{output}"
    for state in states:
        copy_stochastic_(state, casted_einsum(subscripts, promote(state), *transitions).clamp_min(0))


@decorator_no_fullgraph
def init_psgd_eigenbasis(Q: list[Tensor]):
    out = []

    for q in Q:
        if q.ndim < 3:
            out.append(None)
            continue

        q32 = promote(q)
        basis = torch.empty_like(q)
        copy_stochastic_(basis, _stable_symmetric_basis(q32.mT @ q32))
        out.append(basis)

    return out


@decorator_no_fullgraph
def get_psgd_eigenbasis(Q: list[Tensor], prev: list[Tensor | None]):
    out = []

    for q, old_basis in zip(Q, prev):
        if q.ndim < 3:
            out.append(None)
            continue
        if old_basis is None:
            raise ValueError(
                "get_psgd_eigenbasis requires a previous basis for matrix blocks; use init_psgd_eigenbasis"
            )

        q32 = promote(q)
        old_basis32 = promote(old_basis)
        Y = q32.mT @ (q32 @ old_basis32)
        basis_raw = no_compile_qr(Y, mode="reduced").Q
        projected = q32 @ promote(basis_raw)
        sort_idx = torch.argsort(compiled_einsum("...ij,...ij->...j", projected, projected), descending=True)
        gather_idx = sort_idx.unsqueeze(-2).expand_as(basis_raw)
        basis_raw = basis_raw.gather(-1, gather_idx)
        signs = compiled_einsum("...ij,...ij->...j", old_basis32, promote(basis_raw))
        signs = torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs)).to(dtype=basis_raw.dtype)
        basis = basis_raw * signs.unsqueeze(-2)
        out.append(basis)

    return out


@decorator_no_fullgraph
def update_psgd_eigenbasis(Q: list[Tensor], Q_basis: list[Tensor], *states: Tensor, exp_avg_sq: Tensor | None = None):
    old_basis = [None if basis is None else promote(basis).clone() for basis in Q_basis]
    new_basis = get_psgd_eigenbasis(Q, Q_basis)
    for basis, new in zip(Q_basis, new_basis):
        if basis is not None:
            copy_stochastic_(basis, new)
    new_basis = [None if old is None else basis for old, basis in zip(old_basis, Q_basis)]

    _transform_projected_state(old_basis, new_basis, *states)
    if exp_avg_sq is not None:
        _transform_projected_square_state(old_basis, new_basis, exp_avg_sq)


def _stable_symmetric_basis(m: Tensor):
    m = promote(m.detach())
    scale = m.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(m.real.dtype).tiny)
    m = m / scale
    m = (m + m.mH) * 0.5
    m.diagonal(dim1=-2, dim2=-1).add_(torch.finfo(m.real.dtype).eps)
    return torch.flip(no_compile_eigh(m)[1], [-1]).contiguous()


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """

    final = []
    for m in mat:
        if m is None:
            final.append(None)
            continue

        basis = torch.empty_like(m)
        copy_stochastic_(basis, _stable_symmetric_basis(m))
        final.append(basis)

    return final


@decorator_knowngood
def _compilable_stochastic_lerp_(x: list[Tensor], y: list[Tensor], a: float | int | Tensor):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        dtype = torch.promote_types(x32.dtype, y32.dtype)
        x32, y32 = x32.to(dtype), y32.to(dtype)
        copy_stochastic_(x_, x32 * (1 - a) + y32 * a)


def get_beta1(group):
    if "beta" in group and group["beta"] is not None: return group["beta"]
    if "betas" in group: return group["betas"][0]
    raise ValueError("Beta not found in group.")


def get_beta2(group):
    if "betas" in group:
        return group["betas"][1]
    return group["beta2"]


def stochastic_lerp_(x: list[Tensor], y: list[Tensor], a: float | int | Tensor):
    x, y = list_guard(x, y)
    if not x:
        return x
    a = scalar_guard(a, x[0])
    _compilable_stochastic_lerp_(x, y, a)
    return x


def list_guard(*xs):
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.append(x)
        else:
            out.append([x])
    if len(xs) == 1:
        return out[0]
    return out


def scalar_guard(*args):
    *xs, ref = args
    out = []
    for x in xs:
        if isinstance(x, float):
            out.append(torch.empty((), dtype=promote(ref.real.dtype), device=ref.device).fill_(x))
        elif isinstance(x, int):
            out.append(torch.empty((), dtype=torch.int64, device=ref.device).fill_(x))
        elif isinstance(x, Tensor):
            dtype = promote(ref.real.dtype) if x.is_floating_point() else x.dtype
            out.append(x.to(device=ref.device, dtype=dtype))
        else:
            out.append(x)
    if len(xs) == 1:
        return out[0]
    return out


def broadcastable_list_guard(*xs):
    xs = list_guard(*xs)
    if all(not x for x in xs):
        return xs
    if any(not x for x in xs):
        raise ValueError("Cannot broadcast empty and non-empty lists")
    for x in xs:
        if isinstance(x[0], Tensor):
            ref = x[0]
            break
    else:
        raise ValueError("No tensor-valued input given")
    xs = [x if isinstance(x[0], Tensor) else list_guard(scalar_guard(*x, ref)) for x in xs]
    max_len = max(len(x) for x in xs)
    if any(len(x) not in (1, max_len) for x in xs):
        raise ValueError(f"Cannot broadcast list lengths {[len(x) for x in xs]}")
    return [x if len(x) > 1 else x * max_len for x in xs]


@decorator_knowngood
def _compilable_stochastic_add_(x: list[Tensor], y: list[Tensor], alpha: float | int | Tensor):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, x32 + y32 * alpha)


def stochastic_add_(x: list[Tensor] | Tensor, y: list[Tensor] | Tensor, alpha: float | int | Tensor = 1):
    x, y = broadcastable_list_guard(x, y)
    if not x:
        return x
    alpha = scalar_guard(alpha, x[0])
    _compilable_stochastic_add_(x, y, alpha)
    return x


@decorator_knowngood
def _compilable_stochastic_add_divide_(x: list[Tensor], y: list[Tensor], alpha: Tensor, divisor: Tensor):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, (x32 + y32 * alpha) / divisor)


def stochastic_add_divide_(
    x: list[Tensor] | Tensor, y: list[Tensor] | Tensor, alpha: float | int | Tensor = 1, divisor: float = 1
):
    x, y = broadcastable_list_guard(x, y)
    if not x:
        return x
    alpha, divisor = scalar_guard(alpha, divisor, x[0])
    _compilable_stochastic_add_divide_(x, y, alpha, divisor)
    return x


@decorator_knowngood
def _compilable_stochastic_multiply_(x: list[Tensor], y: list[Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, x32 * y32)


def stochastic_multiply_(x: list[Tensor] | Tensor, y: list[Tensor] | Tensor):
    x, y = broadcastable_list_guard(x, y)
    if not x:
        return x
    _compilable_stochastic_multiply_(x, y)
    return x


def _apply_division_backend(x32: Tensor, y32: Tensor, eps: Tensor, backend: DivisionBackend) -> Tensor:
    if backend is DivisionBackend.eps_add:
        return x32 / (y32 + eps)
    if backend is DivisionBackend.eps_clamp:
        return x32 / y32.clamp(min=eps)
    if backend is DivisionBackend.atan2:
        return torch.atan2(x32.abs() / atan2_scale, y32.abs()) * x32.sign() * y32.sign() * atan2_scale
    if backend is DivisionBackend.nan_to_0:
        return torch.nan_to_num(torch.divide(x32, y32), nan=0.0, posinf=0.0, neginf=0.0)
    raise AssertionError(f"Unhandled division backend: {backend}")


@decorator_knowngood
def _compilable_stochastic_divide_(x: list[Tensor], y: list[Tensor], eps: Tensor, backend: DivisionBackend):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, _apply_division_backend(x32, y32, eps, backend))


def stochastic_divide_with_eps_(
    x: list[Tensor] | Tensor,
    y: list[Tensor] | Tensor,
    eps: float = 1e-6,
    *,
    backend: DivisionBackendLike = None,
):
    x, y = broadcastable_list_guard(x, y)
    if not x:
        return x
    eps = scalar_guard(eps, y[0])
    backend_enum = _normalize_division_backend(backend)
    _compilable_stochastic_divide_(x, y, eps, backend_enum)
    return x


def stochastic_divide_(
    x: list[Tensor] | Tensor,
    y: list[Tensor] | Tensor,
    *,
    backend: DivisionBackendLike = None,
    eps: float = 1e-12,
):
    stochastic_divide_with_eps_(x, y, eps, backend=backend)


@decorator_knowngood
def update_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    """
    Simplified by @francois-rozet in commit 704ccc4bab52429f945df421647ec82c54cdd65f
    Re-commited due to faulty merge
    """
    if grad.dim() == 2 and (not precondition_1d or grad.shape[1] > max_precond_dim):
        return

    grad = promote(grad)
    g0 = einsum_base[: grad.dim() - 1]
    for idx, m in enumerate(GG):
        if not isinstance(m, Tensor):
            continue
        b = einsum_base[idx]
        g1 = g0.replace(b, b.upper())
        work = grad.to(torch.promote_types(grad.dtype, promote(m.dtype)))
        outer_product = compiled_einsum(f"...{g0},...{g1}->...{b + b.upper()}", work, work)
        stochastic_lerp_(m, outer_product, 1 - beta)


@decorator_knowngood
def update_ggt_kl(grad, GG, Q, eigenvalues, beta, eps, *, heavy: bool = False):
    grad = promote(grad)
    Q = [None if q is None else promote(q) for q in Q]
    values = [None if value is None else promote(value) for value in eigenvalues]
    modes = einsum_base[: grad.dim() - 1]
    updates = []
    for i, (m, eigenvalue) in enumerate(zip(GG, eigenvalues)):
        if not isinstance(m, Tensor):
            continue
        work = project(grad, [q if j != i else None for j, q in enumerate(Q)], False)
        work = work.to(torch.promote_types(work.dtype, promote(m.dtype)))
        for j, value in enumerate(values):
            if j == i or value is None:
                continue
            inv = torch.where(value > eps, value.rsqrt(), 0.0) if heavy else value.clamp_min(eps).rsqrt()
            shape = [1] * work.ndim
            shape[0], shape[j + 1] = inv.shape[0], -1
            work = work * inv.view(shape)

        normalizer = 1
        for j in range(len(GG)):
            if j != i:
                normalizer *= grad.shape[j + 1]
        reduce_dims = tuple(j + 1 for j in range(len(GG)) if j != i)
        projected = project(work, [q if j == i else None for j, q in enumerate(Q)], False)
        estimate = projected.square().sum(reduce_dims) / normalizer if reduce_dims else projected.square()
        other = modes.replace(modes[i], modes[i].upper())
        outer = compiled_einsum(f"...{modes},...{other}->...{modes[i] + modes[i].upper()}", work, work) / normalizer
        updates.append((m, eigenvalue, outer, estimate))

    for m, eigenvalue, outer, estimate in updates:
        stochastic_lerp_(m, outer, 1 - beta)
        stochastic_lerp_(eigenvalue, estimate, 1 - beta)


@decorator_knowngood
def _kl_shampoo_kron_scale(grad: Tensor, eigenvalues: list[Tensor | None], eps: float, heavy: bool):
    out = promote(grad)
    for idx, value in enumerate(eigenvalues):
        if value is None:
            continue
        value = promote(value)
        inv = torch.where(value > eps, value.rsqrt(), 0.0) if heavy else value.clamp_min(eps).rsqrt()
        shape = [1] * out.ndim
        shape[0] = inv.shape[0]
        shape[idx + 1] = -1
        out = out * inv.view(shape)
    return out


def kl_shampoo_precondition(grad, Q, eigenvalues, eps, *, heavy: bool = False):
    return project(_kl_shampoo_kron_scale(project(grad, Q, back=False), eigenvalues, eps, heavy), Q, back=True)


class _ULPState:
    __slots__ = ("correction", "smax")
    _SMAX = {torch.int8: 127.0, torch.int16: 32767.0}

    def __init__(self, correction, smax):
        self.correction = correction
        self.smax = smax

    def decode(self, x):
        ls = (_log_ulp(x) - 1).float()
        return x.float() + _scale_by_exp2(self.correction.float() / self.smax, ls)

    @staticmethod
    def _bf16_to_f32(x):
        # Decode from bits so Inductor cannot erase the preceding narrowing roundtrip.
        return x.view(dtype=torch.int16).to(torch.int32).bitwise_left_shift(16).view(dtype=torch.float32)

    @staticmethod
    def _fp16_to_f32(x):
        # Expand binary16 fields explicitly, including subnormals, for the same compiler barrier.
        bits = x.view(dtype=torch.int16).to(torch.int32).bitwise_and(0xFFFF)
        sign = bits.bitwise_and(0x8000)
        exponent = bits.bitwise_right_shift(10).bitwise_and(0x1F)
        mantissa = bits.bitwise_and(0x03FF)
        normal_bits = (
            sign.bitwise_left_shift(16)
            .bitwise_or((exponent + 112).bitwise_left_shift(23))
            .bitwise_or(mantissa.bitwise_left_shift(13))
        )
        special_bits = sign.bitwise_left_shift(16).bitwise_or(0x7F800000).bitwise_or(mantissa.bitwise_left_shift(13))
        converted = torch.where(exponent == 0x1F, special_bits, normal_bits).view(dtype=torch.float32)
        subnormal = mantissa.float() * 2**-24
        subnormal = torch.where(sign != 0, -subnormal, subnormal)
        return torch.where(exponent == 0, subnormal, converted)

    def compute_correction(self, fp32, narrow, generator=None):
        narrow_f32 = self._bf16_to_f32(narrow) if narrow.dtype == torch.bfloat16 else self._fp16_to_f32(narrow)
        e = fp32 - narrow_f32
        ls = (_log_ulp(narrow) - 1).float()
        e_norm = _scale_by_exp2(e, -ls)
        scaled = e_norm.clamp(-1.0, 1.0) * self.smax
        # SR on the int correction (the bits below the correction's resolution)
        # narrow is RNE so |e| ≤ ULP/2, keeping `scaled` in [-smax, smax]; SR
        # adds at most 1 unit of correction (= ULP/(2*smax)) of error and is
        # unbiased on the lowest representable bits.
        noise = (
            torch.rand_like(scaled)
            if generator is None
            else torch.rand(scaled.shape, dtype=scaled.dtype, device=scaled.device, generator=generator)
        )
        rounded = (scaled + noise).floor()
        self.correction.copy_(rounded.to(self.correction.dtype))

    def encode(self, fp32, target, generator=None):
        # RNE on the narrow keeps |error| ≤ ULP/2 so the correction range stays
        # ±ULP/2; SR is applied on the int correction inside compute_correction.
        rounded = fp32.to(target.dtype)
        set_(target, rounded)
        if generator is None:
            self.compute_correction(fp32, target)
        else:
            self.compute_correction(fp32, target, generator)


def _promote_leaf(x):
    if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16, torch.int8):
        return torch.float32
    if isinstance(x, Tensor):
        ecc = getattr(x, "_ecc", None)
        if ecc is not None:
            return ecc.decode(x)
        if x.dtype in (torch.bfloat16, torch.float16):
            return x.float()
    return x


def promote(x):
    if isinstance(x, (Tensor, torch.dtype)):
        return _promote_leaf(x)
    if isinstance(x, list):
        return [promote(v) for v in x]
    if isinstance(x, tuple):
        return tuple(promote(v) for v in x)
    if isinstance(x, dict):
        return {k: promote(v) for k, v in x.items()}
    return tree_map(_promote_leaf, x)


def promote_detach(x, should_promote):
    if x is None:
        return x
    if should_promote:
        x = promote(x)
    return x.detach()


def detach(x):
    if isinstance(x, Tensor):
        return x.detach()
    return x


def _preconditioner_matrices(grad, max_precond_dim, precondition_1d, state_dtype, init_factor=0.0):
    if grad.is_complex():
        raise TypeError("SOAP requires real parameters")
    matrices = []
    if grad.numel() > 1 and (grad.ndim > 2 or precondition_1d):
        n = grad.shape[0]
        for sh in grad.shape[1:]:
            if sh > max_precond_dim or sh == 1:
                matrices.append(None)
            elif init_factor > 0:
                matrices.append(
                    torch.eye(sh, device=grad.device, dtype=state_dtype).expand(n, sh, sh).contiguous() * init_factor
                )
            else:
                matrices.append(torch.zeros(n, sh, sh, device=grad.device, dtype=state_dtype))
    else:
        matrices.append(None)
    return matrices


def init_preconditioner(
    grad,
    state,
    max_precond_dim,
    precondition_1d,
    init_factor,
    *,
    state_dtype,
):
    grad = promote(grad)
    state["GG"] = _preconditioner_matrices(grad, max_precond_dim, precondition_1d, state_dtype, init_factor)

    if init_factor <= 0:
        update_ggt(grad, state["GG"], max_precond_dim, precondition_1d, 0)
    state["Q"] = get_orthogonal_matrix(state["GG"])


def init_kl_preconditioner(
    grad,
    state,
    max_precond_dim,
    precondition_1d,
    init_factor,
    beta,
    eps,
    *,
    state_dtype,
    heavy=False,
):
    if init_factor <= 0 or not math.isfinite(init_factor):
        raise ValueError("init_factor must be finite and positive")
    grad = promote(grad)
    GG = _preconditioner_matrices(grad, max_precond_dim, precondition_1d, state_dtype)
    Q = [
        None if m is None else torch.eye(m.shape[-1], device=m.device, dtype=m.dtype).expand_as(m).contiguous()
        for m in GG
    ]
    eigen_dtype = promote(state_dtype)
    eigenvalues = [
        None if m is None else torch.full(m.shape[:-1], init_factor, device=m.device, dtype=eigen_dtype) for m in GG
    ]
    update_ggt_kl(grad, GG, Q, eigenvalues, beta, eps, heavy=heavy)
    for value in eigenvalues:
        if value is not None:
            value.fill_(init_factor)
    state.update(GG=GG, Q=get_orthogonal_matrix(GG), eigenvalues=eigenvalues)


@decorator
def project(grad, Q, back: bool):
    """
    :param grad:
    :param Q:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    grad = promote(grad)
    param = einsum_base[: grad.dim() - 1]
    preconditioners = ",".join(
        ["..." + (g + g.upper())[:: -1 if back else 1] for m, g in zip(Q, param) if m is not None]
    )
    if preconditioners:
        out = "".join([c.upper() if c.upper() in preconditioners else c for c in param])
        out = casted_einsum(
            f"...{param},{preconditioners}->...{out}",
            grad,
            *[promote(q) for q in Q if q is not None],
        )
        grad = out
    return grad


@contextlib.contextmanager
def patch_backward():
    tensor_backward = torch.Tensor.backward
    autograd_backward = torch.autograd.backward

    def tensor_backward_with_graph(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        return tensor_backward(self, gradient, retain_graph, True, inputs=inputs)

    def autograd_backward_with_graph(
        tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None
    ):
        return autograd_backward(tensors, grad_tensors, retain_graph, True, grad_variables, inputs)

    torch.Tensor.backward = tensor_backward_with_graph
    torch.autograd.backward = autograd_backward_with_graph
    try:
        yield
    finally:
        torch.Tensor.backward = tensor_backward
        torch.autograd.backward = autograd_backward


def hasattr_none(obj, name):
    return getattr(obj, name, None) is not None


def set_temporary(group: dict, tensor: Tensor, **kwargs):
    if not kwargs:
        return
    state = group.setdefault("_tmp", {}).setdefault(id(tensor), {"tensor": tensor})
    state.update(kwargs)


def get_temporary(group: dict, tensor: Tensor):
    tmp = group.get("_tmp")
    return None if tmp is None else tmp.get(id(tensor))


def take_temporary(group: dict, tensor: Tensor, *keys):
    state = get_temporary(group, tensor)
    if state is None:
        return None if len(keys) == 1 else (None,) * len(keys)
    out = tuple(state.pop(key, None) for key in keys)
    if len(state) == 1:
        group["_tmp"].pop(id(tensor), None)
    return out[0] if len(keys) == 1 else out


class ExactHVPFailed(ValueError):
    pass


use_default = object()


def _tensor_key(x: Tensor):
    return id(x)


class StatefulOptimizer(torch.optim.Optimizer):
    ema_decay: float = 0.001
    compile_step: bool = False
    hessian_approx: bool = False
    precond_schedule: Callable | float | None = None
    finite_differences: bool = False
    fallback_to_finite_differences: bool = True
    _fallback_enabled: bool = False
    hvp_interval: int = 1
    consume_grad: bool = True

    _INSTANCE_ATTRS = (
        "compile_step",
        "finite_differences",
        "fallback_to_finite_differences",
        "hvp_interval",
        "hessian_approx",
        "consume_grad",
    )

    def __init__(self, params, defaults, use_ema: bool = False):
        for attr in self._INSTANCE_ATTRS:
            if attr in defaults:
                val = defaults.pop(attr)
                if val is not use_default:
                    setattr(self, attr, val)
        defaults.setdefault("multi_tensor", True)
        super().__init__(params, defaults)
        self.use_ema = use_ema
        self.mapping = {}
        self.mapping_inverse = {}

        self.inner_group = {}
        self._is_preconditioning = None

        if self.hessian_approx and self.compile_step:
            raise ValueError("Hessian approximation can't be used with compile_step.")
        self.register_state_dict_post_hook(StatefulOptimizer._store_stats)
        self.register_load_state_dict_pre_hook(StatefulOptimizer._load_stats)

    def _store_stats(self, state_dict: dict[str, any]):
        topology = self._checkpoint_topology() if hasattr(self, "_checkpoint_topology") else None
        state_dict["heavyball"] = {
            "inner_group": self.inner_group.copy(),
            "_fallback_enabled": self._fallback_enabled,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "precond_schedule": getattr(self, "_precond_schedule_spec", None),
            "attrs": {name: getattr(self, name) for name in self._INSTANCE_ATTRS},
            "topology": topology,
        }

    def _load_stats(self, state_dict):
        sd = state_dict["heavyball"]
        topology = sd["topology"]
        for saved, current in zip(state_dict["param_groups"], self.param_groups, strict=True):
            if saved.get("param_ecc") != current.get("param_ecc"):
                raise ValueError("Checkpoint param_ecc must match the optimizer")
        self.inner_group = sd["inner_group"].copy()
        self._fallback_enabled = sd["_fallback_enabled"]
        self.use_ema = sd["use_ema"]
        self.ema_decay = sd["ema_decay"]
        for name, value in sd["attrs"].items():
            setattr(self, name, value)
        if hasattr(self, "_reset_compiled_chains"):
            self._reset_compiled_chains()
        spec = sd["precond_schedule"]
        if spec is not None:
            kind, value = spec
            if kind == "constant":
                self.precond_schedule = value
            elif kind == "none":
                self.precond_schedule = None
            elif kind == "soap":
                self.precond_schedule = get_soap_precond_schedule(value)
            elif kind == "psgd":
                self.precond_schedule = precond_update_prob_schedule(**value)
            self._precond_schedule_spec = spec
        self._loaded_topology = topology
        states = state_dict["state"]
        values = {}
        for saved_group, current_group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for saved_id, param in zip(saved_group["params"], current_group["params"], strict=True):
                if saved_id in states:
                    values[param] = states[saved_id]
        self._loaded_state_values = values

    def get_groups(self, group):
        return [group]

    def state_(self, arg: Tensor, fail: bool = True):
        key = _tensor_key(arg)
        if key not in self.mapping_inverse:
            self._init_mapping()
        if key not in self.mapping_inverse and arg in self.mapping and len(self.mapping[arg]) == 1:
            return self.state.setdefault(arg, {}).setdefault(0, {})
        if key not in self.mapping_inverse:
            if not fail:
                return {}
            raise KeyError("Tensor has no tracked state.")
        state_param, index = self.mapping_inverse[key]
        return self.state.setdefault(state_param, {}).setdefault(index, {})

    def _root_state(self, param):
        return self.state.setdefault(param, {}).setdefault("_root", {})

    def _clear_views(self, p):
        for view in self.mapping.pop(p, ()):
            self.mapping_inverse.pop(_tensor_key(view), None)

    def _set_views(self, p, group):
        if p in self.mapping:
            return self.mapping[p]
        source = p.detach() if group.get("merge_dims", False) else p
        self.mapping[p] = p_views = merge_group(group, source)
        for i, pv in enumerate(p_views):
            self.mapping_inverse[_tensor_key(pv)] = (p, i)
        return p_views

    @contextlib.contextmanager
    def _contiguous_param(self, p, group):
        original = p.data if group.get("merge_dims", False) and not p.data.is_contiguous() else None
        if original is not None:
            self._clear_views(p)
            p.data = original.contiguous()
        try:
            yield
        finally:
            if original is not None:
                original.copy_(p.data)
                p.data = original
                self._clear_views(p)

    def _init_mapping(self, group: dict | None = None):
        if group is None:
            for group in self.param_groups:
                self._init_mapping(group)
            return

        for p in group["params"]:
            if p not in self.mapping:
                self._set_views(p, group)

    def split_p_and_g_in_group(
        self,
        group: dict,
        skip_none: bool = True,
        should_promote: bool = True,
        raw: bool = False,
    ):
        tmp = group.get("_tmp")
        for p in group["params"]:
            if raw:
                grad = getattr(p, "grad", None)
                if grad is None and skip_none:
                    continue
                if grad is not None and grad.numel() == 0:
                    if self.consume_grad:
                        p.grad = None
                    continue
                if self.consume_grad:
                    p.grad = None
                yield p, grad
                continue

            state = None if tmp is None else tmp.get(id(p))
            grad = None if state is None else state.pop("grad", None)
            if grad is None:
                grad = getattr(p, "grad", None)
            if grad is None and skip_none:
                continue
            if grad is not None and grad.numel() == 0:
                if self.consume_grad:
                    p.grad = None
                continue

            if self.consume_grad:
                p.grad = None

            if group.get("merge_dims", False) and not p.data.is_contiguous():
                original = p.data
                set_temporary(group, p, restore_data=original)
                self._clear_views(p)
                p.data = original.contiguous()

            p_views = self._set_views(p, group)

            if state is None:
                vector = hessian_vector = None
            else:
                vector = state.pop("vector", None)
                hessian_vector = state.pop("hessian_vector", None)
                if len(state) == 1:
                    tmp.pop(id(p), None)
            grad = itertools.repeat(None, len(p_views)) if grad is None else merge_group(group, grad)
            vs = itertools.repeat(None, len(p_views)) if vector is None else merge_group(group, vector)
            hvs = itertools.repeat(None, len(p_views)) if hessian_vector is None else merge_group(group, hessian_vector)

            for pv, g, v, hv in zip(p_views, grad, vs, hvs):
                g = promote_detach(g, should_promote)
                if g is not None and not self.consume_grad:
                    g = g.clone(memory_format=torch.preserve_format)
                v = promote_detach(v, should_promote)
                hv = promote_detach(hv, should_promote)
                if v is not None or hv is not None:
                    set_temporary(group, pv, vector=v, hessian_vector=hv)
                yield pv, g

    def state_size(self) -> int:
        total_bytes = 0
        seen: set[int] = set()

        def _add(x):
            nonlocal total_bytes
            if isinstance(x, Tensor) and id(x) not in seen:
                seen.add(id(x))
                total_bytes += x.numel() * x.element_size()

        for st in self.state.values():
            tree_map(_add, st)
        return total_bytes

    def _step(self, group):
        raise NotImplementedError

    def ema_update(self):
        with torch.no_grad():
            for group in self.param_groups:
                k = group["ema_step"] = group.get("ema_step", -1) + 1
                w = beta_debias(1 - self.ema_decay, k + 1)
                param_ecc = group.get("_param_ecc_config")
                for p in group["params"]:
                    state = self._root_state(p)
                    if "param_ema" not in state:
                        state["param_ema"] = torch.zeros_like(
                            p.data, dtype=promote(p.dtype), memory_format=torch.contiguous_format
                        )
                    with self._contiguous_param(p, group):
                        ema_views = merge_group(group, state["param_ema"])
                        for pv, ema in zip(self._set_views(p, group), ema_views):
                            view_state = self.state_(pv)
                            with contextlib.ExitStack() as stack:
                                if param_ecc is not None:
                                    correction = view_state["param::ecc"].view_as(pv)
                                    stack.enter_context(param_ecc.attached([pv], [correction]))
                                ema32, p32 = promote(ema), promote(pv)
                                dtype = torch.promote_types(ema32.dtype, p32.dtype)
                                ema32, p32 = ema32.to(dtype), p32.to(dtype)
                                copy_stochastic_(ema, ema32 + (p32 - ema32) * (1 - w))

    def copy_emas_to_params(self):
        with torch.no_grad():
            for group in self.param_groups:
                param_ecc = group.get("_param_ecc_config")
                for p in group["params"]:
                    state = self._root_state(p)
                    if "param_ema" in state:
                        with self._contiguous_param(p, group):
                            ema_views = merge_group(group, state["param_ema"])
                            for pv, ema in zip(self._set_views(p, group), ema_views):
                                view_state = self.state_(pv)
                                with contextlib.ExitStack() as stack:
                                    if param_ecc is not None:
                                        correction = view_state["param::ecc"].view_as(pv)
                                        stack.enter_context(param_ecc.attached([pv], [correction]))
                                    value = promote(pv).clone()
                                    copy_stochastic_(pv, promote(ema))
                                    copy_stochastic_(ema, value)

    copy_params_to_emas = copy_emas_to_params

    def _finite_differences_hvp(self, closure):
        cuda_devices = sorted(
            {p.device.index for group in self.param_groups for p in group["params"] if p.device.type == "cuda"}
        )
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
        with torch.enable_grad():
            loss = closure()

        consumed, probes, entries = [], [], []
        complete = False
        try:
            for group in self.param_groups:
                for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True):
                    consumed.append((p, g))
                    g = g.detach().clone(memory_format=torch.preserve_format)
                    vector = torch.randn_like(p, dtype=promote(p.dtype))
                    p_scale = promote(p.detach()).abs().amax()
                    v_scale = promote(vector).abs().amax()
                    original = p.data.clone()
                    delta = (
                        math.sqrt(torch.finfo(p.dtype).eps)
                        * (1 + p_scale)
                        / torch.where(v_scale != 0, v_scale, torch.ones_like(v_scale))
                    )
                    delta = torch.where(v_scale != 0, delta, torch.zeros_like(delta))
                    probes.append((group, p, g, vector, original, delta))

            if not probes:
                raise ValueError("No parameter has gradients")

            by_device = {}
            for *_, candidate in probes:
                by_device.setdefault(candidate.device, []).append(candidate)
            delta = max(torch.stack(values).amax().item() for values in by_device.values())
            delta = delta or 1.0
            for group, p, grad, vector, original, _ in probes:
                limits = torch.finfo(p.dtype)
                perturbed = (promote(original) + promote(vector) * delta).clamp(limits.min, limits.max).to(p.dtype)
                vector = (promote(perturbed) - promote(original)) / delta
                entries.append((group, p, grad, vector, original))
                p.data.copy_(perturbed)

            post_cpu_rng = torch.random.get_rng_state()
            post_cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
            try:
                torch.random.set_rng_state(cpu_rng)
                for device, state in cuda_rng.items():
                    torch.cuda.set_rng_state(state, device)
                with torch.enable_grad():
                    closure()
            finally:
                torch.random.set_rng_state(post_cpu_rng)
                for device, state in post_cuda_rng.items():
                    torch.cuda.set_rng_state(state, device)

            second = {
                id(p): (p, g.detach().clone(memory_format=torch.preserve_format))
                for group in self.param_groups
                for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True)
            }
            first_ids = {id(p) for _, p, *_ in entries}
            if first_ids != second.keys():
                for key in second.keys() - first_ids:
                    second[key][0].grad = None
                raise ExactHVPFailed("Finite-difference gradient set changed")
            for group, p, grad, vector, _ in entries:
                hessian_vector = (promote(second[id(p)][1]) - promote(grad)) / delta
                set_temporary(group, p, grad=grad, vector=vector, hessian_vector=hessian_vector)
                p.grad = None if self.consume_grad else grad.clone(memory_format=torch.preserve_format)
            complete = True
            return loss
        finally:
            for _, p, _, _, original, _ in probes:
                p.data.copy_(original)
            if not complete:
                for p, grad in consumed:
                    p.grad = grad

    def _double_backward_hvp(self, closure):
        with torch.enable_grad(), patch_backward():
            loss = closure()

        params, grads, saved_grads, groups = [], [], [], []
        try:
            for group in self.param_groups:
                for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True):
                    params.append(p)
                    grads.append(g)
                    saved_grads.append(g.detach().clone(memory_format=torch.preserve_format))
                    groups.append(group)
        except BaseException:
            for p, g in zip(params, saved_grads):
                p.grad = g
            raise

        complete = False
        try:
            if not params:
                raise ValueError("No parameter has gradients")
            vs = [torch.randn_like(p) for p in params]
            with torch.enable_grad():
                try:
                    hvs = torch.autograd.grad(
                        grads, params, vs, create_graph=False, retain_graph=False, allow_unused=True
                    )
                except RuntimeError as error:
                    raise ExactHVPFailed(str(error.args)) from error

            unused = [list(p.shape) for p, hv in zip(params, hvs) if hv is None]
            if unused:
                raise ExactHVPFailed(f"Parameters with the following shapes have no 2nd order derivative: {unused}")

            for group, p, g, saved, v, hv in zip(groups, params, grads, saved_grads, vs, hvs):
                set_temporary(group, p, grad=detach(g), vector=detach(v), hessian_vector=detach(hv))
                p.grad = None if self.consume_grad else saved
            complete = True
            return loss
        finally:
            if not complete:
                for p, grad in zip(params, saved_grads):
                    p.grad = grad

    def _handle_closure(self, closure):
        hessian_approx = self.hessian_approx and self._is_preconditioning

        if closure is None:
            if hessian_approx:
                raise ValueError("Hessian approximation requires a closure.")
            return None

        if not hessian_approx:
            with torch.enable_grad():
                return closure()

        step = self.inner_group.get("total_hvp_steps", 0) + 1
        if (step - 1) % self.hvp_interval != 0:
            with torch.enable_grad():
                loss = closure()
        elif self.finite_differences or self._fallback_enabled:
            loss = self._finite_differences_hvp(closure)
        else:
            cpu_rng = cuda_rng = None
            if self.fallback_to_finite_differences:
                cpu_rng = torch.random.get_rng_state()
                cuda_devices = sorted(
                    {p.device.index for group in self.param_groups for p in group["params"] if p.device.type == "cuda"}
                )
                cuda_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
            try:
                loss = self._double_backward_hvp(closure)
            except NotImplementedError as e:
                if not self.fallback_to_finite_differences:
                    raise
                if not any(isinstance(arg, str) and _cudnn_double_backward_pattern.match(arg) for arg in e.args):
                    raise
                warn_once(
                    "CUDNN doesn't support double-backward for some models (including RNNs). "
                    f"Falling back to finite_differences.\n{_fd_error}{e}"
                )
            except RuntimeError as e:
                if not self.fallback_to_finite_differences:
                    raise
                if not any(
                    isinstance(arg, str) and _torch_compile_double_backward_pattern.match(arg) for arg in e.args
                ):
                    raise
                warn_once(
                    f"torch.compile does not support double-backward. Disabling it may be beneficial, depending on "
                    f"the model.\n{_fd_error}{e}"
                )
            except ExactHVPFailed as e:
                if not self.fallback_to_finite_differences:
                    raise
                warn_once(f"Exact HVP calculation failed.\n{_fd_error}{e}")
            else:
                self.inner_group["total_hvp_steps"] = step
                return loss
            torch.random.set_rng_state(cpu_rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            loss = self._finite_differences_hvp(closure)
            self._fallback_enabled = True
        self.inner_group["total_hvp_steps"] = step
        return loss

    def _cleanup_temporary_tensors(self):
        for group in self.param_groups:
            tmp = group.pop("_tmp", None)
            if tmp is None:
                continue
            for state in tmp.values():
                original = state.get("restore_data")
                if original is None:
                    continue
                tensor = state["tensor"]
                original.copy_(tensor.data)
                tensor.data = original
                self._clear_views(tensor)

    def step(self, closure: Callable | None = None):
        previous_is_preconditioning = self._is_preconditioning
        schedule = None
        if self.precond_schedule is None:
            self._is_preconditioning = False
            precond_prob = 0.0
        else:
            schedule_step = self.inner_group.get("cumulative_prob_prob_step", 0) + 1
            precond_prob = (
                self.precond_schedule(schedule_step) if callable(self.precond_schedule) else self.precond_schedule
            )
            previous = self.inner_group.get("cumulative_prob", 0)
            compensation = self.inner_group.get("cumulative_prob_compensation", 0)
            adjusted = precond_prob - compensation
            cumulative = previous + adjusted
            compensation = (cumulative - previous) - adjusted
            self._is_preconditioning = int(cumulative) > int(previous)
            schedule = schedule_step, precond_prob, cumulative, compensation
        try:
            loss = self._handle_closure(closure)
        except BaseException:
            self._is_preconditioning = previous_is_preconditioning
            self._cleanup_temporary_tensors()
            raise
        if schedule is not None:
            step, probability, cumulative, compensation = schedule
            self.inner_group["cumulative_prob_prob_step"] = step
            self.inner_group["cumulative_prob_last_prob"] = probability
            self.inner_group["cumulative_prob"] = cumulative
            self.inner_group["cumulative_prob_compensation"] = compensation

        try:
            with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
                for group in self.param_groups:
                    shapes = getattr(self, "_orig_shapes", None) or {}
                    if "param_count" not in group or shapes:
                        group["param_count"] = sum(
                            getattr(shapes.get(id(p)), "total", p.numel()) for p in group["params"]
                        )
                    group["is_preconditioning"] = self._is_preconditioning
                    group["_precond_prob"] = precond_prob
                    self._step(group)
        finally:
            self._cleanup_temporary_tensors()
        if self.use_ema:
            self.ema_update()
        return loss


def copy_stochastic_list_(target: list[Tensor], source: list[Tensor]):
    for t, s in zip(target, source):
        copy_stochastic_(t, s)


@decorator_knowngood
def _lerp(state: list[Tensor], grad: list[Tensor], beta):
    beta = promote(beta)
    out = []
    for s, g in zip(state, grad):
        s, g = promote(s), promote(g)
        dtype = torch.promote_types(s.dtype, g.dtype)
        s, g = s.to(dtype), g.to(dtype)
        out.append(s * beta + g * (1 - beta))
    copy_stochastic_list_(state, out)
    return out


@decorator_knowngood
def _compilable_adam_(
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor | None,
    eps: Tensor,
):
    if step is not None:
        beta1, beta2 = beta_debias(beta1, step), beta_debias(beta2, step)
    g32 = list(map(promote, grad))
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, g32, beta2, eps, [None])
    return [e_ / d_ for e_, d_ in zip(exp_avg32, denom)]


def adam_(
    exp_avg: list[Tensor] | Tensor,
    exp_avg_sq: list[Tensor] | Tensor,
    grad: list[Tensor] | Tensor,
    beta1: float,
    beta2: float,
    step: int | Tensor | None,
    eps: float = 1e-8,
) -> list[Tensor]:
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    if not grad:
        return grad
    dtype = promote(exp_avg[0].dtype)
    if step is None:
        beta1 = (
            beta1.to(dtype=dtype)
            if isinstance(beta1, Tensor) and beta1.device.type == "cpu"
            else scalar_guard(beta1, exp_avg[0])
        )
        beta2 = (
            beta2.to(dtype=dtype)
            if isinstance(beta2, Tensor) and beta2.device.type == "cpu"
            else scalar_guard(beta2, exp_avg[0])
        )
    else:
        beta1, beta2, step = scalar_guard(beta1, beta2, step, exp_avg[0])
    eps = scalar_guard(eps, exp_avg[0])
    return _compilable_adam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)


@decorator_knowngood
def _compilable_unscaled_adam_(
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    g32 = list(map(promote, grad))
    denom = _compilable_exp_avg_sq_(exp_avg_sq, g32, beta2, eps, [None])
    g32 = [g_ / d_ for g_, d_ in zip(g32, denom)]
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    return [e_ * d_ for e_, d_ in zip(exp_avg32, denom)]


def unscaled_adam_(
    exp_avg: list[Tensor] | Tensor,
    exp_avg_sq: list[Tensor] | Tensor,
    grad: list[Tensor] | Tensor,
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
) -> list[Tensor]:
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    if not grad:
        return grad
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    return _compilable_unscaled_adam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)


@decorator_knowngood
def _fused_compilable_adam_(
    y: list[Tensor],
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor | None,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
    cautious_decay: bool,
):
    if step is not None:
        beta1, beta2 = beta_debias(beta1, step), beta_debias(beta2, step)
    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    exp_avg32 = _lerp(exp_avg, u32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    _compilable_update_(y, [e_ / d_ for e_, d_ in zip(exp_avg32, denom)], decay, lr, caution, cautious_decay, g32)


def fused_adam_(
    y: list[Tensor],
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: float,
    beta2: float,
    step: int | Tensor | None,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
    cautious_decay: bool = False,
):
    y, exp_avg, exp_avg_sq, update, grad = list_guard(y, exp_avg, exp_avg_sq, update, grad)
    if not y:
        return
    dtype = promote(y[0].dtype)
    if step is None:
        beta1 = (
            beta1.to(dtype=dtype)
            if isinstance(beta1, Tensor) and beta1.device.type == "cpu"
            else scalar_guard(beta1, y[0])
        )
        beta2 = (
            beta2.to(dtype=dtype)
            if isinstance(beta2, Tensor) and beta2.device.type == "cpu"
            else scalar_guard(beta2, y[0])
        )
    else:
        beta1, beta2, step = scalar_guard(beta1, beta2, step, y[0])
    lr, eps, decay = scalar_guard(lr, eps, decay, y[0])
    _fused_compilable_adam_(
        y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, decay, lr, eps, caution, cautious_decay
    )


def nadam_(
    param: list[Tensor] | Tensor,
    exp_avg: list[Tensor] | Tensor,
    exp_avg_sq: list[Tensor] | Tensor,
    mu_product: list[Tensor] | Tensor,
    update: list[Tensor] | Tensor,
    beta1: float,
    beta2: float,
    step: int,
    momentum_decay: float,
    eps: float,
    weight_decay: float,
    decoupled_weight_decay: bool,
) -> list[Tensor]:
    param, exp_avg, exp_avg_sq, mu_product, update = list_guard(param, exp_avg, exp_avg_sq, mu_product, update)
    if not param:
        return update

    beta1_t, beta2_t, step_t, eps_t = scalar_guard(beta1, beta2, step, eps, param[0])
    update32 = promote(update)
    param32 = promote(param)

    _nadam_prepare_weight_decay(update32, param32, weight_decay, decoupled_weight_decay)
    mu_t, mu_next_t = _nadam_moments(beta1_t, step_t, momentum_decay)
    update32, mu_product32 = _nadam_compute_update(
        exp_avg, exp_avg_sq, mu_product, update32, beta1_t, beta2_t, step_t, eps_t, mu_t, mu_next_t
    )

    copy_stochastic_list_(mu_product, mu_product32)
    return update32


@decorator_knowngood
def _fused_compilable_nadam_(
    param: list[Tensor],
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    mu_product: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    lr: Tensor,
    eps: Tensor,
    mu: Tensor,
    mu_next: Tensor,
    weight_decay: float,
    decoupled_weight_decay: bool,
    caution: bool,
    cautious_decay: bool,
):
    update32 = promote(update)
    grad32 = promote(grad)
    param32 = promote(param)

    decay = _nadam_prepare_weight_decay(update32, param32, weight_decay, decoupled_weight_decay)
    caution_grad = grad32 if decoupled_weight_decay else update32
    update32, mu_product32 = _nadam_compute_update(
        exp_avg, exp_avg_sq, mu_product, update32, beta1, beta2, step, eps, mu, mu_next
    )

    decay_t = scalar_guard(decay, param[0])
    _compilable_update_(param, update32, decay_t, lr, caution, cautious_decay, caution_grad)
    return mu_product32


def fused_nadam_(
    param: list[Tensor] | Tensor,
    exp_avg: list[Tensor] | Tensor,
    exp_avg_sq: list[Tensor] | Tensor,
    mu_product: list[Tensor] | Tensor,
    update: list[Tensor] | Tensor,
    grad: list[Tensor] | Tensor,
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    momentum_decay: float,
    weight_decay: float,
    decoupled_weight_decay: bool,
    caution: bool,
    cautious_decay: bool = False,
):
    param, exp_avg, exp_avg_sq, mu_product, update, grad = list_guard(
        param, exp_avg, exp_avg_sq, mu_product, update, grad
    )
    if not param:
        return

    beta1_t, beta2_t, step_t, lr_t, eps_t = scalar_guard(beta1, beta2, step, lr, eps, param[0])
    mu_t, mu_next_t = _nadam_moments(beta1_t, step_t, momentum_decay)
    mu_product32 = _fused_compilable_nadam_(
        param,
        exp_avg,
        exp_avg_sq,
        mu_product,
        update,
        grad,
        beta1_t,
        beta2_t,
        step_t,
        lr_t,
        eps_t,
        mu_t,
        mu_next_t,
        weight_decay,
        decoupled_weight_decay,
        caution,
        cautious_decay,
    )
    copy_stochastic_list_(mu_product, mu_product32)


@decorator_knowngood
def _compilable_ademamix_update_(
    exp_avg_fast: list[Tensor],
    exp_avg_slow: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    step: Tensor,
    alpha: Tensor,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    update32 = promote(update)
    fast32 = _lerp(exp_avg_fast, update32, beta1)
    slow32 = _lerp(exp_avg_slow, update32, beta3)

    denom = _compilable_exp_avg_sq_(exp_avg_sq, update32, beta2, eps, [None])
    return [(f_ + s_ * alpha) / d_ for f_, s_, d_ in zip(fast32, slow32, denom)]


@decorator_knowngood
def _fused_compilable_ademamix_(
    y: list[Tensor],
    exp_avg_fast: list[Tensor],
    exp_avg_slow: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    step: Tensor,
    alpha: Tensor,
    lr: Tensor,
    eps: Tensor,
    decay: Tensor,
    caution: bool,
    cautious_decay: bool,
):
    grad32 = list(map(promote, grad))
    update32 = _compilable_ademamix_update_(
        exp_avg_fast, exp_avg_slow, exp_avg_sq, update, beta1, beta2, beta3, step, alpha, eps
    )
    _compilable_update_(y, update32, decay, lr, caution, cautious_decay, grad32)


def fused_ademamix_(
    y: list[Tensor],
    exp_avg_fast: list[Tensor],
    exp_avg_slow: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    betas: tuple[float, float, float],
    step: int,
    lr: float,
    eps: float,
    decay: float,
    alpha: float,
    caution: bool,
    cautious_decay: bool = False,
    beta3_warmup: int | None = None,
    alpha_warmup: int | None = None,
):
    y, exp_avg_fast, exp_avg_slow, exp_avg_sq, update, grad = list_guard(
        y, exp_avg_fast, exp_avg_slow, exp_avg_sq, update, grad
    )
    if not y:
        return

    ref = y[0]
    beta1_f, beta2_f, beta3_f, alpha_f = _compute_ademamix_hparams(betas, step, alpha, beta3_warmup, alpha_warmup)
    beta1_t, beta2_t, beta3_t, alpha_t, step_t, lr_t, eps_t, decay_t = scalar_guard(
        beta1_f, beta2_f, beta3_f, alpha_f, step, lr, eps, decay, ref
    )

    _fused_compilable_ademamix_(
        y,
        exp_avg_fast,
        exp_avg_slow,
        exp_avg_sq,
        update,
        grad,
        beta1_t,
        beta2_t,
        beta3_t,
        step_t,
        alpha_t,
        lr_t,
        eps_t,
        decay_t,
        caution,
        cautious_decay,
    )


def ademamix_(
    exp_avg_fast: list[Tensor] | Tensor,
    exp_avg_slow: list[Tensor] | Tensor,
    exp_avg_sq: list[Tensor] | Tensor,
    grad: list[Tensor] | Tensor,
    betas: tuple[float, float, float],
    step: int,
    eps: float,
    alpha: float,
    beta3_warmup: int | None = None,
    alpha_warmup: int | None = None,
):
    exp_avg_fast, exp_avg_slow, exp_avg_sq, grad = list_guard(exp_avg_fast, exp_avg_slow, exp_avg_sq, grad)
    if not grad:
        return grad

    ref = grad[0]
    beta1_f, beta2_f, beta3_f, alpha_f = _compute_ademamix_hparams(betas, step, alpha, beta3_warmup, alpha_warmup)
    beta1_t, beta2_t, beta3_t, alpha_t, step_t, eps_t = scalar_guard(beta1_f, beta2_f, beta3_f, alpha_f, step, eps, ref)

    update32 = _compilable_ademamix_update_(
        exp_avg_fast, exp_avg_slow, exp_avg_sq, grad, beta1_t, beta2_t, beta3_t, step_t, alpha_t, eps_t
    )
    return update32


@decorator_knowngood
def _compilable_laprop_(
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    gp32 = list(map(promote, grad))
    denom = _compilable_exp_avg_sq_(exp_avg_sq, gp32, beta2, eps, [None])
    return _lerp(exp_avg, [g_ / d_ for g_, d_ in zip(gp32, denom)], beta1)


def laprop_(
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    grad: list[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    if not grad:
        return grad
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    return _compilable_laprop_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)


@decorator_knowngood
def _fused_compilable_laprop_(
    y: list[Tensor],
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    lr: Tensor,
    decay: Tensor,
    caution: bool,
    cautious_decay: bool,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    u32, gp32 = [list(map(promote, x)) for x in [update, grad]]
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    u32 = _lerp(exp_avg, [u_ / d_ for u_, d_ in zip(u32, denom)], beta1)
    _compilable_update_(y, u32, decay, lr, caution, cautious_decay, gp32)


def fused_laprop_(
    y: list[Tensor],
    exp_avg: list[Tensor],
    exp_avg_sq: list[Tensor],
    update: list[Tensor],
    grad: list[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    decay: float,
    caution: bool,
    cautious_decay: bool = False,
    eps: float = 1e-8,
):
    y, exp_avg, exp_avg_sq, update, grad = list_guard(y, exp_avg, exp_avg_sq, update, grad)
    if not y:
        return
    beta1, beta2, step, lr, eps, decay = scalar_guard(beta1, beta2, step, lr, eps, decay, exp_avg[0])
    _fused_compilable_laprop_(
        y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, lr, decay, caution, cautious_decay, eps
    )


@decorator_knowngood
def _fused_compilable_adopt_(
    y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution, cautious_decay
):
    g32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad, exp_avg_sq]]
    u32 = []
    for u, state in zip(update, exp_avg_sq32):
        u = promote(u)
        u32.append(u.to(torch.promote_types(u.dtype, state.dtype)))

    beta1 = beta_debias(beta1, step)
    m_new = _lerp(exp_avg, [u_ / eps_sqrt(d_, eps) for u_, d_ in zip(u32, exp_avg_sq32)], beta1)
    _compilable_update_(y, m_new, decay, lr, caution, cautious_decay, g32)

    beta2 = beta_debias(beta2, step + 1)
    stochastic_lerp_(exp_avg_sq, [u_ * u_ for u_ in u32], 1 - beta2)


def fused_adopt_(
    y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution, cautious_decay=False
):
    y, update, grad, exp_avg_sq, exp_avg = list_guard(y, update, grad, exp_avg_sq, exp_avg)
    if not y:
        return
    beta1, beta2, step, lr, eps, decay = scalar_guard(beta1, beta2, step, lr, eps, decay, exp_avg[0])
    _fused_compilable_adopt_(
        y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution, cautious_decay
    )


@decorator_knowngood
def _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps):
    exp_avg_sq32 = list(map(promote, exp_avg_sq))
    g32 = []
    for g, state in zip(grad, exp_avg_sq32):
        g = promote(g)
        g32.append(g.to(torch.promote_types(g.dtype, state.dtype)))

    beta1 = beta_debias(beta1, step)
    m_new = _lerp(exp_avg, [g_ / eps_sqrt(d_, eps) for g_, d_ in zip(g32, exp_avg_sq32)], beta1)

    beta2 = beta_debias(beta2, step + 1)
    stochastic_lerp_(exp_avg_sq, [g_ * g_ for g_ in g32], 1 - beta2)
    return m_new


def adopt(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps: float = 1e-8):
    grad, exp_avg_sq, exp_avg = list_guard(grad, exp_avg_sq, exp_avg)
    if not grad:
        return grad
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    return _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps)


def stochastic_round_list_(ref: list[Tensor], source: list[Tensor]):
    ref, source = list_guard(ref, source)
    return [stochastic_round_(target, value) for target, value in zip(ref, source)]


def _stochastic_round(ref: Tensor, source: Tensor | None = None, generator: torch.Generator | None = None):
    if source is None:
        source = ref
        dtype = torch.bfloat16
    else:
        dtype = ref.dtype
    if dtype != torch.bfloat16 or source.dtype not in (torch.float16, torch.float32, torch.float64):
        return source.to(dtype)

    source = source.to(torch.float32).view(dtype=torch.int32)
    if generator is None:
        noise = sum(torch.randint_like(source, low=0, high=(1 << 16)) for _ in range(dither_steps))
    else:
        noise = sum(
            torch.randint(
                0,
                1 << 16,
                source.shape,
                dtype=source.dtype,
                device=source.device,
                generator=generator,
            )
            for _ in range(dither_steps)
        )
    noise = noise + source - (dither_steps - 1) * (1 << 15)  # center | x - (N-1)*delta/2
    noise = noise.bitwise_and(-65536)  # FFFF0000 mask, preserves sign+exp+7 mantissa bits
    return noise.view(dtype=torch.float32).bfloat16()


@decorator_knowngood
def stochastic_round_(ref: Tensor, source: Tensor | None = None):
    return _stochastic_round(ref, source)


def copy_stochastic_(target: Tensor, source: Tensor):
    ecc = getattr(target, "_ecc", None)
    if ecc is not None:
        ecc.encode(source, target)
        return
    if target.dtype == torch.bfloat16 and source.dtype in (torch.float16, torch.float32, torch.float64):
        source = stochastic_round_(target, source)
    set_(target, source)


@decorator_knowngood
def _compilable_update_(
    p: list[Tensor],
    u: list[Tensor],
    decay: Tensor,
    lr: Tensor,
    caution: bool,
    cautious_decay: bool,
    g: list[Tensor | None],
):
    for u_, g_, p_ in zip(u, g, p):  # lr is data-dependent -> can't compile a multi-tensor op
        u_ = promote(u_.view_as(p_))
        p32_ = promote(p_)
        if caution and g_ is not None:
            u_ = _compilable_cautioning(g_, u_)
        d = decay * _strictly_aligned(p32_, u_).to(p32_.dtype) if cautious_decay else decay
        p32_ = p32_ * (1 - d * lr) + u_ * -lr
        copy_stochastic_(p_, p32_)


def update_param_(
    param: list[Tensor],
    update: list[Tensor],
    lr: float,
    decay: float,
    caution: bool = False,
    cautious_decay: bool = False,
    grad: list[Tensor] = None,
):
    param, update = list_guard(param, update)
    if not param:
        return
    grad = list_guard(grad)
    lr, decay = scalar_guard(lr, decay, param[0])
    if not caution or len(grad) != len(param):
        grad = [None] * len(param)
    _compilable_update_(param, update, decay, lr, caution, cautious_decay, grad)


def precond_schedule(step: Tensor | Real, precond_scheduler):
    power, curvature = precond_scheduler
    if isinstance(step, Tensor):
        log_step = torch.log10(step.clamp(min=1))
    else:
        log_step = math.log10(max(step, 1))
    return 1 / ((log_step * power) ** curvature + 1)


def get_soap_precond_schedule(precond_scheduler):
    return functools.partial(precond_schedule, precond_scheduler=precond_scheduler)


def _max_idx(x: list[int]):
    return len(x) - 1 - np.argmax(x[::-1])  # we want to start counting from the back, as torch is fan-out/fan-in


@decorator_knowngood
def stable_exp(x: Tensor):
    return x.exp()


def _lse_mean(x: Tensor, pow: float, eps: float) -> Tensor:
    # ln(mean(x ** pow) ** (1 / pow / 2))
    normalization = math.log(x.numel())
    x = x.double()
    x = x.abs()
    x = x.clamp(min=eps)
    x = x.log()
    x = x * pow
    x = x.flatten()
    x = x.logsumexp(dim=0)  # log(sum(exp( log(x) * P ) - more stable than sum(x ** P)
    x = x - normalization  # sum -> mean (divide by x.numel() in log space)
    return x / pow / 2


@decorator_knowngood
def mean_root(x: torch.Tensor, pow: float, eps=1e-12):
    # 1 / (mean(x ** pow) ** (1 / pow / 2))
    return stable_exp(-_lse_mean(x, pow, eps))


@decorator_knowngood
def divided_root(x: torch.Tensor, y: torch.Tensor, pow0: float, pow1: float, eps=1e-12):
    # mean(x ** pow0) ** (1 / pow0 / 2) / mean(y ** pow1) ** (1 / pow1 / 2)
    return stable_exp(_lse_mean(x, pow0, eps) - _lse_mean(y, pow1, eps))


def precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector):
    scale_scale = 1 if scale_scale is None else scale_scale
    if scale is not None:
        return scale
    if hessian_vector is None:
        scale = mean_root(grad, 4) * scale_scale
    else:
        scale = divided_root(vector, hessian_vector, 2, 4) * scale_scale
    scale = scale ** (0.5 if scale_power is None else scale_power)
    signal = grad if hessian_vector is None else hessian_vector
    return torch.where(promote(signal).abs().amax() == 0, torch.ones_like(scale), scale)


def _precond_scale_tensor(scale, device, dtype=None):
    if not isinstance(scale, Tensor):
        scale = torch.tensor(scale, dtype=torch.float64, device=device)
    else:
        scale = scale.detach().to(device=device)
    if scale.numel() != 1 or scale.is_complex() or not bool(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("precond_init_scale must be a finite positive scalar")
    if dtype is None:
        return scale
    stored = scale.to(dtype)
    if not bool(torch.isfinite(stored) & (stored > 0)):
        raise ValueError(f"precond_init_scale is not representable in {dtype}")
    return stored.to(promote(dtype))


def init_lra(
    grad, param_count, scale, scale_scale, scale_power, rank, hessian_vector, vector, dtype=None, eps: float = 10
):
    # "+10 to 1) avoid /0; 2) make sure that norm(U*V') << 1 even when rank_of_approximation=1" from @lixilinx at
    # https://github.com/lixilinx/psgd_torch/blob/590cd3f125552998ed20028be096652540e2a200/preconditioned_stochastic_gradient_descent.py#L829C11-L829C14
    scale = precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector)
    dtype = dtype if dtype is not None else grad.dtype
    scale = _precond_scale_tensor(scale, grad.device, dtype)
    uv_scale = (param_count * (rank + eps)) ** -0.5
    U = torch.randn((*grad.shape, rank), dtype=dtype, device=grad.device) * uv_scale
    V = torch.randn((*grad.shape, rank), dtype=dtype, device=grad.device) * uv_scale
    d = (torch.ones_like(grad, dtype=promote(dtype), device=grad.device) * scale).to(dtype)
    return U, V, d


def init_Q_exprs(
    grad,
    scale,
    scale_scale,
    scale_power,
    max_size,
    min_ndim_triangular,
    memory_save_mode,
    hessian_vector,
    vector,
    dtype=None,
):
    """
    For a scalar or tensor `grad`, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.

    precond init scale computation from
    https://github.com/lixilinx/psgd_torch/blob/1943e66596111e78157ca1b72b31c1dfdf0653ef/preconditioned_stochastic_gradient_descent.py#L2208-L2227
    """
    dtype = dtype if dtype is not None else grad.dtype
    scale = precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector)
    scale = _precond_scale_tensor(scale, grad.device)
    n = grad.shape[0]
    shape = grad.shape[1:]

    if len(shape) == 0:  # scalar param: bucket of N scalars
        scale = _precond_scale_tensor(scale, grad.device, dtype)
        Q = [(scale * torch.ones_like(grad, dtype=promote(dtype))).to(dtype)]
        return Q

    scale = scale ** (1 / len(shape))

    dim_diag = [False for _ in shape]
    if memory_save_mode is None:
        pass
    elif memory_save_mode == "one_diag":
        dim_diag[_max_idx(shape)] = True
    elif memory_save_mode == "smart_one_diag":
        sorted_shape = sorted(shape)
        if len(shape) >= 2 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[_max_idx(shape)] = True
    elif memory_save_mode == "one_triu":
        shape_ranks = np.argsort(np.argsort(shape))  # ranks
        dim_diag = (shape_ranks != 0).tolist()  # only triu the smallest
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
            "[None, 'one_diag', 'one_triu', 'all_diag', 'smart_one_diag']"
        )

    Q = []
    for size, dim_d in zip(shape, dim_diag):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            # use diagonal matrix as preconditioner for this dim
            q_dtype = promote(dtype)
            factor = _precond_scale_tensor(scale, grad.device, q_dtype)
            Q.append(factor * torch.ones(n, size, dtype=q_dtype, device=grad.device))
        else:
            # use triangular matrix as preconditioner for this dim
            factor = _precond_scale_tensor(scale, grad.device, dtype)
            eye = torch.eye(size, dtype=promote(dtype), device=grad.device).expand(n, size, size)
            Q.append((factor * eye).to(dtype).contiguous())
    return Q


@decorator_knowngood
def psgd_balance_Q(Q):
    log_norms = torch.stack([promote(q.abs().amax(dim=tuple(range(1, q.ndim)))) for q in Q]).log()
    scales = (log_norms.mean(dim=0) - log_norms).exp()
    scales = torch.where(torch.isfinite(scales).all(dim=0) & (scales > 0).all(dim=0), scales, 1)
    for q, scale in zip(Q, scales):
        shape = [1] * q.ndim
        shape[0] = -1
        copy_stochastic_(q, promote(q) * scale.view(shape))


@decorator_knowngood
def _lra_flatten_and_balance(U: list[Tensor], V: list[Tensor], d: list[Tensor]):
    u_norm = stable_l2_norm_list(U).double()
    v_norm = stable_l2_norm_list(V).double()
    scale = ((u_norm.log() - v_norm.log()) / 2).exp()
    scale = torch.where(torch.logical_and(torch.isfinite(scale), scale > 1e-6), scale, 1)
    stochastic_multiply_(U, [1 / scale] * len(U))
    stochastic_multiply_(V, [scale] * len(V))
    return _flatten_lra(U, V, d)


def _flatten_lra(U, V, d):
    return multi_flatten((U, 1), (V, 1), (d, 0))


def _lra_dtype(U, V, *xs):
    if (
        U.dtype == V.dtype
        and U.dtype in (torch.bfloat16, torch.float16)
        and all(not x.is_complex() and x.dtype != torch.float64 for x in xs)
    ):
        return U.dtype
    return functools.reduce(torch.promote_types, (promote(x.dtype) for x in (U, V, *xs)))


@decorator
def low_rank_mm(U: Tensor, V: Tensor, x: Tensor) -> Tensor:
    dtype = _lra_dtype(U, V, x)
    return x + compiled_einsum("br,gr,g->b", U.to(dtype), V.to(dtype), x.to(dtype)).to(x.dtype)


@decorator_knowngood
def _compilable_d_step(
    d: Tensor,
    d_orig: list[Tensor],
    invQtv: Tensor,
    vector: Tensor,
    inverse_precond_vector: Tensor,
    hessian_vector: Tensor,
    precond_hessian_vector: Tensor,
    eps: Tensor,
    step: Tensor,
):
    precond_hessian_vector = promote(precond_hessian_vector)
    hessian_vector = promote(hessian_vector)
    vector = promote(vector)
    inverse_precond_vector = promote(inverse_precond_vector)
    invQtv = promote(invQtv)
    inverse_precond_vector = invQtv - inverse_precond_vector

    nablaD = promote(d).square() * precond_hessian_vector * hessian_vector - vector * inverse_precond_vector

    a0 = promote(d) * precond_hessian_vector
    a1 = vector
    b0 = inverse_precond_vector / promote(d)
    b1 = hessian_vector

    score = torch.hypot(a0, a1).log2() + torch.hypot(b0, b1).log2()
    idx = score.flatten().argmax()
    a = a0.index_select(0, idx).double().square() + a1.index_select(0, idx).double().square()
    b = b0.index_select(0, idx).double().square() + b1.index_select(0, idx).double().square()
    divisor = (a * b).sqrt().clamp(min=eps)
    step = -step / divisor

    # fused update(s)
    apply_flat_add(d_orig, nablaD, step)


def update_lra_precond_(
    U: list[Tensor],
    V: list[Tensor],
    d: list[Tensor],
    vector: Tensor,
    hessian_vector: Tensor,
    eps: float,
    step: float,
    delayed: bool,
    precond_u: bool,
):
    """
    Adapted from https://github.com/lixilinx/psgd_torch/blob/6dbea94915679d08a289928e6431b6ce07931aaf/preconditioned_stochastic_gradient_descent.py#L657
    """
    U_orig, V_orig, d_orig = U, V, d

    U, V, d = _lra_flatten_and_balance(U, V, d)

    dtype = _lra_dtype(U, V, d, vector, hessian_vector)
    U, V, d, vector, hessian_vector = U.to(dtype), V.to(dtype), d.to(dtype), vector.to(dtype), hessian_vector.to(dtype)

    eps = scalar_guard(eps, vector)

    Qh = low_rank_mm(U, V, d * hessian_vector)
    Ph = low_rank_mm(V, U, Qh)
    rank = U.size(1)

    VtU = compiled_einsum("br,bn->rn", V, U)  # (rank, rank)
    I = torch.eye(rank, dtype=VtU.dtype, device=VtU.device)
    IpVtU = I + VtU
    invQtv = vector / d

    LU, pivots = torch.linalg.lu_factor(promote(IpVtU))

    solve_dtype = LU.dtype
    rhs = (U.T @ invQtv).view(-1, 1).to(solve_dtype)
    correction = torch.linalg.lu_solve(LU, pivots, rhs, adjoint=True).to(V.dtype)
    invQtv = invQtv - (V @ correction).flatten()
    rhs = (V.T @ invQtv).view(-1, 1).to(solve_dtype)
    solution = torch.linalg.lu_solve(LU, pivots, rhs).to(U.dtype)
    invPv = (U @ solution).flatten()

    eps, step = scalar_guard(eps, step, vector)
    _compilable_d_step(d, d_orig, invQtv, vector, invPv, hessian_vector, Ph, eps, step)

    a, b = Qh, invQtv

    precond = V if precond_u else U
    atV = compiled_einsum("b,br->r", a, precond)  # o == one
    btV = compiled_einsum("b,br->r", b, precond)
    atVVt = compiled_einsum("r,br->b", atV, precond)
    btVVt = compiled_einsum("r,br->b", btV, precond)
    divisor = stable_l2_norm(a) * stable_l2_norm(atVVt) + stable_l2_norm(b) * stable_l2_norm(btVVt)
    precond_step = step / divisor.clamp(min=eps)
    if precond_u:
        a = compiled_einsum("b,r,rg->bg", a, atV, IpVtU)
        b = compiled_einsum("b,r,rg->bg", b, btV, IpVtU)
    else:
        a = a + compiled_einsum("br,r->b", V, atV)
        b = b + compiled_einsum("br,r->b", V, btV)
        a = compiled_einsum("b,r->br", a, atV)
        b = compiled_einsum("b,r->br", b, btV)
    apply_flat_add(U_orig if precond_u else V_orig, b - a, precond_step)
    return (U, V, d) if delayed else _flatten_lra(U_orig, V_orig, d_orig)


def lra_precond(U: Tensor, V: Tensor, d: Tensor, g: Tensor):
    """
    As-is from https://github.com/lixilinx/psgd_torch/blob/6dbea94915679d08a289928e6431b6ce07931aaf/preconditioned_stochastic_gradient_descent.py#L744
    """
    d = promote(d)
    new_g = low_rank_mm(U, V, d * promote(g))
    return d * low_rank_mm(V, U, new_g)


@decorator_knowngood
def dampen_grad(g: Tensor, damp: float = 1e-9):
    g = promote(g)
    v = torch.randn_like(g)
    damping = damp + torch.finfo(g.dtype).eps * g.abs()
    return v, g + damping * v


@decorator_knowngood
def _compilable_lra_update_(
    params: list[Tensor],
    update: list[Tensor],
    U: Tensor,
    V: Tensor,
    d: Tensor,
    lr: Tensor,
    decay: Tensor,
    caution: bool,
    cautious_decay: bool,
    grads: list[Tensor],
):
    update = lra_precond(U, V, d, flatten(update))
    start = 0
    update = update.flatten()
    for p, g in zip(params, grads):
        size = p.numel()
        update_param_(p, update[start : start + size].view_as(p), lr, decay, caution, cautious_decay, g)
        start += size


def apply_lra_update(
    params: list[Tensor],
    update: list[Tensor],
    U: Tensor,
    V: Tensor,
    d: Tensor,
    lr: float,
    decay: float,
    caution: bool,
    grads: list[Tensor],
    cautious_decay: bool = False,
):
    params, grads = list_guard(params, grads)
    update = list_guard(update)
    if not params:
        return
    lr, decay = scalar_guard(lr, decay, params[0])
    _compilable_lra_update_(params, update, U, V, d, lr, decay, caution, cautious_decay, grads)


@decorator_knowngood
def apply_flat_update(params: list[Tensor], update: Tensor):
    start = 0
    update = update.flatten()
    for p in params:
        size = p.numel()
        copy_stochastic_(p, update[start : start + size].view_as(p))
        start += size


@decorator_knowngood
def zero_(x: list[Tensor]):
    for i in x:
        i.zero_()


@decorator_knowngood
def apply_flat_add(params: list[Tensor], update: Tensor, alpha: Tensor):
    start = 0
    update = update.flatten()
    for p in params:
        size = p.numel()
        stochastic_add_([p], [update[start : start + size].view_as(p)], alpha)
        start += size


@decorator_knowngood
def extract_from_flat_update(params: list[Tensor], update: Tensor):
    start = 0
    outputs = []
    update = update.flatten()
    for p in params:
        size = p.numel()
        outputs.append(update[start : start + size].view_as(p))
        start += size
    return outputs


@decorator_knowngood
def flatten(x: list[Tensor], remaining: int = 0) -> Tensor:
    last_dim = x[0].shape[-remaining:] if remaining else []
    tensors = [i.reshape(-1, *last_dim) for i in x if i.numel()]
    if not tensors:
        return x[0].new_empty((0, *last_dim))
    return torch.cat(tensors, 0)


@decorator_knowngood
def multi_flatten(*xs: tuple[list[Tensor], int]):
    return [flatten(x, i) for x, i in xs]


@decorator_knowngood
def dampen_multiple(g: list[Tensor], damp: float = 1e-9):
    vs = []
    gs = []
    for g_ in g:
        v, g = dampen_grad(g_, damp)
        vs.append(v)
        gs.append(g)
    return flatten(vs), flatten(gs)


def casted_einsum(expr: str, *args: Tensor) -> Tensor:
    dtype = functools.reduce(torch.promote_types, (promote(arg.dtype) for arg in args))
    return compiled_einsum(expr, *[a.to(dtype) for a in args])


@decorator_knowngood
def _psgd_calc_scalars_(Qs: list[Tensor], conjB: Tensor):
    triangular_qs = []
    conjB = promote(conjB)
    for i, q in enumerate(Qs):
        q = promote(q)
        if q.dim() <= 2:
            if conjB.ndim == 1:
                conjB = conjB / q
            else:
                shape = list(q.shape[:1]) + [1] * (conjB.ndim - 1)
                shape[i + 1] = -1
                conjB = conjB / q.view(shape)
        else:
            triangular_qs.append((i + 1, q))
    return triangular_qs, conjB


def ndim_tuple(Q: list[Tensor]) -> tuple:
    return tuple(q.ndim for q in Q)


def psgd_calc_A_and_conjB(G: Tensor, Q, conjB: Tensor | None):  # conjB ("V", "vector") == randn during hvp/whitening
    if conjB is None:
        conjB = torch.randn_like(G)
    expr_fn = cached_precond_grad_expr.__wrapped__ if is_compiling() else cached_precond_grad_expr
    exprA = expr_fn(ndim_tuple(Q), G.ndim)  # calcA expr and cached precond expr are the same
    A = casted_einsum(exprA, *promote(Q), promote(G)).contiguous()
    qs, conjB = _psgd_calc_scalars_(Q, conjB)
    n = G.shape[0]
    for i, tri_q in qs:
        conjB = conjB.movedim(i, -1).contiguous()
        moved_shape = conjB.shape
        dtype = torch.promote_types(tri_q.dtype, conjB.dtype)
        flat = conjB.reshape(n, -1, moved_shape[-1]).to(dtype)
        flat = no_compile_solve_triangular(tri_q.to(dtype), flat, upper=True, left=False)
        conjB = flat.reshape(moved_shape).movedim(-1, i).contiguous()
    return A, conjB


def _empty_spectral_value(A: Tensor) -> Tensor:
    return torch.zeros(A.shape[:-2] if A.ndim >= 2 else (), device=A.device, dtype=promote(A.real.dtype))


def max_singular_value_exact(A: Tensor) -> Tensor:
    if A.numel() == 0:
        return _empty_spectral_value(A)
    if A.ndim < 2:
        return promote(A).abs().max()
    return torch.linalg.svdvals(promote(A)).amax(dim=-1)


@decorator_no_fullgraph
def max_singular_value_power_iter(A: Tensor, iterations: int = 5):
    if A.numel() == 0:
        return _empty_spectral_value(A)
    if A.ndim < 2:
        return max_singular_value_exact(A)

    A = promote(A)
    scale = A.abs().amax(dim=(-2, -1))
    scaled = A / torch.where(scale != 0, scale, 1)[..., None, None]
    k = min(2, A.shape[-2])
    indices = scaled.norm(dim=-1).topk(k, dim=-1).indices
    x = stable_l2_normalize(scaled.gather(-2, indices[..., None].expand(*A.shape[:-2], k, A.shape[-1])), dim=-1)

    def mv(v):
        return (scaled.mH @ (scaled @ v.mT)).mT

    for _ in range(iterations):
        x = stable_l2_normalize(mv(x), dim=-1)
    estimate = (x.conj() * mv(x)).sum(dim=-1).real.clamp_min(0).sqrt().amax(dim=-1)
    return estimate * scale


@torch.compiler.disable
def max_singular_value_cholesky(A: Tensor, max_abs: Tensor | None = None):
    if A.numel() == 0:
        return _empty_spectral_value(A)
    if A.ndim < 2:
        return max_singular_value_exact(A)
    if max_abs is None:
        max_abs = A.abs().amax(dim=(-2, -1), keepdim=True)
    max_abs = promote(max_abs).clamp_min(torch.finfo(promote(A.real.dtype)).tiny)

    min_dim = min(A.shape[-2:])
    if min_dim <= 1:
        return (promote(A) / max_abs).norm(dim=(-2, -1)) * max_abs.squeeze(-1).squeeze(-1)
    k = min(min_dim, 2 ** math.ceil(math.log2(math.log2(min_dim))))
    scaled = promote(A) / max_abs
    indices = scaled.abs().square().sum(-2).topk(k, largest=True).indices
    Y = scaled.gather(-1, indices.unsqueeze(-2).expand(*A.shape[:-1], k))
    Q = torch.linalg.qr(Y, mode="reduced").Q
    Z = scaled.mH @ Q
    W = torch.linalg.qr(Z, mode="reduced").Q
    return max_singular_value_exact(Z.mH @ W) * max_abs.squeeze(-1).squeeze(-1)


@decorator_no_fullgraph
def max_singular_value(A: Tensor, max_svd: int = 0, use_cholesky: bool = False, power_iter: int = 16) -> Tensor:
    if A.numel() == 0:
        return _empty_spectral_value(A)
    if A.ndim < 2:
        return promote(A).abs().max()
    if min(A.shape[-2:]) <= max_svd:
        return max_singular_value_exact(A)
    if use_cholesky or power_iter < 0:
        return max_singular_value_cholesky(A)
    return max_singular_value_power_iter(A, iterations=power_iter)


@decorator_no_fullgraph
def max_eigenvalue_spd(A: Tensor, power_iter: int = 4) -> Tensor:
    return max_singular_value_power_iter(A, iterations=power_iter)


@decorator_knowngood
def clamped_max_singular_value(
    A: Tensor, min: float, max_svd: int = 0, use_cholesky: bool = False, power_iter: int = 16
) -> Tensor:
    return max_singular_value(A, max_svd, use_cholesky, power_iter).clamp_min(min)


@decorator_knowngood
def _balance_to_triu(Q: "TriuOrLine"):
    if isinstance(Q[0], tuple):
        psgd_balance_Q([o[1] for o in Q])
        return line_to_triu(Q)
    psgd_balance_Q(Q)
    return Q


@functools.lru_cache(maxsize=None)
def calcG_expr(q_dim, g_dim):
    exprs = []
    base = einsum_base[: g_dim - 1]
    for i, q in enumerate(q_dim):
        new = list(base)
        if q == 3:
            new[i] = "Z"
            out = f"{base[i]}Z"
        elif q == 2:
            out = base[i]
        else:
            out = ""
        exprs.append(f"...{base},...{''.join(new)}->...{out}")
    return exprs


def _update_lb(ell: Tensor, lb_state: Tensor, beta: Tensor) -> Tensor:
    ell = promote(ell)
    dtype = ell.dtype
    lower_bound = promote(lb_state)
    ell = ell.maximum(lower_bound + (ell - lower_bound) * (1 - beta))
    copy_stochastic_(lb_state, ell)
    return ell.to(dtype)


@decorator_no_fullgraph
def psgd_update_precond(
    G: Tensor,
    precond_lr: float,
    oq: "TriuOrLine",
    store_triu_as_line: bool,
    V: Tensor,
    running_lower_bound: list[Tensor],
    lower_bound_beta: float,
    power_iter: int,
) -> None:
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    Q = _balance_to_triu(oq)
    expr_fn = calcG_expr.__wrapped__ if is_compiling() else calcG_expr
    exprGs = expr_fn(ndim_tuple(Q), G.ndim)
    precond_lr, lower_bound_beta = scalar_guard(precond_lr, lower_bound_beta, G)

    A, conjB = psgd_calc_A_and_conjB(G, Q, V)
    del V

    for oq_i, q, exprG, lb_state in zip(oq, Q, exprGs, running_lower_bound):
        term1 = promote(compiled_einsum(exprG, A, A))
        term2 = promote(compiled_einsum(exprG, conjB, conjB))
        q_ = promote(q)
        dtype = functools.reduce(torch.promote_types, (term1.dtype, term2.dtype, q_.dtype))
        term1, term2, q_ = term1.to(dtype), term2.to(dtype), q_.to(dtype)

        if q.ndim < 3:
            sum_terms = term1 + term2
            reduced = sum_terms if q.ndim == 1 else sum_terms.amax(dim=-1)
            ell = _update_lb(reduced, lb_state, lower_bound_beta)
            update = q_ * (term1 - term2)
        else:
            ell = _update_lb(max_eigenvalue_spd(term1 + term2, power_iter), lb_state, lower_bound_beta)
            update = (term1 - term2).triu() @ q_
            if store_triu_as_line:
                update = triu_to_line([update])[0][1]

        real_oq = oq_i[1] if isinstance(oq_i, tuple) else oq_i
        ell = ell.view(-1, *([1] * (update.ndim - 1)))
        copy_stochastic_(real_oq, promote(real_oq) - update / ell * precond_lr)
    return None


@decorator_knowngood
def psgd_pro_update_precond(
    G: Tensor,
    precond_lr: float,
    Q: list[Tensor],
    running_lower_bound: list[Tensor],
    lower_bound_beta: float,
    power_iter: int,
    dampening: float,
    max_step_size: float = 1 / 8,
) -> None:
    """Update Kronecker product preconditioner Q with Q0.5EQ1.5 (PRO) method."""
    G = promote(G)
    psgd_balance_Q(Q)
    expr_fn = calcG_expr.__wrapped__ if is_compiling() else calcG_expr
    exprGs = expr_fn(ndim_tuple(Q), G.ndim)
    precond_lr, lower_bound_beta = scalar_guard(precond_lr, lower_bound_beta, G)

    damping = dampening + torch.finfo(G.dtype).eps * G.abs()
    Pg = psgd_precond_grad(G + damping * torch.randn_like(G), Q)

    total_numel = G.numel()
    for q, exprG, lb_state in zip(Q, exprGs, running_lower_bound):
        q_ = promote(q)
        covariance_PP = compiled_einsum(exprG, Pg, Pg)
        dtype = torch.promote_types(q_.dtype, covariance_PP.dtype)
        q_, covariance_PP = q_.to(dtype), covariance_PP.to(dtype)

        if q.ndim < 3:
            target_energy = total_numel / max(1, q.numel())
            reduced = covariance_PP if q.ndim == 1 else covariance_PP.amax(dim=-1)
            ell = _update_lb(reduced + target_energy, lb_state, lower_bound_beta)
            ell_b = ell if q.ndim == 1 else ell.unsqueeze(-1)
            copy_stochastic_(q, q_ - q_ * (covariance_PP - target_energy) / ell_b * precond_lr)
            continue

        target_energy = total_numel / (q.shape[0] * q.shape[-1])
        ell = max_eigenvalue_spd(covariance_PP, power_iter)
        ell = _update_lb(ell + target_energy, lb_state, lower_bound_beta)
        ell_b = ell.unsqueeze(-1).unsqueeze(-1)
        q_ = q_ - (covariance_PP @ q_ - target_energy * q_) / ell_b * precond_lr

        R = (q_.mT - q_).contiguous()
        r_scale = max_singular_value_power_iter(R, power_iter).clamp_min(torch.finfo(R.dtype).smallest_normal)
        R = R / r_scale.unsqueeze(-1).unsqueeze(-1)
        RQ = R @ q_
        RRQ = R @ RQ
        c1 = RQ.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        c2 = RRQ.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        a = torch.where(c2 < 0, (-c1 / c2).clamp(min=0, max=max_step_size), max_step_size)
        a_b = a.unsqueeze(-1).unsqueeze(-1)
        copy_stochastic_(q, q_ + a_b * RQ + (0.5 * a_b * a_b) * RRQ)


@decorator_knowngood
def _householder_vec_e1_to_v(v: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Return w such that H = I - 2 w w^T is orthogonal and H e1 = v (v unit).
    Applying from the right: G @ H = G - 2 (G @ w) w^T.
    If v is (numerically) e1, returns w=0 and H=I.
    """
    v = stable_l2_normalize(v, eps=eps)
    e1 = torch.zeros_like(v)
    e1[0] = 1.0
    w = e1 - v
    return torch.where(stable_l2_norm(w) >= eps, stable_l2_normalize(w), torch.zeros_like(w))


@decorator_knowngood
def eigvecs_product_rank1(G: Tensor, v: Tensor, w: Tensor | None = None, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """
    Compute Y = G @ V where V is an eigenvector matrix for P = λ I + σ v v^T,
    using the Householder reflector with first column v. Never materializes V.

    Args:
        G: shape (..., d) - gradient row(s) you want to rotate into eigenbasis.
        v: shape (d,)     - current unit direction (top eigenvector of P).
        w: optional Householder vector w; pass to reuse across calls.

    Returns:
        (Y, w) where:
          Y has shape (..., d) and equals G @ eigenvectors(P),
          w is the Householder vector you can cache & reuse.
    """
    if w is None:
        w = _householder_vec_e1_to_v(v, eps)
    dtype = torch.promote_types(promote(G).dtype, promote(w).dtype)
    G, w = promote(G).to(dtype), promote(w).to(dtype)
    Y = G - 2.0 * compiled_einsum("...i,i,j->...j", G, w, w)
    return Y, w


@decorator_knowngood
def oja_update(v: Tensor, g: Tensor, lr: float = 1e-2, eps: float = 1e-12) -> Tensor:
    """
    One Oja step to track the top eigendirection of the gradient covariance.
    v <- v + lr * ((g^T v) g - (g^T v)^2 v); then renormalize.
    """
    dtype = torch.promote_types(promote(v).dtype, promote(g).dtype)
    v, g = stable_l2_normalize(promote(v).to(dtype), eps=eps), promote(g).to(dtype)
    scale = g.abs().amax()
    safe = torch.where(scale != 0, scale, 1)
    g = g / safe
    gv = g @ v
    residual = g - (gv / (v @ v)) * v
    roundoff = 8 * torch.finfo(dtype).eps * torch.linalg.vector_norm(g)
    stationary = (gv.abs() <= roundoff) | (torch.linalg.vector_norm(residual) <= roundoff.clamp_min(eps))
    coefficient = lr * gv
    log_coefficient = coefficient.abs().log() + 2 * scale.log()
    sign = coefficient.sign()
    small = v + sign * log_coefficient.clamp_max(0).exp() * residual
    large = sign * residual + (-log_coefficient).clamp_max(0).exp() * v
    out = torch.where(log_coefficient > 0, large, small)
    return stable_l2_normalize(torch.where(stationary, v, out), eps=eps)


@decorator_knowngood
def _clip(x, clip_at, normalizer=1.0):
    x32 = promote(x)
    clip_at = torch.as_tensor(clip_at, device=x32.device, dtype=x32.real.dtype).clamp_min(0)
    scale, norm = stable_l2_components(x32)
    scaled_norm = norm / normalizer
    _apply_clip([x], [x32], scale, scaled_norm, clip_at)


@decorator_knowngood
def _clip_list(xs, clip_at, normalizer=1.0):
    values = [promote(x) for x in xs if x.numel()]
    if not values:
        return
    dtype = functools.reduce(torch.promote_types, (x.dtype for x in values))
    values = [x.to(dtype) for x in values]
    scale = torch.stack([x.abs().amax() for x in values]).amax()
    clip_at = torch.as_tensor(clip_at, device=scale.device, dtype=scale.dtype).clamp_min(0)
    safe = torch.where(scale != 0, scale, 1)
    norm = torch.stack([(x / safe).abs().square().sum() for x in values]).sum().sqrt()
    scaled_norm = norm / normalizer
    _apply_clip(xs, values, scale, scaled_norm, clip_at)


def _apply_clip(xs, values, scale, scaled_norm, clip_at):
    safe_scale = torch.where(scale != 0, scale, 1)
    safe_norm = torch.where(scaled_norm != 0, scaled_norm, 1)
    factor = (clip_at / safe_scale) / safe_norm
    direct = factor.abs() >= torch.finfo(factor.dtype).tiny
    for x, value in zip((x for x in xs if x.numel()), values):
        clipped = torch.where(direct, value * factor, value / safe_scale * (clip_at / safe_norm))
        copy_stochastic_(x, torch.where(scaled_norm > clip_at / safe_scale, clipped, value))


def _distributed_clip_list(xs, clip_at, rms, ref, dtype):
    values = [promote(x).to(dtype) for x in xs if x.numel()]
    zero = torch.zeros((), device=ref.device, dtype=torch.float64)
    scale = torch.stack([x.abs().amax() for x in values]).amax() if values else zero
    safe = torch.where(scale != 0, scale, 1)
    square_sum = torch.stack([(x / safe).abs().square().sum() for x in values]).sum() if values else zero
    local = torch.stack(
        (
            scale.double(),
            square_sum.double(),
            torch.as_tensor(sum(x.numel() for x in values), device=ref.device).double(),
        )
    )
    gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local)
    gathered = torch.stack(gathered)

    scale = gathered[:, 0].amax()
    scale_ratio = torch.where(scale != 0, gathered[:, 0] / scale, 0)
    square_sum = (gathered[:, 1] * scale_ratio.square()).sum()
    numel = gathered[:, 2].sum()
    normalizer = numel.sqrt() if rms else 1
    scaled_norm = square_sum.sqrt() / torch.where(numel > 0, normalizer, 1)
    real_dtype = values[0].real.dtype if values else torch.empty((), dtype=dtype).real.dtype
    scale = scale.to(real_dtype)
    scaled_norm = scaled_norm.to(real_dtype)
    threshold = torch.as_tensor(clip_at, device=ref.device, dtype=real_dtype).clamp_min(0)
    _apply_clip(xs, values, scale, scaled_norm, threshold)
    return numel


@decorator_knowngood
def _compilable_l2_clip_(xs, clip_at):
    for x in xs:
        _clip(x, clip_at)


@decorator_knowngood
def _compilable_normalize_(xs, eps, rms):
    for x in xs:
        if not x.numel():
            continue
        value = promote(x)
        scale, norm = stable_l2_components(value)
        safe = torch.where(scale != 0, scale, 1)
        scaled = value / safe
        if rms:
            norm = norm / math.sqrt(value.numel())
        denominator = torch.maximum(norm, eps / safe)
        copy_stochastic_(x, torch.where(denominator != 0, scaled / denominator, scaled))


def l2_normalization_(x, eps: float = 1e-8):
    x = list_guard(x)
    if not x:
        return x
    _compilable_normalize_(x, scalar_guard(eps, x[0]), False)
    return x


def l2_clip_(x, clip_at: float = 1.0):
    x = list_guard(x)
    if not x:
        return x
    clip_at = scalar_guard(clip_at, x[0])
    _compilable_l2_clip_(x, clip_at)
    return x


@decorator_knowngood
def _compilable_rmsnorm_clip_(xs, clip_at):
    for x in xs:
        if not x.numel():
            continue
        _clip(x, clip_at, math.sqrt(x.numel()))


def rmsnorm_clip_(x, clip_at: float = 1.0):
    x = list_guard(x)
    if not x:
        return x
    clip_at = scalar_guard(clip_at, x[0])
    _compilable_rmsnorm_clip_(x, clip_at)
    return x


@decorator_knowngood
def _compilable_global_rmsnorm_clip_(x, clip_at):
    numel = sum([i.numel() for i in x])
    if not numel:
        return
    _clip_list(x, clip_at, math.sqrt(numel))


def global_rmsnorm_clip(x, clip_at: float = 1.0):
    x = list_guard(x)
    if not x:
        return x
    _compilable_global_rmsnorm_clip_(x, clip_at)
    return x


@decorator_knowngood
def _compilable_global_l2norm_clip_(x, clip_at):
    _clip_list(x, clip_at)


def global_l2norm_clip(x, clip_at: float = 1.0):
    x = list_guard(x)
    if not x:
        return x
    _compilable_global_l2norm_clip_(x, clip_at)
    return x


def rmsnorm_normalize_(x, eps: float = 1e-6):
    x = list_guard(x)
    if not x:
        return x
    _compilable_normalize_(x, scalar_guard(eps, x[0]), True)
    return x


@decorator_knowngood
def _compilable_mu_law_compress_(x, mu):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """

    for x_ in x:
        xa = (promote(x_).abs() * mu).log1p() / torch.log1p(mu)
        xa = xa.copysign(x_)
        copy_stochastic_(x_, xa)


def mu_law_compress(x, mu=127.0):
    """
    μ-law compression
    Args:
        x: Input tensor
        mu: Compression parameter (default 127.0 for behavior similar to trust_region=1.5)
    """
    x = list_guard(x)
    if not x:
        return x
    mu = scalar_guard(mu, x[0])
    _compilable_mu_law_compress_(x, mu)
    return x


@decorator_knowngood
def _compilable_a_law_compress_(x, A):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """
    for x_ in x:
        xa = promote(x_).abs() * A
        xa = torch.where(xa < 1, xa, 1 + xa.log())
        xa = xa.copysign(x_)
        xa = xa * (1 / (1 + torch.log(A)))
        copy_stochastic_(x_, xa)


def a_law_compress(x, A=87.6):
    """
    A-law compression
    Args:
        x: Input tensor
        A: Compression parameter (default 87.6 - European PCM standard)
    :param x:
    :param A:
    :return:
    """
    x = list_guard(x)
    if not x:
        return x
    A = scalar_guard(A, x[0])
    _compilable_a_law_compress_(x, A)
    return x


@decorator_knowngood
def _compilable_softsign_compress_(x):
    for x_ in x:
        value = promote(x_)
        copy_stochastic_(x_, 2.0 * (value / (1.0 + value.abs())))


def softsign_compress(x):
    x = list_guard(x)
    if not x:
        return x
    _compilable_softsign_compress_(x)
    return x


_NUM_MANTISSA_BITS = {torch.float16: 10, torch.bfloat16: 7}
_EXPONENT_BIAS = {torch.float16: 15, torch.bfloat16: 127}


def _log_ulp(x):
    m = _NUM_MANTISSA_BITS[x.dtype]
    bias = _EXPONENT_BIAS[x.dtype]
    exp = (x.view(torch.int16) & 0x7FFF) >> m
    return torch.where(
        exp == 0, torch.tensor(1 - bias - m, device=x.device, dtype=torch.int32), exp.to(torch.int32) - (bias + m)
    )


def _scale_by_exp2(x, log_scale):
    # Split to avoid intermediate overflow when log_scale is large
    h = (log_scale / 2.0).floor()
    return (x * torch.exp2(h)) * torch.exp2(log_scale - h)


@decorator_knowngood
def _compilable_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    ema32 = _lerp(ema, p, ema_decay)
    _lerp(p, ema32, 1 - weight_decay)


def weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    if not p:
        return
    ema_decay, weight_decay = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_weight_decay_to_ema_(p, ema, ema_decay, weight_decay)


@decorator_knowngood
def _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    weight_decay = torch.as_tensor(weight_decay, device=p[0].device, dtype=promote(p[0]).dtype).clamp_min(0)
    ema32 = _lerp(ema, p, ema_decay)
    for p_, e_ in zip(p, ema32):
        delta = promote(p_) - e_
        copy_stochastic_(p_, e_ + delta.sign() * (delta.abs() - weight_decay).clamp_min(0))


def l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    if not p:
        return
    ema_decay, weight_decay = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay)


@decorator_knowngood
def _compilable_sign_(grad: list[Tensor], graft: bool):
    for g_ in grad:
        gs = g_.sign()
        if graft:
            gs = _compilable_grafting(g_, gs)
        copy_stochastic_(g_, gs)


def sign_(grad: list[Tensor], graft: bool = True):
    grad = list_guard(grad)
    _compilable_sign_(grad, graft)
    return grad


@decorator_knowngood
def _compilable_trust_region_clip_(grad, lerp, scale):
    for x_ in grad:
        x = promote(x_)
        magnitude = x.abs()
        ratio = magnitude / scale
        nonzero = ratio != 0
        log_small = magnitude * torch.where(nonzero, ratio.log1p() / ratio, 1)
        tanh_small = magnitude * torch.where(nonzero, ratio.tanh() / ratio, 1)
        log_large = scale * F.softplus(magnitude.log() - scale.log())
        tanh_large = scale * ratio.tanh()
        small = magnitude <= scale
        log_term = torch.where(small, log_small, log_large)
        tanh_term = torch.where(small, tanh_small, tanh_large)
        out = (log_term * (1 - lerp) + tanh_term * lerp).copysign(x).clamp(min=-2, max=2)
        copy_stochastic_(x_, out)


def trust_region_clip_(grad, lerp=0.9, scale=1.5):
    grad = list_guard(grad)
    if not grad:
        return grad
    lerp, scale = scalar_guard(lerp, scale, grad[0])
    _compilable_trust_region_clip_(grad, lerp, scale)
    return grad


@decorator
def triu_to_line(Q_list: list[Tensor]):
    out = []
    for q in Q_list:
        if q.dim() < 3:
            out.append((None, q))
        else:
            rows, cols = torch.triu_indices(q.shape[-2], q.shape[-1], device=q.device)
            out.append((tuple(q.shape), q[..., rows, cols]))
    return out


@decorator_knowngood
def line_to_triu(Q_list: list[tuple[list[int] | None, Tensor]]):
    new = []
    for shape, q in Q_list:
        if shape is not None:
            d0, d1 = shape[-2], shape[-1]
            rows, cols = torch.triu_indices(d0, d1, device=q.device)
            full_shape = q.shape[:-1] + (d0, d1)
            q_mat = torch.zeros(full_shape, device=q.device, dtype=q.dtype)
            q_mat[..., rows, cols] = q
            q = q_mat
        new.append(q)
    return new


_warned = set()


def warn_once(msg):
    if msg not in _warned:
        warnings.warn(msg)
        _warned.add(msg)


@functools.lru_cache(maxsize=None)
def cached_precond_grad_expr(Q_dim, grad_dim):
    expr = [f"...{c.upper()}{c}" if q_ == 3 else f"...{c}" if q_ == 2 else "..." for c, q_ in zip(einsum_base, Q_dim)]
    expr = ",".join(expr)
    grad_expr = "".join(c for c, _ in zip(einsum_base, range(grad_dim - 1)))
    out_expr = "".join(c.upper() if c.upper() in expr else c for c in grad_expr)
    return f"{expr},...{grad_expr}->...{out_expr}"


@decorator_knowngood
def precond_grad_cached_(
    ea: Tensor,
    cached_q: list[Tensor],
    caution: bool = False,
    grad: Tensor | None = None,
):
    args = [promote(q) for q in cached_q]
    args = args + [promote(ea)]
    expr_fn = cached_precond_grad_expr.__wrapped__ if is_compiling() else cached_precond_grad_expr
    expr = expr_fn(ndim_tuple(cached_q), ea.ndim)
    out = casted_einsum(expr, *args)
    return _compilable_cautioning(grad, out) if caution else out


TriuOrLine = list[Tensor] | list[tuple[list[int] | None, Tensor]]


@decorator_knowngood
def _compilable_fused_precond_grad_cached_(
    ea: Tensor, param, lr, grad, decay, caution, cautious_decay, cached_q: list[Tensor]
):
    precond = precond_grad_cached_(ea, cached_q, caution=caution, grad=grad)
    update_param_(param, precond, lr, decay, caution=False, cautious_decay=cautious_decay)


def fused_precond_grad_cached_(
    ea: Tensor, param, lr, grad, decay, caution, cached_q: list[Tensor], cautious_decay: bool = False
):
    lr, decay = scalar_guard(lr, decay, param[0])
    _compilable_fused_precond_grad_cached_(ea, param, lr, grad, decay, caution, cautious_decay, cached_q)


@functools.lru_cache(maxsize=None)
def precond_grad_expr(Q_dim, grad_dim):
    expr = [
        f"...{c2}{c.upper()},...{c2}{c}" if q_ == 3 else f"...{c},...{c}" if q_ == 2 else "...,..."
        for c, c2, q_ in zip(einsum_base, einsum_base[13:], Q_dim)
    ]
    expr = ",".join(expr)
    grad_expr = "".join(c for c, _ in zip(einsum_base, range(grad_dim - 1)))
    out_expr = "".join(c.upper() if c.upper() in expr else c for c in grad_expr)
    return f"{expr},...{grad_expr}->...{out_expr}"


@decorator_knowngood
def psgd_precond_grad(
    ea: Tensor,
    preconds: TriuOrLine,
    caution: bool = False,
    grad: Tensor | None = None,
    store_triu_as_line: bool = False,
    sqrt: bool = False,
):
    """``sqrt`` applies each Kronecker factor once (Q·ea, the square-root factor of P = QᵀQ that
    ``calcA`` forms while fitting Q) instead of twice (the full P·ea); see ``QSGD``."""
    if store_triu_as_line:
        preconds = line_to_triu(preconds)
    args = [promote(q) for q in preconds]
    if sqrt:
        expr_fn = cached_precond_grad_expr.__wrapped__ if is_compiling() else cached_precond_grad_expr
        out = casted_einsum(expr_fn(ndim_tuple(args), ea.ndim), *args, promote(ea))
    else:
        expr_fn = precond_grad_expr.__wrapped__ if is_compiling() else precond_grad_expr
        expr = expr_fn(ndim_tuple(args), ea.ndim)
        out = casted_einsum(expr, *[a for a in args for _ in (0, 1)], promote(ea))
    return _compilable_cautioning(grad, out) if caution else out


@decorator_knowngood
def _compilable_fused_psgd_precond_grad(
    ea: Tensor,
    param,
    lr,
    grad,
    decay,
    caution,
    cautious_decay,
    preconds: TriuOrLine,
    store_triu_as_line: bool = False,
    sqrt: bool = False,
):
    precond = psgd_precond_grad(
        ea, preconds, caution=caution, grad=grad, store_triu_as_line=store_triu_as_line, sqrt=sqrt
    )
    update_param_(param, precond, lr, decay, caution=False, cautious_decay=cautious_decay)


def fused_psgd_precond_grad(
    ea: Tensor,
    param,
    lr,
    grad,
    decay,
    caution,
    preconds: TriuOrLine,
    store_triu_as_line: bool = False,
    cautious_decay: bool = False,
    sqrt: bool = False,
):
    lr, decay = scalar_guard(lr, decay, param[0])
    _compilable_fused_psgd_precond_grad(
        ea, param, lr, grad, decay, caution, cautious_decay, preconds, store_triu_as_line, sqrt
    )


@decorator_knowngood
def _compilable_mars_correction_(g: Tensor, old_g: Tensor, a: Tensor):
    out = [promote(g_) * (1 - a) + promote(old_) * a for g_, old_ in zip(g, old_g)]
    copy_stochastic_list_(old_g, g)
    return out


def mars_correction(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g, old_g)
    if not g:
        return g
    a = scalar_guard(a, g[0])
    return _compilable_mars_correction_(g, old_g, a)


@decorator_knowngood
def _compilable_orthogonalization(weight: list[Tensor], grad: list[Tensor], eps: Tensor, graft: bool = True):
    """
    Implements OrthoGrad from "Grokking at the Edge of Numerical Stability" (https://arxiv.org/abs/2501.04697)
    """

    outputs = []
    for w, g in zip(weight, grad):
        if not g.numel():
            outputs.append(promote(g))
            continue
        w, g32 = promote(w), promote(g)
        w_scale = w.abs().amax()
        g_scale = g32.abs().amax()
        w_scale = torch.where(w_scale != 0, w_scale, 1)
        g_scale = torch.where(g_scale != 0, g_scale, 1)
        w_scaled, g_scaled = w / w_scale, g32 / g_scale
        denominator = w_scaled.square().sum() + promote(eps) / w_scale / w_scale
        numerator = (w_scaled * g_scaled).sum()
        proj = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(denominator))
        out = (g_scaled - proj * w_scaled) * g_scale

        if graft:
            out = _compilable_grafting(g32, out)
        copy_stochastic_(g, out)
        outputs.append(out)
    return outputs


def orthogonalize_grad_to_param(weight, grad, eps, graft=True):
    weight, grad = list_guard(weight, grad)
    if not weight:
        return grad
    eps = scalar_guard(eps, weight[0])
    return _compilable_orthogonalization(weight, grad, eps, graft)


@decorator_knowngood
def _compilable_caution_scaled(g: Tensor, update: Tensor):
    aligned = _strictly_aligned(g, update)
    update = update.masked_fill(~aligned, 0)
    scale = aligned.numel() / aligned.sum().clamp(min=1)
    update.mul_(scale)
    return update


_compilable_cautioning = _compilable_caution_scaled


def caution(g, update):
    return _compilable_cautioning(g, update)


@decorator_knowngood
def _compilable_hyperball_(
    p: list[Tensor],
    u: list[Tensor],
    init_norm: list[Tensor],
    lr: Tensor,
    decay: float,
    caution: bool,
    cautious_decay: bool,
    g: list[Tensor],
):
    for op, u_, n_, g_ in zip(p, u, init_norm, g):
        u_ = promote(u_.view_as(op))
        p_ = promote(op)
        if isinstance(decay, Tensor) or decay != 0:
            u_ = _add_weight_decay(u_, p_, decay, cautious_decay)
        if caution:
            u_ = _compilable_cautioning(g_, u_)
        if n_.numel() == 2:
            norm_scale, scaled_norm = n_.unbind()
        else:
            norm_scale, scaled_norm = n_, torch.ones_like(n_)
        p_ = p_ - stable_l2_normalize(u_, eps=torch.finfo(u_.dtype).tiny) * (lr * scaled_norm) * norm_scale
        p_ = stable_l2_normalize(p_, eps=torch.finfo(p_.dtype).tiny) * scaled_norm * norm_scale
        copy_stochastic_(op, p_)


def hyperball_step_(param, update, init_norm, lr, decay, caution, grad, cautious_decay=False):
    param, update, init_norm, grad = list_guard(param, update, init_norm, grad)
    if not param:
        return
    lr = scalar_guard(lr, param[0])
    if not caution:
        grad = [None] * len(param)
    _compilable_hyperball_(param, update, init_norm, lr, decay, caution, cautious_decay, grad)


def _inner_precond_update_prob_schedule(
    n: int, max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.999, flat_start: float = 1000
):
    return max(min_prob, max_prob * decay ** max(n - flat_start, 0))


def precond_update_prob_schedule(
    max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.999, flat_start: float = 1000
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at `max_prob` for 1000 steps then exponentially anneal down to
    `min_prob` by ~4000 steps. Default settings work very well for most models and
    training regimes.
    """
    return functools.partial(
        _inner_precond_update_prob_schedule, max_prob=max_prob, min_prob=min_prob, decay=decay, flat_start=flat_start
    )


def merge_group(group, *tensors):
    if not group.get("merge_dims", False):
        return tensors
    if isinstance(tensors[0], list):
        return [merge_group(group, *t) for t in tensors]

    out = []
    for t in tensors:
        append_or_extend(
            out,
            dim_merger(
                t,
                group["max_size_triangular"] if "max_size_triangular" in group else group["max_precond_dim"],
                group.get("split", False),
            ),
        )
    return out


@decorator_knowngood
def _compilable_d_adapt_(grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor]):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        if not s.numel():
            copy_stochastic_(u_, u * d)
            continue
        next_s = s + u * d
        scale = torch.maximum(s.abs().amax(), next_s.abs().amax())
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        s_scaled, next_s_scaled = s / scale, next_s / scale
        denominator = next_s_scaled.abs().sum().clamp_min(torch.finfo(s.dtype).tiny)
        next_d = d * (g * (s_scaled / denominator)).sum()
        next_d = torch.maximum(next_d, d)
        copy_stochastic_(u_, u * d)
        copy_stochastic_(d_, next_d)
        copy_stochastic_(s_, next_s)


def d_adaptation(grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor]):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    if not grads:
        return
    _compilable_d_adapt_(grads, update, state, delta)


@decorator_knowngood
def _compilable_lr_adapt_(
    grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor], lr_lr: Tensor
):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        if not s.numel():
            copy_stochastic_(u_, u * d.sigmoid())
            copy_stochastic_(s_, u)
            continue
        lr_grad = d.sigmoid() * (-d).sigmoid()
        lr_grad = (s * (g * lr_grad)).mean()
        d = d - lr_grad * lr_lr
        copy_stochastic_(d_, d)
        copy_stochastic_(u_, u * d.sigmoid())
        copy_stochastic_(s_, u)


def lr_adaptation(grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor], lr_lr: float):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    if not grads:
        return
    lr_lr = scalar_guard(lr_lr, grads[0])
    _compilable_lr_adapt_(grads, update, state, delta, lr_lr)


@decorator_knowngood
def _compilable_pointwise_lr_adapt_(
    grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor], lr_lr: Tensor
):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        lr_grad = d.sigmoid() * (-d).sigmoid()
        lr_grad = lr_grad * s * g
        d = d - lr_grad * lr_lr
        copy_stochastic_(d_, d)
        copy_stochastic_(u_, u * d.sigmoid())
        copy_stochastic_(s_, u)


def pointwise_lr_adaptation(
    grads: list[Tensor], update: list[Tensor], state: list[Tensor], delta: list[Tensor], lr_lr: float
):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    if not grads:
        return
    lr_lr = scalar_guard(lr_lr, grads[0])
    _compilable_pointwise_lr_adapt_(grads, update, state, delta, lr_lr)


def hook_optimizer_into_model(model, optimizer, *args, **kwargs):
    optimizers = {}

    def _step(p: Tensor):
        o = optimizers[p]
        o.step()
        o.zero_grad()

    for p in model.parameters():
        if not p.requires_grad:
            continue
        o = optimizers[p] = optimizer([p], *args, **kwargs)
        if o.hessian_approx:
            raise ValueError("Optimizer hooks cannot use exact Hessian-vector products because they have no closure")
        p.register_post_accumulate_grad_hook(_step)

    return optimizers


def fused_hook(parameters, optimizer, *args, **kwargs):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")

    o = optimizer(parameters, *args, **kwargs)
    if o.hessian_approx:
        raise ValueError("Optimizer hooks cannot use exact Hessian-vector products because they have no closure")
    step_fn = o.step
    o.step = functools.partial(
        warn_once, msg="You're trying to call `step` on a fused optimizer. This will not do anything."
    )

    queued_task = None

    def _step(p: Tensor):
        nonlocal queued_task
        task = torch._C._current_graph_task_id()
        if queued_task == task:
            return
        queued_task = task

        def _step_all():
            nonlocal queued_task
            try:
                step_fn()
                o.zero_grad()
            finally:
                if queued_task == task:
                    queued_task = None

        torch.autograd.Variable._execution_engine.queue_callback(_step_all)

    o._fused_hook_handles = [p.register_post_accumulate_grad_hook(_step) for p in parameters]

    return o


@decorator_knowngood
def _compilable_caution_no_scale(g: Tensor, update: Tensor):
    return update.masked_fill(~_strictly_aligned(g, update), 0)


def disable_caution_scaling():
    global _compilable_cautioning
    _compilable_cautioning = _compilable_caution_no_scale


@decorator_knowngood
def _compilable_sam_step(active: list[Tensor], ball_size: Tensor, adaptive: bool):
    if adaptive:
        log_products = [
            promote(p.grad).to(ball_size).abs().log() + promote(p).to(ball_size).abs().log() for p in active
        ]
        log_norm_sq = [torch.logsumexp(2 * value.flatten(), dim=0) for value in log_products]
        log_norm = 0.5 * functools.reduce(torch.logaddexp, log_norm_sq)
        valid = torch.isfinite(log_norm) & (ball_size != 0)
        for p, log_product in zip(active, log_products):
            p_ = promote(p).to(ball_size)
            delta = promote(p.grad).to(ball_size).sign() * torch.exp(
                ball_size.abs().log() + log_product + p_.abs().log() - log_norm
            )
            stochastic_add_(p.data, torch.where(valid, delta, torch.zeros_like(delta)))
        for p in active:
            p.grad.zero_()
        return

    grad_scale = torch.stack([promote(p.grad).abs().amax().to(ball_size) for p in active]).amax()
    grad_scale = torch.where(grad_scale == 0, torch.ones_like(grad_scale), grad_scale)

    adjusted = []
    norms = []
    for p in active:
        grad = promote(p.grad).to(ball_size) / grad_scale
        adjusted.append(grad)
        scale = grad.abs().amax().to(ball_size)
        denom = torch.where(scale == 0, torch.ones_like(scale), scale)
        norms.append(torch.linalg.vector_norm(grad / denom) * scale)

    norm = functools.reduce(torch.hypot, norms)
    scale = ball_size / torch.where(norm != 0, norm, torch.ones_like(norm))
    scale = torch.where(norm != 0, scale, torch.zeros_like(scale))
    for p, grad in zip(active, adjusted):
        stochastic_add_(p.data, grad, scale)
    for p in active:
        p.grad.zero_()


def sam_step(parameters, ball_size, adaptive: bool = True):
    parameters = list(parameters)
    active = [p for p in parameters if hasattr_none(p, "grad") and p.grad.numel()]
    old_params = [p.detach().clone() for p in parameters]
    if not active:
        return old_params
    dtype = functools.reduce(torch.promote_types, (promote(p.grad).dtype for p in active))
    ball_size = torch.as_tensor(ball_size, device=active[0].device, dtype=dtype)
    _compilable_sam_step(active, ball_size, adaptive)
    return old_params
