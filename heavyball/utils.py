import functools
import gc
import math
import random
import string
import warnings
from typing import Callable, List, Optional, Tuple, Union
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor
from torch._dynamo import config
from torch._dynamo.exc import TorchDynamoException
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

config.cache_size_limit = 2**16

compile_mode = "max-autotune-no-cudagraphs"
dynamic = False
compile_mode_recommended_to_none = None
zeroth_power_mode = "qr"  # 'qr' is baseline, 'newtonschulz' converges better and faster
tiny_bf16 = torch.finfo(torch.bfloat16).tiny


def decorator(func):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or compile_mode_recommended_to_none is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=True, dynamic=dynamic, mode=compile_mode_recommended_to_none)(func)
        return compiled(*args, **kwargs)

    return _fn


def decorator_knowngood(func: Callable):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or compile_mode is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=True, dynamic=dynamic, mode=compile_mode)(func)
        return compiled(*args, **kwargs)

    return _fn


einsum_base = string.ascii_lowercase


def is_compiling():
    try:
        return torch.compiler.is_compiling()
    except TorchDynamoException:
        return True


def set_(dst: Tensor, src: Tensor):
    dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


def _ignore_warning(msg):
    warnings.filterwarnings("ignore", f".*{msg}.*")


def set_torch(benchmark_limit: int = 32, einsum_strategy: str = "auto"):
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.benchmark_limit = benchmark_limit
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
    opt_einsum.set_flags(True, einsum_strategy)

    # Torch calls these for 2nd-order optimization in HeavyBall, but they are explicitly handled.
    _ignore_warning(
        "Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak"
    )
    _ignore_warning(
        "We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak"
    )


def tree_apply(fn):
    def _fn(*args):
        return tree_map(fn, *args)

    return _fn


@tree_apply
def promote(x):
    if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16):
        return torch.float32
    if isinstance(x, Tensor) and x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


# --- Merged Helper Functions (Incorporating fixes/additions from File 2) ---


def list_guard(*xs):
    # Merged: Added None handling from File 2
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.append(x)
        elif x is None:  # Handle None gracefully
            out.append(None)
        else:
            out.append([x])
    if len(xs) == 1:
        return out[0]
    return out


def scalar_guard(*args):
    # Merged: Added None handling from File 2
    *xs, ref = args
    out = []
    for x in xs:
        if x is None:  # Handle None
            out.append(None)
        elif isinstance(x, float):
            # Ensure ref is a Tensor before accessing dtype/device
            if not isinstance(ref, Tensor):
                raise TypeError(f"Reference object must be a Tensor for scalar_guard, got {type(ref)}")
            out.append(torch.empty((), dtype=promote(ref.dtype), device=ref.device).fill_(x))
        elif isinstance(x, int):
            if not isinstance(ref, Tensor):
                raise TypeError(f"Reference object must be a Tensor for scalar_guard, got {type(ref)}")
            out.append(torch.empty((), dtype=torch.int64, device=ref.device).fill_(x))
        else:
            out.append(x)
    if len(xs) == 1:
        return out[0]
    return out


def min_dtype(xs: List[Tensor]):
    # Merged: Added None check from File 2
    dtypes = [x.dtype for x in xs if x is not None]  # Check for None
    if not dtypes:  # Handle case where all inputs are None or list is empty
        return torch.float32
    for d in (torch.float32, torch.bfloat16, torch.float16):
        # Ensure all actual dtypes are comparable
        valid_dtypes = {d, torch.float32, torch.float64}
        if all(x_dtype in valid_dtypes for x_dtype in dtypes):
            # Check if all dtypes are either the target 'd' or higher precision
            if all(x_dtype in (d, torch.float32, torch.float64) for x_dtype in dtypes):
                return d
    return torch.float32


def get_beta1(group):
    # Merged: Added 'momentum' check from File 2
    beta = None
    if "beta" in group:
        beta = group["beta"]
    if beta is None and "momentum" in group:  # Added check from File 2
        beta = group["momentum"]
    if beta is None and "betas" in group:
        beta = group["betas"][0]
    if beta is None:
        raise ValueError("Momentum/Beta1 ('beta', 'momentum', or 'betas'[0]) not found in group.")
    return beta


def get_beta2(group):
    # Kept File 1's version - File 2 had no significant functional change here.
    if "palm" in group and group["palm"] is True and "beta2_scale" in group:
        step = max(group.get("step", 1), 1)
        return 1 - step ** -group["beta2_scale"]
    if "betas" in group:
        return group["betas"][1]
    raise ValueError("Beta2 ('betas'[1] or 'beta2_scale' with 'palm') not found in group.")


def beta_debias(beta, step):
    # Added safeguard for step=0 or beta=1 to avoid NaN/ZeroDivisionError
    if step <= 0:
        return 0.0  # Or handle as appropriate, 0 might make sense for debias at step 0
    if beta == 1.0:
        return 1.0  # Avoid 0/0
    beta_pow_step = beta**step
    if beta_pow_step == 1.0:
        return 1.0  # Avoid division by zero if beta**step underflows/overflows to 1
    return 1 - (1 - beta) / (1 - beta_pow_step)


def eps_sqrt(item, eps):
    # Ensure eps is non-negative
    eps_val = eps.item() if isinstance(eps, Tensor) else eps
    if eps_val < 0:
        raise ValueError(f"eps must be non-negative, got {eps_val}")
    # Clamp after sqrt for numerical stability if item is slightly negative due to precision
    return item.clamp(min=0.0).sqrt().clamp(min=eps)


# --- Stochastic Operations (From File 1, using updated list/scalar guards) ---


@decorator_knowngood
def _compilable_stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        if x32.dtype != y32.dtype:
            # Ensure compatibility before operation
            if torch.is_complex(x32) or torch.is_complex(y32):
                # Handle complex types if necessary, or raise error
                raise TypeError(
                    "Complex types not directly supported in _compilable_stochastic_lerp_ without explicit handling."
                )
            common_dtype = torch.promote_types(x32.dtype, y32.dtype)
            x32 = x32.to(common_dtype)
            y32 = y32.to(common_dtype)
            # Fallback to float32 if promotion leads to unexpected types?
            # Or trust promote_types for now.
        # Use the promoted dtype for calculation consistency
        res = x32 * (1 - a.to(x32.dtype)) + y32 * a.to(x32.dtype)
        copy_stochastic_(x_, res)


def stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    x, y = list_guard(x, y)
    if not x:
        return  # Handle empty lists
    a = scalar_guard(a, x[0])
    _compilable_stochastic_lerp_(x, y, a)


@decorator_knowngood
def _compilable_stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        # Promote alpha to the computation dtype
        alpha_promoted = alpha.to(promote(x_.dtype))
        # Ensure compatible types for addition
        if x32.dtype != y32.dtype:
            common_dtype = torch.promote_types(x32.dtype, y32.dtype)
            x32 = x32.to(common_dtype)
            y32 = y32.to(common_dtype)
            alpha_promoted = alpha_promoted.to(common_dtype)  # Ensure alpha matches
        res = x32 + y32 * alpha_promoted
        copy_stochastic_(x_, res)


def stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor] = 1):
    x, y = list_guard(x, y)
    if not x:
        return  # Handle empty lists
    alpha = scalar_guard(alpha, x[0])
    _compilable_stochastic_add_(x, y, alpha)


@decorator_knowngood
def _compilable_stochastic_multiply_(x: List[Tensor], y: List[Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        # Ensure compatible types for multiplication
        if x32.dtype != y32.dtype:
            common_dtype = torch.promote_types(x32.dtype, y32.dtype)
            x32 = x32.to(common_dtype)
            y32 = y32.to(common_dtype)
        copy_stochastic_(x_, x32 * y32)


def stochastic_multiply_(x: List[Tensor], y: List[Tensor]):
    x, y = list_guard(x, y)
    if not x:
        return  # Handle empty lists
    _compilable_stochastic_multiply_(x, y)


@decorator_knowngood
def stochastic_round_(ref: Tensor, source: Tensor):
    if source.dtype == torch.bfloat16 or ref.dtype == source.dtype:
        return source.to(ref.dtype)  # Ensure output matches ref dtype even if source is bf16

    # Ensure source is float32 for the integer manipulation
    source_f32 = source.float()

    if ref.dtype == torch.bfloat16:
        # Check for NaN/Inf before conversion
        if torch.isnan(source_f32).any() or torch.isinf(source_f32).any():
            # Handle NaN/Inf appropriately, e.g., clamp or return bf16 representation
            # Simple approach: convert directly, letting bf16 handle NaN/Inf
            return source_f32.bfloat16()

        # Proceed with stochastic rounding for bf16
        # Use torch.rand_like for better randomness properties if possible
        # Ensure low/high are within int32 range
        rand_int = torch.randint_like(source_f32, dtype=torch.int32, low=0, high=(1 << 16))  # 65536

        # View source as int32
        source_int32 = source_f32.view(dtype=torch.int32)

        # Add random integer
        result_int32 = source_int32 + rand_int

        # Bitwise AND to clear the lower 16 bits (mantissa for bf16)
        # -65536 is 0xFFFF0000 in signed 32-bit two's complement
        result_int32.bitwise_and_(-65536)

        # View back as float32 and then convert to bfloat16
        return result_int32.view(dtype=torch.float32).bfloat16()

    elif ref.dtype == torch.float16:
        # Similar stochastic rounding for float16 if needed (clearing lower 10 bits of mantissa + 1 sign + 5 exp = 16 bits?)
        # FP16: 1 sign, 5 exponent, 10 mantissa. Need to clear lower 10 bits.
        # Mask is 0xFFFFF800? No, the bits aren't aligned like bf16->fp32.
        # Direct conversion might be preferred unless specific stochastic FP16 is implemented.
        # For now, just convert, matching original behavior for non-bf16.
        return source.to(ref.dtype)

    else:  # Default: standard conversion for fp32, fp64 etc.
        return source.to(ref.dtype)


@decorator_knowngood
def _compilable_copy_stochastic_(target: Tensor, source: Tensor):
    # Use the potentially enhanced stochastic_round_
    stochastic_result = stochastic_round_(target, source)
    # Ensure the result is compatible before copy_
    if target.shape != stochastic_result.shape:
        # This shouldn't happen if stochastic_round_ preserves shape
        raise RuntimeError(
            f"Shape mismatch in copy_stochastic: target {target.shape}, source {source.shape}, result {stochastic_result.shape}"
        )
    if target.dtype != stochastic_result.dtype:
        # Convert stochastic result explicitly if stochastic_round_ didn't match target exactly
        stochastic_result = stochastic_result.to(target.dtype)

    target.copy_(stochastic_result)


def copy_stochastic_(target: Tensor, source: Tensor):
    # Simplified logic - _compilable_copy_stochastic_ handles the rounding
    # No need for the explicit bf16 check here if stochastic_round_ handles it
    _compilable_copy_stochastic_(target, source)
    # Removed the set_ call, as _compilable_copy_stochastic_ now does the copy_


def copy_stochastic_list_(target: List[Tensor], source: List[Tensor]):
    # Added check for list lengths
    if len(target) != len(source):
        raise ValueError(f"Target and source lists must have the same length, got {len(target)} and {len(source)}")
    for t, s in zip(target, source):
        if t is None or s is None:
            # Decide how to handle None: skip? error?
            if t is not None or s is not None:
                warnings.warn("Mismatch in None entries during copy_stochastic_list_")
            continue  # Skip if either is None, or handle as needed
        copy_stochastic_(t, s)


@decorator_knowngood
def _lerp(state: List[Tensor], grad: List[Tensor], beta):
    # Ensure state and grad lists are of the same length
    if len(state) != len(grad):
        raise ValueError(f"State and grad lists must have the same length in _lerp, got {len(state)} and {len(grad)}")

    # Promote beta once
    beta_promoted = promote(beta)
    if not isinstance(beta_promoted, Tensor):
        # Ensure beta is a tensor for calculations
        if not state:
            raise ValueError("Cannot determine device/dtype for beta without state tensors.")
        beta_promoted = scalar_guard(beta_promoted, state[0])  # Use scalar_guard to create tensor

    # Create promoted state list (ea32)
    ea32 = [promote(s) for s in state]
    # Promote grad list
    grad_promoted = [promote(g) for g in grad]

    # Perform stochastic lerp using promoted lists and beta
    # Pass 1 - beta_promoted to stochastic_lerp_
    one_minus_beta = 1 - beta_promoted
    stochastic_lerp_(ea32, grad_promoted, one_minus_beta)  # ea32 updated inplace

    # Copy results back to original state list
    copy_stochastic_list_(state, ea32)

    return ea32  # Return the updated promoted state


# --- Core Optimizer Logic Components (Mostly from File 1) ---


@decorator_knowngood
def _compilable_exp_avg_sq_(
    state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor, out: List[Optional[Tensor]]
):
    # Ensure grad is promoted for squaring
    g32 = [promote(g) for g in grad]
    # Square the promoted gradients
    g32_sq = torch._foreach_mul(g32, g32)

    # Perform lerp: state = state * beta2 + g32_sq * (1 - beta2)
    # Note: _lerp expects weight for the *second* argument (grad)
    # We want: state = lerp(state, g32_sq, 1 - beta2)
    s32 = _lerp(state, g32_sq, 1 - beta2)  # state is updated inplace by _lerp

    # Calculate denominator: sqrt(state) clamped at eps
    denom = [eps_sqrt(d, eps) for d in s32]  # Use the updated s32 from _lerp

    if out is None or out[0] is None:  # Check if output computation is needed
        return denom

    # Copy denominator to output list if provided
    copy_stochastic_list_(out, denom)
    return out


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    # Use list_guard correctly for potentially None 'out'
    guarded = list_guard(state, grad, out)
    state, grad = guarded[0], guarded[1]
    out = guarded[2]  # out is now [None] if originally None, or [tensor] if tensor

    if not state:
        return None  # Handle empty input
    beta2, eps = scalar_guard(beta2, eps, state[0])
    result = _compilable_exp_avg_sq_(state, grad, beta2, eps, out)
    # Return structure should match input 'out'
    if out is None:
        return result  # Returns list of tensors (denom)
    else:
        return out  # Returns the list [tensor] (updated inplace)


@decorator_knowngood
def _compilable_scale_by_exp_avg_sq_(state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor):
    g32 = [promote(g) for g in grad]  # Promote grad
    # Compute denominator using _compilable_exp_avg_sq_ but don't need the output saved elsewhere
    denom = _compilable_exp_avg_sq_(state, g32, beta2, eps, [None])  # state updated inplace
    # Divide promoted grad by denominator
    out = torch._foreach_div(g32, denom)
    # Copy result back to original grad list
    copy_stochastic_list_(grad, out)


def scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps):
    grad, exp_avg_sq = list_guard(grad, exp_avg_sq)
    if not grad:
        return grad  # Handle empty input
    beta2, eps = scalar_guard(beta2, eps, grad[0])
    _compilable_scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps)
    return grad


@decorator_knowngood
def _compilable_exp_avg_(state, grad, beta):
    # Perform lerp: state = state * beta + grad * (1 - beta)
    # _lerp updates state inplace and returns the promoted version
    lerped_state = _lerp(state, grad, 1 - beta)  # state updated inplace

    # Copy the result (updated state) into grad
    copy_stochastic_list_(grad, lerped_state)


def scale_by_exp_avg_(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad  # Handle empty input
    beta = scalar_guard(beta, state[0])
    _compilable_exp_avg_(state, grad, beta)
    return grad


# --- Adam / AdamW / Variants (From File 1) ---


@decorator_knowngood
def _compilable_adam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    # Debias betas
    # Use clamp(min=1) for step to avoid issues with step=0
    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    beta2_debiased = beta_debias(beta2, step_clamped)

    g32 = [promote(g) for g in grad]

    # Update exp_avg: exp_avg = lerp(exp_avg, g32, 1 - beta1_debiased)
    exp_avg32 = _lerp(exp_avg, g32, 1 - beta1_debiased)  # exp_avg updated inplace

    # Update exp_avg_sq & get denominator: denom = sqrt(exp_avg_sq).clamp(eps)
    # Note: _compilable_exp_avg_sq_ expects beta2, not 1-beta2
    # It calculates: state = lerp(state, g32*g32, 1 - beta2) -> returns sqrt(state).clamp(eps)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, g32, beta2_debiased, eps, [None])  # exp_avg_sq updated inplace

    # Calculate update: u32 = exp_avg32 / denom
    u32 = torch._foreach_div(exp_avg32, denom)

    # Copy update back to grad list
    copy_stochastic_list_(grad, u32)


def adam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    # Use map for cleaner guarding
    exp_avg, exp_avg_sq, grad = map(list_guard, (exp_avg, exp_avg_sq, grad))
    if not exp_avg:
        return grad  # Handle empty lists
    beta1, beta2, step_t, eps_t = scalar_guard(
        beta1, beta2, step, eps, exp_avg[0]
    )  # Use different names for tensor versions
    _compilable_adam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step_t, eps_t)
    return grad


@decorator_knowngood
def _compilable_update_(
    p: List[Tensor], u: List[Tensor], decay: Tensor, lr: Tensor, caution: bool, g: List[Optional[Tensor]]
):
    # Ensure decay and lr are promoted if they are floats/ints
    decay_t, lr_t = scalar_guard(decay, lr, p[0])
    decay_p = promote(decay_t)
    lr_p = promote(lr_t)

    for i, p_ in enumerate(p):
        u_ = u[i]
        g_ = g[i] if g is not None else None  # Handle optional grad

        u_promoted = promote(u_.view_as(p_))  # Reshape and promote update
        p32_ = promote(p_)  # Promote parameter

        if caution:
            if g_ is None:
                raise ValueError("Gradient must be provided for cautioning.")
            # Promote gradient only if cautioning is active
            g_promoted = promote(g_)
            u_promoted = _compilable_cautioning(g_promoted, u_promoted)  # Cautioning

        # Apply update: p = p * (1 - decay * lr) - u * lr
        # Ensure calculation happens in promoted dtype
        decay_term = decay_p * lr_p
        p32_updated = p32_ * (1 - decay_term.to(p32_.dtype)) + u_promoted * (-lr_p.to(p32_.dtype))

        # Copy back with stochastic rounding
        copy_stochastic_(p_, p32_updated)


def update_param_(
    param: List[Tensor], update: List[Tensor], lr: float, decay: float, caution: bool = False, grad: List[Tensor] = None
):
    # Use proper list_guard for optional grad
    guarded = list_guard(param, update, grad)
    param, update = guarded[0], guarded[1]
    grad = guarded[2]  # grad is now [None] or list of tensors

    if not param:
        return  # Handle empty lists
    # scalar_guard needs a valid tensor reference
    lr_t, decay_t = scalar_guard(lr, decay, param[0])

    # Pass grad list directly, _compilable_update_ handles None inside
    _compilable_update_(param, update, decay_t, lr_t, caution, grad)


@decorator_knowngood
def _fused_compilable_adam_(
    y: List[Tensor],  # Typically parameters
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],  # Typically gradients used for EMA updates
    grad: List[Tensor],  # Typically original gradients for cautioning
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    decay: Tensor,  # Weight decay
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
    # Debias betas
    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    beta2_debiased = beta_debias(beta2, step_clamped)

    # Promote the 'update' tensor list (used for EMA calculations)
    u32 = [promote(up) for up in update]

    # Update exp_avg: exp_avg = lerp(exp_avg, u32, 1 - beta1_debiased)
    exp_avg32 = _lerp(exp_avg, u32, 1 - beta1_debiased)  # exp_avg updated inplace

    # Update exp_avg_sq & get denominator
    # Pass u32 (promoted update) to _compilable_exp_avg_sq_
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2_debiased, eps, [None])  # exp_avg_sq updated inplace

    # Calculate final update: final_u32 = exp_avg32 / denom
    final_u32 = torch._foreach_div(exp_avg32, denom)

    # Apply update to parameters 'y'
    # Pass original 'grad' list for potential cautioning
    _compilable_update_(y, final_u32, decay, lr, caution, grad)


def fused_adam_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    # Guard all lists
    y, exp_avg, exp_avg_sq, update, grad = list_guard(y, exp_avg, exp_avg_sq, update, grad)
    if not y:
        return  # Handle empty case
    beta1, beta2, step_t, decay_t, lr_t, eps_t = scalar_guard(beta1, beta2, step, decay, lr, eps, y[0])
    _fused_compilable_adam_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step_t, decay_t, lr_t, eps_t, caution)


# --- Laprop (From File 1) ---


@decorator_knowngood
def _compilable_laprop_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    beta2_debiased = beta_debias(beta2, step_clamped)

    gp32 = [promote(g) for g in grad]  # Promoted grad

    # Update exp_avg_sq & get denominator
    denom = _compilable_exp_avg_sq_(exp_avg_sq, gp32, beta2_debiased, eps, [None])  # exp_avg_sq updated inplace

    # Scale gradient: gp32 = gp32 / denom
    gp32_scaled = torch._foreach_div(gp32, denom)

    # Update exp_avg: exp_avg = lerp(exp_avg, gp32_scaled, 1 - beta1_debiased)
    # We want the *result* of this lerp to be copied back to grad
    exp_avg32 = _lerp(exp_avg, gp32_scaled, 1 - beta1_debiased)  # exp_avg updated inplace

    # Copy the result (updated exp_avg) to grad
    copy_stochastic_list_(grad, exp_avg32)


def laprop_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    if not exp_avg:
        return grad  # Handle empty
    beta1, beta2, step_t, eps_t = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_laprop_(exp_avg, exp_avg_sq, grad, beta1, beta2, step_t, eps_t)
    return grad


@decorator_knowngood
def _fused_compilable_laprop_(
    y: List[Tensor],  # Params
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],  # Grad used for EMA updates
    grad: List[Tensor],  # Original grad for cautioning
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    lr: Tensor,
    decay: Tensor,
    caution: bool,
    eps: Tensor,
):
    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    beta2_debiased = beta_debias(beta2, step_clamped)

    u32 = [promote(up) for up in update]  # Promoted update tensor

    # Update exp_avg_sq & get denominator using u32
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2_debiased, eps, [None])  # exp_avg_sq updated inplace

    # Scale update: u32_scaled = u32 / denom
    u32_scaled = torch._foreach_div(u32, denom)

    # Update exp_avg: exp_avg = lerp(exp_avg, u32_scaled, 1 - beta1_debiased)
    final_u32 = _lerp(exp_avg, u32_scaled, 1 - beta1_debiased)  # exp_avg updated inplace

    # Apply the final update (result of the lerp) to parameters 'y'
    _compilable_update_(y, final_u32, decay, lr, caution, grad)


def fused_laprop_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    decay: float,
    caution: bool,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, grad, y, update = list_guard(exp_avg, exp_avg_sq, grad, y, update)  # Guard 'update' as well
    if not exp_avg:
        return  # Handle empty
    beta1, beta2, step_t, lr_t, decay_t, eps_t = scalar_guard(beta1, beta2, step, lr, decay, eps, exp_avg[0])
    _fused_compilable_laprop_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step_t, lr_t, decay_t, caution, eps_t)


# --- AdOpt (From File 1) ---


@decorator_knowngood
def _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps):
    g32 = [promote(g) for g in grad]
    exp_avg_sq32 = [promote(s) for s in exp_avg_sq]

    # Store current exp_avg to use as the update output
    update = [e.clone() for e in exp_avg]  # Clone needed if exp_avg is modified later

    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    # Note: AdOpt paper might use different debiasing or step for beta2
    beta2_debiased = beta_debias(beta2, step_clamped)  # Check AdOpt paper for exact formula if needed

    # Calculate denominator = sqrt(exp_avg_sq).clamp(eps)
    denom = [eps_sqrt(d, eps) for d in exp_avg_sq32]

    # Update exp_avg: exp_avg = lerp(exp_avg, g32 / denom, 1 - beta1_debiased)
    g_div_denom = torch._foreach_div(g32, denom)
    _lerp(exp_avg, g_div_denom, 1 - beta1_debiased)  # exp_avg updated inplace

    # Update exp_avg_sq: exp_avg_sq = lerp(exp_avg_sq, g32 * g32, 1 - beta2_debiased)
    g32_sq = torch._foreach_mul(g32, g32)
    _lerp(exp_avg_sq, g32_sq, 1 - beta2_debiased)  # exp_avg_sq updated inplace

    # Copy the *original* exp_avg (before its update) stored in 'update' to grad
    copy_stochastic_list_(grad, update)


def adopt(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps: float = 1e-8):
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    if not exp_avg:
        return grad  # Handle empty
    beta1, beta2, step_t, eps_t = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step_t, eps_t)
    return grad


@decorator_knowngood
def _fused_compilable_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    # Promote inputs used in calculations
    u32 = [promote(u) for u in update]  # The 'update' here is the one applied to params
    g32 = [promote(g) for g in grad]  # Original grad
    exp_avg_sq32 = [promote(s) for s in exp_avg_sq]  # Current exp_avg_sq

    # Apply the provided 'update' (which is exp_avg from previous step) to parameters 'y'
    _compilable_update_(y, u32, decay, lr, caution, g32)  # y updated inplace

    # Now update the EMAs based on the current grad 'g32'
    step_clamped = step.clamp(min=1)
    beta1_debiased = beta_debias(beta1, step_clamped)
    # AdOpt original paper might use step+1 for beta2 debias - check if needed
    # step_plus_1 = (step + 1).clamp(min=1) # Or just use step_clamped if consistent
    beta2_debiased = beta_debias(beta2, step_clamped)  # Using step_clamped for now

    # Denominator for exp_avg update
    denom = [eps_sqrt(d, eps) for d in exp_avg_sq32]  # Use current exp_avg_sq

    # Update exp_avg: exp_avg = lerp(exp_avg, g32 / denom, 1 - beta1_debiased)
    g_div_denom = torch._foreach_div(g32, denom)
    _lerp(exp_avg, g_div_denom, 1 - beta1_debiased)  # exp_avg updated inplace

    # Update exp_avg_sq: exp_avg_sq = lerp(exp_avg_sq, g32 * g32, 1 - beta2_debiased)
    g32_sq = torch._foreach_mul(g32, g32)
    # Pass exp_avg_sq list directly to _lerp
    _lerp(exp_avg_sq, g32_sq, 1 - beta2_debiased)  # exp_avg_sq updated inplace


def fused_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    exp_avg, exp_avg_sq, grad, y, update = list_guard(exp_avg, exp_avg_sq, grad, y, update)
    if not exp_avg:
        return  # Handle empty
    beta1, beta2, step_t, lr_t, eps_t, decay_t = scalar_guard(beta1, beta2, step, lr, eps, decay, exp_avg[0])
    _fused_compilable_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step_t, lr_t, eps_t, decay_t, caution)


# --- Momentum Variants (From File 1) ---


@decorator_knowngood
def _compilable_heavyball_momentum_(state, grad, beta):
    # Promote state and grad
    s32 = [promote(s) for s in state]
    g32 = [promote(g) for g in grad]
    beta_p = promote(scalar_guard(beta, state[0]))  # Promote beta

    # state = state * beta + grad
    s32_updated = torch._foreach_mul(s32, beta_p)
    s32_updated = torch._foreach_add(s32_updated, g32)  # Add grad (original implementation)

    # Copy result back to state
    copy_stochastic_list_(state, s32_updated)
    # Copy result also to grad
    copy_stochastic_list_(grad, s32_updated)


def heavyball_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad  # Handle empty
    beta_t = scalar_guard(beta, state[0])
    _compilable_heavyball_momentum_(state, grad, beta_t)
    return grad


@decorator_knowngood
def _compilable_nesterov_momentum_(state, grad, beta):
    s32 = [promote(s) for s in state]
    g32 = [promote(g) for g in grad]
    beta_p = promote(scalar_guard(beta, state[0]))  # Promote beta

    # Update state: state = state * beta + grad
    s32_updated = torch._foreach_mul(s32, beta_p)
    s32_updated = torch._foreach_add(s32_updated, g32)

    # Calculate Nesterov gradient: grad_nesterov = grad + state_updated * beta
    # Use g32 (original promoted grad)
    g32_nesterov = [g + s * beta_p for g, s in zip(g32, s32_updated)]

    # Copy updated state back
    copy_stochastic_list_(state, s32_updated)
    # Copy Nesterov gradient back to grad list
    copy_stochastic_list_(grad, g32_nesterov)


def nesterov_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad  # Handle empty
    beta_t = scalar_guard(beta, state[0])
    _compilable_nesterov_momentum_(state, grad, beta_t)
    return grad


@decorator_knowngood
def _compilable_nesterov_ema_(state, grad, beta):
    # Calculate EMA: ema = lerp(state, grad, 1 - beta)
    ema32 = _lerp(state, grad, 1 - beta)  # state updated inplace

    # Update grad: grad = grad + ema
    # Need to promote original grad again as it wasn't returned by _lerp
    g32_original = [promote(g) for g in grad]
    grad_updated = [g + e for g, e in zip(g32_original, ema32)]

    # Copy result back to grad
    copy_stochastic_list_(grad, grad_updated)


def nesterov_ema(state, grad, beta):
    state, grad = list_guard(state, grad)
    if not state:
        return grad  # Handle empty
    beta_t = scalar_guard(beta, state[0])
    _compilable_nesterov_ema_(state, grad, beta_t)
    return grad


# --- Gradient Clipping / Normalization / Modification (From File 1) ---


@decorator_knowngood
def _compilable_grafting(magnitude, direction):
    # Promote inputs
    mag_p = promote(magnitude)
    dir_p = promote(direction)

    # Calculate norms using promoted tensors
    norm_mag = mag_p.norm()
    norm_dir = dir_p.norm().clamp(min=1e-6)  # Clamp denominator

    # Calculate result in promoted dtype
    result = dir_p * (norm_mag / norm_dir)

    # Return result (no inplace modification), caller handles stochastic copy
    return result


@decorator_knowngood
def _compilable_agc_(parameters: List[Tensor], gradients: List[Tensor], clip_val: Tensor, minimum: Tensor, eps: Tensor):
    # Promote parameters and gradients
    p32 = [promote(p) for p in parameters]
    g32 = [promote(g) for g in gradients]

    # Calculate norms
    p_norm = torch._foreach_norm(p32)
    g_norm = torch._foreach_norm(g32)

    # Clamp norms
    p_norm_clamped = torch._foreach_maximum(p_norm, minimum)
    g_norm_clamped = torch._foreach_maximum(g_norm, eps)

    # Calculate scale factor: scale = (p_norm / g_norm) * clip_val
    scale = torch._foreach_div(p_norm_clamped, g_norm_clamped)
    scale = torch._foreach_mul(scale, clip_val)

    # Clamp scale factor: scale = min(scale, 1.0)
    one_tensor = torch.ones_like(scale[0])  # Create a tensor with 1.0
    scale_final = torch._foreach_minimum(scale, one_tensor)

    # Apply scaling to gradients
    g32_scaled = torch._foreach_mul(g32, scale_final)

    # Copy result back to original gradient list
    copy_stochastic_list_(gradients, g32_scaled)


def adaptive_gradient_clipping_(
    parameters: List[Tensor], gradients: List[Tensor], clip_val: float, minimum: float = 1e-3, eps: float = 1e-8
):
    if clip_val <= 0:
        return gradients
    parameters, gradients = list_guard(parameters, gradients)
    if not parameters:
        return gradients  # Handle empty
    # Ensure clip_val, minimum, eps are tensors
    clip_val_t, minimum_t, eps_t = scalar_guard(clip_val, minimum, eps, parameters[0])
    _compilable_agc_(parameters, gradients, clip_val_t, minimum_t, eps_t)
    return gradients


@decorator_knowngood
def _compilable_l2_clip_(x, clip_at):
    ref = x  # Keep original list for stochastic rounding reference
    x_promoted = [promote(t) for t in x]

    # Calculate L2 norm for each tensor
    norms = torch._foreach_norm(x_promoted)

    # Create scale factor: scale = clip_at / max(norm, clip_at)
    # Use foreach_maximum to get max(norm, clip_at)
    max_norm_clip = torch._foreach_maximum(norms, clip_at)
    # Calculate scale = clip_at / max_norm_clip
    scale_factors = torch._foreach_div(clip_at, max_norm_clip)  # Ensure clip_at is tensor

    # Apply scale factor
    out = torch._foreach_mul(x_promoted, scale_factors)

    # Perform stochastic rounding back to original dtype
    # Use the 'ref' list which has the original tensors (and their dtypes)
    stochastic_out = [stochastic_round_(r, o) for r, o in zip(ref, out)]

    # This function should modify inplace, so copy back
    copy_stochastic_list_(x, stochastic_out)
    # Return the modified list (consistent with other _compilable funcs)
    # Although the name suggests inplace, returning allows chaining if needed
    return x


def l2_clip_(x, clip_at: float = 1.0):
    # Renamed from l2_normalization_ in original File 1? No, l2_normalization_ was separate.
    # This is l2_clip_.
    x = list_guard(x)
    if not x:
        return x
    if clip_at <= 0:
        raise ValueError("clip_at must be positive for l2_clip_")
    clip_at_t = scalar_guard(clip_at, x[0])
    _compilable_l2_clip_(x, clip_at_t)  # Modifies x inplace
    return x


def _max_idx(x: List[int]):
    return len(x) - 1 - np.argmax(x[::-1])  # we want to start counting from the back, as torch is fan-out/fan-in


def l2_normalization_(x, clip_at: float = 1e-8):
    # This normalizes, doesn't clip based on param norm like AGC
    # Normalizes grad to have L2 norm = 1, clamping norm below at clip_at
    x = list_guard(x)
    if not x:
        return x
    if clip_at <= 0:
        raise ValueError("clip_at (minimum norm) must be positive for l2_normalization_")
    clip_at_t = scalar_guard(clip_at, x[0])

    ref = x
    x_promoted = [promote(t) for t in x]
    norms = torch._foreach_norm(x_promoted)
    # Clamp norm at minimum value
    norms_clamped = torch._foreach_maximum(norms, clip_at_t)
    # Calculate scale = 1 / norm_clamped
    scales = torch._foreach_reciprocal(norms_clamped)  # Use reciprocal
    # Apply scale
    out = torch._foreach_mul(x_promoted, scales)

    # Stochastic round and copy back (similar to l2_clip_)
    stochastic_out = [stochastic_round_(r, o) for r, o in zip(ref, out)]
    copy_stochastic_list_(x, stochastic_out)
    return x


@decorator_knowngood
def _compilable_rmsnorm_clip_(x, clip_at):
    ref = x
    x_promoted = [promote(t) for t in x]

    # Calculate RMS norm: norm / sqrt(numel)
    norms = torch._foreach_norm(x_promoted)
    numels_sqrt = [math.sqrt(t.numel()) for t in x_promoted]  # Use original tensor numel
    # Need numels_sqrt as tensors for foreach_div
    numels_sqrt_t = [scalar_guard(n, norms[0]) for n in numels_sqrt]  # Convert to tensors
    rms_norms = torch._foreach_div(norms, numels_sqrt_t)

    # Similar to L2 clip: scale = clip_at / max(rms_norm, clip_at)
    max_norm_clip = torch._foreach_maximum(rms_norms, clip_at)
    scale_factors = torch._foreach_div(clip_at, max_norm_clip)  # Ensure clip_at is tensor

    # Apply scale factor
    out = torch._foreach_mul(x_promoted, scale_factors)

    # Stochastic round and copy back
    stochastic_out = [stochastic_round_(r, o) for r, o in zip(ref, out)]
    copy_stochastic_list_(x, stochastic_out)
    return x


def rmsnorm_clip_(x, clip_at: float = 1.0):
    x = list_guard(x)
    if not x:
        return x
    if clip_at <= 0:
        raise ValueError("clip_at must be positive for rmsnorm_clip_")
    clip_at_t = scalar_guard(clip_at, x[0])
    _compilable_rmsnorm_clip_(x, clip_at_t)  # Modifies inplace
    return x


def rmsnorm_normalize_(x, clip_at: float = 1e-6):
    # Normalizes grad to have RMS norm = 1, clamping RMS norm below at clip_at
    x = list_guard(x)
    if not x:
        return x
    if clip_at <= 0:
        raise ValueError("clip_at (minimum RMS norm) must be positive for rmsnorm_normalize_")
    clip_at_t = scalar_guard(clip_at, x[0])

    ref = x
    x_promoted = [promote(t) for t in x]

    norms = torch._foreach_norm(x_promoted)
    numels_sqrt = [math.sqrt(t.numel()) for t in x_promoted]
    numels_sqrt_t = [scalar_guard(n, norms[0]) for n in numels_sqrt]
    rms_norms = torch._foreach_div(norms, numels_sqrt_t)

    # Clamp RMS norm at minimum value
    rms_norms_clamped = torch._foreach_maximum(rms_norms, clip_at_t)

    # Calculate scale = 1 / rms_norm_clamped
    scales = torch._foreach_reciprocal(rms_norms_clamped)

    # Apply scale
    out = torch._foreach_mul(x_promoted, scales)

    # Stochastic round and copy back
    stochastic_out = [stochastic_round_(r, o) for r, o in zip(ref, out)]
    copy_stochastic_list_(x, stochastic_out)
    return x


@decorator_knowngood
def _compilable_sign_(grad: List[Tensor], graft: bool):
    # Keep ref for grafting and final copy
    promoted_grad = [promote(g) for g in grad]

    # Get sign
    signs = [g.sign() for g in promoted_grad]

    if graft:
        # Graft the signs onto the original magnitude
        # Grafting needs original promoted grad and the signs
        grafted_signs = [_compilable_grafting(g_p, s) for g_p, s in zip(promoted_grad, signs)]
        copy_stochastic_list_(grad, grafted_signs)  # Copy grafted result
    else:
        # Copy raw signs
        copy_stochastic_list_(grad, signs)


def sign_(grad: List[Tensor], graft: bool = True):
    grad = list_guard(grad)
    if not grad:
        return grad
    _compilable_sign_(grad, graft)  # Modifies inplace
    return grad


@decorator_knowngood
def _compilable_trust_region_clip_(grad, lerp, scale):
    # Promote inputs needed for calculation
    lerp_p = promote(lerp)
    scale_p = promote(scale)
    one_minus_lerp = 1 - lerp_p

    for x_ in grad:
        x = promote(x_)
        x_scaled = x / scale_p

        # tanh(x_scaled)
        tanh_x = x_scaled.tanh()

        # sign(x_scaled) * log1p(|x_scaled|)
        abs_x_scaled = x_scaled.abs()
        log1p_abs = abs_x_scaled.log1p()
        # Use tanh_x.sign() as sign(x_scaled) for consistency, handle 0?
        # copysign(log1p_abs, x_scaled) is safer
        signed_log1p = log1p_abs.copysign(x_scaled)

        # Combine: signed_log1p * (1 - lerp) + tanh_x * lerp
        combined = signed_log1p * one_minus_lerp + tanh_x * lerp_p

        # Rescale: combined * scale
        rescaled = combined * scale_p

        # Clamp
        clamped = rescaled.clamp(min=-2.0, max=2.0)  # Use float literals

        # Copy back
        copy_stochastic_(x_, clamped)


def trust_region_clip_(grad, lerp=0.9, scale=1.5):
    grad = list_guard(grad)
    if not grad:
        return grad
    lerp_t, scale_t = scalar_guard(lerp, scale, grad[0])
    _compilable_trust_region_clip_(grad, lerp_t, scale_t)  # Modifies inplace
    return grad


@decorator_knowngood
def _compilable_cautioning(g: Tensor, update: Tensor):
    # Ensure g and update are promoted and compatible
    g_p = promote(g)
    u_p = promote(update)
    if g_p.dtype != u_p.dtype:
        common_dtype = torch.promote_types(g_p.dtype, u_p.dtype)
        g_p = g_p.to(common_dtype)
        u_p = u_p.to(common_dtype)

    # Mask where signs differ (signbit is True for negative)
    mask = g_p.signbit() ^ u_p.signbit()

    # Zero out update where signs differ
    update_masked = u_p.masked_fill(mask, 0)

    # Calculate scale factor
    numel = mask.numel()
    # Sum of non-masked elements (where signs are the same)
    # mask.sum() is number of elements where signs differ
    non_masked_count = (numel - mask.sum()).clamp(min=1)  # Avoid division by zero
    scale_factor = numel / non_masked_count
    scale_factor = scale_factor.to(update_masked.dtype)  # Ensure scale factor matches dtype

    # Apply scale factor
    update_scaled = update_masked.mul(scale_factor)

    # Return scaled update (caller handles stochastic copy)
    return update_scaled


def caution(g, update):
    # Needs promotion handled by caller or inside if used standalone
    # Assume g, update are single tensors for this wrapper
    g_p = promote(g)
    u_p = promote(update)
    return _compilable_cautioning(g_p, u_p)  # Returns the cautioned tensor


@decorator_knowngood
def _compilable_caution_no_scale(g: Tensor, update: Tensor):
    # Added version without scaling from File 2's end
    g_p = promote(g)
    u_p = promote(update)
    if g_p.dtype != u_p.dtype:
        common_dtype = torch.promote_types(g_p.dtype, u_p.dtype)
        g_p = g_p.to(common_dtype)
        u_p = u_p.to(common_dtype)
    mask = g_p.signbit() ^ u_p.signbit()
    update_masked = u_p.masked_fill(mask, 0)
    return update_masked


def disable_caution_scaling():
    # Added from File 2's end
    global _compilable_cautioning
    print("Switching cautioning function to _compilable_caution_no_scale")
    _compilable_cautioning = _compilable_caution_no_scale


# --- Preconditioning Helpers (Mostly from File 1) ---


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
    (Copied from File 1/2 - identical)
    """
    # Handle 0-dim or 1-dim tensors
    if grad.dim() <= 1:
        if not split:
            return grad
        else:  # Splitting a 1D tensor if needed
            if grad.shape[0] > max_precond_dim:
                return list(grad.split(max_precond_dim, dim=0))
            else:
                return [grad]

    original_shape = grad.shape
    new_shape = []
    cum_size = 1

    # Iterate from the *second to last* dimension backwards
    for s in original_shape[1:][::-1]:
        temp_size = cum_size * s
        if temp_size > max_precond_dim and cum_size > 1:
            # If adding current dim 's' exceeds limit, and we already accumulated something
            new_shape.append(cum_size)  # Finalize the accumulated dimension
            cum_size = s  # Start new accumulation with current dim
        elif temp_size > max_precond_dim and cum_size == 1:
            # If adding current dim 's' exceeds limit, but nothing was accumulated yet
            new_shape.append(s)  # Add dim 's' by itself
            cum_size = 1  # Reset accumulation
        else:
            # If adding 's' doesn't exceed limit, accumulate it
            cum_size = temp_size

    # Add the last accumulated size (or the last single dimension if > max_precond_dim)
    if cum_size > 1:
        new_shape.append(cum_size)
    elif not new_shape and original_shape[1:][::-1]:  # Handle case where loop doesn't run or add anything
        # This case needs careful thought. If all dims are > max_precond_dim?
        # Let's assume the logic handles it by adding individual large dims.
        # If cum_size is 1, it means the last element was > max_precond_dim or was 1.
        # The check 'if cum_size > 1' correctly handles not adding a trailing 1.
        pass

    # Construct the final shape: [first_dim, *merged_dims_reversed]
    final_shape = [original_shape[0], *new_shape[::-1]]

    # Ensure the product of dimensions matches the original number of elements
    if math.prod(final_shape) != grad.numel():
        raise RuntimeError(
            f"Dim merger failed: Original shape {original_shape} ({grad.numel()}) != New shape {final_shape} ({math.prod(final_shape)})"
        )

    try:
        new_grad = grad.reshape(final_shape)
    except Exception as e:
        print(f"Error reshaping {original_shape} to {final_shape}")
        raise e

    if not split:
        return new_grad

    # --- Splitting Logic (if split=True) ---
    grads_to_process = [new_grad]

    # Iterate dimensions from right to left (excluding batch dim 0)
    current_dim_index = len(final_shape) - 1
    while current_dim_index > 0:
        next_grads_to_process = []
        dim_size = final_shape[current_dim_index]

        for g_ in grads_to_process:
            if dim_size <= max_precond_dim:
                # If current dim is small enough, pass tensor through
                next_grads_to_process.append(g_)
            else:
                # If dim needs splitting, split along this dimension
                # Add the split chunks to the list for the *next* iteration (or current if processing left-to-right)
                # We process right-to-left on shape, so add splits to be processed for *this* dim check
                split_chunks = g_.split(max_precond_dim, dim=current_dim_index)
                next_grads_to_process.extend(split_chunks)

        grads_to_process = next_grads_to_process  # Update list for next dimension check
        current_dim_index -= 1  # Move to the next dimension to the left

    # After checking all dimensions, grads_to_process contains the final list
    # This recursive call seems redundant now. The iterative approach should handle it.
    # --- Recursive part removal ---
    # if len(grads_to_process) == 1 and not split: # Should just return new_grad earlier
    #     return new_grad
    # if len(grads_to_process) == 1 and split and final_shape == original_shape: # No merge/split happened
    #      return new_grad # Return the reshaped grad

    # Return the list of potentially split tensors
    # Check if any splitting actually occurred
    if len(grads_to_process) == 1 and grads_to_process[0].shape == final_shape:
        # If only one tensor remains and its shape is the merged shape, return the single tensor
        return grads_to_process[0]  # Return tensor, not list[tensor]
    else:
        # Otherwise, return the list of split tensors
        return grads_to_process  # Return list[tensor]


@decorator
def update_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    """
    Simplified by @francois-rozet in commit 704ccc4bab52429f945df421647ec82c54cdd65f
    Re-commited due to faulty merge
    """
    # Ensure beta is treated correctly (float or tensor)
    beta_t = scalar_guard(beta, grad)[0]  # Use scalar_guard, get tensor
    one_minus_beta = 1 - beta_t

    # Handle cases where GG update is skipped
    if grad.dim() <= 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return  # Skip update for 1D tensors under specified conditions

    grad_p = promote(grad)  # Promote grad once for einsum

    for idx, m in enumerate(GG):
        if not isinstance(m, Tensor):
            continue  # Skip if placeholder is None

        # Check dimension compatibility before einsum
        if idx >= grad_p.dim():
            warnings.warn(f"GG index {idx} out of bounds for grad dim {grad_p.dim()}. Skipping GG update.")
            continue
        if m.dim() != 2 or m.shape[0] != m.shape[1] or m.shape[0] != grad_p.shape[idx]:
            warnings.warn(
                f"Shape mismatch: GG[{idx}] shape {m.shape} incompatible with grad shape {grad_p.shape} at index {idx}. Skipping GG update."
            )
            continue

        # Einsum indices
        b = einsum_base[idx]  # Current dim letter
        g0_indices = einsum_base[: grad_p.dim()]  # Indices for first grad ('abc...')
        # Indices for second grad, replacing current dim letter with uppercase ('aBc...')
        g1_indices = list(g0_indices)
        g1_indices[idx] = b.upper()
        g1_indices = "".join(g1_indices)

        # Output indices: current dim letter, twice, one uppercase ('bB')
        out_indices = b + b.upper()

        subscripts = f"{g0_indices},{g1_indices}->{out_indices}"

        try:
            outer_product = torch.einsum(subscripts, grad_p, grad_p)
        except Exception as e:
            warnings.warn(
                f"Einsum failed in update_ggt with subscripts '{subscripts}' for grad shape {grad_p.shape}. Error: {e}. Skipping GG update."
            )
            continue

        # Ensure outer product matches matrix m's dtype for lerp
        outer_product = outer_product.to(m.dtype)

        # Update GG matrix 'm' using stochastic lerp
        # lerp(m, outer_product, 1 - beta) -> m = m * beta + outer_product * (1 - beta)
        stochastic_lerp_([m], [outer_product], one_minus_beta)  # Use list version


def get_orthogonal_matrix(mat, max_eps: float = 1e-3, min_eps: float = 1e-30):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    (From File 1)
    """
    final = []
    for m in mat:
        if m is None:
            final.append(None)
            continue

        # Ensure matrix is on the correct device and potentially move to CPU if OOM
        # Keep original device/dtype to return to later
        original_device = m.device
        original_dtype = m.dtype
        m_compute = promote(m.data)  # Promote for computation stability

        # Add epsilon iteratively
        eps = min_eps
        eigvec = None
        while True:
            try:
                # Ensure matrix is symmetric for eigh
                m_sym = (m_compute + m_compute.T) / 2
                # Add regularization term
                eye = torch.eye(m_sym.shape[0], device=m_sym.device, dtype=m_sym.dtype)
                matrix_to_decompose = m_sym + eps * eye

                # Compute eigenvalues and eigenvectors
                _eigval, eigvec = torch.linalg.eigh(matrix_to_decompose)

                # Check for NaNs/Infs in eigenvectors (sign of instability)
                if torch.isnan(eigvec).any() or torch.isinf(eigvec).any():
                    raise RuntimeError("NaN or Inf encountered in eigenvectors.")

                # If successful, break loop
                break
            except torch.linalg.LinAlgError as e:  # Catch specific linalg errors
                warnings.warn(f"torch.linalg.eigh failed with eps={eps:.2e}: {e}. Increasing eps.")
                if m_compute.dtype != torch.double:
                    warnings.warn("Attempting eigh with double precision.")
                    m_compute = m_compute.double()
                    eps = min_eps  # Reset eps for double precision
                    continue  # Retry with double precision
                elif eps < max_eps:
                    eps = min(eps * 10, max_eps)  # Increase eps more aggressively
                    warnings.warn(f"Increased eps to {eps:.2e}.")
                else:
                    warnings.warn(f"Eigh failed even with max_eps={max_eps:.2e} and double precision. Raising error.")
                    raise e
            except torch.OutOfMemoryError:
                if m_compute.device.type == "cpu":
                    raise  # Already on CPU, can't recover
                else:
                    warnings.warn("OutOfMemoryError during eigh on GPU, moving matrix to CPU.")
                    m_compute = m_compute.cpu()
                    # Reset eps? Maybe not necessary if OOM is primary issue.
            except RuntimeError as e:  # Catch other runtime errors (like convergence)
                warnings.warn(f"RuntimeError during eigh with eps={eps:.2e}: {e}. Increasing eps.")
                if m_compute.dtype != torch.double:
                    warnings.warn("Attempting eigh with double precision.")
                    m_compute = m_compute.double()
                    eps = min_eps  # Reset eps
                    continue
                elif eps < max_eps:
                    eps = min(eps * 10, max_eps)
                    warnings.warn(f"Increased eps to {eps:.2e}.")
                else:
                    warnings.warn(
                        f"Eigh failed with RuntimeError even with max_eps={max_eps:.2e} and double precision. Raising error."
                    )
                    raise e
            # Clean cache if we moved to CPU or changed precision
            clean()

        # Ensure eigenvector is on original device and dtype
        if eigvec is None:  # Should not happen if loop breaks successfully
            raise RuntimeError("Eigh finished loop but eigvec is None.")
        eigvec = eigvec.to(device=original_device, dtype=original_dtype)

        # Flip to match convention (descending eigenvalues assumed)
        eigvec = torch.flip(eigvec, dims=[1])
        final.append(eigvec)

    return final


# @decorator_knowngood # Einsum makes this hard to compile reliably? Keep original decorator status.
def get_orthogonal_matrix_QR(GG: List[Tensor], Q: List[Tensor], exp_avg: Optional[Tensor] = None):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition, and updates exp_avg in-place from old to new eigenspace.
    (From File 1)
    """
    if isinstance(Q, list) and not Q:
        return  # Nothing to do if Q is empty

    # Validate exp_avg dimensions if provided
    if exp_avg is not None:
        # Allow exp_avg dim to be less if some Q entries are None?
        num_valid_q = sum(1 for q_ in Q if q_ is not None)
        if exp_avg.dim() != num_valid_q:
            raise ValueError(f"exp_avg dim {exp_avg.dim()} does not match number of non-None Q matrices {num_valid_q}")
        # Check einsum compatibility
        if exp_avg.dim() >= len(einsum_base):  # Use >= as we need 2*dim letters
            raise ValueError(f"exp_avg.dim() {exp_avg.dim()} is too large for einsum base string '{einsum_base}'")

    new_qs = []
    valid_q_indices = []  # Track indices of non-None Qs for exp_avg update

    for i, (m, q) in enumerate(zip(GG, Q)):
        if m is None or q is None:  # Skip if GG or Q matrix is None
            new_qs.append(None)
            continue

        valid_q_indices.append(i)  # Store index of this valid Q

        # Promote matrices for computation
        m_p = promote(m.data)
        q_old_p = promote(q.data)

        # Power iteration step: tmp = m @ q_old
        tmp = m_p @ q_old_p

        # Estimate eigenvalues: dot product of old basis vectors with transformed vectors
        # Ensure einsum uses promoted types
        est_eig = torch.einsum("ij,ij->j", q_old_p, tmp)

        # Sort eigenvectors based on estimated eigenvalues (descending)
        sort_idx = torch.argsort(est_eig, descending=True)

        # Reorder tmp columns based on sorted eigenvalues
        tmp_sorted = tmp[:, sort_idx]

        # Orthogonalize the sorted basis using QR decomposition
        try:
            q_new_p, _ = torch.linalg.qr(tmp_sorted)
            # Convert back to original dtype and store
            new_qs.append(q_new_p.to(q.dtype))
        except torch.linalg.LinAlgError as e:
            warnings.warn(f"QR decomposition failed for GG[{i}]: {e}. Using old Q matrix.")
            new_qs.append(q.data.clone())  # Keep old Q if QR fails

    # --- Update exp_avg using einsum if provided ---
    if exp_avg is None:
        # If no exp_avg, just update Q inplace with new_qs
        for i, q_new in enumerate(new_qs):
            if Q[i] is not None and q_new is not None:
                copy_stochastic_(Q[i], q_new)
            elif Q[i] is not None and q_new is None:
                warnings.warn(f"Q[{i}] was previously valid but became None after update.")
                # Decide how to handle: Keep old Q? Set Q[i] to None? Keep old for now.
            elif Q[i] is None and q_new is not None:
                warnings.warn(f"Q[{i}] was previously None but became valid after update.")
                Q[i] = q_new  # Assign the new Q
        return

    # --- Einsum calculation for exp_avg update ---

    # Filter Q and new_qs to only include non-None pairs corresponding to valid_q_indices
    Q_valid = [Q[i] for i in valid_q_indices if Q[i] is not None and new_qs[i] is not None]
    new_qs_valid = [new_qs[i] for i in valid_q_indices if Q[i] is not None and new_qs[i] is not None]
    exp_avg_indices = [i for i, idx in enumerate(valid_q_indices) if Q[idx] is not None and new_qs[idx] is not None]

    if not Q_valid:  # If no valid Q pairs remain after QR/filtering
        warnings.warn("No valid Q matrices left for exp_avg update via einsum.")
        # Update Q inplace with potentially failed QR results?
        for i, q_new in enumerate(new_qs):
            if Q[i] is not None and q_new is not None:
                copy_stochastic_(Q[i], q_new)
            # Handle None transitions as above if needed
        return

    # Define einsum strings based on the *actual* dimensions corresponding to valid Qs
    in_str_list = list(einsum_base[: exp_avg.dim()])  # Original letters for exp_avg dims
    out_str_list = list(einsum_base[exp_avg.dim() : 2 * exp_avg.dim()])  # Target letters

    from_shampoo_parts = []
    to_shampoo_parts = []
    final_out_str_list = list(in_str_list)  # Start with input letters

    # Map einsum letters only to the dimensions that have valid Q matrices
    for einsum_idx, q_old, q_new in zip(exp_avg_indices, Q_valid, new_qs_valid):
        in_char = in_str_list[einsum_idx]
        out_char = out_str_list[einsum_idx]
        interim_char = in_char.upper()  # Use uppercase for intermediate dimension

        from_shampoo_parts.append(interim_char + in_char)  # e.g., "Aa"
        to_shampoo_parts.append(interim_char + out_char)  # e.g., "Ao"
        final_out_str_list[einsum_idx] = out_char  # Mark this dimension's output letter

    in_str = "".join(in_str_list)
    from_shampoo = ",".join(from_shampoo_parts)
    to_shampoo = ",".join(to_shampoo_parts)
    out_str = "".join(final_out_str_list)

    subscripts = f"{in_str},{from_shampoo},{to_shampoo}->{out_str}"

    # Prepare arguments for einsum: exp_avg, old Qs, new Qs
    einsum_args = [exp_avg] + Q_valid + new_qs_valid

    # Promote args to common dtype for einsum
    common_dtype = promote(min_dtype(einsum_args))  # Promote result of min_dtype
    einsum_args_promoted = [arg.to(common_dtype) for arg in einsum_args]

    try:
        exp_avg_new = torch.einsum(subscripts, *einsum_args_promoted)
        # Copy result back to original exp_avg tensor
        copy_stochastic_(exp_avg, exp_avg_new)
    except Exception as e:
        warnings.warn(
            f"Einsum failed for exp_avg update with subscripts '{subscripts}'. Error: {e}. Skipping exp_avg update."
        )
        # Still update Q matrices even if exp_avg update fails

    # Update Q inplace with new_qs (potentially including failed QR results)
    for i, q_new in enumerate(new_qs):
        if Q[i] is not None and q_new is not None:
            copy_stochastic_(Q[i], q_new)
        # Handle None transitions if needed


def init_preconditioner(grad, state, max_precond_dim, precondition_1d):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    (From File 1)
    """
    state["GG"] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.numel() <= 1 or (grad.ndim <= 1 and not precondition_1d):
        # Handle scalar or 1D case where preconditioning is skipped
        state["GG"].append(None)
    else:
        # Check each dimension
        for sh in grad.shape:
            # Conditions for using diagonal (None placeholder):
            # 1. Dimension size > max_precond_dim
            # 2. Dimension size == 1 (cannot precondition a dim of size 1)
            # 3. It's a 1D tensor, precondition_1d is True, but size > max_precond_dim (handled by outer if)
            if sh > max_precond_dim or sh == 1:
                state["GG"].append(None)
            else:
                # Initialize with zeros (identity * 0)
                state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

    # Perform initial GGT update (effectively populates GG with G*G^T scaled by 0)
    # Pass beta=0 for initialization
    update_ggt(grad, state["GG"], max_precond_dim, precondition_1d, 0.0)

    # Initialize orthogonal matrices Q using eigh
    state["Q"] = get_orthogonal_matrix(state["GG"])


def update_preconditioner(grad, Q, GG, exp_avg, max_precond_dim, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    (From File 1)
    """
    # Update GG = beta * GG + (1 - beta) * grad * grad^T
    update_ggt(grad, GG, max_precond_dim, precondition_1d, beta)  # beta is for GG EMA

    # Update Q (eigenbasis) if scheduled
    if update_precond:
        # Use QR-based update which also updates exp_avg if provided
        get_orthogonal_matrix_QR(GG, Q, exp_avg)


@decorator  # Keep original decorator status
def project(grad, Q, back: bool):
    """
    :param grad: Gradient tensor
    :param Q: List of orthogonal matrices (or None)
    :param back: whether to project to Shampoo eigenbases (False) or back to original space (True)
    :return: Projected gradient
    (From File 1)
    """
    grad_dim = grad.dim()
    if grad_dim == 0:
        return grad  # Cannot project scalar

    if len(Q) != grad_dim:
        raise ValueError(f"Number of Q matrices ({len(Q)}) must match gradient dimension ({grad_dim})")

    param_indices = einsum_base[:grad_dim]
    preconditioner_parts = []
    q_matrices_to_use = []

    for i, q_mat in enumerate(Q):
        if q_mat is not None:
            # Check Q matrix validity
            if q_mat.dim() != 2 or q_mat.shape[0] != q_mat.shape[1] or q_mat.shape[0] != grad.shape[i]:
                raise ValueError(
                    f"Invalid Q matrix at index {i}: shape {q_mat.shape} incompatible with grad shape {grad.shape}"
                )

            current_dim_char = param_indices[i]
            next_dim_char = current_dim_char.upper()  # Use uppercase for the transformed dimension index

            # Forward projection (back=False): Q^T @ grad -> einsum 'ab,Aa->Ab'
            # Backward projection (back=True): Q @ grad -> einsum 'ab,Ab->ab'
            if back:  # Project back to original space
                # Subscript: Matrix index (e.g., 'aA'), input grad index ('A...'), output grad index ('a...')
                # Einsum uses Q[i] which is U. We need U^T for backward? No, Shampoo uses Q x Q^T form.
                # Backward projection is G_hat = Q_L G' Q_R^T. Einsum: G'_ab, QL_aA, QR_bB -> G_AB
                # Let's stick to the formula in the code: [(g + g.upper())[:: -1 if back else 1]]
                # if back=True: upper + lower -> 'Aa'
                subscript = next_dim_char + current_dim_char
            else:  # Project to eigenbasis
                # if back=False: lower + upper -> 'aA'
                subscript = current_dim_char + next_dim_char

            preconditioner_parts.append(subscript)
            q_matrices_to_use.append(q_mat)

    if not preconditioner_parts:  # If no Q matrices were provided
        return grad

    preconditioners = ",".join(preconditioner_parts)

    # Determine output indices based on projection direction
    out_indices_list = list(param_indices)
    for i, q_mat in enumerate(Q):
        if q_mat is not None:
            current_dim_char = param_indices[i]
            next_dim_char = current_dim_char.upper()
            if back:
                out_indices_list[i] = current_dim_char  # 'a'
            else:
                out_indices_list[i] = next_dim_char  # 'A'
    out_indices = "".join(out_indices_list)

    # Construct final einsum string
    einsum_str = f"{param_indices},{preconditioners}->{out_indices}"

    # Promote inputs for einsum
    grad_p = promote(grad)
    q_matrices_promoted = [promote(q) for q in q_matrices_to_use]

    # Perform einsum
    try:
        out = torch.einsum(einsum_str, grad_p, *q_matrices_promoted)
        # Convert back to original gradient dtype
        return out.to(grad.dtype)
    except Exception as e:
        warnings.warn(f"Einsum failed in project with string '{einsum_str}'. Error: {e}. Returning original gradient.")
        return grad


# --- PSGD Specific Functions (File 1 Base + File 2 Additions) ---


@decorator
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # From File 1
    if G.dim() != 2:  # Changed from assert to raise error
        raise ValueError(f"Input matrix must be 2D, got shape {G.shape}")

    a, b, c = (3.4445, -4.7750, 2.0315)  # Newton-Schulz constants for inverse sqrt approx?

    # Preserve float64 if present, otherwise use bfloat16 for intermediate, ensure float for norm
    compute_dtype = torch.bfloat16 if G.dtype != torch.float64 else torch.float64
    X = G.to(compute_dtype)
    original_dtype = G.dtype

    # Normalize: ensure spectral norm <= 1 approx.
    norm_X = X.float().norm()  # Use float32/64 for norm stability
    X = X / (norm_X + eps)

    # Handle non-square matrices by transposing if needed
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    # Iterative refinement
    for _ in range(steps):
        # Avoid recomputing X @ X.T if possible
        XT = X.T
        A = X @ XT  # A = X.X^T
        A_sq = A @ A  # A^2
        # B = b * A + c * A^2
        B = A * b + A_sq * c
        # X_new = a * X + B @ X
        X = X * a + B @ X
        # Note: This seems to compute U of SVD? Check source paper.
        # If it computes U, result should be U @ V.T -> orthogonal matrix.

    # Transpose back if necessary
    if transposed:
        X = X.T

    # Convert back to original dtype
    return X.to(original_dtype)


def ortho(x):
    # From File 1
    mode = zeroth_power_mode  # Use global setting
    x_p = promote(x)  # Promote for numerical stability

    if mode == "qr":
        # Handle potential non-invertibility or shape issues? qr expects full rank typically.
        try:
            Q, _ = torch.linalg.qr(x_p)
            return Q.to(x.dtype)
        except torch.linalg.LinAlgError as e:
            warnings.warn(f"QR failed in ortho: {e}. Falling back to SVD.")
            mode = "svd"  # Fallback to SVD

    if mode == "svd":
        try:
            u, _s, vh = torch.linalg.svd(x_p, full_matrices=False)  # Use full_matrices=False if appropriate
            # Result is u @ vh (v.T)? Check documentation. torch.linalg.svd returns Vh
            result = u @ vh
            return result.to(x.dtype)
        except torch.linalg.LinAlgError as e:
            warnings.warn(f"SVD failed in ortho: {e}. Returning input.")
            return x  # Return original if SVD fails

    # Allow Newton-Schulz as an option if zeroth_power_mode is set
    if mode == "newtonschulz":
        if x.dim() != 2:
            warnings.warn("Newton-Schulz only applicable to 2D matrices. Returning input.")
            return x
        return zeropower_via_newtonschulz5(x)  # Already handles dtype

    raise NotImplementedError(f"Unknown zeroth_power_mode: {mode}")


@decorator_knowngood
def inplace_orthogonal_(x: Tensor, mode: str, out: Tensor, scale_mode: str):
    # From File 1 - uses zeropower_via_newtonschulz5
    # Ensure mode is valid
    valid_modes = {"newtonschulz", "qr", "svd"}
    if mode not in valid_modes:
        raise NotImplementedError(f"Unknown zeroth_power_mode: {mode}")

    # Ensure scale_mode is valid
    valid_scale_modes = {"none", "scale", "graft"}
    if scale_mode not in valid_scale_modes:
        raise NotImplementedError(f"Unknown scale_mode: {scale_mode}")

    # Promote input for computation? Depends on the ortho method.
    # QR/SVD benefit from promotion. NewtonSchulz handles it internally.
    if mode in ("qr", "svd"):
        x_compute = promote(x)
    else:  # NewtonSchulz
        x_compute = x  # Pass original dtype

    # Calculate orthogonal matrix y
    if mode == "newtonschulz" or x.shape[0] != x.shape[1]:
        if x.dim() != 2:
            raise ValueError("Newton-Schulz requires 2D input.")
        y = zeropower_via_newtonschulz5(x_compute, 5)  # NS handles promotion/dtype
    elif mode == "qr":
        try:
            y, _ = torch.linalg.qr(x_compute)
        except torch.linalg.LinAlgError:
            warnings.warn("QR failed in inplace_orthogonal_, falling back to SVD.")
            try:
                u, _s, vh = torch.linalg.svd(x_compute, full_matrices=False)
                y = u @ vh
            except torch.linalg.LinAlgError:
                warnings.warn("SVD fallback failed in inplace_orthogonal_. Using input.")
                y = x_compute  # Use input as last resort
    elif mode == "svd":
        try:
            u, _s, vh = torch.linalg.svd(x_compute, full_matrices=False)
            y = u @ vh
        except torch.linalg.LinAlgError:
            warnings.warn("SVD failed in inplace_orthogonal_. Using input.")
            y = x_compute

    # Apply scaling
    y = y.to(promote(x.dtype))  # Ensure y is promoted for scaling math
    x_p = promote(x)  # Promote original x for scaling comparison/grafting

    if scale_mode == "none":
        pass
    elif scale_mode == "scale":
        # Scale factor based on aspect ratio
        scale_factor = math.sqrt(max(1.0, x.size(0) / x.size(1)))
        y = y * scale_factor
    elif scale_mode == "graft":
        # Graft magnitude of original x onto direction y
        y = _compilable_grafting(x_p, y)  # Grafting uses promoted inputs

    # Copy final result (with stochastic rounding) to output tensor 'out'
    copy_stochastic_(out, y)


_warned = set()


def warn_once(msg):
    # From File 1/2
    if msg not in _warned:
        warnings.warn(msg, stacklevel=2)  # Show caller info
        _warned.add(msg)


def psgd_should_update(
    group, prob: Union[float, callable], rng: Optional[random.Random] = None, name: str = "cumulative_prob"
):
    # From File 1/2
    step_key = f"{name}_step"  # Use consistent key naming
    prob_key = name  # Use 'cumulative_prob' or provided name

    group[step_key] = group.get(step_key, 0) + 1
    current_step = group[step_key]

    if callable(prob):
        current_prob = prob(current_step)
    else:
        current_prob = prob

    if current_prob < 0 or current_prob > 1:
        # Allow > 1 for cumulative mode, but warn if not stochastic?
        if group.get("stochastic_schedule", False) and (current_prob < 0 or current_prob > 1):
            raise ValueError(f"Probability must be between 0 and 1 for stochastic schedule, got {current_prob}")
        elif not group.get("stochastic_schedule", False) and current_prob < 0:
            raise ValueError(f"Cumulative probability increment must be non-negative, got {current_prob}")

    if group.get("stochastic_schedule", False):  # Check group setting
        if rng is None:
            raise ValueError("Random number generator (rng) must be provided for stochastic schedule.")
        return rng.random() < current_prob
    else:  # Cumulative probability mode
        cumulative_prob_old = group.get(prob_key, 0.0)
        cumulative_prob_new = cumulative_prob_old + current_prob
        group[prob_key] = cumulative_prob_new
        # Update occurs if the integer part increases
        return math.floor(cumulative_prob_new) > math.floor(cumulative_prob_old)


# --- PSGD Low-Rank Adaptation (LRA) Functions (NEW from File 2) ---
# Note: These functions assume inputs U, V, d, v, h are column vectors or matrices
# where appropriate, typically flattened/reshaped parameters/grads/HVPs.


@decorator_knowngood
def IpUVtmatvec(U, V, x):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Returns (I + U @ V.t()) @ x. All variables are either matrices or column vectors.
    (From File 2)
    """
    if U.numel() == 0 or V.numel() == 0:  # Handle rank 0 case or empty U/V
        return x
    # Ensure dimensions are compatible for matrix multiplication
    # U: (N, R), V: (N, R), x: (N, 1) or (N,) -> V.t(): (R, N)
    # V.t() @ x: (R, N) @ (N, 1) -> (R, 1)
    # U @ (V.t() @ x): (N, R) @ (R, 1) -> (N, 1)
    try:
        vt_x = V.t() @ x
        u_vt_x = U @ vt_x
        return x + u_vt_x
    except RuntimeError as e:
        raise RuntimeError(f"Dimension mismatch in IpUVtmatvec: U={U.shape}, V={V.shape}, x={x.shape}. Error: {e}")


@decorator_knowngood
def precond_grad_psgd_lra_(U, V, d, g):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    """
    Preconditioning gradient g with Q = (I + U*V')*diag(d).
    Equivalent to P*g = Q'*Q*g = d*(I + VU')* (I + UV')*d*g

    All variables here are either matrices or column vectors.
    (From File 2) - Applies the preconditioner P = Q'Q
    """
    # Calculate Q*g = (I + UV')*(d*g)
    d_g = d * g
    Q_g = IpUVtmatvec(U, V, d_g)

    # Calculate P*g = d*(I + VU')*Q_g
    # Note the switch: V and U are swapped in the second IpUVtmatvec call
    P_g_intermediate = IpUVtmatvec(V, U, Q_g)
    P_g = d * P_g_intermediate

    return P_g


@decorator_knowngood
def psgd_lra_update_precond_(U, V, d, v, h, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Update preconditioner Q = (I + U*V')*diag(d) with (vector, Hessian-vector product) = (v, h).
    State variables U, V and d are updated inplace.

    U, V, d, v, and h are column vectors or matrices where applicable.
    Based on preconditioned_stochastic_gradient_descent.update_precond_UVd_math_
    (From File 2, with minor cleanups/checks)
    """
    # --- Input validation ---
    if not all(isinstance(t, Tensor) for t in [U, V, d, v, h]):
        raise TypeError("U, V, d, v, h must all be tensors.")
    if U.shape[1] != V.shape[1]:  # Rank check
        raise ValueError(f"U and V must have the same rank (number of columns): U={U.shape}, V={V.shape}")
    rank = U.shape[1]
    if U.numel() > 0 and (
        U.shape[0] != V.shape[0] or U.shape[0] != d.shape[0] or U.shape[0] != v.shape[0] or U.shape[0] != h.shape[0]
    ):
        raise ValueError(f"Dimension mismatch: U={U.shape}, V={V.shape}, d={d.shape}, v={v.shape}, h={h.shape}")
    if step_normalizer not in ["1st", "2nd"]:
        raise ValueError(f"step_normalizer must be '1st' or '2nd', got {step_normalizer}")
    if tiny <= 0:
        raise ValueError(f"tiny must be positive, got {tiny}")

    # --- Promote for computation ---
    # Determine compute dtype (e.g., float32 or float64)
    compute_dtype = promote(min_dtype([U, V, d, v, h]))  # Promote the result of min_dtype
    U_p, V_p, d_p, v_p, h_p = [t.to(compute_dtype) for t in (U, V, d, v, h)]

    # --- Balance U and V norms (optional, stochastic) ---
    if rank > 0 and torch.rand([]) < 0.01:
        normU = torch.linalg.vector_norm(U_p).clamp(min=tiny)  # Use promoted, clamp min
        normV = torch.linalg.vector_norm(V_p).clamp(min=tiny)
        # Avoid division by zero/sqrt of zero
        if normV > tiny and normU > tiny:
            rho = torch.sqrt(normU / normV)
            # Update promoted tensors, changes will be copied back at the end
            U_p.div_(rho)
            V_p.mul_(rho)

    # --- Calculate P*h = Q'*Q*h ---
    # Qh = (I + UV')*d*h
    Qh = IpUVtmatvec(U_p, V_p, d_p * h_p)
    # Ph = d*(I + VU')*Qh
    Ph = d_p * IpUVtmatvec(V_p, U_p, Qh)

    # --- Calculate inv(P)*v implicitly ---
    # invPv = inv(Q)*inv(Q')*v where Q = (I+UV')d, Q' = d(I+VU')
    # inv(Q) = d^{-1} * (I - U*inv(I+V'U)*V')
    # inv(Q') = (I - V*inv(I+U'V)*U') * d^{-1}

    # invQtv = inv(Q')*v = (I - V*inv(I+U'V)*U')*(v/d)
    v_div_d = v_p / d_p.clamp(min=tiny)  # Clamp d_p in denominator
    invQtv = v_div_d
    if rank > 0:
        try:
            UtV = U_p.t() @ V_p
            I_UtV = torch.eye(rank, dtype=compute_dtype, device=U_p.device) + UtV
            # Use LU solve for stability/efficiency
            LU_UtV, pivots_UtV = torch.linalg.lu_factor(I_UtV)
            Ut_invQtv = U_p.t() @ invQtv
            solved_term = torch.linalg.lu_solve(LU_UtV, pivots_UtV, Ut_invQtv)
            invQtv = invQtv - V_p @ solved_term
        except torch.linalg.LinAlgError as e:
            warn_once(f"LU solve failed for UtV in psgd_lra_update (invQtv): {e}. Using approximation.")
            # Fallback: Use invQtv = v_div_d if solve fails

    # invPv = inv(Q)*invQtv = d^{-1}*(I - U*inv(I+V'U)*V')*invQtv
    invPv_intermediate = invQtv
    if rank > 0:
        try:
            VtU = V_p.t() @ U_p
            I_VtU = torch.eye(rank, dtype=compute_dtype, device=U_p.device) + VtU
            # Use LU solve
            LU_VtU, pivots_VtU = torch.linalg.lu_factor(I_VtU)
            Vt_invQtv = V_p.t() @ invQtv
            solved_term = torch.linalg.lu_solve(LU_VtU, pivots_VtU, Vt_invQtv)
            invPv_intermediate = invPv_intermediate - U_p @ solved_term
        except torch.linalg.LinAlgError as e:
            warn_once(f"LU solve failed for VtU in psgd_lra_update (invPv): {e}. Using approximation.")
            # Fallback: Use invPv_intermediate = invQtv if solve fails

    invPv = invPv_intermediate / d_p.clamp(min=tiny)  # Final division by d

    # --- Update d ---
    nablaD = Ph * h_p - v_p * invPv  # Gradient w.r.t. diagonal d
    # Calculate step size mu based on normalizer
    if step_normalizer == "2nd":
        # norm_Ph_v = || [Ph; v] ||_F per element? No, vector norm needed?
        # PSGD paper uses max norm. Let's use max abs value for simplicity/consistency.
        # max | Ph*h - v*invPv | ? Or something else?
        # Original code used torch.max(torch.sqrt(Ph*Ph + v*v)*torch.sqrt(h*h + invPv*invPv))? Seems complex.
        # Let's use max abs of gradient for now, like '1st'. Revisit if needed.
        # mu_denom = torch.max(torch.abs(nablaD)) # Simpler approach based on '1st'
        # Let's try to replicate the original intent more closely:
        norm_Ph_v = torch.sqrt(Ph.pow(2) + v_p.pow(2)).clamp(min=tiny)
        norm_h_invPv = torch.sqrt(h_p.pow(2) + invPv.pow(2)).clamp(min=tiny)
        # The original had min(rsqrt*rsqrt). max(norm*norm) seems wrong.
        # Let's assume it normalizes by the product of norms, perhaps max of element-wise product?
        mu_denom = torch.max(norm_Ph_v * norm_h_invPv)  # Max of element-wise product
        mu = step / (mu_denom + tiny)
    else:  # '1st' normalizer
        mu = step / (torch.max(torch.abs(nablaD)) + tiny)

    # Update d: d = d - mu * d * nablaD (multiplicative update)
    d_p.sub_(mu * d_p * nablaD)
    d_p.clamp_(min=tiny)  # Ensure d remains positive

    # --- Update U or V (stochastically) ---
    if rank > 0:
        a, b = Qh, invQtv  # Use intermediate results
        # Recompute/get I + V'U needed for updates
        VtU = V_p.t() @ U_p  # Use updated U_p, V_p if balanced
        I = torch.eye(rank, dtype=compute_dtype, device=U_p.device)
        IpVtU = I + VtU

        # Precompute LU for IpVtU for potential use in both U and V updates
        try:
            LU_IpVtU, pivots_IpVtU = torch.linalg.lu_factor(IpVtU)
            lu_success = True
        except torch.linalg.LinAlgError:
            warn_once("LU factorization failed for IpVtU in psgd_lra_update. Skipping U/V update.")
            lu_success = False

        if lu_success and torch.rand([]) < 0.5:  # Update U
            atV = a.t() @ V_p
            btV = b.t() @ V_p

            # Calculate nablaU approximation = a @ atV @ inv(IpVtU) - b @ btV @ inv(IpVtU)
            try:
                # Use lu_solve: term = M @ X where X = solve(A, B) -> X = solve(IpVtU, atV.T).T
                term_a = a @ torch.linalg.lu_solve(LU_IpVtU, pivots_IpVtU, atV.t()).t()
                term_b = b @ torch.linalg.lu_solve(LU_IpVtU, pivots_IpVtU, btV.t()).t()
                nablaU_approx = term_a - term_b

                # Calculate step size mu_uv
                if step_normalizer == "2nd":
                    # Use || nablaU_approx ||_F ? Seems too expensive.
                    # Use norms of components like original code?
                    norm_a = torch.linalg.vector_norm(a).clamp(min=tiny)
                    norm_b = torch.linalg.vector_norm(b).clamp(min=tiny)
                    # || a @ atV @ inv || ~= norm(a) * norm(atV) / sigma_min(IpVtU)?
                    # Let's use simpler approximation from File 2:
                    atVVt = a @ atV @ V_p.t()  # Reconstruct part of the gradient term? Seems complex.
                    btVVt = b @ btV @ V_p.t()
                    norm_atVVt = torch.linalg.vector_norm(atVVt).clamp(min=tiny)
                    norm_btVVt = torch.linalg.vector_norm(btVVt).clamp(min=tiny)
                    mu_uv_denom = norm_a * norm_atVVt + norm_b * norm_btVVt
                    mu_uv = step / (mu_uv_denom + tiny)

                else:  # '1st' normalizer - use max abs of approx gradient
                    mu_uv = step / (torch.max(torch.abs(nablaU_approx)) + tiny)

                # Update U: U = U - mu_uv * nablaU_approx
                U_p.sub_(mu_uv * nablaU_approx)

            except torch.linalg.LinAlgError as e:
                warn_once(f"LU solve failed during U update in psgd_lra: {e}. Skipping U update.")

        elif lu_success:  # Update V
            atU = a.t() @ U_p
            btU = b.t() @ U_p

            # Calculate nablaV = (a + V @ atU.t()) @ atU - (b + V @ btU.t()) @ btU
            nablaV = (a + V_p @ atU.t()) @ atU - (b + V_p @ btU.t()) @ btU

            # Calculate step size mu_uv
            if step_normalizer == "2nd":
                # Similar norm approximation as in File 2 for V
                norm_a = torch.linalg.vector_norm(a).clamp(min=tiny)
                norm_b = torch.linalg.vector_norm(b).clamp(min=tiny)
                UUta = U_p @ atU.t()  # Part of the gradient expression
                UUtb = U_p @ btU.t()
                norm_UUta = torch.linalg.vector_norm(UUta).clamp(min=tiny)
                norm_UUtb = torch.linalg.vector_norm(UUtb).clamp(min=tiny)
                mu_uv_denom = norm_a * norm_UUta + norm_b * norm_UUtb
                mu_uv = step / (mu_uv_denom + tiny)
            else:  # '1st' normalizer
                mu_uv = step / (torch.max(torch.abs(nablaV)) + tiny)

            # Update V: V = V - mu_uv * nablaV
            V_p.sub_(mu_uv * nablaV)

    # --- Copy updated promoted tensors back to original tensors ---
    copy_stochastic_(U, U_p)
    copy_stochastic_(V, V_p)
    copy_stochastic_(d, d_p)
    # No return needed as update is inplace


@decorator_knowngood
def _compilable_schedule_free_(
    p: List[Tensor],
    z: List[Tensor],
    ckp1: Tensor,  # weight / weight_sum
    update: List[Tensor],  # Typically grad or preconditioned grad
    lr: Tensor,
    beta1: Tensor,
    decay: Tensor,  # Weight decay factor (needs promotion if float)
    grad: List[Tensor],  # Original grad for cautioning
    caution: bool,
):
    # Promote scalar inputs once
    ckp1_p = promote(ckp1)
    lr_p = promote(lr)
    beta1_p = promote(beta1)
    decay_p = promote(decay)  # Promote decay if it's float/int

    one_tensor = torch.ones_like(ckp1_p)  # For 1 - ckp1

    for i in range(len(p)):
        op_ = p[i]  # Original parameter
        oz_ = z[i]  # Original z state
        u_ = update[i]  # Update direction
        g_ = grad[i]  # Original gradient

        # Promote tensors for this iteration
        u_promoted = promote(u_.view_as(op_))  # Reshape update and promote
        p_promoted = promote(op_)
        z_promoted = promote(oz_)

        # Apply weight decay to update if non-zero
        # u = u + p * decay (Check original paper for sign/placement)
        # Seems like AdamW style decay is applied differently here.
        # This looks like u_eff = u + p * decay, then update uses u_eff.
        if decay_p.item() != 0:  # Check value after promotion
            u_promoted = u_promoted + p_promoted * decay_p.to(p_promoted.dtype)

        # Apply cautioning if enabled
        if caution:
            if g_ is None:
                raise ValueError("Gradient must be provided for schedule_free cautioning.")
            g_promoted = promote(g_)
            u_promoted = _compilable_cautioning(g_promoted, u_promoted)

        # Update parameter p: p = lerp(z, p, ckp1) + update * (lr * beta1 * (1 - ckp1) - lr)
        # p_new = p_old * ckp1 + z_old * (1 - ckp1) + u * lr * (beta1 * (1-ckp1) - 1)

        # Calculate lerped p: p_lerped = z_promoted * (1 - ckp1_p) + p_promoted * ckp1_p
        one_minus_ckp1 = one_tensor - ckp1_p
        p_lerped = z_promoted * one_minus_ckp1 + p_promoted * ckp1_p

        # Calculate update coefficient: coeff = lr * (beta1 * (1 - ckp1) - 1)
        update_coeff = lr_p * (beta1_p * one_minus_ckp1 - one_tensor)

        # Apply update
        p_new = p_lerped + u_promoted * update_coeff.to(p_lerped.dtype)

        # Update state z: z = z - update * lr
        z_new = z_promoted + u_promoted * (-lr_p.to(z_promoted.dtype))

        # Copy back results
        copy_stochastic_(op_, p_new)
        copy_stochastic_(oz_, z_new)


def schedule_free_(
    lr: float,
    weight_lr_power: float,
    weight_sum: float,
    beta1: float,
    parameters: List[Tensor],
    z: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    caution: bool = False,
    r: float = 0.0,  # Weight step power
    step: int = 0,
    decay: float = 0.0,  # Weight decay
):
    # Calculate weight for this step
    # Ensure step >= 1 for power calculation
    clamped_step = max(step, 1)
    try:
        # Use abs(lr) for power
        weight = (abs(lr) ** weight_lr_power) * (clamped_step**r)
    except OverflowError:
        # Handle potential overflow if step or r are very large
        warnings.warn(
            f"Overflow calculating weight in schedule_free: lr={lr}, power={weight_lr_power}, step={clamped_step}, r={r}. Using large value."
        )
        weight = float("inf")

    # Check for inf/nan weight
    if not math.isfinite(weight):
        warnings.warn(f"Weight became non-finite ({weight}). Resetting weight_sum and weight.")
        weight_sum = 0.0
        weight = 1.0  # Assign a default finite weight

    # Update weight sum
    weight_sum_new = weight_sum + weight

    # Calculate ckp1 = weight / weight_sum_new
    # Handle potential division by zero if weight_sum_new is 0 (only if initial weight_sum=0 and weight=0)
    if weight_sum_new > 0:
        ckp1 = weight / weight_sum_new
    else:
        ckp1 = 0.0  # Should only happen at step 0 if lr=0?

    # Guard inputs
    update, parameters, z, grad = list_guard(update, parameters, z, grad)
    if not parameters:
        return weight_sum_new  # Handle empty case

    # Convert scalars to tensors
    lr_t, ckp1_t, beta1_t, decay_t = scalar_guard(lr, ckp1, beta1, decay, grad[0])  # grad[0] for ref

    # Call compilable function
    _compilable_schedule_free_(parameters, z, ckp1_t, update, lr_t, beta1_t, decay_t, grad, caution)

    # Return the updated weight sum
    return weight_sum_new


# --- MARS Correction (From File 1) ---


@decorator_knowngood
def _compilable_mars_correction_(g: List[Tensor], old_g: List[Tensor], a: Tensor):
    # Store a copy of the current gradient g
    g_copy = [g_.clone() for g_ in g]

    # Perform lerp: g = g * (1 - a) + old_g * a
    # _lerp(target, source, weight_for_source)
    _lerp(g, old_g, a)  # g is updated inplace

    # Copy the stored original gradient into old_g
    copy_stochastic_list_(old_g, g_copy)


def mars_correction(g, old_g, gamma, beta1):  # Renamed beta1 from File 1's beta
    # Calculate coefficient 'a'
    if beta1 < 0 or beta1 >= 1:
        raise ValueError(f"beta1 must be in [0, 1) for MARS correction, got {beta1}")
    # Avoid division by zero if beta1 is exactly 1 (though disallowed above)
    a_float = -gamma * beta1 / (1 - beta1)

    g, old_g = list_guard(g), list_guard(old_g)
    if not g:
        return  # Handle empty
    a = scalar_guard(a_float, g[0])  # Convert coefficient to tensor

    _compilable_mars_correction_(g, old_g, a)


# --- OrthoGrad (From File 1) ---


@decorator_knowngood
def _compilable_orthogonalization(weight: List[Tensor], grad: List[Tensor], eps: Tensor, graft: bool = True):
    """
    Implements OrthoGrad from "Grokking at the Edge of Numerical Stability" (https://arxiv.org/abs/2501.04697)
    """
    eps_p = promote(eps)  # Promote eps once

    for w, g in zip(weight, grad):
        w_p = promote(w)
        g_p = promote(g)

        # Ensure compatible dtypes for dot products
        common_dtype = promote(min_dtype([w, g]))  # Use min_dtype then promote
        w_c = w_p.to(common_dtype)
        g_c = g_p.to(common_dtype)

        # Calculate projection scalar: proj = (w * g).sum() / ((w * w).sum() + eps)
        dot_wg = torch.sum(w_c * g_c)
        dot_ww = torch.sum(w_c * w_c)
        proj = dot_wg / (dot_ww + eps_p.to(dot_ww.dtype))  # Add eps to denominator

        # Calculate orthogonalized gradient: out = g - proj * w
        # Ensure proj is correct dtype for multiplication
        out = g_c - proj.to(g_c.dtype) * w_c

        if graft:
            # Graft the magnitude of the original gradient (g_p) onto the new direction (out)
            # Ensure grafting uses compatible types
            out = _compilable_grafting(g_p, out.to(g_p.dtype))  # Graft needs promoted original grad

        # Copy result back to original grad tensor
        copy_stochastic_(g, out)


def orthogonalize_grad_to_param(weight, grad, eps, graft=True):
    weight, grad = list_guard(weight, grad)
    if not weight:
        return grad  # Handle empty
    eps_t = scalar_guard(eps, weight[0])
    _compilable_orthogonalization(weight, grad, eps_t, graft)  # Modifies grad inplace
    return grad


# --- Mu/A-Law Compression (From File 1) ---


@decorator_knowngood
def _compilable_mu_law_compress_(x, mu):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """
    mu_p = promote(mu)  # Promote mu once
    log1p_mu = torch.log1p(mu_p)  # Precompute log(1+mu)

    for x_ in x:
        x_p = promote(x_)
        # xa = sign(x) * log(1 + mu * |x|) / log(1 + mu)
        xa_abs = x_p.abs()
        xa_term = torch.log1p(mu_p * xa_abs)  # log(1 + mu*|x|)
        xa_scaled = xa_term / log1p_mu  # Divide by log(1+mu)
        # Apply original sign
        xa_signed = xa_scaled.copysign(x_p)
        # Copy back
        copy_stochastic_(x_, xa_signed)


def mu_law_compress(x, mu=127.0):
    """
    μ-law compression
    Args:
        x: Input tensor or list of tensors
        mu: Compression parameter (default 127.0)
    """
    x = list_guard(x)
    if not x:
        return x
    if mu < 0:
        raise ValueError("mu must be non-negative for mu-law compression")
    mu_t = scalar_guard(mu, x[0])
    _compilable_mu_law_compress_(x, mu_t)  # Modifies inplace
    return x


@decorator_knowngood
def _compilable_a_law_compress_(x, A):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """
    A_p = promote(A)  # Promote A once
    log_A = torch.log(A_p)  # log(A)
    denom = 1 + log_A  # 1 + log(A)

    for x_ in x:
        x_p = promote(x_)
        x_abs = x_p.abs()
        Ax = A_p * x_abs

        # Apply A-law formula based on magnitude
        # where(Ax < 1, Ax / denom, (1 + log(Ax)) / denom ) * sign(x)
        is_small = Ax < 1.0
        term_small = Ax
        # Use log1p for potentially better stability if Ax is near 1? No, paper uses log.
        term_large = 1 + torch.log(Ax.clamp(min=tiny_bf16))  # Clamp log input away from 0

        compressed_abs = torch.where(is_small, term_small, term_large) / denom

        # Apply sign
        xa_signed = compressed_abs.copysign(x_p)
        # Copy back
        copy_stochastic_(x_, xa_signed)


def a_law_compress(x, A=87.6):
    """
    A-law compression
    Args:
        x: Input tensor or list of tensors
        A: Compression parameter (default 87.6)
    """
    x = list_guard(x)
    if not x:
        return x
    if A < 1:
        raise ValueError("A must be >= 1 for A-law compression")
    A_t = scalar_guard(A, x[0])
    _compilable_a_law_compress_(x, A_t)  # Modifies inplace
    return x


# --- EMA Weight Decay (From File 1) ---


@decorator_knowngood
def _compilable_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    # Promote inputs
    ema_decay_p = promote(ema_decay)
    weight_decay_p = promote(weight_decay)

    # Calculate EMA: ema = lerp(ema, p, ema_decay)
    # Remember _lerp uses weight for *second* argument
    # We want ema = ema*(1-ema_decay) + p*ema_decay
    # So call _lerp(ema, p, ema_decay)
    ema32 = _lerp(ema, p, ema_decay_p)  # ema updated inplace, returns promoted ema

    # Update parameter p using EMA and weight decay
    # p = lerp(p, ema, weight_decay) ? Seems too simple.
    # AdamW: p = p - lr * wd * p
    # Decoupled WD: p = p * (1 - lr * wd)
    # This looks like: p = p * (1 - weight_decay) + ema * weight_decay
    # p = lerp(p, ema, weight_decay) -> Yes.
    # Need promoted p list
    p_list = list_guard(p)  # Ensure p is a list
    p_promoted = [promote(pi) for pi in p_list]
    # Use stochastic_lerp_ directly as we don't need _lerp's copy-back here
    stochastic_lerp_(p_promoted, ema32, weight_decay_p)  # p_promoted updated inplace

    # Copy the final p back to original p list
    copy_stochastic_list_(p, p_promoted)


def weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    if not p:
        return
    ema_decay_t, weight_decay_t = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_weight_decay_to_ema_(p, ema, ema_decay_t, weight_decay_t)  # p and ema modified inplace


@decorator_knowngood
def _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    # Promote scalars
    ema_decay_p = promote(ema_decay)
    weight_decay_p = promote(weight_decay)

    # Calculate EMA: ema = lerp(ema, p, ema_decay)
    ema32 = _lerp(ema, p, ema_decay_p)  # ema updated inplace, returns promoted ema

    # Apply L1 decay to parameter p
    for i in range(len(p)):
        p_ = p[i]
        e_ = ema32[i]  # Use the updated ema

        p32 = promote(p_)  # Promote current p

        # Calculate difference: diff = p - ema
        diff = p32 - e_.to(p32.dtype)
        # Calculate update: p_new = p - sign(diff) * weight_decay
        p32_updated = p32 - diff.sign() * weight_decay_p.to(p32.dtype)

        # Copy back
        copy_stochastic_(p_, p32_updated)


def l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    if not p:
        return
    ema_decay_t, weight_decay_t = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay_t, weight_decay_t)  # p and ema modified inplace


# --- Triu/Line Conversion (From File 1) ---


@decorator  # Keep original decorator
def triu_to_line(Q_list: List[Tensor]):
    out = []
    for q in Q_list:
        if q is None:  # Handle None
            out.append((None, None))
        elif q.dim() < 2:  # Diagonal preconditioner (1D tensor)
            out.append((None, q))
        else:  # Triangular matrix (2D)
            # Ensure it's square
            if q.shape[0] != q.shape[1]:
                raise ValueError(f"Matrix must be square for triu_to_line, got shape {q.shape}")
            # Get upper triangle indices
            indices = torch.triu_indices(*q.shape, device=q.device)
            # Extract elements and store shape + line
            out.append((list(q.shape), q[tuple(indices)]))  # Store shape as list
    return out


def _triu_shape(numel):
    # Calculate n from n*(n+1)/2 = numel
    # n^2 + n - 2*numel = 0
    # Use quadratic formula: n = (-1 + sqrt(1 + 8*numel)) / 2
    if numel < 0:
        raise ValueError("Number of elements cannot be negative.")
    # Handle numel=0 case -> n=0
    if numel == 0:
        return (0, 0)

    discriminant = 1 + 8 * numel
    # Check if discriminant is a perfect square
    sqrt_discriminant = math.isqrt(discriminant)  # Integer square root
    if sqrt_discriminant * sqrt_discriminant != discriminant:
        raise ValueError(f"Number of elements {numel} does not correspond to a triangular matrix.")

    # Calculate n
    n = (-1 + sqrt_discriminant) / 2
    if n != int(n):  # Check if n is an integer
        raise ValueError(f"Number of elements {numel} does not correspond to a triangular matrix.")

    n = int(n)
    # Final check: n*(n+1)/2 == numel
    if n * (n + 1) // 2 != numel:
        raise ValueError(f"Calculated dimension {n} is incorrect for {numel} elements.")

    return n, n


@decorator  # Keep original decorator
def line_to_triu(Q_list: List[Tuple[Optional[List[int]], Optional[Tensor]]]):
    new = []
    for shape_info, q_line in Q_list:
        if q_line is None:  # Handle None from input
            new.append(None)
            continue

        if shape_info is not None:  # It was a triangular matrix
            # Determine shape from numel if needed (shape_info should be correct)
            try:
                if not isinstance(shape_info, (list, tuple)) or len(shape_info) != 2:
                    raise TypeError("Shape info must be a list/tuple of length 2")
                shape = tuple(shape_info)  # Use provided shape
                # Sanity check shape against numel
                n, m = shape
                if n != m or n * (n + 1) // 2 != q_line.numel():
                    # Try calculating from numel as fallback
                    calc_n, calc_m = _triu_shape(q_line.numel())
                    if n != calc_n or m != calc_m:
                        raise ValueError(
                            f"Shape info {shape} inconsistent with line numel {q_line.numel()} (expected {calc_n}x{calc_m})"
                        )
                    shape = (calc_n, calc_m)

            except ValueError as e:
                raise ValueError(f"Error determining shape for line tensor with {q_line.numel()} elements: {e}")

            # Create zero matrix
            x = torch.zeros(shape, device=q_line.device, dtype=q_line.dtype)
            # Get indices and fill
            indices = torch.triu_indices(*shape, device=q_line.device)
            x[tuple(indices)] = q_line
            new.append(x)
        else:  # It was a diagonal (1D) tensor or None
            new.append(q_line)
    return new


def update_triu_(q_state, materialised):
    # q_state: List[Tuple[Optional[List[int]], Tensor]] (line format)
    # materialised: List[Tensor] (matrix format)
    try:
        materialised_lines = triu_to_line(materialised)
    except ValueError as e:
        raise ValueError(f"Error converting materialised matrices to lines in update_triu_: {e}")

    if len(q_state) != len(materialised_lines):
        raise ValueError(f"Length mismatch: q_state ({len(q_state)}) vs materialised ({len(materialised_lines)})")

    for (shape0, q), (shape1, m) in zip(q_state, materialised_lines):
        # Check for None consistency
        if (q is None) != (m is None):
            raise ValueError("Inconsistent None presence between q_state and materialised lines.")
        if q is None:
            continue  # Skip if both are None

        # Check shape consistency if shape info exists
        if shape0 is not None and shape1 is not None and tuple(shape0) != tuple(shape1):
            raise ValueError(f"Shape mismatch: q_state expects {shape0}, materialised gave {shape1}")
        elif shape0 is not None and shape1 is None:
            raise ValueError("Shape mismatch: q_state expects 2D, materialised gave 1D")
        elif shape0 is None and shape1 is not None:
            raise ValueError("Shape mismatch: q_state expects 1D, materialised gave 2D")

        # Copy materialised line (m) to state line (q)
        copy_stochastic_(q, m)


# --- PSGD Preconditioning Application (From File 1) ---


@decorator_knowngood
def precond_grad_cached_(
    expr: str, ea: Tensor, *cached_q: Tensor, caution: bool = False, grad: Optional[Tensor] = None, cast: bool = True
):
    # Promote ea (effective accumulator/gradient)
    ea_p = promote(ea)
    grad_p = promote(grad) if grad is not None else None

    if caution:
        if grad_p is None:
            raise ValueError("Original gradient 'grad' must be provided for cautioning in precond_grad_cached_")
        ea_p = _compilable_cautioning(grad_p, ea_p)  # Caution the promoted effective grad

    # Determine minimum dtype for einsum
    # Needs list conversion for cached_q
    q_list = list(cached_q)
    all_tensors = q_list + [ea_p]
    md = promote(min_dtype(all_tensors))  # Promote the dtype itself

    # Prepare args: promoted Qs + promoted (and maybe cautioned) ea
    args = [q.to(md) for q in q_list] + [ea_p.to(md)]

    # Perform einsum
    try:
        new = torch.einsum(expr, *args)
    except Exception as e:
        # Provide more context on error
        shapes = [arg.shape for arg in args]
        raise RuntimeError(f"Einsum failed in precond_grad_cached_ with expr '{expr}' and shapes {shapes}. Error: {e}")

    # Cast back to original effective grad dtype if requested
    if cast:
        return new.to(ea.dtype)
    else:
        return new  # Return in computed dtype


@decorator_knowngood
def _compilable_fused_precond_grad_cached_(expr: str, ea: Tensor, param, lr, grad, decay, caution, *cached_q: Tensor):
    # Calculate preconditioned gradient, DO NOT cast back yet
    precond = precond_grad_cached_(expr, ea, *cached_q, caution=caution, grad=grad, cast=False)

    # Update parameter using the preconditioned gradient (which is in computed dtype)
    # update_param_ handles promotion and stochastic copy back to param's dtype
    # Pass precond (single tensor) and param (single tensor) as lists
    # grad is original grad for cautioning inside update_param if needed (though caution applied above)
    # Let update_param handle decay/lr promotion
    update_param_(
        [param], [precond], lr.item(), decay.item(), caution=False, grad=[grad]
    )  # caution=False here as already done


def fused_precond_grad_cached_(
    expr: str, ea: Tensor, param, lr: float, grad: Tensor, decay: float, caution: bool, *cached_q: Tensor
):
    # Convert scalar lr/decay to tensors relative to param
    lr_t, decay_t = scalar_guard(lr, decay, param)  # param is single tensor
    # Ensure param and grad are passed correctly
    _compilable_fused_precond_grad_cached_(expr, ea, param, lr_t, grad, decay_t, caution, *cached_q)


@decorator_knowngood
def psgd_precond_grad(expr: str, ea: Tensor, *preconds: Tensor, caution: bool = False, grad: Optional[Tensor] = None):
    """Applies Q'Q"""
    # Promote ea and grad
    ea_p = promote(ea)
    grad_p = promote(grad) if grad is not None else None

    if caution:
        if grad_p is None:
            raise ValueError("Original gradient 'grad' must be provided for cautioning in psgd_precond_grad")
        ea_p = _compilable_cautioning(grad_p, ea_p)

    # Determine computation dtype
    precond_list = list(preconds)
    all_tensors = precond_list + [ea_p]
    md = promote(min_dtype(all_tensors))

    # Prepare args: Qs, Qs again, ea
    args_q = [q.to(md) for q in precond_list]
    args = args_q + args_q + [ea_p.to(md)]  # Q' Q g -> uses Q twice

    # Perform einsum
    try:
        new = torch.einsum(expr, *args)
    except Exception as e:
        shapes = [arg.shape for arg in args]
        raise RuntimeError(f"Einsum failed in psgd_precond_grad with expr '{expr}' and shapes {shapes}. Error: {e}")

    # Return result cast back to original ea dtype
    return new.to(ea.dtype)


@decorator_knowngood
def _compilable_fused_psgd_precond_grad(expr: str, ea: Tensor, param, lr, grad, decay, caution, *preconds: Tensor):
    # Calculate preconditioned gradient (Q'Q applied)
    precond = psgd_precond_grad(expr, ea, *preconds, caution=caution, grad=grad)
    # precond is already cast back to ea.dtype

    # Update parameter
    update_param_([param], [precond], lr.item(), decay.item(), caution=False, grad=[grad])


def fused_psgd_precond_grad(
    expr: str, ea: Tensor, param, lr: float, grad: Tensor, decay: float, caution: bool, *preconds: Tensor
):
    lr_t, decay_t = scalar_guard(lr, decay, param)
    _compilable_fused_psgd_precond_grad(expr, ea, param, lr_t, grad, decay_t, caution, *preconds)


# --- PSGD Initialization and Update Steps (From File 1) ---
# Note: These seem specific to a different PSGD variant (Triangular/Diagonal) than LRA


@decorator_knowngood
def mean_root(x: torch.Tensor, pow: float):
    if pow <= 0:
        raise ValueError("Power must be positive for mean_root.")
    # Calculate mean of powers, then root
    x_f = x.float()  # Use float32 for calculation
    mean_pow = x_f.pow(pow).mean()
    # Root is mean_pow ^ (1/pow) -> used for scaling? Paper uses mean(|g|^4)^(-1/8)
    # Formula here: mean(x^pow)^(-1 / pow / 2)
    root_val = mean_pow.pow(-1.0 / (pow * 2.0))  # Check exponent carefully
    # Stochastic round back to original type? Assume x is the reference.
    return stochastic_round_(x, root_val)


@decorator_knowngood
def divided_root(x, y, pow0, pow1):
    if pow0 <= 0 or pow1 <= 0:
        raise ValueError("Powers must be positive for divided_root.")
    # mean(|v|^p)^(1/p/2) * mean(|h|^q)^(-1/q/2)
    mean_x = x.float().pow(pow0).mean().pow(1.0 / (pow0 * 2.0))
    mean_y = y.float().pow(pow1).mean().pow(-1.0 / (pow1 * 2.0))  # Negative root for y
    result = mean_x * mean_y
    # Stochastic round back to x's type? Assume x is ref.
    return stochastic_round_(x, result)


def init_Q_exprs(grad, scale, max_size, min_ndim_triangular, memory_save_mode, hessian_vector, vector, dtype=None):
    """
    Initialize preconditioner Q (Diagonal/Triangular) and einsum expressions.
    (From File 1) - Seems incompatible/alternative to LRA above. Keeping as is.
    """
    # Determine initial scale factor if not provided
    if scale is None:
        if hessian_vector is None or vector is None:  # Need both v and h for divided_root
            # Use mean root of grad if HVP info is missing
            if grad is None:
                raise ValueError("Cannot initialize Q scale without gradient or HVP info.")
            warn_once("Hessian/vector info missing for init_Q_exprs scale, using mean_root(grad).")
            scale = mean_root(grad, 4)  # mean(|g|^4)^(-1/8)
        else:
            # Use divided root of vector and hessian_vector
            scale = divided_root(vector, hessian_vector, 2, 4)  # mean(|v|^2)^(1/4) * mean(|h|^4)^(-1/8)

    # Setup
    letters = string.ascii_lowercase + string.ascii_uppercase
    dtype_out = dtype if dtype is not None else grad.dtype
    shape = grad.shape

    # Handle scalar gradient
    if len(shape) == 0:
        Q = [scale * torch.ones_like(grad, dtype=dtype_out)]  # Use dtype_out
        # No indices needed for scalar einsum
        exprA = ",->"  # Update Q
        exprGs = [",->"]  # Update step for each Q (only one)
        exprP = ",,->"  # Apply preconditioner
        return [Q, (exprA, tuple(exprGs), exprP)]

    # Handle tensor gradient
    if len(shape) > len(string.ascii_lowercase):  # Check against available letters
        raise ValueError(
            f"Tensor dimension {len(shape)} exceeds available einsum letters ({len(string.ascii_lowercase)})"
        )

    # Distribute scale across dimensions: scale_per_dim = scale ^ (1 / num_dims)
    scale_per_dim = scale ** (1.0 / len(shape))

    # Determine which dimensions use diagonal preconditioning
    use_diag = [False] * len(shape)
    if memory_save_mode is not None:
        valid_modes = {"one_diag", "smart_one_diag", "all_diag"}
        if memory_save_mode not in valid_modes:
            raise ValueError(f"Invalid memory_save_mode: {memory_save_mode}")

        if memory_save_mode == "all_diag":
            use_diag = [True] * len(shape)
        elif memory_save_mode == "one_diag":
            # Apply to the 'last' dimension according to _max_idx convention
            max_idx = _max_idx(shape)  # Find index of largest dim from the end
            use_diag[max_idx] = True
        elif memory_save_mode == "smart_one_diag":
            # Apply if largest dim is strictly larger than second largest
            if len(shape) >= 2:
                sorted_indices = np.argsort(shape)
                if shape[sorted_indices[-1]] > shape[sorted_indices[-2]]:
                    # Find the original index of the largest dimension
                    max_val = shape[sorted_indices[-1]]
                    # Find first occurrence from right using _max_idx logic
                    max_idx = len(shape) - 1 - np.argmax(np.array(shape[::-1]) == max_val)
                    use_diag[max_idx] = True

    # Initialize Q matrices and build einsum expressions
    Q = []
    # For exprA (updating Q): Q_ik, Q_jk, G_...ijk... -> G'_...k... (?) Check PSGD paper
    # File 1 has: ", ".join(piece1A) + "," + piece2A + "->" + piece3A
    piece1A, piece2A, piece3A = ([], "", "")  # Inputs: Qs, A; Output: conjB? No, seems related to G update term.
    exprGs = []  # For calculating terms like T1 = einsum(exprG, A, A)
    # For exprP (applying Q'Q): Q_ia, Q_ib, G_...ab... -> G'_...ij... (?) Check PSGD paper
    # File 1 has: ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    piece1P, piece2P, piece3P, piece4P = ([], [], "", "")  # Inputs: Qs, Qs, G; Output: G'

    for i, size in enumerate(shape):
        is_diag = use_diag[i]
        # Conditions for using diagonal: explicit flag, size=1, size>max_size, ndim < min_ndim_triangular
        use_diag_final = is_diag or size == 1 or size > max_size or len(shape) < min_ndim_triangular

        if use_diag_final:
            # Diagonal Q: stored as 1D tensor
            # Promote dtype for diagonal elements? File 1 uses promote(dtype_out). Let's use that.
            Q.append(scale_per_dim * torch.ones(size, dtype=promote(dtype_out), device=grad.device))
            diag_char = letters[i]  # Letter for this dimension

            # exprA: Diagonal Q applied to A? -> A_...i... * Q_i -> B_...i...
            piece1A.append(diag_char)  # Q_i
            piece2A += diag_char  # A_...i...
            piece3A += diag_char  # Output_...i...

            # exprGs: Calculate update term for this Q_i
            # Need outer product T = G G^T, then extract diagonal Q update term
            # File 1 uses: piece1 = indices with i->i+13; subscripts = p1, p1 -> i+13
            # This looks like einsum('...a...,...a...->a', G, G) to get diag(G G^T)_i
            # No, File 1 is einsum('...A...,...A...->A', T1, T2)? Check term calculation.
            # Let's assume exprGs is for term = einsum(exprG, A, A) or conjB, conjB
            idx_G_1 = list(letters[: len(shape)])
            idx_G_1[i] = letters[
                i + len(shape)
            ]  # Use uppercase for temporary index? No, +13 in orig. Let's use +len(shape)
            idx_G_1 = "".join(idx_G_1)
            subscripts = f"{idx_G_1},{idx_G_1}->{letters[i + len(shape)]}"  # sum over all except i -> diag update?
            exprGs.append(subscripts)

            # exprP: Apply Q'Q where Q is diagonal -> g'_...i... = Q_i * Q_i * g_...i...
            # File 1 uses piece1P=[i+13], piece2P=[i+13], piece3P=i+13, piece4P=i+13
            # This doesn't look right for Q'Q g. Let's assume Q_i, Q_i, g_...i... -> g'_...i...
            idx_P_q = letters[i + len(shape)]  # Index for Q
            idx_P_g = letters[i]  # Index for g dimension
            piece1P.append(idx_P_q)  # First Q_i
            piece2P.append(idx_P_q)  # Second Q_i
            piece3P += idx_P_g  # Input g index ...i...
            piece4P += idx_P_g  # Output g' index ...i...

        else:
            # Triangular Q: stored as 2D tensor (lower or upper?) PSGD uses upper.
            Q.append(scale_per_dim * torch.eye(size, dtype=dtype_out, device=grad.device))
            # Need indices for matrix multiplication in einsum
            char1 = letters[i]  # Original index 'a'
            char2 = letters[i + len(shape)]  # New index 'b' after Q applied
            char3 = letters[i + 2 * len(shape)]  # New index 'c' after Q' applied?

            # exprA: Update triangular Q_ab
            # File 1: piece1A=[ab], piece2A=b, piece3A=a
            # This looks like Q_ab, A_...b... -> conjB_...a... ? Matches conjB calc.
            piece1A.append(char1 + char2)  # Q_ab
            piece2A += char2  # A_...b...
            piece3A += char1  # Output_...a...

            # exprGs: Calculate update term for Q_ab (matrix T1 or T2)
            # File 1: piece1=indices i->b; piece2=indices i->c; subscripts = p1, p2 -> bc
            # Looks like einsum('...b...,...c...->bc', A, A) or conjB, conjB
            idx_G_1 = list(letters[: len(shape)])
            idx_G_1[i] = char2
            idx_G_1 = "".join(idx_G_1)
            idx_G_2 = list(letters[: len(shape)])
            idx_G_2[i] = char3
            idx_G_2 = "".join(idx_G_2)
            subscripts = f"{idx_G_1},{idx_G_2}->{char2 + char3}"  # Outer product for Q update
            exprGs.append(subscripts)

            # exprP: Apply Q'Q -> g'_...c... = sum_{a,b} Q_ba * Q_ca * g_...a... ? No, Q'*Q is matrix.
            # Apply Q' then Q. Q is upper triangular.
            # Step 1: y_...b... = sum_a Q_ab * g_...a... (Apply Q - uses char1, char2)
            # Step 2: g'_...c... = sum_b Q'_cb * y_...b... = sum_b Q_bc * y_...b... (Apply Q' - uses char2, char3)
            # File 1 uses: piece1P=[ac], piece2P=[ab], piece3P=b, piece4P=c
            # Looks like Q_ac, Q_ab, G_...b... -> G'_...c...
            # This applies Q' (Q_ac) and Q (Q_ab) together. Let's trust this.
            # Indices: Q1=ac, Q2=ab, G=...b..., Out=...c...
            piece1P.append(
                char1 + char3
            )  # Q'_ca -> Q_ac? Assumes Q symmetric? No, Q is triangular. Q'_ca = Q_ac if upper.
            piece2P.append(char1 + char2)  # Q_ab
            piece3P += char2  # Input G index ...b...
            piece4P += char3  # Output G' index ...c...

    # Finalize einsum expressions
    exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
    exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    return [Q, (exprA, tuple(exprGs), exprP)]


@decorator
def psgd_balance_Q(Q_in):
    # From File 1
    # Filter out None values before calculating norms
    valid_Q = [q for q in Q_in if q is not None]
    if not valid_Q:
        return  # Nothing to balance

    # Calculate inf norm for each valid Q
    norms = torch.stack([q.norm(float("inf")) for q in valid_Q])

    # Clamp norms to avoid issues with zero norms?
    norms = norms.clamp(min=tiny_bf16)  # Use global tiny

    # Calculate geometric mean of norms
    log_norms = norms.log()
    geometric_mean = log_norms.mean().exp()

    # Calculate scaling factors: target_norm / current_norm
    scale_factors = geometric_mean / norms
    scale_factors_list = list(scale_factors)  # Convert to list for foreach

    # Apply scaling factors using foreach_mul_
    # Need to handle None in original Q_in list
    scaled_valid_Q = torch._foreach_mul(valid_Q, scale_factors_list)  # Modify valid_Q list

    # Copy scaled results back to the original Q_in list structure
    valid_idx = 0
    for i in range(len(Q_in)):
        if Q_in[i] is not None:
            # Use copy_ instead of direct assignment if Q_in should be modified inplace
            Q_in[i].copy_(scaled_valid_Q[valid_idx])
            valid_idx += 1


@decorator_knowngood
def dampen_grad(g: Tensor, damp: float = 2**-13):
    # From File 1 - used by PSGD triangular/diag variant
    v = torch.randn_like(g)  # Random vector
    # Calculate mean absolute value of gradient
    # Use float for mean calculation stability?
    mean_abs_g = g.abs().float().mean()
    # Dampened grad = g + damp * mean(|g|) * v
    g_damp = g + damp * mean_abs_g.to(g.dtype) * v
    return v, g_damp  # Return random vector and dampened grad


def psgd_calc_A_and_conjB(exprA, G, Q, V=None):
    """Calculates A = G applied through Q, and conjB = V applied through Q^-1"""
    # From File 1 - used by PSGD triangular/diag variant
    order = G.dim()
    if order == 0:
        raise ValueError("Cannot calculate A/conjB for scalar gradient.")

    if V is None:
        V, G_damp = dampen_grad(G)  # Dampen G if V not provided
        G_compute = G_damp  # Use dampened G for A calculation
    else:
        G_compute = G  # Use original G if V is provided

    # Calculate conjB = Q^-T V (applied element-wise for diag, solve for triangular)
    # Start with V, permuted so the last dim aligns with matrix dimensions later
    # Permute: move dim 0 to end, shift others left. e.g., (0,1,2) -> (1,2,0)
    permute_dims = list(range(1, order)) + [0]
    conjB = V.permute(*permute_dims).contiguous()  # Ensure contiguous after permute
    conjB = promote(conjB)  # Promote conjB for calculations

    # Promote Q matrices and G for computation
    Q_p = [promote(q) if q is not None else None for q in Q]
    G_p = promote(G_compute)

    # Determine computation dtype
    valid_tensors = [t for t in Q_p if t is not None] + [G_p, conjB]
    md = promote(min_dtype(valid_tensors))

    # Calculate A = einsum(exprA, Qs..., G)
    try:
        einsum_args_A = [q.to(md) for q in Q_p if q is not None] + [G_p.to(md)]
        A = torch.einsum(exprA, *einsum_args_A)
        A = A.to(G_compute.dtype)  # Cast back to original compute dtype
    except Exception as e:
        shapes = [arg.shape for arg in einsum_args_A]
        raise RuntimeError(
            f"Einsum failed for A in psgd_calc_A_and_conjB with expr '{exprA}', shapes {shapes}. Error: {e}"
        )

    # Calculate conjB = Q^-T V by applying inverse transpose of each Q
    # Iterate through Q matrices (promoted)
    for i, q in enumerate(Q_p):
        if q is None:  # Skip if no preconditioning for this dim
            # Should we transpose conjB even if Q is None? Yes, to maintain structure.
            if i < order - 1:
                conjB = torch.transpose(conjB, i, order - 1)
            continue

        q = q.to(md)  # Ensure q is in compute dtype

        if q.dim() <= 1:  # Diagonal Q: inv(Q^T) = inv(Q) = 1/Q
            # Apply division carefully, ensure broadcasting works.
            # conjB has shape e.g., (d1, d2, d0). We need to divide by q (shape d0).
            # Unsqueeze q to match the last dim of conjB.
            q_unsqueezed = q.view([1] * i + [-1])  # Shape (1, 1, d0) for i=2
            conjB = conjB / q_unsqueezed.clamp(min=tiny_bf16)  # Clamp denominator
        else:  # Triangular Q (assume upper)
            # Apply inv(Q^T) = inv(Q)^T. Solve Q^T x = conjB_slice ?
            # Or solve Q y = conjB_slice, then transpose?
            # PSGD paper uses solve_triangular(Q, ..., upper=True, left=False)
            # This solves X Q = B -> X = B Q^-1. Matches applying inv(Q).
            # We need inv(Q^T). Since Q is upper, Q^T is lower.
            # Let's follow File 1's solve_triangular call structure.
            # It solves X * Q = B where X has shape (..., N), Q is (N, N), B has shape (..., N)
            # Input conjB needs to be reshaped to (Batch, N) where N = q.size(0)
            num_features = q.size(0)
            conjB_reshaped = conjB.reshape(-1, num_features)
            try:
                # left=False solves X @ Q = B for X
                # upper=True means Q is upper triangular
                solved_X = torch.linalg.solve_triangular(
                    q, conjB_reshaped.T, upper=True, left=True, unitriangular=False
                ).T
                # Original used left=False, which solves Q @ X = B. Let's retry that.
                # conjB is effectively B. We want X = Q^-1 B.
                # Let's check PSGD_torch source if possible. File 1 has left=False.
                # If left=False: Solves Q @ X.T = B.T -> X.T = Q^-1 B.T -> X = B @ Q^-T. This is what we want!
                solved_X_T = torch.linalg.solve_triangular(
                    q, conjB_reshaped.T, upper=True, left=False, unitriangular=False
                )
                solved_X = solved_X_T.T  # Transpose result to get B @ Q^-T

            except torch.linalg.LinAlgError as e:
                raise RuntimeError(f"solve_triangular failed for conjB update (dim {i}, shape {q.shape}): {e}")

            # Reshape back to original conjB structure (permuted)
            conjB = solved_X.reshape_as(conjB)

        # Transpose conjB to prepare for the next dimension's solve
        # Moves current dim 'i' to the end, shifts 'i+1' to 'order-1' left.
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)  # Use transpose, not permute

    # Cast final A and conjB back to original G's dtype? Or compute dtype?
    # Let's return in compute dtype (md) as they are intermediates.
    return A.to(md), conjB.to(md)  # Return promoted results


def psgd_lb(A, max_abs):
    """Lower bound calculation for PSGD update step."""
    # From File 1 - used by PSGD triangular/diag variant
    # A is likely conjB * conjB term (symmetric)
    # Normalize A by max_abs
    A_norm = A / max_abs.clamp(min=tiny_bf16)  # Clamp denominator

    # Power iteration to estimate largest eigenvalue (lambda_max)
    # a0 = column norms squared = diag(A^T A)? No, einsum("ij,ij->j", A, A) is sum_i A_ij^2
    # This assumes A comes from einsum(exprG, conjB, conjB) -> shape (N,N)?
    if A_norm.dim() != 2:
        raise ValueError("psgd_lb expects a 2D matrix.")

    # Start power iteration with a vector based on max column norm
    col_norm_sq = torch.einsum("ij,ij->j", A_norm, A_norm)
    i = torch.argmax(col_norm_sq)  # Index of column with largest norm
    x = A_norm[:, i].contiguous()  # Initial vector x (shape N)

    # One step of power iteration: x = A * x / || A * x || -> estimate dominant eigenvector
    # File 1 does: x = A^T x, normalize, x = A x, norm -> lambda_max estimate
    x = torch.einsum(
        "i,ij->j", x, A_norm
    )  # x = x^T @ A (A assumed symmetric here?) Or just matvec? Matvec is einsum("ij,j->i", A, x)
    # Let's assume File 1 intended x = A^T @ x
    x_norm = x.norm().clamp(min=tiny_bf16)
    x = x / x_norm  # Normalize vector

    # Apply A again: y = A @ x
    y = torch.einsum("ij,j->i", A_norm, x)  # y = A @ x

    # Estimate eigenvalue lambda = ||y|| = ||Ax|| (Rayleigh quotient x'Ax/x'x simplifies as ||x||=1)
    lambda_max_est = y.norm()

    # Rescale by original max_abs
    lower_bound = lambda_max_est * max_abs
    return lower_bound


@decorator  # Keep original decorator
def psgd_update_precond(Q, exprs, G, precond_lr, oq, store_triu_as_line, V):
    """Update Kronecker product preconditioner Q (Diagonal/Triangular) with pair (V, G)."""
    # From File 1 - used by PSGD triangular/diag variant
    exprA, exprGs, _ = exprs  # Unpack einsum expressions

    # Calculate A and conjB intermediates
    try:
        # Ensure G, V are compatible and Q matches G dim
        if len(Q) != G.dim():
            raise ValueError("Q length must match G dim.")
        if V is not None and V.shape != G.shape:
            raise ValueError("V shape must match G shape.")

        A, conjB = psgd_calc_A_and_conjB(exprA, G, Q, V)
    except Exception as e:
        warn_once(f"Failed to calculate A/conjB in psgd_update_precond: {e}. Skipping update.")
        return

    # Iterate through each Q matrix and update
    for i, (q, exprG, o) in enumerate(zip(Q, exprGs, oq)):
        if q is None:
            continue  # Skip if this dimension isn't preconditioned

        # Calculate update terms T1 and T2 using einsum
        try:
            # Ensure A, conjB have compatible dimensions for exprG
            term1 = promote(torch.einsum(exprG, A, A))  # T1 = einsum(exprG, A, A)
            term2 = promote(torch.einsum(exprG, conjB, conjB))  # T2 = einsum(exprG, conjB, conjB)
        except Exception as e:
            warn_once(
                f"Einsum failed for terms in psgd_update_precond (dim {i}, expr '{exprG}'): {e}. Skipping update for this Q."
            )
            continue

        # Calculate update delta: delta = precond_lr * (T1 - T2) / lb(T1 + T2) * Q
        T_sum = term1 + term2
        T_diff = term1 - term2

        # Calculate lower bound for normalization
        norm_inf = T_sum.norm(float("inf")).clamp(min=tiny_bf16)
        if norm_inf == 0:  # Avoid division by zero if T_sum is zero
            # If T_sum is zero, T1=T2=0? Or T1=-T2? If T1=-T2, delta is proportional to T1.
            # If T_sum=0, norm_inf=0, lb=0. Update becomes infinite/NaN. Skip update?
            warnings.warn(f"T1+T2 norm is zero for Q dim {i}. Skipping update.")
            continue

        if q.dim() < 2:  # Diagonal Q
            lower_bound = norm_inf  # Use simple norm for diagonal
        else:  # Triangular Q
            lower_bound = psgd_lb(T_sum, norm_inf)  # Use power iteration estimate

        # Calculate scaled difference
        update_delta = precond_lr * T_diff / lower_bound.clamp(min=tiny_bf16)

        # Apply pre-multiplication by Q for triangular case
        if q.dim() >= 2:
            # Ensure dtypes match for mm
            q_compute = q.to(update_delta.dtype)
            update_delta = torch.mm(update_delta, q_compute)
            # Keep only upper triangle for update
            torch.triu(update_delta, out=update_delta)  # Modify inplace

        # Prepare state 'o' (which is the Q state, potentially in line format)
        q_state = o  # o is the state tensor (or tuple for line format)

        # Convert update_delta to line format if necessary
        if store_triu_as_line:
            if q.dim() >= 2:
                # Extract upper triangle indices
                indices = torch.triu_indices(*update_delta.shape, device=update_delta.device)
                update_delta_line = update_delta[tuple(indices)]
                # Ensure state 'o' is also in line format (o should be (shape_info, line_tensor))
                if not isinstance(o, tuple) or len(o) != 2:
                    raise TypeError("State 'o' must be tuple (shape, line_tensor) if store_triu_as_line=True")
                q_state = o[1]  # Get the line tensor from state
                update_delta_to_apply = update_delta_line
            else:  # Diagonal case
                if not isinstance(o, torch.Tensor):
                    raise TypeError("State 'o' must be tensor for diagonal Q if store_triu_as_line=True")
                q_state = o
                update_delta_to_apply = update_delta
        else:  # Not storing as line
            if not isinstance(o, torch.Tensor):
                raise TypeError("State 'o' must be tensor if store_triu_as_line=False")
            q_state = o
            update_delta_to_apply = update_delta

        # Apply update: q_state = q_state - update_delta_to_apply
        stochastic_add_([q_state], [update_delta_to_apply], alpha=-1)


# --- Optimizer Class and EMA (From File 1, _handle_closure kept from File 1) ---


def modify_closure(closure):
    """
    Modifies the closure function to use create_graph=True in backward().
    (From File 1) - Used by original _handle_closure.
    """
    if not callable(closure):
        raise TypeError("closure must be a callable function.")

    # Use context manager for patching if available and cleaner
    # Requires `from unittest.mock import patch`
    @functools.wraps(closure)
    def wrapper(*args, **kwargs):
        original_backward = torch.Tensor.backward

        def patched_backward(self, *b_args, **b_kwargs):
            # print("Using patched backward with create_graph=True") # Debug print
            b_kwargs["create_graph"] = True
            return original_backward(self, *b_args, **b_kwargs)

        with patch.object(torch.Tensor, "backward", patched_backward):
            result = closure(*args, **kwargs)
        return result

    return wrapper  # Return the wrapped closure


def merge_group(group, *tensors):
    # From File 1/2 - Handles dim merging based on group settings
    # Check if merging is enabled
    if not group.get("merge_dims", False):
        # If only one tensor passed, return it directly, else return tuple
        return tensors[0] if len(tensors) == 1 else tensors

    # Handle list of tensors input (e.g., merge_group(g, list_of_grads))
    if isinstance(tensors[0], list):
        # Apply merge_group to each tensor in the list
        # This seems wrong. If input is (group, [t1, t2]), it should merge t1 and t2?
        # Let's assume input is always (group, t1, t2, ...)
        pass  # Continue to process tensors individually

    # Determine max_precond_dim from group settings
    max_dim = group.get("max_size_triangular")  # Check PSGD specific key first
    if max_dim is None:
        max_dim = group.get("max_precond_dim")  # Fallback to general key
    if max_dim is None:
        raise ValueError("Missing 'max_size_triangular' or 'max_precond_dim' in group for merge_group.")

    split = group.get("split", False)  # Get split flag

    # Process each tensor
    out = []
    for t in tensors:
        if t is None:  # Handle None tensors
            append_or_extend(out, None)  # Add None placeholder
        else:
            merged_or_split = dim_merger(t, max_dim, split)
            append_or_extend(out, merged_or_split)

    # Return structure depends on input and output of dim_merger
    # dim_merger returns Tensor if not split, List[Tensor] if split.
    # append_or_extend builds a flat list 'out'.
    # We need to reconstruct the original structure if multiple tensors were passed.

    # Let's simplify: Apply dim_merger to each tensor individually.
    results = []
    for t in tensors:
        if t is None:
            results.append(None)
        else:
            results.append(dim_merger(t, max_dim, split))

    # If input was single tensor, return single result (tensor or list)
    if len(tensors) == 1:
        return results[0]
    else:
        # If multiple tensors input, return tuple of results
        return tuple(results)


class StatefulOptimizer(torch.optim.Optimizer):
    """
    Base optimizer class from File 1, with minor clarifications.
    _handle_closure is the ORIGINAL version from File 1.
    """

    ema_decay: float = 0.001
    compile_step: bool = False
    hessian_approx: bool = False  # Controls whether HVP is computed
    precond_schedule: Union[Callable, float, None] = None  # Schedule for preconditioner updates
    stochastic_schedule: bool = False  # Use stochastic application of schedule probability
    finite_differences: bool = False  # Use finite differences for HVP (if hessian_approx=True)

    def __init__(self, params, defaults, foreach: bool = True, use_ema: bool = False):
        super().__init__(params, {**defaults, "foreach": foreach})
        self.use_ema = use_ema
        self.mapping = {}  # For merged dimensions?
        self._inner_group = {"stochastic_schedule": self.stochastic_schedule}  # State for psgd_should_update
        self._precond_rng = random.Random(0x12312)  # RNG for stochastic schedule
        self._is_preconditioning = False  # Flag set during step

        if self.hessian_approx and self.compile_step:
            # compile_step likely refers to torch.compile(optimizer.step)
            # HVP calculation often involves graph breaks or complex backward calls
            # which might interfere with compiling the step function.
            warnings.warn(
                "Using hessian_approx=True with compile_step=True might lead to issues. Proceed with caution."
            )
            # raise ValueError("Hessian approximation can't be used with compile_step.") # Keep warning from File 1

    def get_groups(self, group):
        # Seems intended to allow splitting a group? Default is no split.
        return [group]

    def state_(self, arg: Tensor):
        # Convenience method to access state for a parameter
        return self.state[arg]

    def mars_correct_list(self, group, p_list, g_list, mars_gamma, beta):
        # Applies MARS correction to a list of gradients
        for p, g in zip(p_list, g_list):
            if g is None:
                continue  # Skip if grad is None
            state = self.state_(p)
            if "mars_old_grad" not in state:
                # Initialize previous gradient state if needed
                state["mars_old_grad"] = torch.zeros_like(g)
            elif state["mars_old_grad"].shape != g.shape:
                # Handle shape changes if they occur mid-training
                warnings.warn(f"Shape mismatch for MARS old_grad for param {id(p)}. Reinitializing.")
                state["mars_old_grad"] = torch.zeros_like(g)

        # Filter out None grads before collecting old_gs
        valid_indices = [i for i, g in enumerate(g_list) if g is not None]
        if not valid_indices:
            return  # Nothing to correct

        p_list_valid = [p_list[i] for i in valid_indices]
        g_list_valid = [g_list[i] for i in valid_indices]
        old_gs = [self.state_(p)["mars_old_grad"] for p in p_list_valid]

        # Apply correction (modifies g_list_valid inplace)
        mars_correction(g_list_valid, old_gs, mars_gamma, beta)

    def split_p_and_g_in_group(
        self, group: dict, skip_none: bool = True, should_promote: bool = True, beta1: float = -1.0
    ):
        # Yields parameter-gradient pairs, potentially handling merged dimensions
        # and MARS correction.
        for p in group["params"]:
            if p.grad is None and skip_none:  # Skip parameter if it has no gradient
                continue

            # Handle dimension merging if enabled in the group
            if p in self.mapping:
                # Use cached merged views if available
                p_views = self.mapping[p]
            else:
                # Merge parameter dimensions based on group settings
                merged_p = merge_group(group, p)
                # Store merged views (tensor or list)
                self.mapping[p] = p_views = list_guard(merged_p)[0]  # Ensure it's a list

            # Get gradient (might be None)
            grad = getattr(p, "grad", None)
            # Detach gradient from parameter to allow modification? No, just clear it.
            # p.grad = None # Clear original grad reference after getting it

            # Merge gradient dimensions similarly to parameter
            if grad is None:
                # If original grad is None, try getting grads from merged views? Unlikely.
                # Assume grad corresponds to original param 'p'
                # If p was merged, grad needs corresponding merging.
                # If grad is None, merged grad is also None (or list of Nones)
                merged_grad = merge_group(group, grad)  # Should handle None input
                grad_views = list_guard(merged_grad)[0]  # Ensure list structure
            else:
                # Ensure gradient requires grad if needed for HVP later? No, handle in closure.
                merged_grad = merge_group(group, grad)
                grad_views = list_guard(merged_grad)[0]

            # Ensure param views and grad views align
            if len(p_views) != len(grad_views):
                raise RuntimeError(
                    f"Mismatch between merged parameter views ({len(p_views)}) and gradient views ({len(grad_views)})"
                )

            # Iterate through potentially merged views
            for pv, gv in zip(p_views, grad_views):
                if skip_none and gv is None:
                    continue

                g_processed = gv  # Start with the gradient view

                # Promote gradient if requested (usually for compute precision)
                if should_promote and g_processed is not None:
                    g_processed = promote(g_processed)

                # Apply MARS correction if enabled and beta1 is provided
                if beta1 >= 0 and group.get("mars", False) and g_processed is not None:
                    # mars_correct_list expects lists
                    self.mars_correct_list(group, [pv], [g_processed], group["mars_gamma"], beta1)
                    # Note: mars_correction modifies g_processed inplace

                yield pv, g_processed  # Yield parameter view and processed gradient view

    def state_size(self) -> int:
        # Calculate total size of optimizer state tensors in bytes
        total_bytes = 0

        def _add(x):
            nonlocal total_bytes
            if isinstance(x, Tensor):
                total_bytes += x.numel() * x.element_size()

        processed_params = set()  # Avoid double counting if group splitting happens
        for group in self.param_groups:
            # Use split_p_and_g? No, just iterate params and access state.
            for p in group["params"]:
                if p in processed_params:
                    continue
                processed_params.add(p)
                if p in self.state:
                    tree_map(_add, self.state[p])  # Apply _add recursively to state dict

        return total_bytes

    def _step(self, group):
        # Placeholder for optimizer specific logic
        raise NotImplementedError

    def ema_update(self):
        # Update EMA of parameters if use_ema=True
        if not self.use_ema:
            return

        with torch.no_grad():
            for group in self.param_groups:
                active_p = [
                    p for p in group["params"] if p.grad is not None
                ]  # Only update EMA for params with grads? Or all? Let's do all.
                active_p = group["params"]

                if not active_p:
                    continue  # Skip group if no parameters

                # Increment EMA step counter for debiasing
                k = group["ema_step"] = group.get("ema_step", -1) + 1
                ema_beta = 1.0 - self.ema_decay  # Beta for EMA update lerp

                # Initialize EMA state if missing
                for p in active_p:
                    if "param_ema" not in self.state_(p):
                        self.state_(p)["param_ema"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    elif self.state_(p)["param_ema"].shape != p.data.shape:
                        # Handle shape changes
                        warnings.warn(f"Shape mismatch for EMA state param {id(p)}. Reinitializing.")
                        self.state_(p)["param_ema"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                # Get current params and their EMAs
                params_data = [p.data for p in active_p]
                emas_data = [self.state_(p)["param_ema"] for p in active_p]

                # Calculate debiased weight for lerp
                # weight = beta_debias(ema_beta, k + 1) # weight for *current* param p
                # _foreach_lerp(a, b, weight) -> a = a*(1-weight) + b*weight
                # We want: ema = ema * ema_beta + p * (1 - ema_beta)
                # So call: _foreach_lerp(ema, p, weight = 1 - ema_beta = ema_decay)
                # Need debiased version of ema_decay?
                # beta_debias(beta, step) = 1 - (1-beta)/(1-beta^step)
                # Let target decay be D (ema_decay). Let beta = 1-D.
                # Debias factor for param 'p' is D / (1 - (1-D)^(k+1))
                debias_factor = self.ema_decay / (1.0 - ema_beta ** (k + 1))
                # Ensure factor is finite
                debias_factor = min(debias_factor, 1.0) if math.isfinite(debias_factor) else 1.0

                # Use foreach_lerp_: Updates first argument inplace
                torch._foreach_lerp_(emas_data, params_data, weight=debias_factor)

    def copy_emas_to_params(self):
        # Swap EMA values into parameters (e.g., for evaluation)
        with torch.no_grad():
            for group in self.param_groups:
                active_p = [p for p in group["params"] if "param_ema" in self.state_(p)]
                if not active_p:
                    continue

                for p in active_p:
                    # Swap p.data and state['param_ema']
                    p_clone = p.data.clone()
                    set_(p.data, self.state_(p)["param_ema"])  # p.data = ema
                    set_(self.state_(p)["param_ema"], p_clone)  # ema = old p.data

    def copy_params_to_emas(self):
        # Copy current parameters into EMA state (e.g., to reset EMA)
        with torch.no_grad():
            for group in self.param_groups:
                active_p = [p for p in group["params"]]  # Copy for all params in group
                if not active_p:
                    continue

                for p in active_p:
                    if "param_ema" in self.state_(p):
                        # Copy p.data into state['param_ema']
                        set_(self.state_(p)["param_ema"], p.data)
                    else:
                        # Initialize EMA state with current param data if not present
                        self.state_(p)["param_ema"] = p.data.clone()

    # --- Using ORIGINAL _handle_closure from File 1 ---
    def _handle_closure(self, closure):
        hessian_approx = self.hessian_approx  # Use instance attribute

        if closure is None:
            if hessian_approx:
                # File 1 behavior: Raise error if HVP needed but no closure
                raise ValueError("Hessian approximation requires a closure.")
            return None  # No closure, no loss, no HVP

        # --- HVP Calculation Logic (Original File 1) ---
        if not hessian_approx:
            # Standard gradient calculation
            with torch.enable_grad():
                loss = closure()
            return loss

        # --- HVP Calculation Enabled ---
        if self.finite_differences:
            # Store original parameter data and gradients
            original_data = {}
            original_grads = {}
            param_list = []  # List of params involved
            for group in self.param_groups:
                for p, g in self.split_p_and_g_in_group(group, skip_none=False, should_promote=False):
                    # Only consider params with gradients for HVP? Or all? Assume all for now.
                    if p not in original_data:
                        param_list.append(p)
                        original_data[p] = p.data.clone()
                        original_grads[p] = g.clone() if g is not None else None
                        # Generate random vector v
                        p.vector = torch.randn_like(p)  # Store vector on param object
                        # Perturb parameter: p = p + v * eps^0.5
                        delta_scale = torch.finfo(p.dtype).eps ** 0.5
                        stochastic_add_(p.data, p.vector, delta_scale)  # Modify p.data inplace

            # Calculate loss and gradients at perturbed point
            with torch.enable_grad():
                loss = closure()  # This closure uses perturbed params

            # Calculate HVP using finite difference
            for p in param_list:
                # Get new gradient (g_perturbed)
                g_perturbed = getattr(p, "grad", None)  # Get grad calculated by closure
                # Need to merge grad if necessary? Assume closure handles grads correctly.
                # Let's re-fetch using split_p_and_g for consistency? No, closure sets p.grad.
                # Assume p.grad is now the perturbed gradient.

                g_orig = original_grads[p]

                if g_perturbed is None or g_orig is None:
                    # Handle cases where grad becomes None or was None
                    p.hessian_vector = torch.zeros_like(p.data)  # Set HVP to zero?
                else:
                    # HVP approx: (g_perturbed - g_original) / delta_scale
                    delta_scale = torch.finfo(p.dtype).eps ** 0.5
                    hvp = (g_perturbed - g_orig) / delta_scale
                    p.hessian_vector = hvp  # Store HVP on param object

                # Restore original parameter data
                p.data.copy_(original_data[p])
                # Restore original gradient? Or keep perturbed one? File 1 code implies restoring.
                # Let's re-set p.grad to the *original* gradient for the subsequent optimizer step.
                setattr(p, "grad", g_orig)
                # del p.orig # File 1 had p.orig, let's assume original_data serves this purpose

        else:  # Exact HVP using autograd.grad
            # Step 1: Compute gradients with create_graph=True
            with torch.enable_grad():
                # Use modify_closure to patch backward call within the user's closure
                loss = modify_closure(closure)()  # Ensure closure is called

            # Prepare for autograd.grad: Collect params with grads and the grads themselves
            params_with_grad = []
            grads_to_diff = []
            for group in self.param_groups:
                # Use split_p_and_g to get potentially merged grads?
                # Or just use p.grad set by modify_closure? Let's use p.grad.
                for p in group["params"]:
                    if p.grad is not None:
                        # Ensure grad requires grad for HVP calculation
                        if not p.grad.requires_grad:
                            # This should ideally happen due to create_graph=True
                            warnings.warn(
                                f"Gradient for param {id(p)} does not require grad after closure with create_graph=True. HVP might be zero."
                            )
                            # Force requires_grad? Might break things if graph was already detached.
                            # p.grad.requires_grad_(True) # Avoid this for now.

                        params_with_grad.append(p)
                        grads_to_diff.append(p.grad)
                    else:
                        # If grad is None, can't compute HVP involving it.
                        pass

            if not params_with_grad:
                warnings.warn("No parameters with gradients found for exact HVP calculation.")
                return loss  # Return loss, HVP attributes won't be set

            # Step 2: Generate random vectors v
            vs = [torch.randn_like(p) for p in params_with_grad]

            # Step 3: Compute HVP: torch.autograd.grad(grads, params, vs)
            with torch.enable_grad():  # Ensure grad is enabled for HVP calc
                try:
                    hvs = torch.autograd.grad(
                        outputs=grads_to_diff,
                        inputs=params_with_grad,
                        grad_outputs=vs,
                        create_graph=False,  # Usually False is sufficient here, don't need HVP of HVP
                        allow_unused=True,  # Allow if some params don't affect grads
                    )
                except RuntimeError as e:
                    warnings.warn(f"autograd.grad failed for HVP calculation: {e}. Setting HVP to zero.")
                    hvs = [None] * len(params_with_grad)  # Set HVP to None on failure

            # Step 4: Store results (v and hv) on parameter objects
            for p, v, hv in zip(params_with_grad, vs, hvs):
                p.vector = v
                # Handle case where HVP is None (due to allow_unused or error)
                p.hessian_vector = hv if hv is not None else torch.zeros_like(p.data)
                # Keep p.grad as it was calculated by modify_closure

        return loss  # Return loss calculated by closure

    def step(self, closure: Optional[Callable] = None):
        # Determine if preconditioning update should happen this step
        if self.precond_schedule is None:
            self._is_preconditioning = False
        else:
            # Use psgd_should_update to check schedule
            self._is_preconditioning = psgd_should_update(self._inner_group, self.precond_schedule, self._precond_rng)

        # --- Handle closure and HVP calculation (if needed) ---
        # _handle_closure calculates loss and computes+stores HVP (v, hv)
        # on param objects if self.hessian_approx and self._is_preconditioning.
        loss = self._handle_closure(closure)

        # --- Perform optimizer step ---
        # Disable dynamo cache limit during step? Seems unusual but keep from File 1.
        with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
            # Clear grads before step? Or after? Standard is zero_grad() before closure.
            # Optimizer step assumes grads are populated. Let's not zero here.

            for group in self.param_groups:
                # Pass preconditioning flag to group for use in _step logic
                group["is_preconditioning"] = self._is_preconditioning
                # Call the specific optimizer's step implementation
                self._step(group)

                # Perform EMA update after parameter update if enabled
                if self.use_ema:
                    # Pass group to ema_update? No, ema_update iterates groups.
                    pass  # ema_update called below outside loop? No, inside.

            # Call EMA update once after processing all groups in the step?
            # File 1 has it inside the group loop. Let's keep it there.
            if self.use_ema:
                self.ema_update()  # Should this be inside the group loop or outside? File 1 is inside.

        # --- Cleanup ---
        # Gradients should typically be zeroed by the user *before* the next forward pass.
        # self.zero_grad() # Optionally zero here, but less flexible for user.

        # Clear HVP info stored on params? Or keep until next HVP calculation?
        # Let's clear them here to avoid stale info if HVP isn't calc'd next step.
        # Only clear if HVP was potentially calculated this step.
        if self.hessian_approx:
            for group in self.param_groups:
                for p in group["params"]:
                    if hasattr(p, "vector"):
                        delattr(p, "vector")
                    if hasattr(p, "hessian_vector"):
                        delattr(p, "hessian_vector")
                    # if hasattr(p, "orig"): delattr(p, "orig") # If finite diff used this

        return loss


# --- Hooking Utilities (From File 1/2) ---


def hook_optimizer_into_model(model, optimizer_cls, *args, **kwargs):
    # Creates separate optimizer instance for each param and hooks into backward.
    optimizers = {}

    def _step_hook(param):  # Hook receives the parameter itself
        # Get optimizer instance for this specific parameter
        if param not in optimizers:
            warnings.warn(f"Optimizer hook called for parameter {id(param)} not found in optimizers dict.")
            return
        opt_instance = optimizers[param]
        # Perform step and zero grad for this single parameter
        opt_instance.step()
        opt_instance.zero_grad(set_to_none=True)  # Use set_to_none=True generally preferred

    # Iterate through model parameters
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # Skip params that don't require gradients

        # Create a dedicated optimizer instance for this parameter
        # Need to ensure optimizer_cls takes params as first arg
        optimizers[p] = optimizer_cls([p], *args, **kwargs)
        # Register the hook
        p.register_post_accumulate_grad_hook(_step_hook)
        # Note: register_hook vs register_post_accumulate_grad_hook
        # post_accumulate runs after gradient accumulation (good for DDP)

    return optimizers  # Return dict mapping param -> its optimizer instance


def fused_hook(parameters, optimizer_cls, *args, **kwargs):
    # Creates one optimizer for all params, steps when last grad arrives.
    # Ensure parameters is a list/tuple
    parameters = list(parameters)
    param_count = len(parameters)
    if param_count == 0:
        return None  # Nothing to optimize

    # Set to track which parameters have received grads in this cycle
    seen_params_in_cycle = set()

    # Create a single optimizer instance for all parameters
    optimizer_instance = optimizer_cls(parameters, *args, **kwargs)

    # Store original step function and replace it with a warning
    original_step_fn = optimizer_instance.step

    def _step_warning_wrapper(*s_args, **s_kwargs):
        warn_once(
            "Attempting to call step() directly on an optimizer managed by fused_hook. Step is handled automatically by the hook."
        )
        # Optionally call original step? No, hook manages it.
        # return original_step_fn(*s_args, **s_kwargs)

    optimizer_instance.step = _step_warning_wrapper

    def _fused_step_hook(param):
        # Add current parameter to the set for this cycle
        seen_params_in_cycle.add(param)

        # Check if all parameters have received their gradients
        if len(seen_params_in_cycle) == param_count:
            # All grads accumulated, perform optimizer step
            try:
                original_step_fn()  # Call original step method
            except Exception as e:
                warnings.warn(f"Error during optimizer step in fused_hook: {e}")
                # Potentially raise error or handle differently
            finally:
                # Zero gradients for the *next* iteration
                optimizer_instance.zero_grad(set_to_none=True)
                # Clear the set for the next accumulation cycle
                seen_params_in_cycle.clear()

    # Register the hook for each parameter
    for p in parameters:
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(_fused_step_hook)

    return optimizer_instance  # Return the single optimizer instance
