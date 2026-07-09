from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

aten = torch.ops.aten


def _overloads(op: Any) -> set:
    """All overloads of an aten op packet (add -> {add, add.Tensor, add.Scalar})."""
    ovs = {op}
    if hasattr(op, "op_overloads"):
        ovs.update(op.op_overloads())
    else:
        ovs.update(getattr(op, o) for o in op.overloads())
    return ovs


_ADD = _overloads(aten.add)
_SUB = _overloads(aten.sub)
_MUL = _overloads(aten.mul)
_RSUB = _overloads(aten.rsub)
_POW = _overloads(aten.pow)


def _arg(node: torch.fx.Node, i: int, name: str, default: Any = None) -> Any:
    return node.args[i] if len(node.args) > i else node.kwargs.get(name, default)


def _call(graph: torch.fx.Graph, target: Any, args: tuple, meta: dict,
          kwargs: dict | None = None) -> torch.fx.Node:
    n = graph.call_function(target, args, kwargs or {})
    n.meta.update(meta)
    return n


def _is_float_tensor(x: Any) -> bool:
    v = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
    return isinstance(v, torch.Tensor) and v.dtype.is_floating_point


# ── 1 - x**y  ->  -expm1(y * log1p(x - 1)) ────────────────────────────────────

def stable_one_minus_pow(graph: torch.fx.Graph) -> int:
    """Rewrite ``1 - x**y`` (0-d float scalar ``x``) to ``where(x > 0, stable, original)``.

    Exact identity, numerically stable when ``x**y`` is near 1 -- the bias-correction
    cancellation ``1 - beta**step`` (beta in (0,1)): ~100-1000x smaller fp32 error.
    The fallback preserves PyTorch pow semantics for bases where the log identity is
    not valid (negative base, 0**0, etc.).
    """
    count = 0
    for node in list(graph.nodes):
        # Match 1 - x**y (the bias-correction cancellation) in either dynamo form.
        if node.target == torch.ops.aten.rsub.Scalar:
            if _arg(node, 2, "alpha", 1) != 1:
                continue
            c = _arg(node, 1, "other", 1)
            inner = _arg(node, 0, "input")
        elif node.target == torch.ops.aten.sub.Tensor:
            if _arg(node, 2, "alpha", 1) != 1:
                continue
            c = _arg(node, 0, "input")
            inner = _arg(node, 1, "other")
        else:
            continue
        if c != 1:  # only 1 - x**y has the near-cancellation worth the rewrite
            continue
        if not (isinstance(inner, torch.fx.Node) and inner.op == "call_function"
                and inner.target in _POW and len(inner.users) == 1):
            continue
        base = _arg(inner, 0, "input")
        bv = base.meta.get("val") if isinstance(base, torch.fx.Node) else None
        if not (isinstance(bv, torch.Tensor) and bv.dim() == 0 and bv.dtype.is_floating_point):
            continue
        expn = _arg(inner, 1, "exponent")
        with graph.inserting_before(node):
            pos = _call(graph, aten.gt.Scalar, (base, 0), {**inner.meta, "val": bv > 0})
            fallback = _call(graph, aten.sub.Tensor, (1, inner), node.meta)
            t = _call(graph, aten.sub.Tensor, (base, 1), inner.meta)
            t = _call(graph, aten.log1p.default, (t,), inner.meta)
            t = _call(graph, aten.mul.Tensor, (expn, t), inner.meta)
            stable = _call(graph, aten.neg.default, (_call(graph, aten.expm1.default, (t,), inner.meta),), node.meta)
            built = _call(graph, aten.where.self, (pos, stable, fallback), node.meta)
        node.replace_all_uses_with(built)
        graph.erase_node(node)
        count += 1

    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


# ── 1 - (1 - x) -> x ─────────────────────────────────────────────────────────

def _one_minus_arg(node: Any) -> Any | None:
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if _arg(node, 2, "alpha", 1) != 1:
        return None
    if node.target in _RSUB:
        return _arg(node, 0, "input") if _arg(node, 1, "other", 1) == 1 else None
    if node.target in _SUB:
        return _arg(node, 1, "other") if _arg(node, 0, "input") == 1 else None
    return None


def cancel_double_one_minus(graph: torch.fx.Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or not _is_float_tensor(node):
            continue
        inner = _one_minus_arg(node)
        x = _one_minus_arg(inner)
        if not (isinstance(inner, torch.fx.Node) and len(inner.users) == 1
                and isinstance(x, torch.fx.Node) and _is_float_tensor(x)):
            continue
        node.replace_all_uses_with(x)
        graph.erase_node(node)
        count += 1

    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


# ── mul + add -> fma ──────────────────────────────────────────────────────────

def _is_mul_operand(x: Any) -> bool:
    """fma multiply input: a tensor node of any dtype (the mul's float result puts the
    arithmetic in the float domain) or a python scalar."""
    if isinstance(x, torch.fx.Node):
        return isinstance(x.meta.get("val"), torch.Tensor)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_addend(x: Any) -> bool:
    """fma addend: a float tensor node or a python scalar."""
    if isinstance(x, torch.fx.Node):
        return _is_float_tensor(x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _single_user_mul(node: Any) -> tuple[Any, Any] | None:
    """A float-result mul with one user; returns its operands, else None.

    An all-integer mul has an integer result and is rejected, so integer arithmetic
    stays exact.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target in _MUL and len(node.users) == 1 and _is_float_tensor(node)):
        return None
    a, b = _arg(node, 0, "input"), _arg(node, 1, "other")
    return (a, b) if _is_mul_operand(a) and _is_mul_operand(b) else None


def _neg(graph: torch.fx.Graph, x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        return _call(graph, aten.neg.default, (x,), x.meta)
    return -x


def fuse_mul_add_to_fma(graph: torch.fx.Graph) -> int:
    """Rewrite ``add``/``sub``/``rsub`` of a product into ``prims.fma`` so Inductor
    emits one ``tl.fma`` (one rounding, deterministic) instead of leaving mul+add
    fusion to Triton's register-pressure heuristic.

    Each add/sub/rsub is normalized to ``lhs + scale*rhs`` (scale a python scalar).
    A single-user mul on either side is the product; a scalar ``scale`` with no mul
    becomes ``fma(rhs, scale, lhs)``. A sign flip negates a scalar operand when free,
    else a tensor operand.
    """
    from torch._inductor import inductor_prims
    fma = getattr(inductor_prims, "fma", None)
    if fma is None:
        return 0

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or not _is_float_tensor(node):
            continue
        t = node.target
        if t in _ADD:
            lhs, rhs, alpha, sgn = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), 1
        elif t in _SUB:
            lhs, rhs, alpha, sgn = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), -1
        elif t in _RSUB:
            # rsub.Scalar/Tensor(input, other, alpha) = other - alpha*input  (same semantics).
            rhs, lhs, alpha, sgn = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), -1
        else:
            continue
        if not isinstance(alpha, (int, float)):
            continue
        scale = sgn * alpha

        lhs_mul = _single_user_mul(lhs)
        rhs_mul = _single_user_mul(rhs)
        flip_product = flip_addend = False
        if rhs_mul and abs(scale) == 1:
            a, b, addend = rhs_mul[0], rhs_mul[1], lhs
            flip_product = scale == -1
        elif lhs_mul and abs(scale) == 1:
            a, b, addend = lhs_mul[0], lhs_mul[1], rhs
            flip_addend = scale == -1
        elif abs(scale) != 1 and not (lhs_mul or rhs_mul):
            a, b, addend = rhs, scale, lhs
            if not _is_mul_operand(a):
                continue
        else:
            continue
        if not _is_addend(addend):
            continue

        with graph.inserting_before(node):
            if flip_product:
                if not isinstance(a, torch.fx.Node):
                    a = -a
                elif not isinstance(b, torch.fx.Node):
                    b = -b
                else:
                    a = _neg(graph, a)
            if flip_addend:
                addend = _neg(graph, addend)
            r = _call(graph, fma, (a, b, addend), node.meta)
        node.replace_all_uses_with(r)
        graph.erase_node(node)
        count += 1

    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def post_grad_custom_pre_pass(graph: torch.fx.Graph) -> int:
    return stable_one_minus_pow(graph) + cancel_double_one_minus(graph) + fuse_mul_add_to_fma(graph)


def inductor_backend(gm: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any) -> Any:
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config

    mode = kwargs.pop("mode", None)
    options = kwargs.pop("options", None)
    if kwargs:
        raise TypeError(f"Unexpected torch.compile backend options: {sorted(kwargs)}")
    merged: dict[str, Any] = {}
    if mode is not None:
        merged.update(torch._inductor.list_mode_options(mode))
    if options:
        merged.update(options)

    prior = inductor_config.post_grad_custom_pre_pass

    def run(graph: torch.fx.Graph) -> int:
        n = 0
        for p in (prior, post_grad_custom_pre_pass):
            if p is None:
                continue
            r = p(graph)
            n += r if isinstance(r, int) else 0
        return n

    with inductor_config.patch({"post_grad_custom_pre_pass": run}):
        return inductor.compile(gm, example_inputs, options=merged or None)


def compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return torch.compile(fn, backend=inductor_backend, **kwargs)
