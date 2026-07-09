from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

aten = torch.ops.aten


def _arg(node: torch.fx.Node, index: int, name: str, default: Any = None) -> Any:
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def _match(target: Any, *packets: Any) -> bool:
    packet = getattr(target, "overloadpacket", target)
    return packet in packets


def _call(graph: torch.fx.Graph, target: Any, args: tuple, meta: dict, kwargs: dict | None = None) -> torch.fx.Node:
    node = graph.call_function(target, args, kwargs or {})
    node.meta.update(meta)
    return node


def _is_float_tensor(x: Any) -> bool:
    v = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
    return isinstance(v, torch.Tensor) and v.dtype.is_floating_point


def _scalar(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.fx.Node):
        value = x.meta.get("val")
        if isinstance(value, torch.Tensor) and value.dim() == 0 and value.dtype.is_floating_point:
            from torch._subclasses.fake_tensor import FakeTensor
            if not isinstance(value, FakeTensor):
                try:
                    return float(value.item())
                except Exception:
                    return None
    return None


@dataclass(frozen=True)
class Affine:
    base: Any
    a: float
    b: float

    def is_const(self) -> bool:
        return self.base is None


def _affine_add(lhs: Affine, rhs: Affine, scale: float = 1.0) -> Affine | None:
    ra, rb = rhs.a * scale, rhs.b * scale
    if lhs.is_const() and rhs.is_const():
        return Affine(None, 0.0, lhs.b + rb)
    if lhs.is_const():
        return Affine(rhs.base, ra, rb + lhs.b)
    if rhs.is_const():
        return Affine(lhs.base, lhs.a, lhs.b + rb)
    if lhs.base is rhs.base:
        return Affine(lhs.base, lhs.a + ra, lhs.b + rb)
    return None


def _affine_leaf(node: torch.fx.Node) -> Affine | None:
    return Affine(node, 1.0, 0.0) if _is_float_tensor(node) else None


def _affine_of(node: Any, cache: dict[Any, Affine | None]) -> Affine | None:
    if not isinstance(node, torch.fx.Node):
        value = _scalar(node)
        return Affine(None, 0.0, value) if value is not None else None
    if node in cache:
        return cache[node]
    cache[node] = None
    if node.op == "placeholder":
        aff = _affine_leaf(node)
        cache[node] = aff
        return aff
    if node.op != "call_function" or not _is_float_tensor(node):
        return None
    aff: Affine | None = None
    target = node.target
    if _match(target, aten.add):
        alpha = _arg(node, 2, "alpha", 1)
        if isinstance(alpha, (int, float)):
            left = _affine_of(_arg(node, 0, "input"), cache)
            right = _affine_of(_arg(node, 1, "other"), cache)
            if left and right:
                aff = _affine_add(left, right, alpha)
    elif _match(target, aten.sub):
        alpha = _arg(node, 2, "alpha", 1)
        if isinstance(alpha, (int, float)):
            left = _affine_of(_arg(node, 0, "input"), cache)
            right = _affine_of(_arg(node, 1, "other"), cache)
            if left and right:
                aff = _affine_add(left, right, -alpha)
    elif _match(target, aten.rsub):
        alpha = _arg(node, 2, "alpha", 1)
        if isinstance(alpha, (int, float)):
            left = _affine_of(_arg(node, 0, "input"), cache)
            right = _affine_of(_arg(node, 1, "other"), cache)
            if left and right:
                aff = _affine_add(right, left, -alpha)
    elif _match(target, aten.mul):
        left, right = _arg(node, 0, "input"), _arg(node, 1, "other")
        left_scalar, right_scalar = _scalar(left), _scalar(right)
        if left_scalar is not None:
            inner = _affine_of(right, cache)
            if inner:
                aff = Affine(inner.base, inner.a * left_scalar, inner.b * left_scalar)
        elif right_scalar is not None:
            inner = _affine_of(left, cache)
            if inner:
                aff = Affine(inner.base, inner.a * right_scalar, inner.b * right_scalar)
    elif target == aten.neg.default:
        inner = _affine_of(_arg(node, 0, "input"), cache)
        if inner:
            aff = Affine(inner.base, -inner.a, -inner.b)
    else:
        aff = _affine_leaf(node)
    cache[node] = aff
    return aff


def _fma() -> Any:
    from torch._inductor import inductor_prims
    return getattr(inductor_prims, "fma", None)


def fold_affine(graph: torch.fx.Graph) -> int:
    seen: dict[tuple[int | None, float, float], torch.fx.Node] = {}
    cache: dict[Any, Affine | None] = {}
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or not _is_float_tensor(node):
            continue
        affine = _affine_of(node, cache)
        if affine is None or affine.is_const():
            continue
        if affine.a == 1.0 and affine.b == 0.0 and isinstance(affine.base, torch.fx.Node):
            if affine.base is not node:
                node.replace_all_uses_with(affine.base)
                graph.erase_node(node)
                count += 1
            continue
        key = (id(affine.base) if affine.base is not None else None, affine.a, affine.b)
        rep = seen.get(key)
        if rep is not None:
            node.replace_all_uses_with(rep)
            graph.erase_node(node)
            count += 1
        else:
            seen[key] = node
    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def _one_minus_inner(node: Any) -> Any | None:
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if _arg(node, 2, "alpha", 1) != 1:
        return None
    if _match(node.target, aten.rsub):
        return _arg(node, 0, "input") if _arg(node, 1, "other", 1) == 1 else None
    if _match(node.target, aten.sub):
        return _arg(node, 1, "other") if _arg(node, 0, "input") == 1 else None
    return None


def _build_stable_one_minus(graph: torch.fx.Graph, base: Any, exponent: Any, inner: torch.fx.Node,
        outer: torch.fx.Node) -> torch.fx.Node:
    one_minus_base = _call(graph, aten.sub.Tensor, (base, 1), inner.meta)
    log1p = _call(graph, aten.log1p.default, (one_minus_base,), inner.meta)
    scaled = _call(graph, aten.mul.Tensor, (exponent, log1p), inner.meta)
    expm1 = _call(graph, aten.expm1.default, (scaled,), inner.meta)
    return _call(graph, aten.neg.default, (expm1,), outer.meta)


def stable_one_minus_pow(graph: torch.fx.Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        inner = _one_minus_inner(node)
        if inner is None:
            continue
        if not (isinstance(inner, torch.fx.Node) and inner.op == "call_function" and _match(inner.target,
                                                                                            aten.pow) and len(
            inner.users) == 1):
            continue
        base = _arg(inner, 0, "input")
        base_value = base.meta.get("val") if isinstance(base, torch.fx.Node) else None
        if not (isinstance(base_value, torch.Tensor) and base_value.dim() == 0 and base_value.dtype.is_floating_point):
            continue
        exponent = _arg(inner, 1, "exponent")
        base_scalar = _scalar(base)
        with graph.inserting_before(node):
            if base_scalar is not None and base_scalar > 0:
                built = _build_stable_one_minus(graph, base, exponent, inner, node)
            elif base_scalar is not None:
                built = _call(graph, aten.sub.Tensor, (1, inner), node.meta)
            else:
                positive = _call(graph, aten.gt.Scalar, (base, 0), {**inner.meta, "val": base_value > 0})
                stable = _build_stable_one_minus(graph, base, exponent, inner, node)
                fallback = _call(graph, aten.sub.Tensor, (1, inner), node.meta)
                built = _call(graph, aten.where.self, (positive, stable, fallback), node.meta)
        node.replace_all_uses_with(built)
        graph.erase_node(node)
        count += 1
    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def cancel_double_one_minus(graph: torch.fx.Graph) -> int:
    return fold_affine(graph)


def _is_mul_operand(x: Any) -> bool:
    if isinstance(x, torch.fx.Node):
        return isinstance(x.meta.get("val"), torch.Tensor)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_addend(x: Any) -> bool:
    if isinstance(x, torch.fx.Node):
        return _is_float_tensor(x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _single_user_mul(node: Any) -> tuple[Any, Any] | None:
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function" and _match(node.target, aten.mul) and len(
        node.users) == 1 and _is_float_tensor(node)):
        return None
    a, b = _arg(node, 0, "input"), _arg(node, 1, "other")
    return (a, b) if _is_mul_operand(a) and _is_mul_operand(b) else None


def _neg(graph: torch.fx.Graph, x: Any) -> Any:
    return _call(graph, aten.neg.default, (x,), x.meta) if isinstance(x, torch.fx.Node) else -x


def fuse_mul_add_to_fma(graph: torch.fx.Graph) -> int:
    fma = _fma()
    if fma is None:
        return 0
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or not _is_float_tensor(node):
            continue
        target = node.target
        if _match(target, aten.add):
            lhs, rhs, alpha, sign = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), 1
        elif _match(target, aten.sub):
            lhs, rhs, alpha, sign = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), -1
        elif _match(target, aten.rsub):
            rhs, lhs, alpha, sign = _arg(node, 0, "input"), _arg(node, 1, "other"), _arg(node, 2, "alpha", 1), -1
        else:
            continue
        if not isinstance(alpha, (int, float)):
            continue
        scale = sign * alpha

        lhs_mul = _single_user_mul(lhs)
        rhs_mul = _single_user_mul(rhs)
        if rhs_mul and abs(scale) == 1:
            product_a, product_b, addend = rhs_mul[0], rhs_mul[1], lhs
            flip_product = scale == -1
            flip_addend = False
        elif lhs_mul and abs(scale) == 1:
            product_a, product_b, addend = lhs_mul[0], lhs_mul[1], rhs
            flip_product = False
            flip_addend = scale == -1
        elif abs(scale) != 1 and not (lhs_mul or rhs_mul):
            product_a, product_b, addend = rhs, scale, lhs
            flip_product = False
            flip_addend = False
            if not _is_mul_operand(product_a):
                continue
        else:
            continue
        if not _is_addend(addend):
            continue

        with graph.inserting_before(node):
            if flip_product:
                if not isinstance(product_a, torch.fx.Node):
                    product_a = -product_a
                elif not isinstance(product_b, torch.fx.Node):
                    product_b = -product_b
                else:
                    product_a = _neg(graph, product_a)
            if flip_addend:
                addend = _neg(graph, addend)
            result = _call(graph, fma, (product_a, product_b, addend), node.meta)
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        count += 1
    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def post_grad_custom_pre_pass(graph: torch.fx.Graph) -> int:
    return stable_one_minus_pow(graph) + fold_affine(graph) + fuse_mul_add_to_fma(graph)


def inductor_backend(graph_module: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any) -> Any:
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config

    mode = kwargs.pop("mode", None)
    options = kwargs.pop("options", None)
    if kwargs:
        raise TypeError(f"Unexpected torch.compile backend options: {sorted(kwargs)}")
    merged: dict[str, Any] = {}
    if mode is not None:
        merged.update(inductor.list_mode_options(mode))
    if options:
        merged.update(options)

    prior = inductor_config.post_grad_custom_pre_pass

    def run(graph: torch.fx.Graph) -> int:
        total = 0
        for pass_fn in (prior, post_grad_custom_pre_pass):
            if pass_fn is None:
                continue
            result = pass_fn(graph)
            total += result if isinstance(result, int) else 0
        return total

    with inductor_config.patch({"post_grad_custom_pre_pass": run}):
        return inductor.compile(graph_module, example_inputs, options=merged or None)


def compile(fn: Any, **kwargs: Any) -> Any:
    return torch.compile(fn, backend=inductor_backend, **kwargs)
