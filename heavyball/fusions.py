from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

aten = torch.ops.aten

# Functional overloads only; this excludes out= and rounding-mode division.
_ADD = (aten.add.Tensor, aten.add.Scalar)
_SUB = (aten.sub.Tensor, aten.sub.Scalar)
_RSUB = (aten.rsub.Tensor, aten.rsub.Scalar)
_MUL = (aten.mul.Tensor, aten.mul.Scalar)
_DIV = (aten.div.Tensor, aten.div.Scalar)
_LINEAR = _ADD + _SUB + _RSUB
_ELEMENTWISE = _LINEAR + _MUL + _DIV + (aten.neg.default,)
_FRESH = _ELEMENTWISE + (aten.full.default,)
_POINTWISE_TAG = getattr(getattr(torch, "Tag", None), "pointwise", None)
_SEEDED_TAG = getattr(getattr(torch, "Tag", None), "nondeterministic_seeded", None)


def _arg(node: torch.fx.Node, index: int, name: str, default: Any = None) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, node.kwargs.get("self", default) if index == 0 else default)


def _schema(node: torch.fx.Node) -> Any:
    return getattr(node.target, "_schema", None) if node.op == "call_function" else None


def _call(graph: torch.fx.Graph, target: Any, args: tuple, meta: dict) -> torch.fx.Node:
    node = graph.call_function(target, args)
    node.meta.update(meta)
    return node


def _is_float_tensor(x: Any) -> bool:
    v = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
    return isinstance(v, torch.Tensor) and v.dtype.is_floating_point


def _number(x: Any) -> int | float | None:
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _scalar(x: Any) -> float | None:
    value = _number(x)
    return float(value) if value is not None else None


def _scaled_add_args(node: torch.fx.Node) -> tuple[Any, Any, int | float] | None:
    target = node.target
    if target not in _LINEAR:
        return None
    alpha = _number(_arg(node, 2, "alpha", 1))
    if alpha is None:
        return None
    left, right = _arg(node, 0, "input"), _arg(node, 1, "other")
    if target in _RSUB:
        left, right = right, left
    return left, right, alpha if target in _ADD else -alpha


def _fma() -> Any:
    from torch._inductor import inductor_prims

    return getattr(inductor_prims, "fma", None)


@dataclass(frozen=True)
class Affine:
    base: torch.fx.Node | None
    a: float
    b: float
    depth: int = 0
    added: bool = False


def _affine_add(lhs: Affine, rhs: Affine, scale: int | float = 1) -> Affine | None:
    ra, rb = rhs.a * scale, rhs.b * scale
    depth = max(lhs.depth, rhs.depth) + 1
    if lhs.base is None:
        return Affine(rhs.base, ra, rb + lhs.b, depth, True) if rhs.base is None or ra > 0 else None
    if rhs.base is None:
        return Affine(lhs.base, lhs.a, lhs.b + rb, depth, True)
    return None


def _affine_leaf(node: torch.fx.Node) -> Affine | None:
    return Affine(node, 1.0, 0.0) if _is_float_tensor(node) else None


def _scale_affine(affine: Affine, scale: int | float) -> Affine:
    return Affine(affine.base, affine.a * scale, affine.b * scale, affine.depth + 1, affine.added)


def _fresh(node: torch.fx.Node) -> bool:
    schema = _schema(node)
    tags = getattr(node.target, "tags", ())
    return (
        schema is not None
        and schema.returns
        and not node.is_impure()
        and _SEEDED_TAG not in tags
        and all(ret.alias_info is None for ret in schema.returns)
        and (node.target in _FRESH or _POINTWISE_TAG in tags or node.target == _fma())
    )


def _barrier(node: torch.fx.Node) -> bool:
    return not _fresh(node) if node.op == "call_function" else node.op in {"call_method", "call_module"}


def _epochs(graph: torch.fx.Graph) -> dict[torch.fx.Node, int]:
    epoch = 0
    result = {}
    for node in graph.nodes:
        result[node] = epoch
        epoch += _barrier(node)
    return result


def _affine_of(
    node: Any, cache: dict[tuple[torch.fx.Node, int], Affine | None], epochs: dict[torch.fx.Node, int], epoch: int
) -> Affine | None:
    if not isinstance(node, torch.fx.Node):
        value = _scalar(node)
        return Affine(None, 0.0, value) if value is not None else None
    key = node, epoch
    if key in cache:
        return cache[key]
    cache[key] = None
    if epochs.get(node, epoch) != epoch:
        affine = _affine_leaf(node)
        cache[key] = affine
        return affine
    if node.op == "placeholder":
        aff = _affine_leaf(node)
        cache[key] = aff
        return aff
    if node.op != "call_function" or not _is_float_tensor(node):
        return None
    aff: Affine | None = None
    args = _scaled_add_args(node)
    target = node.target
    if args is not None:
        left, right, scale = args
        left = _affine_of(left, cache, epochs, epoch)
        right = _affine_of(right, cache, epochs, epoch)
        if left and right:
            aff = _affine_add(left, right, scale)
    elif target in _MUL:
        left, right = _arg(node, 0, "input"), _arg(node, 1, "other")
        scale, inner_node = _scalar(left), right
        if scale is None:
            scale, inner_node = _scalar(right), left
        if scale is not None and scale > 0:
            inner = _affine_of(inner_node, cache, epochs, epoch)
            if inner:
                aff = _scale_affine(inner, scale)
    elif target in _DIV:
        scale = _scalar(_arg(node, 1, "other"))
        if scale is not None and scale > 0:
            inner = _affine_of(_arg(node, 0, "input"), cache, epochs, epoch)
            if inner:
                aff = _scale_affine(inner, 1.0 / scale)
    if aff is None:
        aff = _affine_leaf(node)
    elif aff.base is not node and (aff.a == 0.0 or aff.a == 1.0 and aff.b == 0.0):
        aff = _affine_leaf(node)
    cache[key] = aff
    return aff


def _emit_affine(
    graph: torch.fx.Graph, affine: Affine, meta: dict, epochs: dict[torch.fx.Node, int], epoch: int
) -> Any:
    def emit(target: Any, args: tuple) -> torch.fx.Node:
        node = _call(graph, target, args, meta)
        epochs[node] = epoch
        return node

    base = affine.base
    if affine.b == 0.0:
        return emit(aten.mul.Tensor, (base, affine.a))
    if affine.a == 1.0:
        return emit(aten.add.Tensor, (base, affine.b))
    scaled = emit(aten.mul.Tensor, (base, affine.a))
    return emit(aten.add.Tensor, (scaled, affine.b))


def _is_affine_form(node: torch.fx.Node, affine: Affine) -> bool:
    if affine.b == 0.0:
        left, right = _arg(node, 0, "input"), _arg(node, 1, "other")
        return node.target in _MUL and left is affine.base and _scalar(right) == affine.a
    args = _scaled_add_args(node)
    if args is None:
        return False
    left, right, scale = args
    if affine.a == 1.0:
        return left is affine.base and scale == 1 and _scalar(right) == affine.b
    return (
        scale == 1
        and _scalar(right) == affine.b
        and isinstance(left, torch.fx.Node)
        and left.target in _MUL
        and _arg(left, 0, "input") is affine.base
        and _scalar(_arg(left, 1, "other")) == affine.a
    )


def _survives_cast(value: float, dtype: torch.dtype) -> bool:
    if not math.isfinite(value):
        return False
    try:
        cast = torch.tensor(value, dtype=dtype, device="cpu").item()
    except (OverflowError, RuntimeError):
        return False
    return math.isfinite(cast) and (value == 0.0 or cast != 0.0)


def _representable(affine: Affine, node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and all(_survives_cast(x, value.dtype) for x in (affine.a, affine.b))


def _finalize(graph: torch.fx.Graph, count: int) -> int:
    if count:
        graph.eliminate_dead_code(lambda node: node.op != "call_function" or not _fresh(node))
        graph.lint()
    return count


def fold_affine(graph: torch.fx.Graph) -> int:
    epochs = _epochs(graph)
    fma = _fma() is not None
    cache: dict[tuple[torch.fx.Node, int], Affine | None] = {}
    count = 0
    for node, node_epoch in tuple(epochs.items()):
        if node.op != "call_function" or not _is_float_tensor(node) or node.meta["val"].dtype == torch.float64:
            continue
        affine = _affine_of(node, cache, epochs, node_epoch)
        if affine is None or affine.base is None or not _representable(affine, node):
            continue

        if affine.a <= 0.0 or affine.b == 0.0 and affine.added:
            continue

        if _is_affine_form(node, affine):
            continue

        cost = 1 if fma or affine.a == 1.0 or affine.b == 0.0 else 2
        if affine.depth > cost:
            with graph.inserting_before(node):
                built = _emit_affine(graph, affine, node.meta, epochs, node_epoch)
            node.replace_all_uses_with(built)
            graph.erase_node(node)
            count += 1

    return _finalize(graph, count)


def _is_mul_operand(x: Any) -> bool:
    if isinstance(x, torch.fx.Node):
        return isinstance(x.meta.get("val"), torch.Tensor)
    return _number(x) is not None


def _single_user_mul(node: Any, epochs: dict[torch.fx.Node, int], epoch: int) -> tuple[Any, Any] | None:
    if not (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in _MUL
        and len(node.users) == 1
        and _is_float_tensor(node)
        and epochs.get(node, epoch) == epoch
    ):
        return None
    a, b = _arg(node, 0, "input"), _arg(node, 1, "other")
    return (a, b) if _is_mul_operand(a) and _is_mul_operand(b) else None


def fuse_mul_add_to_fma(graph: torch.fx.Graph) -> int:
    fma = _fma()
    if fma is None:
        return 0
    epochs = _epochs(graph)
    count = 0
    for node, epoch in epochs.items():
        if node.op != "call_function" or not _is_float_tensor(node):
            continue
        args = _scaled_add_args(node)
        if args is None:
            continue
        lhs, rhs, scale = args
        if node.target not in _ADD or scale <= 0:
            continue

        if abs(scale) != 1:
            product_a, product_b, addend = rhs, scale, lhs
            if not _is_float_tensor(product_a) or product_a.meta["val"].dtype != node.meta["val"].dtype:
                continue
        else:
            lhs_mul = _single_user_mul(lhs, epochs, epoch)
            rhs_mul = _single_user_mul(rhs, epochs, epoch)
            if lhs_mul and rhs_mul:
                continue
            if rhs_mul:
                product_a, product_b, addend = rhs_mul[0], rhs_mul[1], lhs
            elif lhs_mul:
                product_a, product_b, addend = lhs_mul[0], lhs_mul[1], rhs
            else:
                continue
        if not _is_mul_operand(addend):
            continue

        with graph.inserting_before(node):
            result = _call(graph, fma, (product_a, product_b, addend), node.meta)
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        count += 1
    return _finalize(graph, count)


def post_grad_custom_pre_pass(graph: torch.fx.Graph) -> int:
    return fold_affine(graph) + fuse_mul_add_to_fma(graph)


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
    return fn if torch._dynamo.config.disable else torch.compile(fn, backend=inductor_backend, **kwargs)
