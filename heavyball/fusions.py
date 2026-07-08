from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_map


aten = torch.ops.aten


def _op_targets(op) -> set[Any]:
    targets = {op}
    if isinstance(op, torch._ops.OpOverloadPacket):
        if hasattr(op, "op_overloads"):
            targets.update(op.op_overloads())
        else:
            targets.update(getattr(op, overload) for overload in op.overloads())
    return targets


_ADD_TARGETS = _op_targets(aten.add)
_SUB_TARGETS = _op_targets(aten.sub)
_MUL_TARGETS = _op_targets(aten.mul)
_RSUB_TARGETS = _op_targets(aten.rsub)


def _get_arg(node: torch.fx.Node, index: int, name: str, default: Any = None) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _is_float_tensor_node(node: Any) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].dtype.is_floating_point
    )


def _fma_target() -> Callable[..., Any] | None:
    try:
        from torch._inductor import inductor_prims
    except ImportError:
        return None
    return getattr(inductor_prims, "fma", None)


def _is_mul_operand(x: Any) -> bool:
    """A usable fma multiply input: a tensor node or a python number.

    The mul node's own result being float (checked by the caller) guarantees the
    arithmetic is in the float domain, so an integer operand (promoted by the mul)
    and a python scalar are both fine -- prims.fma accepts scalar operands.
    """
    if isinstance(x, torch.fx.Node):
        return isinstance(x.meta.get("val"), torch.Tensor)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _single_user_mul(node: Any) -> tuple[Any, Any] | None:
    """Match a float-result mul with no other users, returning its operands.

    Operands may be tensor nodes or python scalars; an all-integer mul has an
    integer result and is rejected by the float-result check, so it stays exact.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target in _MUL_TARGETS and len(node.users) == 1
            and _is_float_tensor_node(node)):
        return None
    a, b = _get_arg(node, 0, "input"), _get_arg(node, 1, "other")
    if _is_mul_operand(a) and _is_mul_operand(b):
        return a, b
    return None


def _is_addend(x: Any) -> bool:
    """An operand usable as the fma addend: a float tensor node or a python number."""
    if isinstance(x, torch.fx.Node):
        return _is_float_tensor_node(x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _negate(graph: torch.fx.Graph, x: Any) -> Any:
    """Exact sign flip: aten.neg for a node, arithmetic negation for a scalar."""
    if isinstance(x, torch.fx.Node):
        neg = graph.call_function(aten.neg.default, (x,))
        neg.meta.update(x.meta)
        return neg
    return -x


def fuse_mul_add_to_fma(graph: torch.fx.Graph) -> int:
    """Rewrite add/sub/rsub of a product into prims.fma so Inductor emits one tl.fma.

      a*b + c, c + a*b        -> fma(a, b, c)
      a*b - c                 -> fma(a, b, -c)
      c - a*b, 1 - a*b        -> fma(-a, b, c)        (rsub: scalar minus product)
      add(x, y, alpha=k!=1)   -> fma(y, k, x)         (prims.fma takes a scalar)

    Subtraction inserts an exact ``aten.neg`` (or arithmetic negation for a scalar
    addend), so the fused multiply-add still rounds once. The addend may be a tensor
    node or a python scalar (e.g. the ``1`` in ``1 - d*lr``); prims.fma accepts scalar
    operands, so scalar addends fuse without an extra node. Plain ``x +/- y`` with no
    product is left alone: it already rounds once.
    """
    fma = _fma_target()
    if fma is None:
        return 0

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        is_rsub = node.target in _RSUB_TARGETS
        if node.target in _ADD_TARGETS:
            sign = 1
        elif node.target in _SUB_TARGETS:
            sign = -1
        elif is_rsub:
            sign = -1  # rsub(input, alpha) == alpha - input
        else:
            continue
        if not _is_float_tensor_node(node):
            continue

        if is_rsub:
            mul = _single_user_mul(_get_arg(node, 0, "input"))
            if mul is None:
                continue
            a, b = mul
            c = _get_arg(node, 1, "alpha", 1)  # addend, usually the scalar 1
            negate_a, negate_c = True, False
        else:
            alpha = _get_arg(node, 2, "alpha", 1)
            scale = sign * alpha
            lhs, rhs = _get_arg(node, 0, "input"), _get_arg(node, 1, "other")
            lhs_mul = _single_user_mul(lhs)
            rhs_mul = _single_user_mul(rhs)
            if rhs_mul and abs(scale) == 1:          # lhs + scale*(a*b)
                a, b, c = rhs_mul[0], rhs_mul[1], lhs
                negate_a, negate_c = scale == -1, False
            elif lhs_mul and abs(scale) == 1:        # a*b + scale*rhs
                a, b, c = lhs_mul[0], lhs_mul[1], rhs
                negate_a, negate_c = False, scale == -1
            elif alpha != 1 and not (lhs_mul or rhs_mul):  # scaled add: x + k*y
                a, b, c = rhs, scale, lhs
                negate_a, negate_c = False, False
            else:
                continue
        if not _is_addend(c):
            continue

        with graph.inserting_before(node):
            if negate_a:
                a = _negate(graph, a)
            if negate_c:
                c = _negate(graph, c)
            replacement = graph.call_function(fma, (a, b, c))
            replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        count += 1

    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def _is_scalar_node(node: Any) -> bool:
    """A 0-d float result computed only from 0-d / scalar operands (a pure scalar op).

    Excludes reductions (tensor -> 0-d): those would need a tensor->fp64 cast, which is
    a data-movement cost. Placeholders and pointwise ops on 0-d inputs qualify.
    """
    if not isinstance(node, torch.fx.Node):
        return False
    val = node.meta.get("val")
    if not (isinstance(val, torch.Tensor) and val.dim() == 0 and val.dtype.is_floating_point):
        return False
    if node.op == "placeholder":
        return True
    if node.op != "call_function":
        return False
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            av = arg.meta.get("val")
            if not (isinstance(av, torch.Tensor) and av.dim() == 0):
                return False  # consumes a real tensor -> reduction / broadcast, skip
    return True


def promote_scalar_ops_to_fp64(graph: torch.fx.Graph) -> int:
    """Recompute every scalar (0-d float) op in float64, casting back at tensor edges.

    Rationale: a 0-d value never moves through HBM, so under a "data movement is the
    only cost" model, scalar intermediates (bias correction, decay factors, schedules)
    are free to compute in float64. This removes their float32 cancellation -- e.g.
    ``1 - beta**step`` loses ~50 ULP at beta=0.999, step=2 in float32, none in float64.

    Mechanism: build a parallel fp64 subgraph for scalar nodes (a placeholder becomes
    ``.to(fp64)``; an op is reissued with fp64 operands), and splice a downcast back to
    the original dtype only where a scalar feeds a non-scalar (tensor) user. Tensors
    therefore never upcast (no extra data movement); the old float32 scalar chain dies.
    """
    scalar_nodes = [n for n in graph.nodes if _is_scalar_node(n)]
    if not scalar_nodes:
        return 0

    fp64_of: dict[torch.fx.Node, torch.fx.Node] = {}

    def fp64(node: torch.fx.Node) -> torch.fx.Node:
        if node in fp64_of:
            return fp64_of[node]
        if node.op == "placeholder":
            with graph.inserting_after(node):
                cast = graph.call_function(aten.to.dtype, (node, torch.float64))
                cast.meta["val"] = node.meta["val"].to(torch.float64)
            fp64_of[node] = cast
            return cast

        def promote(x: Any) -> Any:
            return fp64(x) if (isinstance(x, torch.fx.Node) and _is_scalar_node(x)) else x

        new_args = tree_map(promote, node.args)
        new_kwargs = {}
        for key, value in node.kwargs.items():
            if key == "dtype" and value in (torch.float16, torch.bfloat16, torch.float32):
                new_kwargs[key] = torch.float64  # constant constructor (full/ones) -> fp64
            else:
                new_kwargs[key] = tree_map(promote, value)
        with graph.inserting_after(node):
            rep = graph.call_function(node.target, new_args, new_kwargs)
            rep.meta["val"] = node.meta["val"].to(torch.float64)
        fp64_of[node] = rep
        return rep

    downcast_of: dict[torch.fx.Node, torch.fx.Node] = {}

    def downcast(node: torch.fx.Node) -> torch.fx.Node:
        if node in downcast_of:
            return downcast_of[node]
        hi = fp64(node)
        dtype = node.meta["val"].dtype
        with graph.inserting_after(hi):
            cast = graph.call_function(aten.to.dtype, (hi, dtype))
            cast.meta["val"] = hi.meta["val"].to(dtype)
        downcast_of[node] = cast
        return cast

    count = 0
    for node in scalar_nodes:
        for user in list(node.users):
            if _is_scalar_node(user):
                continue  # stays inside the fp64 subgraph via fp64()
            dc = downcast(node)
            user.args = tree_map(lambda x: dc if x is node else x, user.args)
            user.kwargs = tree_map(lambda x: dc if x is node else x, user.kwargs)
            count += 1

    if count:
        graph.eliminate_dead_code()
        graph.lint()
    return count


def post_grad_custom_pre_pass(graph: torch.fx.Graph) -> int:
    return fuse_mul_add_to_fma(graph) + promote_scalar_ops_to_fp64(graph)


def _chain_graph_passes(*passes: Callable[[torch.fx.Graph], Any] | None) -> Callable[[torch.fx.Graph], int]:
    def chained(graph: torch.fx.Graph) -> int:
        count = 0
        for graph_pass in passes:
            if graph_pass is None:
                continue
            result = graph_pass(graph)
            if isinstance(result, int):
                count += result
        return count

    return chained


def _compile_options(mode: str | None, options: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = {}
    if mode is not None:
        merged.update(torch._inductor.list_mode_options(mode))
    if options:
        merged.update(options)
    return merged or None


def inductor_backend(gm: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any) -> Callable[..., Any]:
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config

    prior_pass = inductor_config.post_grad_custom_pre_pass
    local_pass = _chain_graph_passes(prior_pass, post_grad_custom_pre_pass)
    options = _compile_options(kwargs.pop("mode", None), kwargs.pop("options", None))
    if kwargs:
        raise TypeError(f"Unexpected torch.compile backend options: {sorted(kwargs)}")

    with inductor_config.patch({"post_grad_custom_pre_pass": local_pass}):
        return inductor.compile(gm, example_inputs, options=options)


def compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return torch.compile(fn, backend=inductor_backend, **kwargs)
