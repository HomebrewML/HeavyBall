"""Narrow Inductor configuration for HeavyBall's functional optimizer graph."""

from collections.abc import Hashable

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

_CSE_TARGETS = frozenset(
    (
        torch.ops.aten._to_copy.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.expm1.default,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.log.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.rsub.Scalar,
        torch.ops.aten.sub.Tensor,
    )
)
_MATRIX_TARGETS = frozenset(
    (
        torch.ops.aten.addbmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.mm.default,
    )
)


def _freeze(value) -> Hashable:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class OptimizerCSE(CustomGraphPass):
    """Deduplicate scalar clock/hyper expressions in elementwise optimizer graphs."""

    def uuid(self) -> str:
        return "heavyball-elementwise-optimizer-scalar-cse-v1"

    def __call__(self, graph: torch.fx.Graph) -> None:
        if any(node.op == "call_function" and node.target in _MATRIX_TARGETS for node in graph.nodes):
            return
        expressions = {}
        removed = 0
        for node in tuple(graph.nodes):
            if node.op != "call_function" or node.target not in _CSE_TARGETS:
                continue
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                continue
            key = (node.target, _freeze(node.args), _freeze(node.kwargs))
            previous = expressions.get(key)
            if previous is None:
                expressions[key] = node
                continue
            node.replace_all_uses_with(previous)
            graph.erase_node(node)
            removed += 1
        if removed:
            torch._dynamo.utils.counters["heavyball"]["fx_cse_removed"] += removed


OPTIMIZER_CSE = OptimizerCSE()
# max_autotune is the minimum we need for fast kernels (it is what the "max-autotune" preset enables).
# We pass it as an option (not mode=) so it composes with our scalar-CSE pass; no cudagraphs, matching
# the prior "max-autotune-no-cudagraphs" behavior.
STEP_COMPILE_OPTIONS = {"max_autotune": True, "post_grad_custom_pre_pass": OPTIMIZER_CSE}
