"""Compiled HeavyBall steps must preserve eager numerics exactly."""

from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, adamw, ademamix, kl_soap_adamw, laprop, soap_adamw

# Keep this deterministic: kron, psgd_pro, and lather use randn probes, while msam has a literal swap.
# Those recipes need explicit RNG plumbing before compiled-vs-eager math can be compared directly.
RECIPES = (
    ("adamw", adamw, False),
    ("ademamix", ademamix, False),
    ("laprop", laprop, False),
    ("soap_adamw", soap_adamw, True),
    ("kl_soap_adamw", kl_soap_adamw, True),
)


def _params() -> list[torch.nn.Parameter]:
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]


def _gradients(params: list[torch.nn.Parameter]) -> tuple[tuple[torch.Tensor, ...], ...]:
    torch.manual_seed(1)
    return tuple(tuple(torch.randn_like(param) for param in params) for _ in range(6))


@pytest.mark.parametrize(("_name", "recipe", "matrix"), RECIPES)
def test_compiled_matches_eager(_name, recipe, matrix):
    """The compiled route matches its eager route after six identical steps."""

    compiled_params = _params()
    eager_params = _params()
    gradients = _gradients(compiled_params)
    try:
        compiled = Engine(compiled_params, recipe)
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            eager = Engine(eager_params, recipe)

        for index, step_gradients in enumerate(gradients):
            for compiled_param, eager_param, gradient in zip(compiled.params, eager.params, step_gradients, strict=True):
                compiled_param.grad.copy_(gradient)
                eager_param.grad.copy_(gradient)
            step_type = "refresh" if matrix and index == 0 else "normal"
            compiled.step(step_type=step_type)
            eager.step(step_type=step_type)

        for compiled_group, eager_group in zip(compiled.groups, eager.groups, strict=True):
            torch.testing.assert_close(
                compiled_group.param_slab.double(), eager_group.param_slab.double(), atol=1e-12, rtol=0
            )
    finally:
        torch._dynamo.reset()
