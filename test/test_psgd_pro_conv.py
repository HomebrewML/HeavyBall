"""PSGD-PRO and QSGD precondition convolutions through matrix merging."""

from unittest.mock import patch

import pytest
import torch

from heavyball import build, psgd_pro_adamw, qsgd_adamw
from heavyball.psgd_pro import psgd_pro, qsgd


@pytest.mark.parametrize(
    ("route", "recipe"),
    ((psgd_pro_adamw, psgd_pro), (qsgd_adamw, qsgd)),
    ids=("psgd_pro", "qsgd"),
)
def test_psgd_pro_conv_routes_preconditions_and_keeps_shape(route, recipe):
    parameter = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], route, lr=1e-3)

    assert optimizer.groups[0].recipe is recipe
    state = optimizer.groups[0].states[0]
    assert tuple(state["Q_0"].shape[1:]) == (4, 4)
    assert tuple(state["Q_1"].shape[1:]) == (27, 27)

    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step(step_type="refresh")
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert tuple(parameter.shape) == (4, 3, 3, 3)


def _run(recipe, initial, gradients, *, as_conv, seed):
    native_shape = (4, 27)
    torch.manual_seed(seed)  # sync Tempo's stateless whitening-probe seed across the two runs
    parameter = torch.nn.Parameter(initial.clone() if as_conv else initial.reshape(native_shape).clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], recipe, lr=1e-3, weight_decay=0.0)

    for step, gradient in enumerate(gradients):
        parameter.grad.copy_((gradient if as_conv else gradient.reshape(native_shape)).clone())
        optimizer.step(step_type="refresh" if step in (0, 3) else "normal")
    return parameter.detach().reshape(native_shape).clone()


@pytest.mark.parametrize("recipe", (psgd_pro, qsgd), ids=("psgd_pro", "qsgd"))
def test_psgd_pro_conv_equals_native_2d_rng_synced(recipe):
    torch.manual_seed(0)
    initial = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    gradients = [torch.randn_like(initial) for _ in range(6)]

    conv = _run(recipe, initial, gradients, as_conv=True, seed=7)
    native = _run(recipe, initial, gradients, as_conv=False, seed=7)
    torch.testing.assert_close(conv, native, rtol=0, atol=0)
