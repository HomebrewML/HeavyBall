"""Convolution preconditioning via bind-time dimension merging."""

from unittest.mock import patch

import pytest
import torch

from heavyball import build, soap_adamw
from heavyball.matrix import matrix_route, merged_matrix_shape


class _Info:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


@pytest.mark.parametrize(
    ("shape", "max_precond_dim", "expected"),
    (
        ((128, 64, 3, 3), 16, (128, 64, 9)),
        ((128, 64, 3, 3), 64, (128, 64, 9)),
        ((128, 64, 3, 3), 2048, (128, 576)),
        ((4, 3, 3, 3), 16, (4, 3, 9)),
        ((4, 3, 3, 3), 64, (4, 27)),
        ((4, 3, 3, 3), 2048, (4, 27)),
        ((8, 4, 2, 2), 16, (8, 16)),
        ((8, 4, 2, 2), 64, (8, 16)),
        ((8, 4, 2, 2), 2048, (8, 16)),
        ((16, 8), 16, (16, 8)),
        ((10,), 64, (10,)),
        ((128, 1000, 3, 3), 2048, (128, 1000, 9)),
        ((1, 5), 16, (1, 5)),
    ),
)
def test_merged_matrix_shape_follows_right_to_left_product_rule(shape, max_precond_dim, expected):
    """Goldens obtained by hand-applying the documented right-to-left product limit."""

    assert merged_matrix_shape(shape, max_precond_dim) == expected


def test_matrix_route_selects_leaves_that_merge_to_2d():
    assert matrix_route(_Info((4, 3, 3, 3))) is True      # merges to (4, 27)
    assert matrix_route(_Info((128, 64, 3, 3))) is True    # merges to (128, 576)
    assert matrix_route(_Info((16, 8))) is True            # native 2D
    assert matrix_route(_Info((10,))) is False             # 1D bias
    assert matrix_route(_Info((128, 1000, 3, 3))) is False  # merges to 3D -> adamw


def _run_soap(leaf0, gradients, *, as_conv, native_shape):
    parameter = torch.nn.Parameter(leaf0.clone() if as_conv else leaf0.reshape(native_shape).clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], soap_adamw, lr=1e-3, weight_decay=0.0)
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_((gradient if as_conv else gradient.reshape(native_shape)).clone())
        optimizer.step(step_type="refresh" if step == 0 else "normal")
    return parameter.detach().reshape(native_shape).clone()


def test_conv_soap_equals_native_2d_because_the_merge_is_a_noop_reshape():
    """A merged convolution trains bit-identically to the equivalent native matrix."""

    torch.manual_seed(0)
    leaf0 = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    gradients = [torch.randn(4, 3, 3, 3, dtype=torch.float64) for _ in range(6)]
    conv = _run_soap(leaf0, gradients, as_conv=True, native_shape=(4, 27))
    native = _run_soap(leaf0, gradients, as_conv=False, native_shape=(4, 27))
    torch.testing.assert_close(conv, native, rtol=0, atol=0)


def test_conv_leaf_is_preconditioned_and_keeps_its_shape():
    """A conv routes to SOAP, allocates a merged factor, preconditions, stays finite and shaped."""

    parameter = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], soap_adamw, lr=1e-3)
    state = optimizer.groups[0].states[0]
    assert "GG_r" in state and tuple(state["exp_avg"].shape) == (1, 4, 27)
    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step(step_type="refresh")
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert tuple(parameter.shape) == (4, 3, 3, 3)
