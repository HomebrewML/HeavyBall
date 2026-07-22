"""LATHER preconditions convolutions via the shared merge wrapper.

Parity by reduction (as test_kron_conv / test_psgd_pro_conv): lather composes
merge_matrix_transform, so a conv equals the native-2D run bit-identically and inherits
the native-2D legacy parity test_lather proves.
"""

from unittest.mock import patch

import torch

from heavyball import build, lather, lather_adamw
from heavyball.matrix import matrix_route, merged_matrix_shape


class _Info:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


def test_conv_merges_to_2d_and_routes_to_lather():
    assert merged_matrix_shape((4, 3, 3, 3), 2048) == (4, 27)
    assert matrix_route(_Info((4, 3, 3, 3))) is True
    assert matrix_route(_Info((128, 1000, 3, 3))) is False  # >2D merge -> adamw


def test_conv_lather_routes_preconditions_and_keeps_shape():
    parameter = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], lather_adamw, lr=1e-3)
    state = optimizer.groups[0].states[0]
    assert any(key.startswith("Q") for key in state) and tuple(state["exp_avg"].shape[1:]) == (4, 27)
    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step(step_type="refresh")
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert tuple(parameter.shape) == (4, 3, 3, 3)


def _run_lather(conv0, gradients, *, as_conv, native_shape, seed):
    torch.manual_seed(seed)  # sync Tempo's stateless whitening-probe seed across the two runs
    parameter = torch.nn.Parameter(conv0.clone() if as_conv else conv0.reshape(native_shape).clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], lather, lr=1e-3, weight_decay=0.0)
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_((gradient if as_conv else gradient.reshape(native_shape)).clone())
        optimizer.step(step_type="refresh" if step == 0 else "normal")
    return parameter.detach().reshape(native_shape).clone()


def test_conv_lather_equals_native_2d_rng_synced():
    torch.manual_seed(0)
    conv0 = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    gradients = [torch.randn(4, 3, 3, 3, dtype=torch.float64) for _ in range(6)]
    conv = _run_lather(conv0, gradients, as_conv=True, native_shape=(4, 27), seed=11)
    native = _run_lather(conv0, gradients, as_conv=False, native_shape=(4, 27), seed=11)
    torch.testing.assert_close(conv, native, rtol=0, atol=0)
