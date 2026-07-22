"""PSGD-Kron preconditions convolutions via bind-time dimension merging.

Parity to legacy is by reduction (as in test_dim_merge): the merge reproduces legacy
dim_merger, and it is a byte-exact reshape drawn before the whitening probe, so
conv-kron equals native-2D kron and inherits the native-2D legacy parity test_kron proves.
"""

from unittest.mock import patch

import torch

from heavyball import build, kron, kron_adamw
from heavyball.matrix import matrix_route, merged_matrix_shape


class _Info:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


def test_conv_merges_to_2d_and_routes_to_kron():
    assert merged_matrix_shape((4, 3, 3, 3), 2048) == (4, 27)
    assert matrix_route(_Info((4, 3, 3, 3))) is True       # conv -> kron branch
    assert matrix_route(_Info((128, 1000, 3, 3))) is False  # merges to 3D -> adamw


def test_conv_kron_routes_preconditions_and_keeps_shape():
    parameter = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], kron_adamw, lr=1e-3)
    state = optimizer.groups[0].states[0]
    assert "Q_0" in state and tuple(state["Q_0"].shape[1:]) == (4, 4)
    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step(step_type="refresh")
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert tuple(parameter.shape) == (4, 3, 3, 3)


def _run_kron(conv0, gradients, *, as_conv, native_shape, seed):
    torch.manual_seed(seed)  # sync Tempo's stateless whitening-probe seed across the two runs
    parameter = torch.nn.Parameter(conv0.clone() if as_conv else conv0.reshape(native_shape).clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = build([parameter], kron, lr=1e-3, weight_decay=0.0)
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_((gradient if as_conv else gradient.reshape(native_shape)).clone())
        optimizer.step(step_type="refresh" if step == 0 else "normal")
    return parameter.detach().reshape(native_shape).clone()


def test_conv_kron_equals_native_2d_rng_synced():
    """The merge is a byte-exact reshape drawn before the probe, so conv == native 2D."""

    torch.manual_seed(0)
    conv0 = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    gradients = [torch.randn(4, 3, 3, 3, dtype=torch.float64) for _ in range(6)]
    conv = _run_kron(conv0, gradients, as_conv=True, native_shape=(4, 27), seed=7)
    native = _run_kron(conv0, gradients, as_conv=False, native_shape=(4, 27), seed=7)
    torch.testing.assert_close(conv, native, rtol=0, atol=0)
