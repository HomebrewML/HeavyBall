"""Regression proofs for mixed diagonal/triangular PSGD-Kron factors."""

from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.kron import kron


def _eager_engine(params, **hyper) -> Engine:
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, kron, **hyper)


def test_oversized_axis_uses_linear_factor_storage():
    parameter = torch.nn.Parameter(torch.randn(4096, 8))
    optimizer = _eager_engine([parameter], max_size_triangular=64)
    state = optimizer.groups[0].states[0]

    assert state["Q_0"].shape == (1, 4096)
    assert state["Q_1"].shape == (1, 8, 8)
    assert all(factor.numel() <= 4096 * 8 for factor in state.values())


def test_mixed_diagonal_triangular_kron_applies_q_transpose_q():
    """A normal step applies the explicit Kronecker product ``QᵀQ`` on both axes."""

    parameter = torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64))
    optimizer = _eager_engine(
        [parameter],
        lr=0.1,
        max_size_triangular=4,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    diagonal = torch.linspace(0.7, 1.3, 5, dtype=torch.float64)
    triangular = torch.tensor(
        ((1.1, -0.2, 0.3), (0.0, 0.8, -0.4), (0.0, 0.0, 1.4)),
        dtype=torch.float64,
    )
    state["Q_0"][0].copy_(diagonal)
    state["Q_1"][0].copy_(triangular)
    before_state = {name: value.clone() for name, value in state.items()}
    gradient = torch.arange(1, 16, dtype=torch.float64).reshape(5, 3) / 7
    expected = diagonal.square().unsqueeze(1) * gradient
    expected = expected @ (triangular.mT @ triangular)

    parameter.grad.copy_(gradient)
    optimizer.step(step_type="normal")

    torch.testing.assert_close(-parameter / 0.1, expected, rtol=1e-12, atol=1e-12)
    for name, value in before_state.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("big_first", (True, False))
def test_kron_diagonal_factor_whitens_a_diagonal_axis_covariance(big_first):
    """A large axis uses a diagonal (not triangular) factor. PSGD-Kron's whitening property, restricted to
    what a diagonal factor can whiten: for a gradient whose large-axis covariance is diagonal and small-axis
    covariance is full, the preconditioned-gradient covariance goes toward the identity. Parametrized over
    axis order so BOTH _refresh_mixed_q branches are exercised -- q0-diagonal/q1-triangular (big axis first)
    and q0-triangular/q1-diagonal (big axis second), which use different einsum index patterns. Independent
    independently of the optimizer implementation."""

    torch.manual_seed(0)
    big, small = 12, 4  # big axis (>8) uses a diagonal factor; small axis (<=8) a triangular one
    diagonal = torch.rand(big, dtype=torch.float64) + 0.2
    full = (lambda factor: factor @ factor.T)(torch.randn(small, small, dtype=torch.float64))
    diagonal_sqrt, full_sqrt = diagonal.sqrt(), torch.linalg.cholesky(full)
    learning_rate = 1e-3
    if big_first:
        shape, diagonal_axis, covariance = (big, small), 0, torch.kron(full, torch.diag(diagonal))

        def draw():
            return (diagonal_sqrt.unsqueeze(1) * torch.randn(big, small, dtype=torch.float64)) @ full_sqrt.T
    else:
        shape, diagonal_axis, covariance = (small, big), 1, torch.kron(torch.diag(diagonal), full)

        def draw():
            return full_sqrt @ (torch.randn(small, big, dtype=torch.float64) * diagonal_sqrt)

    param = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float64))
    optimizer = _eager_engine([param], lr=learning_rate, weight_decay=0.0, precond_lr=0.1, max_size_triangular=8)
    assert optimizer.groups[0].states[0][f"Q_{diagonal_axis}"].dim() == 2  # the big-axis factor is diagonal
    preconditioned = []
    for step in range(2500):
        before = param.detach().clone()
        param.grad.copy_(draw())
        optimizer.step()
        if step >= 1800:
            preconditioned.append((-(param.detach() - before) / learning_rate).reshape(-1))
    samples = torch.stack(preconditioned)
    identity = torch.eye(big * small, dtype=torch.float64)
    raw = (covariance - identity).norm() / identity.norm()
    whitened = (samples.T @ samples / samples.shape[0] - identity).norm() / identity.norm()
    assert raw > 1  # the injected gradient is correlated
    assert whitened < 0.7  # the diagonal + triangular factors drive its covariance toward the identity
