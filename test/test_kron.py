"""Proofs for the slab-native gradient-whitening PSGD-Kron port."""

from unittest.mock import patch

import torch

from heavyball.core import Engine
from heavyball.kron import kron


def test_kron_refresh_only_updates_Q():
    """Normal steps apply PSGD but retain both Q factors and lower bounds exactly."""

    torch.manual_seed(63)
    param = torch.nn.Parameter(torch.randn(3, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(
            [param],
            kron,
            lr=0.02,
            precond_lr=0.05,
            lower_bound_beta=0.9,
            dampening=1e-6,
            weight_decay=0.0,
        )
    state = optimizer.groups[0].states[0]
    initial = {name: value.clone() for name, value in state.items()}

    before = param.detach().clone()
    param.grad.copy_(torch.randn_like(param))
    optimizer.step(step_type="normal")
    assert not torch.equal(param, before)
    for name, value in initial.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    param.grad.copy_(torch.randn_like(param))
    torch.manual_seed(964)
    optimizer.step(step_type="refresh")
    refreshed = {name: value.clone() for name, value in state.items()}
    assert not torch.equal(refreshed["Q_0"], initial["Q_0"])
    assert not torch.equal(refreshed["Q_1"], initial["Q_1"])

    param.grad.copy_(torch.randn_like(param))
    optimizer.step(step_type="normal")
    for name, value in refreshed.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def test_kron_normal_and_refresh_are_stable_fullgraphs():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        parameter = torch.nn.Parameter(torch.randn(3, 4))
        optimizer = Engine(
            [parameter],
            kron,
            lr=0.01,
            precond_lr=0.05,
            dampening=1e-6,
            weight_decay=0.0,
        )
        state = optimizer.groups[0].states[0]
        initial_q = state["Q_0"].clone()
        parameter.grad.copy_(torch.arange(1, 13, dtype=torch.float32).reshape(3, 4))

        optimizer.step(step_type="normal")
        torch.testing.assert_close(state["Q_0"], initial_q, rtol=0, atol=0)
        normal_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="normal")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == normal_graphs == 1

        optimizer.step(step_type="refresh")
        assert not torch.equal(state["Q_0"], initial_q)
        refresh_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="refresh")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == refresh_graphs == 2
        assert torch.isfinite(parameter).all()
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()


def test_lower_bound_weights_history_like_li():
    """The running spectral lower bound weights HISTORY by beta (Li psgd_torch: max(ell, beta*L +
    (1-beta)*ell)), so the safety floor decays slowly when curvature falls. A reversed lerp (weighting the
    NEW estimate 90% at beta=0.9) would collapse the floor to 1.9; the legacy-parity gate missed this
    because a short random trajectory rarely has a falling spectral estimate. Derived from Li, not legacy."""
    import torch

    from heavyball.kron import _next_lower_bound

    beta = torch.tensor(0.9, dtype=torch.float64)
    lower_bound = torch.tensor([10.0], dtype=torch.float64)
    for ell_value in (10.0, 1.0):  # rising selects ell; falling must hold 90% of the 10.0 history -> 9.1
        ell = torch.tensor([ell_value], dtype=torch.float64)
        li_expected = torch.maximum(beta * lower_bound + (1 - beta) * ell, ell)  # Li's formula, independent
        _, lower_bound = _next_lower_bound(ell, lower_bound, beta)
        torch.testing.assert_close(lower_bound, li_expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(lower_bound, torch.tensor([9.1], dtype=torch.float64), rtol=0, atol=1e-12)


def test_kron_whitens_the_gradient_covariance():
    """PSGD-Kron's defining purpose, checked independently of legacy AND of the shipped code: for a
    stationary Kronecker-structured gradient distribution the preconditioner drives the covariance of the
    preconditioned gradient toward the identity. cov(vec(G)) = right (x) left is far from I; after Q
    converges the preconditioned-gradient covariance is near I. This catches a whitening bug even if
    shipped and legacy shared it (which a shipped-vs-legacy parity test cannot)."""

    torch.manual_seed(0)
    rows, cols = 6, 4
    left = (lambda factor: factor @ factor.T)(torch.randn(rows, rows, dtype=torch.float64))
    right = (lambda factor: factor @ factor.T)(torch.randn(cols, cols, dtype=torch.float64))
    left_sqrt = torch.linalg.cholesky(left)
    right_sqrt = torch.linalg.cholesky(right)
    learning_rate = 1e-3
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        param = torch.nn.Parameter(torch.zeros(rows, cols, dtype=torch.float64))
        optimizer = Engine([param], kron, lr=learning_rate, weight_decay=0.0, precond_lr=0.1)
        preconditioned = []
        for step in range(1500):
            before = param.detach().clone()
            param.grad.copy_(left_sqrt @ torch.randn(rows, cols, dtype=torch.float64) @ right_sqrt.T)
            optimizer.step()
            if step >= 900:  # the terminal sgd_commit is param -= lr * (P @ grad), so recover P @ grad
                preconditioned.append((-(param.detach() - before) / learning_rate).reshape(-1))
    samples = torch.stack(preconditioned)
    identity = torch.eye(rows * cols, dtype=torch.float64)
    raw = (torch.kron(right, left) - identity).norm() / identity.norm()
    whitened = (samples.T @ samples / samples.shape[0] - identity).norm() / identity.norm()
    assert raw > 10  # the injected gradient is strongly correlated
    assert whitened < 0.7  # PSGD-Kron drives its preconditioned-gradient covariance toward the identity
