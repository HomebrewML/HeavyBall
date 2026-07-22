"""Regression proofs for mixed diagonal/triangular PSGD-Kron factors."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

import heavyball_legacy.chainable as legacy_chainable
from heavyball.core import Engine
from heavyball.kron import kron
from heavyball.transforms import Tempo


@contextmanager
def _legacy_eager():
    import heavyball_legacy.utils as legacy

    previous = legacy.compile_mode
    legacy.compile_mode = None
    try:
        yield
    finally:
        legacy.compile_mode = previous


def _eager_engine(params, **hyper) -> Engine:
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, kron, **hyper)


def _legacy_group(*, refresh: bool, step: int, max_size_triangular: int) -> dict:
    return {
        "caution": False,
        "dampening": 1e-6,
        "is_preconditioning": refresh,
        "lower_bound_beta": 0.9,
        "max_size_triangular": max_size_triangular,
        "memory_save_mode": None,
        "min_ndim_triangular": 2,
        "momentum_into_precond_update": True,
        "precond_grad_accum": False,
        "precond_init_scale": 1.0,
        "precond_init_scale_scale": 1.0,
        "precond_init_scale_power": None,
        "precond_lr": 0.05,
        "precond_update_power_iterations": 2,
        "q_dtype": "float64",
        "step": step,
        "step_count": step,
        "store_triu_as_line": False,
    }


def _legacy_slot(state: dict, label: str):
    prefix = f"scale_by_psgd_{label}_"
    for key, value in state.items():
        if key.startswith(prefix):
            return value
        if key.startswith("__bucket_") and isinstance(value, dict):
            found = _legacy_slot(value, label)
            if found is not None:
                return found
    return None


def _legacy_scale_by_psgd():
    (transform,) = legacy_chainable.set_indices((legacy_chainable.scale_by_psgd,), retain=False)
    return transform


def test_oversized_axis_uses_linear_factor_storage():
    parameter = torch.nn.Parameter(torch.randn(4096, 8))
    optimizer = _eager_engine([parameter], max_size_triangular=64)
    state = optimizer.groups[0].states[0]

    assert state["Q_0"].shape == (1, 4096)
    assert state["Q_1"].shape == (1, 8, 8)
    assert all(factor.numel() <= 4096 * 8 for factor in state.values())


def test_mixed_diagonal_triangular_kron_matches_legacy():
    threshold = 4
    tolerance = 1e-10

    with _legacy_eager():
        torch.manual_seed(52)
        legacy_param = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))
        legacy_state: dict = {}
        transform = _legacy_scale_by_psgd()

        def state_fn(_param):
            return legacy_state

        bootstrap = torch.randn_like(legacy_param)
        transform(
            state_fn,
            _legacy_group(refresh=False, step=0, max_size_triangular=threshold),
            [bootstrap.clone()],
            [bootstrap.clone()],
            [legacy_param],
        )

        opt_param = torch.nn.Parameter(legacy_param.detach().clone())
        optimizer = _eager_engine(
            [opt_param],
            lr=0.1,
            precond_lr=0.05,
            lower_bound_beta=0.9,
            dampening=1e-6,
            max_size_triangular=threshold,
            weight_decay=0.0,
        )
        state = optimizer.groups[0].states[0]
        legacy_q0, legacy_q1 = _legacy_slot(legacy_state, "Q")
        state["Q_0"].copy_(legacy_q0)
        state["Q_1"].copy_(legacy_q1)

        assert state["Q_0"].ndim == 2
        assert state["Q_1"].ndim == 3

        for step, refresh in enumerate((False, True, False, True), start=1):
            gradient = torch.randn_like(legacy_param)
            before = opt_param.detach().clone()
            opt_param.grad.copy_(gradient)
            probe_seed = 900 + step
            torch.manual_seed(probe_seed)
            probe = torch.randn(1, *gradient.shape, dtype=gradient.dtype, device=gradient.device)

            def fixed_dampen(value: torch.Tensor, damp: float):
                vector = probe.to(value)
                damping = damp + torch.finfo(value.dtype).eps * value.abs()
                return vector, value + damping * vector

            with patch("heavyball_legacy.chainable.utils.dampen_grad", fixed_dampen):
                expected = transform(
                    state_fn,
                    _legacy_group(refresh=refresh, step=step, max_size_triangular=threshold),
                    [gradient.clone()],
                    [gradient.clone()],
                    [legacy_param],
                )[0]

            with patch.object(Tempo, "randn_like", lambda _tempo, value: probe.to(value)):
                optimizer.step(step_type="refresh" if refresh else "normal")
            actual = (before - opt_param.detach()) / 0.1
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

            legacy_q0, legacy_q1 = _legacy_slot(legacy_state, "Q")
            torch.testing.assert_close(state["Q_0"], legacy_q0, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(state["Q_1"], legacy_q1, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("big_first", (True, False))
def test_kron_diagonal_factor_whitens_a_diagonal_axis_covariance(big_first):
    """A large axis uses a diagonal (not triangular) factor. PSGD-Kron's whitening property, restricted to
    what a diagonal factor can whiten: for a gradient whose large-axis covariance is diagonal and small-axis
    covariance is full, the preconditioned-gradient covariance goes toward the identity. Parametrized over
    axis order so BOTH _refresh_mixed_q branches are exercised -- q0-diagonal/q1-triangular (big axis first)
    and q0-triangular/q1-diagonal (big axis second), which use different einsum index patterns. Independent
    of legacy AND the shipped code."""

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
