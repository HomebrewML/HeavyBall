"""Proofs for the slab-native PSGD-PRO and QSGD ports."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import heavyball_legacy.chainable as legacy_chainable
from heavyball import adamw, psgd_pro_adamw, qsgd_adamw
from heavyball.core import Engine
from heavyball.psgd_pro import psgd_pro, psgd_pro_init, qsgd
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


def _eager_engine(params, recipe=psgd_pro, **hyper) -> Engine:
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, recipe, **hyper)


def _legacy_group(dtype: torch.dtype, *, refresh: bool, sqrt: bool, step: int) -> dict:
    return {
        "caution": False,
        "dampening": 1e-6,
        "is_preconditioning": refresh,
        "lower_bound_beta": 0.9,
        "max_size_triangular": 8,
        "memory_save_mode": None,
        "min_ndim_triangular": 2,
        "momentum_into_precond_update": True,
        "precond_grad_accum": False,
        "precond_init_scale": 1.0,
        "precond_init_scale_scale": 1.0,
        "precond_init_scale_power": None,
        "precond_lr": 0.05,
        "precond_update_power_iterations": 2,
        "q_dtype": str(dtype).removeprefix("torch."),
        "sqrt": sqrt,
        "step": step,
        "step_count": step,
        "store_triu_as_line": False,
    }


def _legacy_slot(state: dict, label: str):
    prefix = f"scale_by_psgd_pro_{label}_"
    for key, value in state.items():
        if key.startswith(prefix):
            return value
        if key.startswith("__bucket_") and isinstance(value, dict):
            found = _legacy_slot(value, label)
            if found is not None:
                return found
    return None


def _legacy_transform():
    (transform,) = legacy_chainable.set_indices((legacy_chainable.scale_by_psgd_pro,), retain=False)
    return transform


def _copy_legacy_state(optimizer: Engine, legacy_state: dict) -> None:
    state = optimizer.groups[0].states[0]
    q0, q1 = _legacy_slot(legacy_state, "Q")
    lower0, lower1 = _legacy_slot(legacy_state, "running_lower_bound")
    state["Q_0"].copy_(q0)
    state["Q_1"].copy_(q1)
    state["running_lower_bound_0"].copy_(lower0)
    state["running_lower_bound_1"].copy_(lower1)


def _assert_matches_legacy(*, sqrt: bool, shape: tuple[int, ...] = (3, 4), engine_max_size_triangular: int | None = None) -> None:
    recipe = qsgd if sqrt else psgd_pro
    with _legacy_eager():
        for dtype, tolerance in ((torch.float64, 1e-11), (torch.float32, 2e-5)):
            torch.manual_seed(52)
            legacy_param = torch.nn.Parameter(torch.randn(*shape, dtype=dtype))
            legacy_state: dict = {}
            transform = _legacy_transform()

            def state_fn(_param):
                return legacy_state

            bootstrap = torch.randn_like(legacy_param)
            transform(
                state_fn,
                _legacy_group(dtype, refresh=False, sqrt=sqrt, step=0),
                [bootstrap.clone()],
                [bootstrap.clone()],
                [legacy_param],
            )

            opt_param = torch.nn.Parameter(legacy_param.detach().clone())
            engine_hyper = dict(lr=0.1, precond_lr=0.05, lower_bound_beta=0.9, dampening=1e-6, weight_decay=0.0)
            if engine_max_size_triangular is not None:
                engine_hyper["max_size_triangular"] = engine_max_size_triangular
            optimizer = _eager_engine([opt_param], recipe, **engine_hyper)
            _copy_legacy_state(optimizer, legacy_state)

            for step, refresh in enumerate((False, True, False, True, True), start=1):
                gradient = torch.randn_like(legacy_param)
                before = opt_param.detach().clone()
                opt_param.grad.copy_(gradient)
                probe_seed = 900 + step

                torch.manual_seed(probe_seed)
                expected = transform(
                    state_fn,
                    _legacy_group(dtype, refresh=refresh, sqrt=sqrt, step=step),
                    [gradient.clone()],
                    [gradient.clone()],
                    [legacy_param],
                )[0]

                torch.manual_seed(probe_seed)
                with patch.object(Tempo, "randn_like", lambda _tempo, value: torch.randn_like(value)):
                    optimizer.step(step_type="refresh" if refresh else "normal")
                actual = (before - opt_param.detach()) / 0.1
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

                state = optimizer.groups[0].states[0]
                legacy_q0, legacy_q1 = _legacy_slot(legacy_state, "Q")
                legacy_lower0, legacy_lower1 = _legacy_slot(legacy_state, "running_lower_bound")
                torch.testing.assert_close(state["Q_0"], legacy_q0, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(state["Q_1"], legacy_q1, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(
                    state["running_lower_bound_0"], legacy_lower0, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    state["running_lower_bound_1"], legacy_lower1, rtol=tolerance, atol=tolerance
                )


def test_psgd_pro_matches_legacy():
    """PRO updates and Q factors directly match legacy with synchronized probes."""

    _assert_matches_legacy(sqrt=False)


def test_qsgd_matches_legacy():
    """QSGD directly matches legacy's apply-once path with synchronized probes."""

    _assert_matches_legacy(sqrt=True)


@pytest.mark.parametrize("recipe", (psgd_pro, qsgd), ids=("psgd_pro", "qsgd"))
def test_psgd_pro_mixed_oversized_compiles_and_matches_eager(recipe):
    """The mixed diagonal/triangular path compiles (compile-first) and the compiled step matches eager."""

    def trajectory(compiled: bool):
        torch.manual_seed(71)
        param = torch.nn.Parameter(torch.randn(12, 4))  # axis0=12 diagonal, axis1=4 triangular at msq=8
        gradients = [torch.randn(12, 4) for _ in range(4)]
        if compiled:
            torch._dynamo.reset()
            optimizer = Engine([param], recipe, lr=0.02, precond_lr=0.05, dampening=1e-6, weight_decay=0.0, max_size_triangular=8)
        else:
            optimizer = _eager_engine([param], recipe, lr=0.02, precond_lr=0.05, dampening=1e-6, weight_decay=0.0, max_size_triangular=8)
        for step, gradient in enumerate(gradients, start=1):
            param.grad.copy_(gradient)
            torch.manual_seed(500 + step)
            optimizer.step(step_type="refresh" if step in (2, 4) else "normal")
        return param.detach().clone()

    compiled = trajectory(compiled=True)
    eager = trajectory(compiled=False)
    assert torch.isfinite(compiled).all()
    torch.testing.assert_close(compiled, eager, rtol=2e-5, atol=2e-5)


def test_psgd_pro_mixed_oversized_axis_matches_legacy():
    """An oversized axis (> max_size_triangular) takes a diagonal factor and matches legacy's mixed update."""

    _assert_matches_legacy(sqrt=False, shape=(12, 4), engine_max_size_triangular=8)


def test_qsgd_mixed_oversized_axis_matches_legacy():
    """QSGD apply-once with a diagonal factor on the oversized axis matches legacy."""

    _assert_matches_legacy(sqrt=True, shape=(12, 4), engine_max_size_triangular=8)


def _run_trajectory(
    dtype: torch.dtype,
    *,
    recipe,
    initial: list[torch.Tensor],
    gradients: list[list[torch.Tensor]],
    probes: list[torch.Tensor],
) -> list[torch.Tensor]:
    params = [torch.nn.Parameter(value.to(dtype).clone()) for value in initial]
    optimizer = _eager_engine(
        params,
        recipe,
        lr=1e-3,
        precond_lr=0.01,
        lower_bound_beta=0.9,
        dampening=1e-6,
        weight_decay=0.0,
    )
    probe_index = 0

    def fixed_probe(_tempo: Tempo, update: torch.Tensor) -> torch.Tensor:
        nonlocal probe_index
        probe = probes[probe_index].to(device=update.device, dtype=update.dtype)
        probe_index += 1
        return probe

    with patch.object(Tempo, "randn_like", fixed_probe):
        for step, step_gradients in enumerate(gradients, start=1):
            for param, gradient in zip(params, step_gradients, strict=True):
                param.grad.copy_(gradient.to(dtype))
            optimizer.step(step_type="refresh" if step in (2, 5, 7) else "normal")
    assert probe_index == len(probes)
    return [param.detach().clone() for param in params]


def _assert_fp64_accuracy(capsys, recipe, label: str, budget: float) -> None:
    torch.manual_seed(41)
    initial = [torch.randn(3, 4, dtype=torch.float64) for _ in range(2)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(7)]
    torch.manual_seed(700)
    probes = [torch.randn(len(initial), *initial[0].shape, dtype=torch.float64) for _ in range(3)]
    truth = _run_trajectory(torch.float64, recipe=recipe, initial=initial, gradients=gradients, probes=probes)
    actual = _run_trajectory(torch.float32, recipe=recipe, initial=initial, gradients=gradients, probes=probes)
    error = max((result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True))
    with capsys.disabled():
        print(f"{label} fp64 max error: {float(error):.9e}")
    assert error <= budget


def test_psgd_pro_fp64_accuracy(capsys):
    _assert_fp64_accuracy(capsys, psgd_pro, "psgd_pro", 1e-6)


def test_qsgd_fp64_accuracy(capsys):
    _assert_fp64_accuracy(capsys, qsgd, "qsgd", 1e-6)


@pytest.mark.parametrize("recipe", (psgd_pro, qsgd), ids=("psgd_pro", "qsgd"))
def test_psgd_pro_refresh_only_updates_Q(recipe):
    """Normal steps apply the current Q factors without changing their state."""

    torch.manual_seed(63)
    param = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = _eager_engine(
        [param],
        recipe,
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


def _compiled_codes(tmp_path: Path, recipe_name: str) -> tuple[str, str]:
    source = f"""
import torch
from heavyball.core import Engine
from heavyball.psgd_pro import {recipe_name}

params = [torch.nn.Parameter(torch.randn(2, 2))]
optimizer = Engine(params, {recipe_name}, lr=0.01, precond_lr=0.05, dampening=1e-6, weight_decay=0.0)
optimizer.groups[0].grad_slab.normal_()
optimizer.step(step_type="normal")
optimizer.step(step_type="refresh")
"""
    environment = dict(
        os.environ,
        TORCH_LOGS="output_code",
        TORCHINDUCTOR_FX_GRAPH_CACHE="0",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / f"{recipe_name}_artifacts"),
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    paths = [Path(path) for path in re.findall(r"Output code written to: (.*\.py)", output)]
    assert paths, output
    artifacts = [path.read_text() for path in dict.fromkeys(paths)]
    normal = next((artifact for artifact in artifacts if "log1p" not in artifact.lower()), None)
    refresh = next((artifact for artifact in artifacts if "log1p" in artifact.lower()), None)
    assert normal is not None, output
    assert refresh is not None, output
    return normal, refresh


@pytest.mark.parametrize("recipe_name", ("psgd_pro", "qsgd"))
def test_psgd_pro_fullgraph_clean(tmp_path, recipe_name):
    """Both host-selected artifacts compile without banned graph operations."""

    normal, refresh = (artifact.lower() for artifact in _compiled_codes(tmp_path, recipe_name))
    assert "log1p" not in normal
    assert "log1p" in refresh
    for artifact in (normal, refresh):
        assert "torch.stack" not in artifact
        assert "while_loop" not in artifact
        assert not re.search(r"torch\\.cond|\\bcond\\b", artifact)
        assert "_local_scalar_dense" not in artifact
        assert ".item(" not in artifact

    source = (Path(__file__).parents[1] / "heavyball" / "psgd_pro.py").read_text()
    assert not re.search(r"_foreach|vmap|while_loop|torch\.cond|dynamic=True|torch\.stack|\.item\(|autocast", source)
    assert "heavyball_legacy.utils" not in source
    assert "heavyball_legacy.chainable" not in source


def test_psgd_pro_rejects_leaves_that_do_not_merge_to_2d():
    with pytest.raises(ValueError, match="dimensions merge to 2D"):
        psgd_pro_init(torch.zeros(4))


@pytest.mark.parametrize("route", (psgd_pro_adamw, qsgd_adamw))
def test_psgd_pro_adamw_routes_matrices_not_vectors(route):
    matrix = torch.nn.Parameter(torch.zeros(2, 3))
    vector = torch.nn.Parameter(torch.zeros(3))
    optimizer = _eager_engine([matrix, vector], route, lr=1e-3, weight_decay=0.0)
    assert optimizer.groups[0].recipe in (psgd_pro, qsgd)
    assert optimizer.groups[1].recipe is adamw


def test_psgd_pro_whitens_the_gradient_covariance():
    """PSGD-PRO's defining purpose, checked independently of legacy AND of the shipped code: for a
    stationary Kronecker-structured gradient distribution the P = QᵀQ preconditioner drives the
    preconditioned-gradient covariance toward the identity. Catches a correlated whitening bug (shared
    by shipped and legacy) that a shipped-vs-legacy parity test cannot."""

    torch.manual_seed(0)
    rows, cols = 6, 4
    left = (lambda factor: factor @ factor.T)(torch.randn(rows, rows, dtype=torch.float64))
    right = (lambda factor: factor @ factor.T)(torch.randn(cols, cols, dtype=torch.float64))
    left_sqrt = torch.linalg.cholesky(left)
    right_sqrt = torch.linalg.cholesky(right)
    learning_rate = 1e-3
    param = torch.nn.Parameter(torch.zeros(rows, cols, dtype=torch.float64))
    optimizer = _eager_engine([param], psgd_pro, lr=learning_rate, weight_decay=0.0, precond_lr=0.1)
    preconditioned = []
    for step in range(1500):
        before = param.detach().clone()
        param.grad.copy_(left_sqrt @ torch.randn(rows, cols, dtype=torch.float64) @ right_sqrt.T)
        optimizer.step()
        if step >= 900:
            preconditioned.append((-(param.detach() - before) / learning_rate).reshape(-1))
    samples = torch.stack(preconditioned)
    identity = torch.eye(rows * cols, dtype=torch.float64)
    raw = (torch.kron(right, left) - identity).norm() / identity.norm()
    whitened = (samples.T @ samples / samples.shape[0] - identity).norm() / identity.norm()
    assert raw > 10
    assert whitened < 0.7


@pytest.mark.parametrize("big_first", (True, False))
def test_psgd_pro_diagonal_factor_whitens_a_diagonal_axis_covariance(big_first):
    """PSGD-PRO with a large axis forced to a diagonal factor, both axis orders, so BOTH branches of the
    mixed-factor refresh are exercised: q0-triangular/q1-diagonal (big axis second) uses distinct einsum
    index patterns from the q0-diagonal/q1-triangular case (big axis first). Verifies the mixed
    preconditioner still whitens a diagonal-axis covariance. Independent of legacy AND the shipped code."""

    torch.manual_seed(0)
    big, small = 12, 4
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
    assert raw > 1
    assert whitened < 0.7
