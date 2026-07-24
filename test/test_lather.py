"""Proofs for the slab-native LATHER port."""

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

import heavyball_legacy.chainable as legacy_chainable
from heavyball.core import Engine
from heavyball.lather import lather
from heavyball.matrix import _project
from heavyball.transforms import Tempo, beta_debias


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
    """Build the exact Engine path without compiling either step artifact."""

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, lather, **hyper)


def _legacy_group(dtype: torch.dtype, *, refresh: bool, step: int) -> dict:
    return {
        "betas": (0.9, 0.999),
        "caution": False,
        "dampening": 1e-6,
        "eps": 1e-8,
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
        "step": step,
        "step_count": step,
        "storage_dtype": str(dtype).removeprefix("torch."),
        "store_triu_as_line": False,
    }


def _legacy_slot(state: dict, label: str):
    prefix = f"scale_by_lather_{label}_"
    for key, value in state.items():
        if key.startswith(prefix):
            return value
        if key.startswith("__bucket_") and isinstance(value, dict):
            found = _legacy_slot(value, label)
            if found is not None:
                return found
    return None


def _legacy_scale_by_lather():
    (transform,) = legacy_chainable.set_indices((legacy_chainable.scale_by_lather,), retain=False)
    return transform


def _copy_legacy_state(optimizer: Engine, legacy_state: dict) -> None:
    state = optimizer.groups[0].states[0]
    q0, q1 = _legacy_slot(legacy_state, "Q")
    basis0, basis1 = _legacy_slot(legacy_state, "Q_basis")
    lower0, lower1 = _legacy_slot(legacy_state, "running_lower_bound")
    state["Q_0"].copy_(q0)
    state["Q_1"].copy_(q1)
    for key, basis in (("Q_basis_0", basis0), ("Q_basis_1", basis1)):
        if basis is None:
            assert key not in state
        else:
            state[key].copy_(basis)
    state["running_lower_bound_0"].copy_(lower0)
    state["running_lower_bound_1"].copy_(lower1)
    state["exp_avg"].copy_(_legacy_slot(legacy_state, "exp_avg"))
    state["exp_avg_sq"].copy_(_legacy_slot(legacy_state, "exp_avg_sq"))


def test_lather_matches_legacy():
    """RNG-synced direct parity with legacy ``scale_by_lather`` and all its state."""

    with _legacy_eager():
        for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
            torch.manual_seed(52)
            legacy_param = torch.nn.Parameter(torch.randn(3, 4, dtype=dtype))
            legacy_state: dict = {}
            transform = _legacy_scale_by_lather()

            def state_fn(_param):
                return legacy_state

            bootstrap = torch.randn_like(legacy_param)
            assert transform(
                state_fn,
                _legacy_group(dtype, refresh=False, step=1),
                [bootstrap.clone()],
                [bootstrap.clone()],
                [legacy_param],
            ) is legacy_chainable._SKIP

            opt_param = torch.nn.Parameter(legacy_param.detach().clone())
            optimizer = _eager_engine(
                [opt_param],
                lr=0.1,
                precond_lr=0.05,
                lower_bound_beta=0.9,
                dampening=1e-6,
                weight_decay=0.0,
            )
            _copy_legacy_state(optimizer, legacy_state)

            for step, refresh in enumerate((False, True, False, True), start=2):
                gradient = torch.randn_like(legacy_param)
                before = opt_param.detach().clone()
                opt_param.grad.copy_(gradient)
                probe_seed = 900 + step

                torch.manual_seed(probe_seed)
                expected = transform(
                    state_fn,
                    _legacy_group(dtype, refresh=refresh, step=step),
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
                legacy_basis0, legacy_basis1 = _legacy_slot(legacy_state, "Q_basis")
                legacy_lower0, legacy_lower1 = _legacy_slot(legacy_state, "running_lower_bound")
                torch.testing.assert_close(state["Q_0"], legacy_q0, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(state["Q_1"], legacy_q1, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(state["Q_basis_0"], legacy_basis0, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(state["Q_basis_1"], legacy_basis1, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(
                    state["running_lower_bound_0"], legacy_lower0, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    state["running_lower_bound_1"], legacy_lower1, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    state["exp_avg"], _legacy_slot(legacy_state, "exp_avg"), rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    state["exp_avg_sq"],
                    _legacy_slot(legacy_state, "exp_avg_sq"),
                    rtol=tolerance,
                    atol=tolerance,
                )


def test_lather_mixed_oversized_axis_matches_legacy(capsys):
    """A diagonal axis omits its basis and remains fp64-equivalent to legacy."""

    tolerance = 1e-10
    max_abs_diff = 0.0
    with _legacy_eager():
        torch.manual_seed(52)
        legacy_param = torch.nn.Parameter(torch.randn(12, 5, dtype=torch.float64))
        legacy_state: dict = {}
        transform = _legacy_scale_by_lather()

        def state_fn(_param):
            return legacy_state

        bootstrap = torch.randn_like(legacy_param)
        assert transform(
            state_fn,
            _legacy_group(torch.float64, refresh=False, step=1),
            [bootstrap.clone()],
            [bootstrap.clone()],
            [legacy_param],
        ) is legacy_chainable._SKIP

        opt_param = torch.nn.Parameter(legacy_param.detach().clone())
        optimizer = _eager_engine(
            [opt_param],
            lr=0.1,
            precond_lr=0.05,
            lower_bound_beta=0.9,
            dampening=1e-6,
            max_size_triangular=8,
            weight_decay=0.0,
        )
        _copy_legacy_state(optimizer, legacy_state)
        state = optimizer.groups[0].states[0]
        assert state["Q_0"].shape == (1, 12)
        assert state["Q_1"].shape == (1, 5, 5)
        assert state["Q_0"][0].ndim == 1
        assert state["Q_1"][0].ndim == 2
        assert "Q_basis_0" not in state
        assert "Q_basis_1" in state

        for step, refresh in enumerate((True, False, False, True, False), start=2):
            gradient = torch.randn_like(legacy_param)
            before = opt_param.detach().clone()
            opt_param.grad.copy_(gradient)
            probe_seed = 900 + step

            torch.manual_seed(probe_seed)
            expected = transform(
                state_fn,
                _legacy_group(torch.float64, refresh=refresh, step=step),
                [gradient.clone()],
                [gradient.clone()],
                [legacy_param],
            )[0]

            torch.manual_seed(probe_seed)
            with patch.object(Tempo, "randn_like", lambda _tempo, value: torch.randn_like(value)):
                optimizer.step(step_type="refresh" if refresh else "normal")
            actual = (before - opt_param.detach()) / 0.1
            max_abs_diff = max(max_abs_diff, float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

            legacy_q0, legacy_q1 = _legacy_slot(legacy_state, "Q")
            _, legacy_basis1 = _legacy_slot(legacy_state, "Q_basis")
            legacy_lower0, legacy_lower1 = _legacy_slot(legacy_state, "running_lower_bound")
            torch.testing.assert_close(state["Q_0"], legacy_q0, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(state["Q_1"], legacy_q1, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(state["Q_basis_1"], legacy_basis1, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(
                state["running_lower_bound_0"], legacy_lower0, rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(
                state["running_lower_bound_1"], legacy_lower1, rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(
                state["exp_avg"], _legacy_slot(legacy_state, "exp_avg"), rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(
                state["exp_avg_sq"],
                _legacy_slot(legacy_state, "exp_avg_sq"),
                rtol=tolerance,
                atol=tolerance,
            )
            assert "Q_basis_0" not in state
            assert "Q_basis_1" in state

    with capsys.disabled():
        print(f"lather mixed oversized fp64 max abs diff: {max_abs_diff:.9e}")


def test_lather_both_oversized_axes_use_linear_factor_storage():
    parameter = torch.nn.Parameter(torch.randn(12, 10))
    optimizer = _eager_engine([parameter], max_size_triangular=8)
    state = optimizer.groups[0].states[0]

    assert state["Q_0"].shape == (1, 12)
    assert state["Q_1"].shape == (1, 10)
    assert "Q_basis_0" not in state
    assert "Q_basis_1" not in state


def _run_trajectory(
    dtype: torch.dtype,
    *,
    initial: list[torch.Tensor],
    gradients: list[list[torch.Tensor]],
    probes: list[torch.Tensor],
) -> list[torch.Tensor]:
    params = [torch.nn.Parameter(value.to(dtype).clone()) for value in initial]
    optimizer = _eager_engine(
        params,
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


def test_lather_fp64_accuracy(capsys):
    """The fp32 LATHER trajectory remains close to shared-probe fp64 truth."""

    torch.manual_seed(41)
    initial = [torch.randn(3, 4, dtype=torch.float64) for _ in range(2)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(7)]
    torch.manual_seed(700)
    probes = [torch.randn(len(initial), *initial[0].shape, dtype=torch.float64) for _ in range(3)]
    truth = _run_trajectory(torch.float64, initial=initial, gradients=gradients, probes=probes)
    actual = _run_trajectory(torch.float32, initial=initial, gradients=gradients, probes=probes)
    error = max((result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True))
    with capsys.disabled():
        print(f"lather fp64 max error: {float(error):.9e}")
    assert error <= 1e-6


def test_lather_refresh_transports():
    """Only refreshes change Q/bases, and the first moment keeps its physical value."""

    param = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer = _eager_engine(
        [param],
        lr=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        precond_lr=0.05,
        lower_bound_beta=0.9,
        dampening=1e-6,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    first = torch.tensor(((3.0, 0.0, 1.0, 0.0), (0.0, 2.0, 0.0, 1.0), (1.0, 0.0, 1.0, 2.0)))
    second = torch.tensor(((0.0, 1.0, 2.0, 0.0), (2.0, 1.0, 0.0, 3.0), (1.0, 3.0, 1.0, 0.0)))
    third = torch.tensor(((1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 3.0, 0.0), (2.0, 0.0, 1.0, 2.0)))
    q_names = (
        "Q_0",
        "Q_1",
        "Q_basis_0",
        "Q_basis_1",
        "running_lower_bound_0",
        "running_lower_bound_1",
    )
    initial = {name: state[name].clone() for name in q_names}

    param.grad.copy_(first)
    optimizer.step(step_type="normal")
    for name, value in initial.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    old_left, old_right = state["Q_basis_0"].clone(), state["Q_basis_1"].clone()
    old_avg, old_avg_sq = state["exp_avg"].clone(), state["exp_avg_sq"].clone()
    age = torch.full((1,), 2, dtype=torch.int64)
    hyper = optimizer.groups[0].hyper
    projected = _project(second.unsqueeze(0), old_left, old_right, back=False)
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    raw_avg = old_avg * beta1 + projected * (1 - beta1)
    raw_avg_sq = old_avg_sq * beta2 + projected.square() * (1 - beta2)
    physical_before_transport = _project(raw_avg, old_left, old_right, back=True)

    param.grad.copy_(second)
    torch.manual_seed(964)
    optimizer.step(step_type="refresh")
    refreshed = {name: state[name].clone() for name in q_names}
    assert not torch.equal(refreshed["Q_0"], initial["Q_0"])
    assert not torch.equal(refreshed["Q_1"], initial["Q_1"])
    assert not torch.equal(refreshed["Q_basis_0"], old_left)
    assert not torch.equal(refreshed["Q_basis_1"], old_right)

    physical_after_transport = _project(
        state["exp_avg"], refreshed["Q_basis_0"], refreshed["Q_basis_1"], back=True
    )
    torch.testing.assert_close(physical_after_transport, physical_before_transport, rtol=2e-5, atol=2e-5)
    left_transition = torch.einsum("nia,nic->nac", old_left, refreshed["Q_basis_0"]).square()
    right_transition = torch.einsum("njb,njd->nbd", old_right, refreshed["Q_basis_1"]).square()
    expected_sq = torch.einsum("nab,nac,nbd->ncd", raw_avg_sq, left_transition, right_transition).clamp_min(0)
    torch.testing.assert_close(state["exp_avg_sq"], expected_sq, rtol=2e-5, atol=2e-5)
    assert not torch.equal(state["exp_avg"], raw_avg)
    assert not torch.equal(state["exp_avg_sq"], raw_avg_sq)

    param.grad.copy_(third)
    optimizer.step(step_type="normal")
    for name, value in refreshed.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def _compiled_codes(tmp_path: Path) -> tuple[str, str]:
    source = """
import torch
from heavyball.core import Engine
from heavyball.lather import lather

params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
optimizer = Engine(params, lather, lr=0.01, precond_lr=0.05, dampening=1e-6, weight_decay=0.0)
optimizer.groups[0].grad_slab.normal_()
optimizer.step(step_type="normal")
optimizer.step(step_type="refresh")
"""
    environment = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "artifacts"))
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
    normal = next((artifact for artifact in artifacts if "linalg_qr" not in artifact.lower()), None)
    refresh = next((artifact for artifact in artifacts if "linalg_qr" in artifact.lower()), None)
    assert normal is not None, output
    assert refresh is not None, output
    return normal, refresh


def test_lather_fullgraph_clean(tmp_path):
    """Normal LATHER omits basis work; refresh is QR-bearing and fullgraph clean."""

    normal, refresh = (artifact.lower() for artifact in _compiled_codes(tmp_path))
    assert "linalg_qr" not in normal
    assert "linalg_eigh" not in normal
    assert "linalg_qr" in refresh
    for artifact in (normal, refresh):
        assert "torch.stack" not in artifact
        assert "while_loop" not in artifact
        assert not re.search(r"torch\\.cond|\\bcond\\b", artifact)
        assert "_local_scalar_dense" not in artifact
        assert ".item(" not in artifact

    source = (Path(__file__).parents[1] / "heavyball" / "lather.py").read_text()
    assert not re.search(
        r"_foreach|vmap|torch\.cond|while_loop|dynamic=True|torch\.stack|\.item\(|autocast|\benabled\b|\bamp\b",
        source,
    )
    assert "heavyball_legacy.utils" not in source
    assert "heavyball_legacy.chainable" not in source
