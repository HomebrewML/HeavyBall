"""Oversized-dimension topology proofs for matrix preconditioners."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

import heavyball_legacy.chainable as legacy_chainable
from heavyball.core import Engine
from heavyball.kl import kl_shampoo_recipe, kl_soap_recipe
from heavyball.matrix import _transport_exp_avg_sq, shampoo_recipe, soap_recipe


@contextmanager
def _legacy_eager():
    import heavyball_legacy.utils as legacy

    previous = legacy.compile_mode
    legacy.compile_mode = None
    try:
        yield legacy
    finally:
        legacy.compile_mode = previous


@contextmanager
def _engine_eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _legacy_slot(state: dict, label: str, prefix: str):
    for key, value in state.items():
        if key.startswith(f"{prefix}{label}_"):
            return value
        if key.startswith("__bucket_") and isinstance(value, dict):
            found = _legacy_slot(value, label, prefix)
            if found is not None:
                return found
    raise KeyError(label)


def _legacy_transform(transform):
    """Create a legacy transform with its regular per-parameter state ABI."""

    (indexed,) = legacy_chainable.set_indices((transform,), retain=False)
    return indexed


def _legacy_group(
    *, beta1: float, beta2: float, shampoo_beta: float | None, init_factor: float, step: int, refresh: bool
) -> dict:
    group = {
        "betas": (beta1, beta2),
        "caution": False,
        "eps": 1e-8,
        "init_factor": init_factor,
        "is_preconditioning": refresh,
        "max_precond_dim": 16,
        "precondition_1d": False,
        "step": step,
        "storage_dtype": "float64",
    }
    if shampoo_beta is not None:
        group["shampoo_beta"] = shampoo_beta
    return group


def _copy_legacy_factors(state: dict[str, torch.Tensor], legacy_state: dict, *, prefix: str, labels: tuple[str, ...]) -> None:
    for label in labels:
        left, right = _legacy_slot(legacy_state, label, prefix)
        for key, value in ((f"{label}_l", left), (f"{label}_r", right)):
            if key in state:
                assert value is not None
                state[key].copy_(value)
            else:
                assert value is None


def _assert_legacy_factors(
    state: dict[str, torch.Tensor], legacy_state: dict, *, prefix: str, labels: tuple[str, ...]
) -> None:
    for label in labels:
        left, right = _legacy_slot(legacy_state, label, prefix)
        for key, value in ((f"{label}_l", left), (f"{label}_r", right)):
            if key in state:
                assert value is not None
                torch.testing.assert_close(state[key], value, rtol=1e-10, atol=1e-10)
            else:
                assert value is None


@pytest.mark.parametrize("shape", ((40, 6), (40, 24)))
@pytest.mark.parametrize(
    ("recipe", "legacy_scale", "prefix"),
    ((soap_recipe, legacy_chainable.scale_by_soap, "scale_by_soap_"),),
    ids=("soap",),
)
def test_skipped_factor_paths_match_legacy(recipe, legacy_scale, prefix, shape):
    """Legacy's ``None`` factor axes agree with the bind-time key topology."""

    torch.manual_seed(410 + shape[1])
    beta1, beta2 = 0.0, 0.0
    shampoo_beta = 0.7
    init_factor = 0.0
    legacy_parameter = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    legacy_state: dict = {}

    def state_fn(_parameter):
        return legacy_state

    transform = _legacy_transform(legacy_scale)
    bootstrap = torch.randn_like(legacy_parameter)
    with _legacy_eager():
        assert transform(
            state_fn,
            _legacy_group(
                beta1=beta1,
                beta2=beta2,
                shampoo_beta=shampoo_beta,
                init_factor=init_factor,
                step=1,
                refresh=False,
            ),
            [bootstrap.clone()],
            [bootstrap.clone()],
            [legacy_parameter],
        ) is legacy_chainable._SKIP

        parameter = torch.nn.Parameter(legacy_parameter.detach().clone())
        hyper = dict(
            lr=0.1,
            beta1=beta1,
            beta2=beta2,
            eps=1e-8,
            max_precond_dim=16,
            weight_decay=0.0,
            shampoo_beta=shampoo_beta,
        )
        with _engine_eager():
            optimizer = Engine([parameter], recipe, **hyper)
        state = optimizer.groups[0].states[0]
        labels = ("Q", "GG")
        _copy_legacy_factors(state, legacy_state, prefix=prefix, labels=labels)
        state["exp_avg"].copy_(_legacy_slot(legacy_state, "exp_avg", prefix))
        state["exp_avg_sq"].copy_(_legacy_slot(legacy_state, "exp_avg_sq", prefix))
        optimizer.groups[0].age.fill_(1)

        for step in range(2, 7):
            refresh = step in (3, 5)
            gradient = torch.randn_like(legacy_parameter)
            expected = transform(
                state_fn,
                _legacy_group(
                    beta1=beta1,
                    beta2=beta2,
                    shampoo_beta=shampoo_beta,
                    init_factor=init_factor,
                    step=step,
                    refresh=refresh,
                ),
                [gradient.clone()],
                [gradient.clone()],
                [legacy_parameter],
            )[0]
            before = parameter.detach().clone()
            old_left = state["Q_l"].clone() if "Q_l" in state else None
            old_right = state["Q_r"].clone() if "Q_r" in state else None
            parameter.grad.copy_(gradient)
            optimizer.step(step_type="refresh" if refresh else "normal")
            actual = (before - parameter) / 0.1
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            _assert_legacy_factors(state, legacy_state, prefix=prefix, labels=labels)
            torch.testing.assert_close(state["exp_avg"], _legacy_slot(legacy_state, "exp_avg", prefix), rtol=1e-10, atol=1e-10)
            expected_sq = _legacy_slot(legacy_state, "exp_avg_sq", prefix)
            if refresh:
                # The factory applies the analytical Hadamard-square second-moment transport that
                # legacy (pre-fix) omits, so its refreshed exp_avg_sq is legacy's value re-based.
                new_left = state["Q_l"] if "Q_l" in state else None
                new_right = state["Q_r"] if "Q_r" in state else None
                expected_sq = _transport_exp_avg_sq(expected_sq, old_left, old_right, new_left, new_right)
            torch.testing.assert_close(state["exp_avg_sq"], expected_sq, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("recipe", (kl_soap_recipe, kl_shampoo_recipe), ids=("kl_soap", "kl_shampoo"))
def test_kl_skipped_left_factor_topology_and_runs_finite(recipe):
    """A tall matrix skips the oversized left (row) factor; the KL None-left-factor path (with the
    seed-only step 1) runs finite. Legacy-independent: KL now diverges from the pre-fix legacy."""

    parameter = torch.nn.Parameter(torch.randn(40, 6, dtype=torch.float64))
    with _engine_eager():
        optimizer = Engine(
            [parameter],
            recipe,
            lr=0.1,
            beta1=0.9,
            beta2=0.8,
            eps=1e-8,
            max_precond_dim=16,
            weight_decay=0.0,
            **({"shampoo_beta": 0.7} if recipe is kl_soap_recipe else {}),
        )
    state = optimizer.groups[0].states[0]
    assert "GG_l" not in state and "eigenvalues_l" not in state
    assert "GG_r" in state and "eigenvalues_r" in state
    for _ in range(4):
        parameter.grad.copy_(torch.randn(40, 6, dtype=torch.float64))
        optimizer.step(step_type="normal")
    assert torch.isfinite(parameter).all()


def _inverse_fourth_root(gram: torch.Tensor, eps: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(gram)
    return (vectors * values.clamp_min(eps).rsqrt().sqrt().unsqueeze(-2)) @ vectors.mT


@pytest.mark.parametrize("shape", ((40, 6), (40, 24)))
def test_shampoo_skipped_factors_follow_legacy_none_topology(shape):
    """Shampoo leaves absent axes as identity, like legacy's ``None`` factor slots."""

    torch.manual_seed(420 + shape[1])
    with _legacy_eager() as legacy:
        factors = legacy._preconditioner_matrices(
            torch.empty((1, *shape), dtype=torch.float64), 16, False, torch.float64
        )
    left_active, right_active = (factor is not None for factor in factors)
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))
    with _engine_eager():
        optimizer = Engine([parameter], shampoo_recipe, lr=0.1, eps=1e-8, max_precond_dim=16, weight_decay=0.0)
    state = optimizer.groups[0].states[0]
    assert ("GG_l" in state, "GG_r" in state) == (left_active, right_active)
    left = torch.eye(shape[0], dtype=torch.float64).unsqueeze(0) if left_active else None
    right = torch.eye(shape[1], dtype=torch.float64).unsqueeze(0) if right_active else None
    gg_left = torch.zeros_like(left) if left_active else None
    gg_right = torch.zeros_like(right) if right_active else None

    for step in range(1, 6):
        refresh = step in (2, 4)
        gradient = torch.randn_like(parameter).unsqueeze(0)
        if gg_left is not None:
            gg_left = gg_left + gradient @ gradient.mT
        if gg_right is not None:
            gg_right = gg_right + gradient.mT @ gradient
        if refresh:
            if gg_left is not None:
                left = _inverse_fourth_root(gg_left, 1e-8)
            if gg_right is not None:
                right = _inverse_fourth_root(gg_right, 1e-8)
        expected = gradient
        if left is not None:
            expected = left @ expected
        if right is not None:
            expected = expected @ right

        before = parameter.detach().clone()
        parameter.grad.copy_(gradient[0])
        optimizer.step(step_type="refresh" if refresh else "normal")
        torch.testing.assert_close((before - parameter) / 0.1, expected[0], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("recipe", "factor_keys", "eigenvalue_keys"),
    (
        (soap_recipe, ("GG_l", "GG_r", "Q_l", "Q_r"), ()),
        (shampoo_recipe, ("GG_l", "GG_r", "L", "R"), ()),
        (kl_soap_recipe, ("GG_l", "GG_r", "Q_l", "Q_r"), ("eigenvalues_l", "eigenvalues_r")),
        (kl_shampoo_recipe, ("GG_l", "GG_r", "Q_l", "Q_r"), ("eigenvalues_l", "eigenvalues_r")),
    ),
    ids=("soap", "shampoo", "kl_soap", "kl_shampoo"),
)
def test_small_leaf_keeps_the_previous_both_factor_initialization(recipe, factor_keys, eigenvalue_keys):
    """Both-active leaves retain the original zero-Gram, identity-basis bytes."""

    parameter = torch.nn.Parameter(torch.zeros(6, 4, dtype=torch.float64))
    with _engine_eager():
        state = Engine([parameter], recipe, max_precond_dim=2048).groups[0].states[0]
    assert set(factor_keys) <= state.keys()
    for key, dimension in ((factor_keys[0], 6), (factor_keys[1], 4)):
        assert torch.equal(state[key], torch.zeros((1, dimension, dimension), dtype=torch.float64))
    for key, dimension in ((factor_keys[2], 6), (factor_keys[3], 4)):
        assert torch.equal(state[key], torch.eye(dimension, dtype=torch.float64).unsqueeze(0))
    if eigenvalue_keys:
        for key, dimension in zip(eigenvalue_keys, (6, 4), strict=True):
            assert torch.equal(state[key], torch.full((1, dimension), 0.1, dtype=torch.float64))


def test_soap_oversized_axis_avoids_large_factor_allocation():
    """A GPT-style skipped axis does not allocate any quadratic state slab."""

    parameter = torch.nn.Parameter(torch.zeros(4096, 8))
    with _engine_eager():
        state = Engine([parameter], soap_recipe, max_precond_dim=64).groups[0].states[0]
    assert "GG_l" not in state
    assert "Q_l" not in state
    assert {"GG_r", "Q_r"} <= state.keys()
    assert all(value.numel() <= 4096 * 8 for value in state.values())
