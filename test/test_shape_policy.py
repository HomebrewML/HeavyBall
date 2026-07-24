"""Oversized-dimension topology proofs for matrix preconditioners."""

import math
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.kl import kl_shampoo_recipe, kl_soap_recipe
from heavyball.matrix import shampoo_recipe, soap_recipe


@contextmanager
def _engine_eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _project(
    matrix: torch.Tensor,
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    back: bool,
) -> torch.Tensor:
    if left is not None:
        matrix = (left if back else left.mT) @ matrix
    if right is not None:
        matrix = matrix @ (right.mT if back else right)
    return matrix


def _transport_variance(
    variance: torch.Tensor,
    old_left: torch.Tensor | None,
    old_right: torch.Tensor | None,
    new_left: torch.Tensor | None,
    new_right: torch.Tensor | None,
) -> torch.Tensor:
    if old_left is not None:
        variance = (old_left.mT @ new_left).square().mT @ variance
    if old_right is not None:
        variance = variance @ (old_right.mT @ new_right).square()
    return variance


@pytest.mark.parametrize(
    ("shape", "active"),
    (((40, 6), (False, True)), ((40, 24), (False, False))),
)
def test_soap_skipped_factor_paths_follow_explicit_policy_and_moment_math(shape, active):
    """Axes above the limit are identities; active axes project and transport Adam state."""

    torch.manual_seed(410 + shape[1])
    parameter = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    with _engine_eager():
        optimizer = Engine(
            [parameter],
            soap_recipe,
            lr=0.1,
            beta1=0.0,
            beta2=0.0,
            eps=1e-8,
            max_precond_dim=16,
            weight_decay=0.0,
            shampoo_beta=0.7,
        )
    state = optimizer.groups[0].states[0]
    assert ("Q_l" in state, "Q_r" in state) == active
    assert ("GG_l" in state, "GG_r" in state) == active

    for refresh in (False, True, False):
        gradient = torch.randn_like(parameter)
        old_left = state["Q_l"].clone() if active[0] else None
        old_right = state["Q_r"].clone() if active[1] else None
        projected = _project(gradient.unsqueeze(0), old_left, old_right, back=False)
        direction = projected / projected.abs().clamp_min(math.sqrt(1e-8))
        expected_update = _project(direction, old_left, old_right, back=True)[0]
        before = parameter.detach().clone()

        parameter.grad.copy_(gradient)
        optimizer.step(step_type="refresh" if refresh else "normal")

        torch.testing.assert_close((before - parameter) / 0.1, expected_update, rtol=1e-10, atol=1e-10)
        new_left = state["Q_l"] if active[0] else None
        new_right = state["Q_r"] if active[1] else None
        physical_avg = _project(projected, old_left, old_right, back=True)
        expected_avg = _project(physical_avg, new_left, new_right, back=False) if refresh else projected
        expected_variance = (
            _transport_variance(projected.square(), old_left, old_right, new_left, new_right)
            if refresh
            else projected.square()
        )
        torch.testing.assert_close(state["exp_avg"], expected_avg, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(
            state["exp_avg_sq"],
            expected_variance.clamp_min(0).sqrt(),
            rtol=1e-10,
            atol=1e-10,
        )
        if refresh:
            for basis in (new_left, new_right):
                if basis is not None:
                    identity = torch.eye(basis.shape[-1], dtype=basis.dtype).expand_as(basis)
                    torch.testing.assert_close(basis.mT @ basis, identity, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("recipe", (kl_soap_recipe, kl_shampoo_recipe), ids=("kl_soap", "kl_shampoo"))
def test_kl_skipped_left_factor_topology_and_runs_finite(recipe):
    """A tall matrix skips the oversized left (row) factor; the KL None-left-factor path (with the
    seed-only step 1) runs finite under the current equations."""

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


@pytest.mark.parametrize(
    ("shape", "active"),
    (((40, 6), (False, True)), ((40, 24), (False, False))),
)
def test_shampoo_skipped_factors_follow_explicit_size_policy(shape, active):
    """Shampoo omits exactly the axes above ``max_precond_dim``."""

    torch.manual_seed(420 + shape[1])
    left_active, right_active = active
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
            expected = torch.full((1, dimension), math.sqrt(0.1), dtype=torch.float64)
            torch.testing.assert_close(state[key], expected, rtol=0, atol=0)


def test_soap_oversized_axis_avoids_large_factor_allocation():
    """A GPT-style skipped axis does not allocate any quadratic state slab."""

    parameter = torch.nn.Parameter(torch.zeros(4096, 8))
    with _engine_eager():
        state = Engine([parameter], soap_recipe, max_precond_dim=64).groups[0].states[0]
    assert "GG_l" not in state
    assert "Q_l" not in state
    assert {"GG_r", "Q_r"} <= state.keys()
    assert all(value.numel() <= 4096 * 8 for value in state.values())
