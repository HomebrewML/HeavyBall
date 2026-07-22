"""Regression coverage for Engine-owned preconditioner refresh cadence."""

from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.kron import kron
from heavyball.matrix import soap_recipe
from heavyball.programs import SAM

_GRADIENTS = (
    torch.tensor(((1.0, -2.0, 3.0), (2.0, 5.0, -1.0))),
    torch.tensor(((2.0, 1.0, -3.0), (-4.0, 1.0, 2.0))),
    torch.tensor(((-1.0, 3.0, 2.0), (5.0, -2.0, 1.0))),
)


def _factor_snapshots(recipe, factor_names: tuple[str, ...], probability: float) -> list[dict[str, torch.Tensor]]:
    parameter = torch.nn.Parameter(torch.tensor(((1.0, 2.0, 3.0), (4.0, 6.0, 9.0))))
    with patch("torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(
            [parameter],
            recipe,
            preconditioner_update_probability=probability,
            weight_decay=0.0,
        )
        state = optimizer.groups[0].states[0]
        snapshots = [{name: state[name].clone() for name in factor_names}]
        for gradient in _GRADIENTS:
            parameter.grad.copy_(gradient)
            optimizer.step()
            snapshots.append({name: state[name].clone() for name in factor_names})
    return snapshots


def _assert_naive_refreshes_and_zero_probability_freezes(recipe, factor_names: tuple[str, ...]) -> None:
    refreshed = _factor_snapshots(recipe, factor_names, probability=1.0)
    frozen = _factor_snapshots(recipe, factor_names, probability=0.0)

    for name in factor_names:
        assert not torch.equal(refreshed[0][name], refreshed[1][name])
        assert not torch.equal(refreshed[1][name], refreshed[2][name])
        for snapshot in frozen[1:]:
            assert torch.equal(snapshot[name], frozen[0][name])


def test_naive_soap_steps_refresh_and_zero_probability_freezes_basis():
    _assert_naive_refreshes_and_zero_probability_freezes(soap_recipe, ("Q_l", "Q_r"))


def test_naive_kron_steps_refresh_and_zero_probability_freezes_factors():
    _assert_naive_refreshes_and_zero_probability_freezes(kron, ("Q_0", "Q_1"))


def test_cadence_matches_explicit_refresh_for_soap():
    initial = torch.tensor(((1.0, 2.0, 3.0), (4.0, 6.0, 9.0)))
    cadence_parameter = torch.nn.Parameter(initial.clone())
    explicit_parameter = torch.nn.Parameter(initial.clone())
    with patch("torch.compile", lambda function, **kwargs: function):
        cadence = Engine(
            [cadence_parameter], soap_recipe, preconditioner_update_probability=1.0, weight_decay=0.0
        )
        explicit = Engine(
            [explicit_parameter], soap_recipe, preconditioner_update_probability=1.0, weight_decay=0.0
        )
        for gradient in _GRADIENTS:
            cadence_parameter.grad.copy_(gradient)
            explicit_parameter.grad.copy_(gradient)
            cadence.step()
            explicit.step(step_type="refresh")
            assert torch.equal(cadence_parameter, explicit_parameter)


def test_rejected_observed_does_not_advance_cadence_or_change_retry_step_type():
    rejected_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    baseline_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    with patch("torch.compile", lambda function, **kwargs: function):
        rejected = Engine(
            [rejected_parameter], kron, lr=0.0, weight_decay=0.0, preconditioner_update_probability=0.5
        )
        baseline = Engine(
            [baseline_parameter], kron, lr=0.0, weight_decay=0.0, preconditioner_update_probability=0.5
        )
        rejected_parameter.grad.fill_(1.0)
        baseline_parameter.grad.fill_(1.0)
        initial_cadence = (
            rejected._cadence.step,
            rejected._cadence.cumulative,
            rejected._cadence.compensation,
        )

        with pytest.raises(ValueError, match="observed must contain one value"):
            rejected.step(observed=[])

        assert (
            rejected._cadence.step,
            rejected._cadence.cumulative,
            rejected._cadence.compensation,
        ) == initial_cadence

        def record_steps(engine, calls):
            compiled_steps = engine.compiled_steps

            def record(step_type):
                def recorded_step():
                    calls.append(step_type)
                    compiled_steps[step_type]()

                return recorded_step

            engine.compiled_steps = {step_type: record(step_type) for step_type in compiled_steps}

        rejected_calls = []
        baseline_calls = []
        record_steps(rejected, rejected_calls)
        record_steps(baseline, baseline_calls)
        rejected.step()
        baseline.step()

    assert rejected_calls == baseline_calls == ["normal"]
    assert (
        rejected._cadence.step,
        rejected._cadence.cumulative,
        rejected._cadence.compensation,
    ) == (
        baseline._cadence.step,
        baseline._cadence.cumulative,
        baseline._cadence.compensation,
    )


def test_sam_uses_base_cadence_and_increments_once_per_step():
    torch.manual_seed(23)
    parameter = torch.nn.Parameter(torch.randn(3, 4))
    with patch("torch.compile", lambda function, **kwargs: function):
        base = Engine([parameter], soap_recipe, preconditioner_update_probability=1.0, weight_decay=0.0)
        optimizer = SAM(base, rho=0.05)
        initial_basis = base.groups[0].states[0]["Q_l"].clone()
        initial_step = base.step_count.clone()

        def closure():
            base.zero_grad()
            loss = parameter.square().sum()
            loss.backward()
            return loss

        calls = 3
        for _ in range(calls):
            optimizer.step(closure)

    assert not torch.equal(base.groups[0].states[0]["Q_l"], initial_basis)
    assert torch.equal(base.step_count, initial_step + calls)


def test_soap_recipe_default_refresh_probability_matches_legacy_frequency():
    assert soap_recipe.defaults["preconditioner_update_probability"] == 0.5
