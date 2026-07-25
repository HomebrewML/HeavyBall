"""Regression coverage for Engine-owned preconditioner refresh cadence."""

import inspect
from unittest.mock import patch

import pytest
import torch

from heavyball import SOAP
from heavyball.core import Engine
from heavyball.kron import kron
from heavyball.matrix import soap_recipe
from heavyball.programs import SAM

_GRADIENTS = (
    torch.tensor(((1.0, -2.0, 3.0), (2.0, 5.0, -1.0))),
    torch.tensor(((2.0, 1.0, -3.0), (-4.0, 1.0, 2.0))),
    torch.tensor(((-1.0, 3.0, 2.0), (5.0, -2.0, 1.0))),
)


@pytest.mark.parametrize("recipe", (soap_recipe, kron), ids=("soap", "kron"))
@pytest.mark.parametrize(
    ("probability", "step_type"),
    ((1.0, "refresh"), (0.0, "normal")),
)
def test_cadence_matches_explicit_step_type(recipe, probability, step_type):
    initial = torch.tensor(((1.0, 2.0, 3.0), (4.0, 6.0, 9.0)))
    cadence_parameter = torch.nn.Parameter(initial.clone())
    explicit_parameter = torch.nn.Parameter(initial.clone())
    with patch("torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(17)
        cadence = Engine(
            [cadence_parameter],
            recipe,
            preconditioner_update_probability=probability,
            weight_decay=0.0,
        )
        torch.manual_seed(17)
        explicit = Engine(
            [explicit_parameter],
            recipe,
            preconditioner_update_probability=probability,
            weight_decay=0.0,
        )
        for gradient in _GRADIENTS:
            cadence_parameter.grad.copy_(gradient)
            explicit_parameter.grad.copy_(gradient)
            cadence.step()
            explicit.step(step_type=step_type)
            assert torch.equal(cadence_parameter, explicit_parameter)


def test_rejected_observed_does_not_advance_cadence_or_change_retry_step_type():
    rejected_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    baseline_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    with patch("torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(29)
        rejected = Engine(
            [rejected_parameter], kron, weight_decay=0.0, preconditioner_update_probability=0.5
        )
        torch.manual_seed(29)
        baseline = Engine(
            [baseline_parameter], kron, weight_decay=0.0, preconditioner_update_probability=0.5
        )
        rejected_parameter.grad.fill_(1.0)
        baseline_parameter.grad.fill_(1.0)

        with pytest.raises(ValueError, match="observed must contain one value"):
            rejected.step(observed=[])

        rejected.step()
        baseline.step()

    assert torch.equal(rejected_parameter, baseline_parameter)


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
    parameter = inspect.signature(SOAP).parameters["preconditioner_update_probability"]
    assert parameter.default == 0.5
