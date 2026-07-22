"""Regression coverage for confirmed cadence and hyperparameter findings."""

from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.kron import kron


def test_naive_whitening_steps_refresh_basis():
    parameter = torch.nn.Parameter(torch.eye(2))
    gradients = (
        torch.tensor(((1.0, -2.0), (3.0, 4.0))),
        torch.tensor(((-2.0, 1.0), (4.0, -3.0))),
    )
    with patch("torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build([parameter], heavyball.whitening)
        initial_basis = optimizer.groups[0].states[0]["Q"].clone()
        for gradient in gradients:
            parameter.grad.copy_(gradient)
            optimizer.step()

    assert not torch.equal(optimizer.groups[0].states[0]["Q"], initial_basis)


def test_callable_probability_builds_and_naively_refreshes_kron_factors():
    parameter = torch.nn.Parameter(torch.eye(2))
    with patch("torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build(
            [parameter], kron, preconditioner_update_probability=lambda step: 1.0
        )
        initial_factors = {
            name: optimizer.groups[0].states[0][name].clone() for name in ("Q_0", "Q_1")
        }
        parameter.grad.copy_(torch.tensor(((1.0, -2.0), (3.0, 4.0))))
        optimizer.step()

    for name, initial_factor in initial_factors.items():
        assert not torch.equal(optimizer.groups[0].states[0][name], initial_factor)


def test_probability_is_host_only_and_not_settable_dynamically():
    parameter = torch.nn.Parameter(torch.eye(2))
    with patch("torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build([parameter], kron)

    assert all(
        not hasattr(group.hyper, "preconditioner_update_probability") for group in optimizer.groups
    )
    with pytest.raises(ValueError, match="not a dynamic hyperparameter"):
        optimizer.set_hyper("preconditioner_update_probability", 0.5)
