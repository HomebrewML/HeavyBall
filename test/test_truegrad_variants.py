"""Regression tests for TrueGrad variants of second-moment optimizers."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

import heavyball


def _eager_compile(function, **kwargs):
    del kwargs
    return function


def _step_pair(truegrad_recipe, base_recipe, observation, **hyperparameters):
    torch.manual_seed(42)
    truegrad_linear = nn.Linear(4, 3)
    base_linear = nn.Linear(4, 3)
    base_linear.load_state_dict(truegrad_linear.state_dict())
    inputs = torch.randn(7, 4)

    with patch("heavyball.core.torch.compile", _eager_compile):
        truegrad_optimizer = heavyball.HeavyBallOptimizer(
            truegrad_linear.parameters(), truegrad_recipe, **hyperparameters
        )
        base_optimizer = heavyball.HeavyBallOptimizer(
            base_linear.parameters(), base_recipe, **hyperparameters
        )

    truegrad_optimizer.zero_grad()
    base_optimizer.zero_grad()
    truegrad_linear(inputs).square().sum().backward()
    base_linear(inputs).square().sum().backward()

    for param in truegrad_linear.parameters():
        truegrad_optimizer.produce(param, "sum_grad_squared", observation(param))

    truegrad_optimizer.step()
    base_optimizer.step()
    return truegrad_linear, base_linear


def _assert_matches_base(truegrad_recipe, base_recipe, **hyperparameters):
    truegrad_linear, base_linear = _step_pair(
        truegrad_recipe,
        base_recipe,
        lambda param: param.grad.square(),
        **hyperparameters,
    )

    for truegrad_param, base_param in zip(
        truegrad_linear.parameters(), base_linear.parameters(), strict=True
    ):
        torch.testing.assert_close(truegrad_param, base_param, rtol=1e-5, atol=1e-6)


def _assert_differs_from_base(truegrad_recipe, base_recipe, **hyperparameters):
    truegrad_linear, base_linear = _step_pair(
        truegrad_recipe,
        base_recipe,
        lambda param: torch.ones_like(param),
        **hyperparameters,
    )

    for truegrad_param, base_param in zip(
        truegrad_linear.parameters(), base_linear.parameters(), strict=True
    ):
        assert not torch.allclose(truegrad_param, base_param, rtol=1e-5, atol=1e-7)


_CASES = (
    (heavyball.truegrad_rmsprop, heavyball.rmsprop, dict(lr=0.05, beta2=0.99, eps=1e-8)),
    (heavyball.truegrad_laprop, heavyball.laprop, dict(lr=0.05, beta1=0.9, beta2=0.99, eps=1e-8)),
    (
        heavyball.truegrad_nadam,
        heavyball.nadam,
        dict(lr=0.05, beta1=0.9, beta2=0.99, eps=1e-8, momentum_decay=0.004),
    ),
)


@pytest.mark.parametrize(("truegrad_recipe", "base_recipe", "hyperparameters"), _CASES)
def test_truegrad_variants_use_the_supplied_observation(truegrad_recipe, base_recipe, hyperparameters):
    _assert_matches_base(truegrad_recipe, base_recipe, **hyperparameters)
    _assert_differs_from_base(truegrad_recipe, base_recipe, **hyperparameters)
