"""PSGD-LRA initialization must use the Engine's synchronized RNG seed."""

from unittest.mock import patch

import torch

import heavyball
from heavyball.core import build


def _lra_trajectory(ambient_seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(ambient_seed)
    parameters = [
        torch.nn.Parameter(torch.ones(2, 3)),
        torch.nn.Parameter(torch.ones(2, 3)),
    ]
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        engine = build(
            parameters,
            heavyball.PSGDLRA.recipe,
            _rng_seed=123,
            lr=0.01,
            rank=2,
            weight_decay=0.0,
        )
    gradient = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    for parameter in parameters:
        parameter.grad.copy_(gradient)
    engine.step(step_type="refresh")
    return tuple(parameter.detach().clone() for parameter in parameters)


def test_lra_uses_synchronized_seed_not_ambient_rng():
    first = _lra_trajectory(1)
    second = _lra_trajectory(999)

    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    assert not torch.equal(first[0], first[1])
