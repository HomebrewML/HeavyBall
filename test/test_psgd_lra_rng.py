"""PSGD-LRA initialization must use the Engine's synchronized RNG seed."""

from unittest.mock import patch

import torch

import heavyball
from heavyball.core import build


def _no_compile(function, **_kwargs):
    return function


def _lra_state(ambient_seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(ambient_seed)
    parameters = [
        torch.nn.Parameter(torch.ones(2, 3)),
        torch.nn.Parameter(torch.full((2, 3), 2.0)),
    ]
    with patch("heavyball.core.torch.compile", _no_compile):
        engine = build(
            parameters,
            heavyball.PSGDLRA.recipe,
            _rng_seed=123,
            rank=2,
        )
    return next(
        state
        for state in engine.groups[0].states
        if {"U", "V", "d"}.issubset(state)
    )


def test_lra_initial_state_uses_synchronized_seed_not_ambient_rng():
    first = _lra_state(1)
    second = _lra_state(999)

    assert torch.equal(first["U"], second["U"])
    assert torch.equal(first["V"], second["V"])
    assert not torch.equal(first["U"][0], first["U"][1])
    assert not torch.equal(first["V"][0], first["V"][1])
