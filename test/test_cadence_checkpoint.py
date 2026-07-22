"""Checkpoint coverage for Engine-owned refresh cadence counters."""

from unittest.mock import patch

import pytest
import torch

from heavyball import Recipe, RefreshCadence, build, kron_adamw
from heavyball.transforms import sgd_commit


def test_cadence_checkpoint_preserves_refresh_sequence():
    probability = 0.3
    before_checkpoint = 2
    after_checkpoint = 12

    reference = RefreshCadence(probability)
    for _ in range(before_checkpoint):
        reference.next_step_type()
    snapshot = (reference.step, reference.cumulative, reference.compensation)
    uninterrupted = [reference.next_step_type() for _ in range(after_checkpoint)]

    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        engine = build(
            [parameter],
            kron_adamw,
            preconditioner_update_probability=probability,
        )
        for _ in range(before_checkpoint):
            parameter.grad.fill_(1)
            engine.step()

        cadence = engine._cadence
        assert cadence is not None
        assert (cadence.step, cadence.cumulative, cadence.compensation) == snapshot

        state_dict = engine.state_dict()
        assert state_dict["format"] == 3
        assert state_dict["cadence"] == {
            "probability": probability,
            "step": snapshot[0],
            "cumulative": snapshot[1],
            "compensation": snapshot[2],
        }

        fresh_parameter = torch.nn.Parameter(torch.ones(2, 2))
        fresh = build(
            [fresh_parameter],
            kron_adamw,
            preconditioner_update_probability=probability,
        )
        fresh.load_state_dict(state_dict)

    fresh_cadence = fresh._cadence
    assert fresh_cadence is not None
    assert (
        fresh_cadence.step,
        fresh_cadence.cumulative,
        fresh_cadence.compensation,
    ) == snapshot
    resumed = [fresh_cadence.next_step_type() for _ in range(after_checkpoint)]
    assert resumed == uninterrupted

    unrestored = RefreshCadence(probability)
    drifted = [unrestored.next_step_type() for _ in range(after_checkpoint)]
    assert drifted != uninterrupted


def test_cadence_checkpoint_restores_numeric_probability():
    source_param = torch.nn.Parameter(torch.ones(2, 2))
    target_param = torch.nn.Parameter(torch.ones(2, 2))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        source = build(
            [source_param], kron_adamw, param_keys=("p",),
            preconditioner_update_probability=0.25,
        )
        target = build(
            [target_param], kron_adamw, param_keys=("p",),
            preconditioner_update_probability=0.75,
        )
        target.load_state_dict(source.state_dict())

    assert target._cadence is not None
    assert target._cadence.probability == 0.25


def test_format_3_cadence_presence_must_match_engine():
    defaults = {"lr": 0.0, "weight_decay": 0.0}
    without_cadence = Recipe((), sgd_commit, defaults)
    with_cadence = Recipe(
        (), sgd_commit, {**defaults, "preconditioner_update_probability": 0.5}
    )

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        for source_recipe, target_recipe in (
            (without_cadence, with_cadence),
            (with_cadence, without_cadence),
        ):
            source = build([torch.nn.Parameter(torch.ones(1))], source_recipe, param_keys=("p",))
            target = build([torch.nn.Parameter(torch.ones(1))], target_recipe, param_keys=("p",))
            with pytest.raises(ValueError, match="cadence presence"):
                target.load_state_dict(source.state_dict())


def test_callable_cadence_probability_fails_loudly_at_checkpoint_boundaries():
    numeric_param = torch.nn.Parameter(torch.ones(2, 2))
    callable_param = torch.nn.Parameter(torch.ones(2, 2))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        numeric = build(
            [numeric_param], kron_adamw, param_keys=("p",),
            preconditioner_update_probability=0.25,
        )
        scheduled = build(
            [callable_param], kron_adamw, param_keys=("p",),
            preconditioner_update_probability=lambda step: 0.25,
        )
        checkpoint = numeric.state_dict()

        with pytest.raises(ValueError, match="callable cadence probability"):
            scheduled.state_dict()
        with pytest.raises(ValueError, match="callable cadence probability"):
            scheduled.load_state_dict(checkpoint)
