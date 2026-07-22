"""A malformed checkpoint is rejected without mutating any already-loaded state (atomic load).

Engine.load_state_dict must stage both tensor copies and scalar fills before touching live state.
These tests cover late failures in each staging path.
"""

from unittest.mock import patch

import copy

import pytest
import torch

from heavyball import adamw
from heavyball.core import build


def test_malformed_fill_value_does_not_mutate_state():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(0)
        target = torch.nn.Parameter(torch.randn(6))
        engine = build([target], adamw, lr=1e-2)
        target.grad.copy_(torch.randn(6))
        engine.step()
        slot = engine.groups[0].states[0]
        name = next(iter(slot))
        before = slot[name].clone()

        torch.manual_seed(1)
        other = torch.nn.Parameter(torch.randn(6))
        other_engine = build([other], adamw, lr=1e-2)
        other.grad.copy_(torch.randn(6) * 5)
        other_engine.step()
        checkpoint = copy.deepcopy(other_engine.state_dict())
        assert not torch.equal(checkpoint["state"][next(iter(checkpoint["state"]))][0][name], before)

        step_key = next(iter(checkpoint["step"]))
        checkpoint["step"][step_key] = "not a number"

        with pytest.raises(ValueError):
            engine.load_state_dict(checkpoint)
        torch.testing.assert_close(slot[name], before, rtol=0, atol=0)


def test_uncopyable_tensor_does_not_mutate_ages_or_state():
    source_param = torch.nn.Parameter(torch.ones(2))
    target_param = torch.nn.Parameter(torch.ones(2))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        source = build([source_param], adamw, param_keys=("p",))
        target = build([target_param], adamw, param_keys=("p",))

    source_param.grad.fill_(3.0)
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())
    ages_before = tuple(group.age.clone() for group in target.groups)
    states_before = tuple(
        tuple({name: value.clone() for name, value in slots.items()} for slots in group.states)
        for group in target.groups
    )
    checkpoint["state"]["p"][0]["exp_avg_sq"] = torch.empty(2, device="meta")

    with pytest.raises(NotImplementedError, match="meta tensor"):
        target.load_state_dict(checkpoint)

    for group, age_before, group_states_before in zip(
        target.groups, ages_before, states_before, strict=True
    ):
        assert torch.equal(group.age, age_before)
        for slots, slots_before in zip(group.states, group_states_before, strict=True):
            for name, value in slots.items():
                assert torch.equal(value, slots_before[name])
