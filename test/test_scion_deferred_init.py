"""Scion preserves supplied parameters until its first optimizer step."""

import copy
from unittest.mock import patch

import torch

import heavyball


def _no_compile(function, **_kwargs):
    return function


def test_scion_reseeds_once_at_first_step():
    torch.manual_seed(44)
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    expected = parameter.detach().clone()
    heavyball.scion_param_init(expected, seed=0, scale=torch.tensor(1.0))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.Scion([parameter], lr=0.0)

    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step()
    assert torch.equal(parameter, expected)

    replacement = torch.full_like(parameter, 3.0)
    with torch.no_grad():
        parameter.copy_(replacement)
    parameter.grad.zero_()
    optimizer.step()
    assert torch.equal(parameter, replacement)


def test_scion_advanced_checkpoint_does_not_reseed_on_resume():
    source_parameter = torch.nn.Parameter(torch.randn(4, 4))
    with patch("heavyball.core.torch.compile", _no_compile):
        source = heavyball.Scion([source_parameter], lr=0.0)
    source_parameter.grad.copy_(torch.randn_like(source_parameter))
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    with patch("heavyball.core.torch.compile", _no_compile):
        target = heavyball.Scion([target_parameter], lr=0.0)
    target.load_state_dict(checkpoint)
    replacement = torch.full_like(target_parameter, 3.0)
    with torch.no_grad():
        target_parameter.copy_(replacement)
    target_parameter.grad.zero_()

    target.step()

    assert torch.equal(target_parameter, replacement)


def test_scion_added_group_initializes_only_new_parameter_on_next_step():
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.Scion([parameter], lr=0.0)
    parameter.grad.copy_(torch.randn_like(parameter))
    optimizer.step()
    existing = parameter.detach().clone()

    added = torch.nn.Parameter(torch.full((4, 4), 9.0))
    before_added = added.detach().clone()
    expected_added = before_added.clone()
    heavyball.scion_param_init(expected_added, seed=1, scale=torch.tensor(1.0))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer.add_param_group({"params": [added]})
    assert torch.equal(parameter, existing)
    assert torch.equal(added, before_added)

    parameter.grad.zero_()
    added.grad.zero_()
    optimizer.step()

    assert torch.equal(parameter, existing)
    assert torch.equal(added, expected_added)


def test_scion_first_step_does_not_apply_stale_gradient():
    # The first step reinitializes parameters from seeded frames; the gradient was computed at the
    # pre-initialization values, so applying it would update the wrong parameters. The first step must
    # reinitialize WITHOUT applying the gradient (regression: it applied the stale gradient).
    torch.manual_seed(0)
    init = torch.randn(6, 6)
    first = torch.nn.Parameter(init.clone())
    second = torch.nn.Parameter(init.clone())
    with patch("heavyball.core.torch.compile", _no_compile):
        opt_first = heavyball.Scion([first], lr=0.05)
        opt_second = heavyball.Scion([second], lr=0.05)
    opt_first.zero_grad()
    first.grad.copy_(torch.ones_like(first))
    opt_first.step()
    opt_second.zero_grad()
    second.grad.copy_(torch.zeros_like(second))
    opt_second.step()
    torch.testing.assert_close(first, second)  # first step skips update -> grad is irrelevant
    first.grad.copy_(torch.ones_like(first))
    opt_first.step()  # second step applies a real update
    assert not torch.equal(first, second)


def test_scion_unobserved_checkpoint_does_not_repeat_initialization():
    source_parameter = torch.nn.Parameter(torch.randn(4, 4))
    with patch("heavyball.core.torch.compile", _no_compile):
        source = heavyball.build([source_parameter], heavyball.scion, lr=0.0)
    source_parameter.grad.zero_()
    source.step(observed=(False,))
    checkpoint = copy.deepcopy(source.state_dict())

    target_parameter = torch.nn.Parameter(torch.full((4, 4), 3.0))
    replacement = target_parameter.detach().clone()
    with patch("heavyball.core.torch.compile", _no_compile):
        target = heavyball.build([target_parameter], heavyball.scion, lr=0.0)
    target.load_state_dict(checkpoint)
    target_parameter.grad.zero_()

    target.step()

    assert torch.equal(target_parameter, replacement)
