"""Facade checkpoint loading must not expose public state from a rejected checkpoint."""

import copy
from unittest.mock import patch

import pytest
import torch

import heavyball


def _no_compile(function, **_kwargs):
    return function


def test_invalid_engine_fingerprint_does_not_mutate_public_groups():
    parameter = torch.nn.Parameter(torch.ones(1))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.AdamW([parameter], lr=0.1, weight_decay=0.0)

    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0]["lr"] = 0.9
    checkpoint["engines"][0]["fingerprint"]["bogus"] = {}

    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.load_state_dict(checkpoint)

    assert optimizer.param_groups[0]["lr"] == 0.1
    parameter.grad.fill_(1.0)
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == 0.1
    torch.testing.assert_close(parameter, torch.tensor([0.9]), rtol=0, atol=1e-6)


def test_valid_checkpoint_adopts_public_learning_rate():
    parameter = torch.nn.Parameter(torch.ones(1))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.AdamW([parameter], lr=0.1)

    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0]["lr"] = 0.9

    optimizer.load_state_dict(checkpoint)

    assert optimizer.param_groups[0]["lr"] == 0.9


def test_invalid_public_group_count_does_not_mutate_engine():
    # Atomicity in the reverse direction: if Torch rejects the public param-group count AFTER the
    # Engine commit, the Engine must roll back too. (Regression: swapping the two commits made this
    # direction non-atomic — Engine lr changed while public lr stayed put.)
    parameter = torch.nn.Parameter(torch.ones(1))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.AdamW([parameter], lr=0.1)
    optimizer.zero_grad()
    parameter.grad.fill_(1.0)
    optimizer.step()
    engine_lr_before = float(optimizer._engine.groups[0].hyper.lr)

    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"].append(dict(checkpoint["param_groups"][0]))  # 2 groups vs 1

    with pytest.raises(ValueError):
        optimizer.load_state_dict(checkpoint)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert float(optimizer._engine.groups[0].hyper.lr) == engine_lr_before


def test_splitopt_child_failure_does_not_mutate_earlier_children():
    # SplitOpt must commit its child optimizers atomically: a failure loading one child must not leave
    # earlier children mutated. (Regression: children were loaded sequentially.)
    first = torch.nn.Parameter(torch.ones(1))
    second = torch.nn.Parameter(torch.ones(1))
    with patch("heavyball.core.torch.compile", _no_compile):
        split = heavyball.SplitOpt([
            {"params": [first], "optimizer": heavyball.AdamW, "lr": 0.1},
            {"params": [second], "optimizer": heavyball.AdamW, "lr": 0.1},
        ])
    split.zero_grad()
    first.grad.fill_(1.0)
    second.grad.fill_(1.0)
    split.step()

    checkpoint = split.state_dict()
    checkpoint["optimizers"][0]["param_groups"][0]["lr"] = 0.9  # would mutate child 0
    checkpoint["optimizers"][1]["engines"][0]["fingerprint"]["bogus"] = {}  # corrupts child 1

    with pytest.raises(ValueError, match="fingerprint"):
        split.load_state_dict(checkpoint)
    assert split.optimizers[0].param_groups[0]["lr"] == 0.1


def test_malformed_public_hyper_rolls_back_engine():
    # The post-load hyper sync (set_hyper -> _scalar) must be inside the atomic transaction: a public
    # group value that fails scalar conversion must roll back the already-committed Engine load.
    parameter = torch.nn.Parameter(torch.ones(1))
    with patch("heavyball.core.torch.compile", _no_compile):
        optimizer = heavyball.AdamW([parameter], lr=0.1)
    optimizer.zero_grad()
    parameter.grad.fill_(1.0)
    optimizer.step()
    engine_lr_before = float(optimizer._engine.groups[0].hyper.lr)

    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0]["lr"] = "not-a-number"  # _sync -> _scalar raises

    with pytest.raises(Exception):
        optimizer.load_state_dict(checkpoint)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert float(optimizer._engine.groups[0].hyper.lr) == engine_lr_before
