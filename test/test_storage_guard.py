from unittest.mock import patch

import pytest
import torch

import heavyball


def _overlapping_parameters():
    storage = torch.zeros(10)
    first = torch.nn.Parameter(storage.narrow(0, 0, 6))
    second = torch.nn.Parameter(storage.narrow(0, 4, 6))
    return first, second


@patch("torch.compile", lambda function, **kwargs: function)
def test_engine_rejects_overlapping_storage():
    params = _overlapping_parameters()
    with pytest.raises(ValueError, match="overlapping storage"):
        heavyball.Engine(params, heavyball.adamw)


@patch("torch.compile", lambda function, **kwargs: function)
def test_engine_accepts_disjoint_storage():
    params = [torch.nn.Parameter(torch.randn(4)) for _ in range(5)]
    engine = heavyball.Engine(params, heavyball.adamw)
    assert len(engine.groups) == 1


@patch("torch.compile", lambda function, **kwargs: function)
def test_engine_accepts_same_storage_non_overlapping():
    storage = torch.zeros(10)
    first = torch.nn.Parameter(storage.narrow(0, 0, 5))
    second = torch.nn.Parameter(storage.narrow(0, 5, 5))
    engine = heavyball.Engine([first, second], heavyball.adamw)
    assert len(engine.groups) == 1


@patch("torch.compile", lambda function, **kwargs: function)
def test_engine_rejects_adjacent_but_overlapping_by_one_element():
    storage = torch.zeros(10)
    first = torch.nn.Parameter(storage.narrow(0, 0, 6))
    second = torch.nn.Parameter(storage.narrow(0, 5, 5))
    with pytest.raises(ValueError, match="overlapping storage"):
        heavyball.Engine([first, second], heavyball.adamw)
