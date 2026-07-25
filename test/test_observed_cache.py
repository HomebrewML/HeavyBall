from unittest.mock import patch

import torch

import heavyball
import heavyball.core as core


def _optimizer():
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        optimizer = heavyball.AdamW(params)
    return optimizer


def test_default_observed_not_rebuilt_every_step(monkeypatch):
    optimizer = _optimizer()
    group = optimizer._engine.groups[0]
    original_tensor = core.torch.tensor
    rebuilds = 0

    def counting_tensor(data, *args, **kwargs):
        nonlocal rebuilds
        result = original_tensor(data, *args, **kwargs)
        if result.dtype == torch.bool and result.shape == group.observed.shape:
            rebuilds += 1
        return result

    monkeypatch.setattr(core.torch, "tensor", counting_tensor)
    for _ in range(6):
        optimizer.step()

    assert rebuilds == 0


def test_changing_observed_mask_still_applies():
    optimizer = _optimizer()
    engine = optimizer._engine
    group = engine.groups[0]

    for mask in ([True, True], [True, False], [True, True]):
        engine.step(observed=mask)
        assert group.observed.tolist() == mask
