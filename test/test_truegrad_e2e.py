"""End-to-end: register_truegrad hooks + TrueGrad facade + real training loop."""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import heavyball


def _train_truegrad(model, optimizer, inputs, target, steps=30):
    handles = heavyball.register_truegrad(model)
    losses = []
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = F.mse_loss(model(inputs), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    return losses


def test_truegrad_adam_trains_small_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.TrueGradAdam(model.parameters(), lr=0.01)
    inputs = torch.randn(16, 8)
    target = torch.randn(16, 4)

    losses = _train_truegrad(model, optimizer, inputs, target)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("facade", ["TrueGradRMSprop", "TrueGradLaProp", "TrueGradNAdam"])
def test_truegrad_variant_trains(facade):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    cls = getattr(heavyball, facade)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = cls(model.parameters(), lr=0.01)
    inputs = torch.randn(16, 8)
    target = torch.randn(16, 4)

    losses = _train_truegrad(model, optimizer, inputs, target)
    assert losses[-1] < losses[0]
