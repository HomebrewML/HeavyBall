"""Matrix-family optimizers handle single-output ([1, N]) weights.

A Linear(N, 1) head (value head, regression, binary classifier) has a [1, N] weight, so one Kronecker
factor is 1x1 -- the degenerate size-1-row path. This guards that path against a heavyball regression.
(The same shape hit a torch<2.13 fake-tensor compile bug; that is a torch issue, fixed in the required
2.13, so this runs eager to test the heavyball math independent of the compiler.)
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

import heavyball

MATRIX_FACADES = ("SOAP", "Shampoo", "KLSOAP", "KLShampoo", "PSGDKron", "PSGDPro", "QSGD", "LATHER", "Whitening")


@pytest.mark.parametrize("name", MATRIX_FACADES)
def test_single_output_head_updates_finitely(name):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1))  # final weight is [1, 8]
    head = model[-1].weight
    assert tuple(head.shape) == (1, 8)
    before = head.detach().clone()
    inputs = torch.randn(64, 16)
    with torch.no_grad():
        targets = model(inputs) + 0.1 * torch.randn(64, 1)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        opt = getattr(heavyball, name)(model.parameters(), lr=1e-3)
        for _ in range(20):
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

    assert torch.isfinite(head).all()
    assert not torch.equal(head, before)
