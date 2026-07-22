"""Gradient accumulation: N backward()s before one step() equals a single step on the summed gradient.

HeavyBall binds param.grad to a slab and backward() accumulates into it in place, so accumulating
micro-batch gradients then stepping must match stepping once on their sum. This gates that common
workflow (large effective batch size) against a regression in the grad-slab handling or the step-time
grad-binding guard. Covers a first-order, an eigenbasis, an orthogonal, and two PSGD-family optimizers.
"""

from unittest.mock import patch

import pytest
import torch

import heavyball


@pytest.mark.parametrize("name", ("AdamW", "SOAP", "Muon", "PSGDKron", "LATHER"))
def test_accumulation_matches_summed_gradient(name):
    torch.manual_seed(0)
    micro_grads = [torch.randn(8, 4) for _ in range(3)]

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(0)
        accumulated = torch.nn.Parameter(torch.randn(8, 4))
        accumulated_opt = getattr(heavyball, name)([accumulated], lr=1e-2)
        accumulated.grad.copy_(micro_grads[0])
        for grad in micro_grads[1:]:
            accumulated.grad.add_(grad)  # backward() accumulates into the slab view in place
        accumulated_opt.step()

        torch.manual_seed(0)
        summed = torch.nn.Parameter(torch.randn(8, 4))
        summed_opt = getattr(heavyball, name)([summed], lr=1e-2)
        summed.grad.copy_(sum(micro_grads))
        summed_opt.step()

    torch.testing.assert_close(accumulated, summed, rtol=0, atol=0)
