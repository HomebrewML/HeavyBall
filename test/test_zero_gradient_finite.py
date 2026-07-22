"""Guard preconditioner finiteness under sustained zero gradients.

The PSGD spectral and learned-Q divisions are unclamped, so this pins their
current zero-gradient safety.
"""

import inspect

import pytest
import torch
import torch.nn as nn

import heavyball


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compile-first GPU path"
)


@pytest.mark.parametrize("name", ("PSGDKron", "PSGDPro", "QSGD", "LATHER", "SOAP", "Muon"))
def test_zero_gradient_stays_finite(name):
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16).cuda())
    lr = inspect.signature(getattr(heavyball, name)).parameters["lr"].default
    opt = getattr(heavyball, name)([p], lr=lr)

    for _ in range(25):
        (p * torch.zeros(16, 16, device="cuda")).sum().backward()
        opt.step()
        opt.zero_grad()

    assert torch.isfinite(p).all(), f"{name}: non-finite under sustained zero gradient"
