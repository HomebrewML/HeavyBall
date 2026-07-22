"""Regression for the (1,N)/(1,K,N) compile crash in PSGD/Kron power iteration.

The crash came from the degenerate 1x1 row factor; leading-singleton leaves now
route to AdamW via ``matrix_route``.
"""

import inspect

import pytest
import torch
import torch.nn as nn

import heavyball


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compile-first GPU path"
)


@pytest.mark.parametrize("name", ("PSGDKron", "PSGDPro", "QSGD", "LATHER"))
@pytest.mark.parametrize("shape", ((1, 16), (1, 3, 16)))
def test_leading_singleton_shape_routing(name, shape):
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(shape).cuda())
    lr = inspect.signature(getattr(heavyball, name)).parameters["lr"].default
    opt = getattr(heavyball, name)([p], lr=lr)
    before = p.detach().clone()

    for _ in range(4):
        g = torch.randn(shape, device="cuda") * 0.1
        (p * g).sum().backward()
        opt.step()
        opt.zero_grad()

    assert torch.isfinite(p).all(), f"{name} {shape}: non-finite"
    assert not torch.equal(p, before), f"{name} {shape}: did not move"
