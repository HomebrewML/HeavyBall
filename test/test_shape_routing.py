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
    initial = torch.randn(shape, device="cuda")
    parameter = nn.Parameter(initial.clone())
    reference = nn.Parameter(initial.clone())
    lr = inspect.signature(getattr(heavyball, name)).parameters["lr"].default
    optimizer = getattr(heavyball, name)([parameter], lr=lr)
    adamw = heavyball.AdamW([reference], lr=lr)

    for _ in range(4):
        gradient = torch.randn(shape, device="cuda") * 0.1
        parameter.grad.copy_(gradient)
        reference.grad.copy_(gradient)
        optimizer.step()
        adamw.step()

    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
