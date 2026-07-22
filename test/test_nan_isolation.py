"""Guard per-parameter NaN isolation in heavyball's shared slab.

``clip_global_norm`` and SAM intentionally couple all parameters through a global
norm, so isolation is a default-recipe property.
"""

import inspect

import pytest
import torch
import torch.nn as nn

import heavyball


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compile-first GPU path")


@pytest.mark.parametrize("name", ("AdamW", "SOAP", "Muon", "PSGDKron", "Scion", "LATHER"))
def test_non_finite_gradient_isolated_per_parameter(name):
    torch.manual_seed(0)
    lin0 = nn.Linear(8, 8).cuda()
    lin1 = nn.Linear(8, 4).cuda()
    optimizer_cls = getattr(heavyball, name)
    lr = inspect.signature(optimizer_cls).parameters["lr"].default
    opt = optimizer_cls(list(lin0.parameters()) + list(lin1.parameters()), lr=lr)

    x0 = torch.randn(16, 8, device="cuda")
    y0 = torch.randn(16, 8, device="cuda")
    x1 = torch.randn(16, 8, device="cuda")
    y1 = torch.randn(16, 4, device="cuda")

    def loss():
        return ((lin0(x0) - y0) ** 2).mean() + ((lin1(x1) - y1) ** 2).mean()

    for _ in range(3):
        loss().backward()
        opt.step()
        opt.zero_grad()

    loss().backward()
    lin0.weight.grad[0, 0] = float("nan")
    opt.step()
    opt.zero_grad()

    assert all(torch.isfinite(p).all() for p in lin1.parameters()), (
        f"{name}: NaN in lin0 grad corrupted lin1 (slab cross-contamination)"
    )

    for _ in range(3):
        loss().backward()
        opt.step()
        opt.zero_grad()

    assert all(torch.isfinite(p).all() for p in lin1.parameters()), (
        f"{name}: NaN in lin0 grad corrupted lin1 (slab cross-contamination) after resume"
    )
    assert not all(torch.isfinite(p).all() for p in lin0.parameters()), (
        f"{name}: injected NaN was absorbed, test vacuous"
    )
