"""Test that PSGD preconditioner lower bound uses fp32, not fp64.

Verifies:
1. Convergence is not degraded with fp32 lower bound
2. No fp64 tensors leak into optimizer state
"""

import pytest
import torch
from torch import nn

import heavyball
from heavyball.utils import clean, set_torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_PSGD_OPTIMIZERS = [
    (heavyball.PSGDPRO, 1e-3, {}),
    (heavyball.PSGDKron, 1e-3, {}),
]


def _train(model, opt, data, target, steps):
    losses = []
    for _ in range(steps):
        p = next(model.parameters())
        d = data.to(p.dtype) if p.dtype != data.dtype else data
        loss = ((model(d) - target.to(d.dtype)) ** 2).mean().float()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("opt_cls,lr,extra_kw", _PSGD_OPTIMIZERS, ids=[t[0].__name__ for t in _PSGD_OPTIMIZERS])
def test_psgd_convergence_fp32_lb(opt_cls, lr, extra_kw):
    set_torch()
    torch.manual_seed(42)
    data = torch.randn(128, 64, device="cuda")
    target = torch.randn(128, 32, device="cuda")

    torch.manual_seed(0)
    model = nn.Linear(64, 32, bias=False, device="cuda")
    opt = opt_cls(model.parameters(), lr=lr, **extra_kw)

    losses = _train(model, opt, data, target, 200)
    assert losses[-1] < losses[0] * 0.5, f"Loss did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"

    del model, opt
    clean()


def _check_no_fp64(obj, path=""):
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        if "step" in path:
            return
        assert obj.dtype != torch.float64, f"fp64 tensor at {path}: dtype={obj.dtype}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_fp64(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_no_fp64(v, f"{path}[{i}]")


@pytest.mark.parametrize("opt_cls,lr,extra_kw", _PSGD_OPTIMIZERS, ids=[t[0].__name__ for t in _PSGD_OPTIMIZERS])
def test_psgd_no_fp64_in_state(opt_cls, lr, extra_kw):
    set_torch()
    torch.manual_seed(42)
    data = torch.randn(32, 64, device="cuda")
    target = torch.randn(32, 32, device="cuda")

    torch.manual_seed(0)
    model = nn.Linear(64, 32, bias=False, device="cuda")
    opt = opt_cls(model.parameters(), lr=lr, **extra_kw)

    _train(model, opt, data, target, 5)

    for param in model.parameters():
        _check_no_fp64(opt.state[param])

    del model, opt
    clean()
