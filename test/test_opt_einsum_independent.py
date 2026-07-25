"""PSGD-PRO and QSGD must not depend on opt_einsum for safe contraction order."""

from unittest.mock import patch

import pytest
import torch

import heavyball


@pytest.mark.parametrize("optimizer_cls", (heavyball.QSGD, heavyball.PSGDPro))
def test_optimizer_stays_finite_without_opt_einsum(monkeypatch, optimizer_cls):
    monkeypatch.setattr(torch.backends.opt_einsum, "enabled", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 256 if device == "cuda" else 96
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(size, size, device=device))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = optimizer_cls([parameter], lr=1e-3)

    for _ in range(3):
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    assert torch.isfinite(parameter).all()
