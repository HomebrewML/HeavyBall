"""PSGD-PRO and QSGD must not depend on opt_einsum for safe contraction order."""

from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.psgd_pro import _apply_once, _apply_once_mixed


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


def test_apply_once_matches_reference_without_opt_einsum(monkeypatch):
    monkeypatch.setattr(torch.backends.opt_einsum, "enabled", False)
    torch.manual_seed(0)
    q0 = torch.randn(2, 13, 13, dtype=torch.float64)
    q1 = torch.randn(2, 17, 17, dtype=torch.float64)
    update = torch.randn(2, 13, 17, dtype=torch.float64)

    reference = torch.einsum("nia,njb,nab->nij", q0, q1, update)
    torch.testing.assert_close(_apply_once(update, q0, q1), reference, rtol=1e-9, atol=1e-9)


def test_apply_once_mixed_matches_reference_without_opt_einsum(monkeypatch):
    monkeypatch.setattr(torch.backends.opt_einsum, "enabled", False)
    torch.manual_seed(0)
    # q0 diagonal (oversized axis), q1 triangular
    q0, q1, update = torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 5, 5, dtype=torch.float64), torch.randn(2, 8, 5, dtype=torch.float64)
    ref = torch.einsum("...a,...Bb,...ab->...aB", q0, q1, update)
    torch.testing.assert_close(_apply_once_mixed(update, q0, q1), ref, rtol=1e-9, atol=1e-9)
    # q0 triangular, q1 diagonal (oversized axis)
    q0, q1, update = torch.randn(2, 5, 5, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 5, 8, dtype=torch.float64)
    ref = torch.einsum("...Aa,...b,...ab->...Ab", q0, q1, update)
    torch.testing.assert_close(_apply_once_mixed(update, q0, q1), ref, rtol=1e-9, atol=1e-9)
