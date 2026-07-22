"""AdaMuon must apply arXiv:2507.11005's RMS-aligned rescaling: the pre-learning-rate update has per-leaf
RMS 0.2 shape-invariantly (the paper's core contribution, letting Adam's LR schedules transfer). The
shared muon_commit aspect scale is only correct for a truly-orthogonal update, which O/sqrt(v) is not; so
Muon/NorMuon/Aurora keep the aspect scale (shape-dependent) and must be unaffected by this fix."""
from unittest.mock import patch

import torch

import heavyball


def _prelr_update_rms(optimizer_cls, shape, lr=0.01):
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float64))
    before = parameter.detach().clone()
    with patch("heavyball.core.torch.compile", lambda f, **k: f):  # scoped: never leak into the session
        optimizer = optimizer_cls([parameter], lr=lr, weight_decay=0.0)
        parameter.square().sum().backward()
        optimizer.step()
    update = -(parameter.detach() - before) / lr  # sgd_commit: W -= lr*update
    return (update.norm() / update.numel() ** 0.5).item()


def test_adamuon_rms_aligned_and_shape_invariant():
    tall = _prelr_update_rms(heavyball.AdaMuon, (64, 16))
    wide = _prelr_update_rms(heavyball.AdaMuon, (16, 64))
    assert abs(tall - 0.2) < 5e-3, tall
    assert abs(wide - 0.2) < 5e-3, wide
    assert abs(tall - wide) < 5e-3, (tall, wide)  # shape-invariant, the paper's point


def test_muon_keeps_aspect_scale_unchanged():
    # Guard: rms_align must NOT leak into muon_commit. Muon stays aspect-scaled -> shape-dependent, not 0.2.
    tall = _prelr_update_rms(heavyball.Muon, (64, 16))
    wide = _prelr_update_rms(heavyball.Muon, (16, 64))
    assert abs(tall - wide) > 5e-2, (tall, wide)
    assert abs(tall - 0.2) > 2e-2, tall
