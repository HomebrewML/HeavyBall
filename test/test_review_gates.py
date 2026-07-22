"""RED release gates for the review's independently reproduced public-contract failures.

Each test states an actual public contract that CURRENTLY fails (xfail strict), so it documents the bug
and turns into a hard failure the moment the contract is met (forcing removal of the marker). These are
the "turn counterexamples into release gates" step: the smallest test stating each reproduced defect.
"""
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


def test_always_step_updates_every_parameter():
    """Documented design choice (not a gate): heavyball updates EVERY parameter every step regardless of
    gradient presence -- there is no per-parameter activity detection. An inactive leaf still advances,
    e.g. weight decay shrinks it. This diverges from torch.optim's skip-grad-None, so MoE / conditional /
    frozen-then-unfrozen users must ensure every optimized parameter receives a gradient, or accept the
    divergence. Kept deliberately (activity decision) for zero step-time cost and no implicit machinery;
    this test guards that the always-step behavior is not silently changed."""
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball
        p = nn.Parameter(torch.tensor([3.0]))
        opt = heavyball.AdamW([p], lr=0.1, weight_decay=0.1)
        opt.zero_grad()  # p.grad is the persistent zero slab; no backward touches it (inactive)
        opt.step()
        assert p.item() != pytest.approx(3.0)  # decayed despite no gradient -> always-step


def test_compile_identity_survives_parameter_order_permutations():
    import heavyball
    torch._dynamo.reset()
    try:
        shapes = [(2,), (3,), (4,), (5,)]
        for perm in list(itertools.permutations(range(4)))[:9]:
            params = [nn.Parameter(torch.randn(*shapes[j])) for j in perm]
            opt = heavyball.AdamW(params, lr=0.1)
            sum((p * p).sum() for p in params).backward()
            opt.step()
    finally:
        torch._dynamo.reset()


@pytest.mark.xfail(strict=True, reason="numerical policy: unscaled squares/sums overflow avoidable "
                                        "intermediates, so representable finite fp64 inputs produce inf/NaN")
def test_matrix_transforms_finite_on_large_finite_input():
    from heavyball.transforms import Tempo, normuon_normalize, normuon_normalize_init, polargrad_direction

    def tempo(dt):
        return Tempo(torch.tensor(1), torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.bool),
                     SimpleNamespace(beta2=torch.tensor(0.95, dtype=dt), eps=torch.tensor(1e-7, dtype=dt)), False)

    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball  # noqa: F401
        torch.manual_seed(0)
        polar_in = torch.randn(1, 8, 4, dtype=torch.float64) * 9e307
        polar_out = polargrad_direction(polar_in.clone(), None, None, {}, tempo(torch.float64))[0]
        normuon_in = torch.randn(1, 8, 4, dtype=torch.float64) * 1e154
        state = {"moment2": normuon_normalize_init(normuon_in)["moment2"]}
        normuon_out = normuon_normalize(normuon_in.clone(), None, None, state, tempo(torch.float64))[0]
    assert torch.isfinite(polar_out).all() and torch.isfinite(normuon_out).all()


def test_lifecycle_swap_bumps_version():
    """eval()/train() rewrite the parameter representation, so they must bump _version like an ordinary
    step -- otherwise a retained autograd graph through the mutated parameter silently returns a wrong
    gradient. Was RED before the shared _bump_versions path covered the swap; guards the fix."""
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball
        p = nn.Parameter(torch.tensor([2.0]))
        opt = heavyball.ScheduleFree([p], lr=0.1)
        for _ in range(3):
            opt.zero_grad(); (p * p).sum().backward(); opt.step()
        opt.train()
        y = (p * p).sum()
        v0 = p._version
        opt.eval()
        assert p._version > v0, "eval() mutated the parameter without bumping _version"
        p.grad = None
        with pytest.raises(RuntimeError):
            y.backward()  # retained graph through the eval-mutated parameter must error
