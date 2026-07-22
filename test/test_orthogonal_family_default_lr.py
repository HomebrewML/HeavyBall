"""Regression test for the orthogonal-family default learning rate.

At the former 0.0025 default these orthogonal-family optimizers under-step
(the aspect-scaled orthogonal update is tiny at 0.0025) and so converge slowly:
on a real MNIST autoencoder they are 1.3-1.6x worse than AdamW at ~100 steps,
though they pass AdamW by ~200+ steps.  Muon's 0.02 converges ~2x faster and
dominates 0.0025 at every horizon tested (verified to 600 steps for Muon; the
siblings share the mechanism).  Oblique is deliberately left at 0.0025 because
it is adam-based and 0.02 measurably OVERSHOOTS (0.051 -> 0.067).
"""

import heavyball
import torch


def test_orthogonal_family_user_facing_default_lr():
    p = torch.nn.Parameter(torch.randn(4, 4))

    assert {g["lr"] for g in heavyball.Muon([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.Aurora([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.MuonLaProp([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.SpEL([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.NorMuon([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.PolarGrad([p]).param_groups} == {0.02}
    assert {g["lr"] for g in heavyball.Oblique([p]).param_groups} == {0.0025}
