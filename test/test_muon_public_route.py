"""Public-route certification for Muon (the highest-use matrix optimizer).

This runs the FULL public optimizer over multiple steps and compares it to an INDEPENDENT reimplementation
of the original Keller Jordan Muon -- heavy-ball Nesterov momentum, Newton-Schulz orthogonalization,
sqrt(max(1, out/in)) update scale, and decoupled weight decay. It is a public-route oracle, not a transform
test: it is exactly the class of check that catches a correct transform sitting in a wrong recipe (the
PolarGrad failure mode). The Newton-Schulz kernel itself is verified separately (test_normuon_polargrad_
equations, test_aurora); this shares it so the comparison isolates the recipe (momentum + scale + decay).

Scope: this certifies the ALGORITHM is the correct Muon (necessary -- it must be Muon, not a different
optimizer), run here in fp64 so the recipe is checked exactly. It is NOT the precision objective. The
objective -- heavyball's low-precision execution being as close or closer to full-fp64 than a naive
same-dtype baseline -- is a separate property of heavyball's stochastic-rounding path, verified in
test_precision_objective. Matching the fp64 algorithm is necessary but is not, by itself, that objective.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from heavyball.transforms import Tempo, orthogonalize


def _ortho(direction):
    tempo = Tempo(torch.tensor(1), torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.bool),
                  SimpleNamespace(eps=torch.tensor(1e-7, dtype=torch.float64)), False)
    return orthogonalize(direction.unsqueeze(0), None, None, {}, tempo)[0][0]


def _kj_muon(grads, shape, lr, beta, wd):
    p, buf = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
    scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
    for g in grads:
        buf = beta * buf + g                 # heavy-ball buffer
        u = _ortho(g + beta * buf) * scale   # Nesterov look-ahead, orthogonalized, scaled
        p = p * (1 - lr * wd) - lr * u       # decoupled weight decay
    return p


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (6, 6), (16, 3)])
def test_muon_public_route_matches_keller_jordan(shape):
    torch.manual_seed(0)
    grads = [torch.randn(*shape, dtype=torch.float64) for _ in range(6)]
    lr, beta, wd = 0.1, 0.95, 0.1
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball
        p = nn.Parameter(torch.zeros(*shape, dtype=torch.float64))
        opt = heavyball.Muon([p], lr=lr, beta1=beta, weight_decay=wd)
        for g in grads:
            opt.zero_grad()
            p.grad.copy_(g)
            opt.step()
        hb = p.detach()
    ref = _kj_muon(grads, shape, lr, beta, wd)
    torch.testing.assert_close(hb, ref, rtol=0, atol=1e-12)
