"""Independent oracle for Stochastic MCSD's SpEL specialization.

SpEL applies its matrix-sign LMO to the Stiefel-tangent projection of an EMA momentum, then
retracts the new point to the manifold.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, spel
from heavyball.transforms import (
    Tempo,
    stiefel_tangent_projection,
)


def _orthonormal_spread(weight):
    tall = weight if weight.shape[-2] >= weight.shape[-1] else weight.mT
    return torch.linalg.svdvals(tall)


def _tempo(count):
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(),
        False,
    )


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_spel_uses_the_stiefel_tangent_projection(shape):
    torch.manual_seed(10)
    tall = torch.linalg.qr(torch.randn(1, max(shape), min(shape), dtype=torch.float64)).Q
    point = tall if shape[0] >= shape[1] else tall.mT
    update = torch.randn_like(point)
    projected = stiefel_tangent_projection(
        update, None, point, {}, _tempo(point.shape[0])
    )[0]

    x = point if shape[0] >= shape[1] else point.mT
    g = update if shape[0] >= shape[1] else update.mT
    xtg = x.mT @ g
    expected = g - x @ ((xtg + xtg.mT) * 0.5)
    if shape[0] < shape[1]:
        expected = expected.mT
    torch.testing.assert_close(projected, expected, rtol=1e-14, atol=1e-14)


def test_spel_reprojects_the_weight_onto_the_stiefel_manifold():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], spel, lr=0.1, beta1=0.9, weight_decay=0.0)
    param.grad.copy_(torch.randn_like(param))
    optimizer.step()
    singular = _orthonormal_spread(param.detach())
    assert singular.min() > 0.9 and singular.max() < 1.1

