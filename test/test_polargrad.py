"""Independent property oracle for PolarGrad: the polar direction scaled by the momentum's nuclear norm.

Verifies PolarGrad's defining behavior against neither the upstream source nor the shipped
transform: the update is the orthogonalized (polar) direction whose scale equals the nuclear norm
(sum of singular values) of the pre-orthogonalization momentum -- steepest descent under the
spectral norm. Property oracle, calibrated against torch.linalg.svdvals.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, polargrad
from heavyball.transforms import Tempo, polargrad_direction


def _tempo(count: int) -> Tempo:
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(),
        False,
    )


def _direction(transform, x):
    return transform(x.clone(), None, None, {}, _tempo(x.shape[0]))[0]


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])
def test_polargrad_scales_the_polar_direction_by_the_nuclear_norm(shape):
    torch.manual_seed(0)
    m = torch.randn(1, *shape, dtype=torch.float64)
    update = _direction(polargrad_direction, m)[0]
    nuclear = torch.linalg.svdvals(m[0]).sum()
    # update = polar(m) * scale, and ||polar||_F == sqrt(min(shape)) up to NS-5 tolerance
    scale = update.norm() / (min(shape) ** 0.5)
    assert scale.item() == pytest.approx(nuclear.item(), rel=0.05)
    normalized = update / scale
    tall = normalized if normalized.shape[-2] >= normalized.shape[-1] else normalized.mT
    singular = torch.linalg.svdvals(tall)
    assert singular.min() > 0.9 and singular.max() < 1.1


def test_polargrad_minimizes_a_convex_quadratic():
    torch.manual_seed(3)
    target = torch.randn(6, 4)
    param = torch.nn.Parameter(torch.zeros(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], polargrad, lr=1e-2, beta1=0.9, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(150):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial


def test_polargrad_compiles_fullgraph():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], polargrad, lr=0.01, beta1=0.9, weight_decay=0.0)
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        assert torch.isfinite(param).all()
    finally:
        torch._dynamo.reset()


def test_polargrad_facade_trains():
    from heavyball import PolarGrad

    assert PolarGrad.recipe is polargrad
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = PolarGrad(model.parameters(), lr=1e-2)
    initial = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(inputs), targets) < initial
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
