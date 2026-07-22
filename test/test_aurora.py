"""Independent property oracle for the Aurora leverage-balanced polar direction.

Verifies Aurora's DEFINING behavior without referencing the upstream Aurora
source or the shipped transform internals: the direction is a polar factor
(orthonormal columns, NS-5 tolerance) whose row leverages are driven toward
uniform (target n/m), measurably tighter than the plain polar. These are
property oracles, not a reimplementation -- a bug shared between the transform
and a naive reference cannot satisfy them.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, aurora
from heavyball.transforms import Tempo, balanced_orthogonalize, orthogonalize


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


def _row_leverage(factor):
    tall = factor if factor.shape[-2] >= factor.shape[-1] else factor.mT
    return (tall**2).sum(dim=-1)


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (16, 3)])
def test_balanced_orthogonalize_is_a_polar_factor(shape):
    torch.manual_seed(0)
    g = torch.randn(1, *shape, dtype=torch.float64)
    factor = _direction(balanced_orthogonalize, g)[0]
    tall = factor if factor.shape[-2] >= factor.shape[-1] else factor.mT
    singular = torch.linalg.svdvals(tall)
    assert singular.min() > 0.95
    assert singular.max() < 1.05


@pytest.mark.parametrize("shape", [(8, 4), (4, 8)])
def test_balanced_orthogonalize_uniformizes_row_leverage(shape):
    torch.manual_seed(1)
    # Skew the tall orientation's rows: this is row-scaling leverage the diagonal
    # balancer can correct. Present it tall, or transposed for the wide case (the
    # balancer tall-orients internally, so the skew must live in the tall's rows).
    tall = torch.randn(1, max(shape), min(shape), dtype=torch.float64)
    tall[0, :2] *= 40.0
    g = tall if shape[0] >= shape[1] else tall.mT
    target = min(shape) / max(shape)
    plain = _row_leverage(_direction(orthogonalize, g)[0])
    balanced = _row_leverage(_direction(balanced_orthogonalize, g)[0])
    assert plain.std() > 0.15  # the input has genuinely non-uniform leverage
    assert balanced.std() < 0.5 * plain.std()  # balancing tightens the leverage spread
    assert balanced.mean().item() == pytest.approx(target, abs=0.05)


def test_balanced_orthogonalize_leaves_square_to_the_plain_polar():
    torch.manual_seed(2)
    g = torch.randn(1, 5, 5, dtype=torch.float64)
    torch.testing.assert_close(
        _direction(balanced_orthogonalize, g),
        _direction(orthogonalize, g),
        rtol=0,
        atol=0,
    )


def test_balanced_orthogonalize_is_transpose_equivariant():
    # An INDEPENDENT wide matrix (not the transpose of a tall case): balanced(Gᵀ) must equal balanced(G)ᵀ,
    # proving tall and wide are handled identically. Guards the NorMuon shape-conditional-axis bug class,
    # which the (8,4)/(4,8) tests above cannot catch since they build wide as the transpose of tall.
    torch.manual_seed(7)
    g = torch.randn(1, 16, 64, dtype=torch.float64)
    balanced = _direction(balanced_orthogonalize, g)
    balanced_transposed = _direction(balanced_orthogonalize, g.mT.contiguous())
    torch.testing.assert_close(balanced_transposed, balanced.mT, rtol=1e-6, atol=1e-6)


def test_aurora_minimizes_a_convex_quadratic():
    torch.manual_seed(3)
    target = torch.randn(6, 4)
    param = torch.nn.Parameter(torch.zeros(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], aurora, lr=0.1, beta1=0.9, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(120):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial


def test_aurora_compiles_fullgraph():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], aurora, lr=0.05, beta1=0.9, weight_decay=0.0)  # real fullgraph compile
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        assert torch.isfinite(param).all()
    finally:
        torch._dynamo.reset()


def test_aurora_facade_trains():
    from heavyball import Aurora

    assert Aurora.recipe is aurora
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = Aurora(model.parameters(), lr=3e-3)
    initial = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(inputs), targets) < initial
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
