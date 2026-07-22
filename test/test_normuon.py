"""Independent property oracle for NorMuon: row/col second-moment normalization with Frobenius preservation.

Verifies NorMuon's defining behavior against neither the upstream source nor the shipped transform:
the orthogonalized direction is divided by its per-output-neuron (row) RMS -- reducing the second
moment over the input axis for tall and wide leaves alike, per NorMuon (arXiv:2510.05491) Alg. 1 --
then rescaled to preserve the orthogonalized update's Frobenius norm. Property oracle, calibrated
independently against the paper.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, normuon
from heavyball.transforms import Tempo, normuon_normalize, orthogonalize


def _tempo(count: int) -> Tempo:
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(beta2=torch.tensor(0.95, dtype=torch.float64), eps=torch.tensor(1e-8, dtype=torch.float64)),
        False,
    )


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_normuon_equalizes_axis_rms_and_preserves_frobenius(shape):
    torch.manual_seed(0)
    m = torch.randn(1, *shape, dtype=torch.float64)
    orth = orthogonalize(m.clone(), None, None, {}, _tempo(1))[0]
    state = normuon_normalize.init(m[0])
    update = normuon_normalize(orth.clone(), None, None, state, _tempo(1))[0]
    assert update.norm().item() == pytest.approx(orth.norm().item(), rel=1e-6)  # Frobenius preserved
    reduce_dim = -1  # per output-neuron (row), reducing over the input axis; NorMuon Alg. 1
    before = orth.square().mean(dim=reduce_dim).sqrt().std()
    after = update.square().mean(dim=reduce_dim).sqrt().std()
    assert after.item() < 0.2 * before.item()  # per-neuron RMS equalized on the first step


def test_normuon_minimizes_a_convex_quadratic():
    torch.manual_seed(3)
    target = torch.randn(6, 4)
    param = torch.nn.Parameter(torch.zeros(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], normuon, lr=0.1, beta1=0.9, beta2=0.95, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(150):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial


def test_normuon_compiles_fullgraph():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], normuon, lr=0.05, beta1=0.9, beta2=0.95, weight_decay=0.0)
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        assert torch.isfinite(param).all()
    finally:
        torch._dynamo.reset()


def test_normuon_facade_trains():
    from heavyball import NorMuon

    assert NorMuon.recipe is normuon
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = NorMuon(model.parameters(), lr=0.05)
    initial = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(inputs), targets) < initial
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
