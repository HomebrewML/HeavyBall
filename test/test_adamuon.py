"""Independent oracle for AdaMuon (arXiv:2507.11005): Muon's orthogonalized direction with RMSprop's
second-moment normalization, then RMS-aligned rescaling to a fixed RMS 0.2 (the paper's core
contribution -- it matches the update magnitude to Adam so Adam's LR schedules transfer), committed
WITHOUT Muon's aspect scale (which is only correct for a truly-orthogonal update; O/sqrt(v) is not).
The structural check pins that composition; behavioral checks confirm it optimizes, compiles fullgraph,
and ships a working facade. See test_adamuon_rms_align.py for the RMS-0.2 shape-invariance property.
"""

from unittest.mock import patch

import torch

from heavyball import Engine, adamuon
from heavyball.transforms import momentum, orthogonalize, rms_align, rmsprop as rmsprop_transform, sgd_commit


def test_adamuon_is_muon_rmsprop_then_rms_aligned():
    matrix = adamuon.then
    assert matrix.chain == (momentum, orthogonalize, rmsprop_transform, rms_align)
    assert matrix.commit is sgd_commit


def test_adamuon_minimizes_a_convex_quadratic():
    torch.manual_seed(3)
    target = torch.randn(6, 4)
    param = torch.nn.Parameter(torch.zeros(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], adamuon, lr=0.1, beta1=0.9, beta2=0.95, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(150):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial


def test_adamuon_compiles_fullgraph():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], adamuon, lr=0.05, beta1=0.9, beta2=0.95, weight_decay=0.0)
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        assert torch.isfinite(param).all()
    finally:
        torch._dynamo.reset()


def test_adamuon_facade_trains():
    from heavyball import AdaMuon

    assert AdaMuon.recipe is adamuon
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = AdaMuon(model.parameters(), lr=0.05)
    initial = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(inputs), targets) < initial
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
