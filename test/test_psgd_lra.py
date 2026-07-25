from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.lra import _lra_precond, lra, make_psgd_lra


@pytest.fixture(autouse=True)
def disable_compile():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def test_lra_precond_matches_dense_qtq():
    batch, flat, rank = 2, 10, 3
    g = torch.randn(batch, flat, dtype=torch.float64)
    U = torch.randn(batch, flat, rank, dtype=torch.float64) * 0.1
    V = torch.randn(batch, flat, rank, dtype=torch.float64) * 0.1
    d = torch.rand(batch, flat, dtype=torch.float64) + 0.5
    identity = torch.eye(flat, dtype=torch.float64).expand(batch, -1, -1)
    q = (identity + U @ V.mT) * d.unsqueeze(-2)
    result = _lra_precond(g, U, V, d)
    expected = ((q.mT @ q) @ g.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result, expected)


def test_lra_different_ranks():
    torch.manual_seed(0)
    initial = torch.randn(4, 4)
    grads = [torch.randn(4, 4) * 0.1 for _ in range(5)]

    results = {}
    for rank in [1, 5, 10]:
        param = torch.nn.Parameter(initial.clone())
        torch.manual_seed(0)
        recipe = replace(
            lra,
            chain=(make_psgd_lra(rank=rank),),
            defaults={**lra.defaults, "rank": rank},
        )
        optimizer = Engine(
            [param],
            recipe,
            lr=1e-3,
            rank=rank,
            precond_lr=0.1,
            dampening=1e-9,
        )
        for step, grad in enumerate(grads):
            param.grad.copy_(grad)
            torch.manual_seed(3000 + step)
            optimizer.step()
        results[rank] = param.detach().clone()

    assert not torch.allclose(results[1], results[5], atol=1e-6)
    assert not torch.allclose(results[5], results[10], atol=1e-6)
