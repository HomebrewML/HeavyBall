from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.core import Engine
from heavyball.lra import _lra_low_rank_mm, _lra_precond, lra, make_psgd_lra, psgd_lra_init


def _no_compile(function, **kwargs):
    return function


@pytest.fixture(autouse=True)
def disable_compile():
    with patch("heavyball.core.torch.compile", _no_compile):
        yield


def test_lra_init_shapes():
    """U, V have (flat_dim, rank), d has (flat_dim,)."""

    ref = torch.randn(64)
    state = psgd_lra_init(ref, rank=5)
    assert state["U"].shape == (64, 5)
    assert state["V"].shape == (64, 5)
    assert state["d"].shape == (64,)


def test_lra_init_shapes_multidim():
    """Higher-rank leaves flatten to 1D."""

    ref = torch.randn(8, 16)
    state = psgd_lra_init(ref, rank=10)
    assert state["U"].shape == (128, 10)
    assert state["d"].shape == (128,)


def test_lra_low_rank_mm_matches_dense():
    batch, flat, rank = 2, 12, 4
    A = torch.randn(batch, flat, rank, dtype=torch.float64)
    B = torch.randn(batch, flat, rank, dtype=torch.float64)
    x = torch.randn(batch, flat, dtype=torch.float64)
    identity = torch.eye(flat, dtype=torch.float64).expand(batch, -1, -1)

    result = _lra_low_rank_mm(A, B, x)

    torch.testing.assert_close(result, _mv_dense(identity + A @ B.mT, x))


def _mv_dense(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def test_lra_precond_identity_init():
    """With d=1, U=0, V=0, preconditioning is identity."""

    batch, flat = 3, 32
    g = torch.randn(batch, flat, dtype=torch.float64)
    U = torch.zeros(batch, flat, 5, dtype=torch.float64)
    V = torch.zeros(batch, flat, 5, dtype=torch.float64)
    d = torch.ones(batch, flat, dtype=torch.float64)
    result = _lra_precond(g, U, V, d)
    torch.testing.assert_close(result, g)


def test_lra_precond_diagonal_only():
    """With U=0, V=0, preconditioning scales by d^2."""

    batch, flat = 2, 16
    g = torch.randn(batch, flat, dtype=torch.float64)
    d = torch.rand(batch, flat, dtype=torch.float64) + 0.5
    U = torch.zeros(batch, flat, 3, dtype=torch.float64)
    V = torch.zeros(batch, flat, 3, dtype=torch.float64)
    result = _lra_precond(g, U, V, d)
    torch.testing.assert_close(result, d * d * g)


def test_lra_precond_matches_dense_qtq():
    """Precond output matches dense Q^T Q multiplication."""

    batch, flat, rank = 2, 10, 3
    g = torch.randn(batch, flat, dtype=torch.float64)
    U = torch.randn(batch, flat, rank, dtype=torch.float64) * 0.1
    V = torch.randn(batch, flat, rank, dtype=torch.float64) * 0.1
    d = torch.rand(batch, flat, dtype=torch.float64) + 0.5
    identity = torch.eye(flat, dtype=torch.float64).expand(batch, -1, -1)
    q = (identity + U @ V.mT) * d.unsqueeze(-2)
    result = _lra_precond(g, U, V, d)
    torch.testing.assert_close(result, _mv_dense(q.mT @ q, g))


def test_lra_trains_small_model():
    """PSGD-LRA decreases loss on a simple model."""

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    optimizer = heavyball.PSGDLRA(model.parameters(), lr=0.01, rank=5)
    inputs = torch.randn(32, 8)
    targets = torch.randn(32, 4)
    losses = []
    for step in range(30):
        optimizer.zero_grad()
        loss = ((model(inputs) - targets) ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        torch.manual_seed(1000 + step)
        optimizer.step()
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert torch.isfinite(torch.tensor(losses)).all()


def test_lra_finite_updates():
    """LRA produces finite updates for 8 steps."""

    param = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.float64))
    optimizer = Engine([param], lra, lr=1e-3, rank=5, precond_lr=0.1, dampening=1e-9)
    for step in range(8):
        param.grad.copy_(torch.randn_like(param) * 0.1)
        torch.manual_seed(2000 + step)
        optimizer.step()
    assert torch.isfinite(param).all()


def test_lra_different_ranks():
    """Different ranks produce different trajectories."""

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
