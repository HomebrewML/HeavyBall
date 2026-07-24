from types import SimpleNamespace
from unittest.mock import patch

import torch

from heavyball import NorMuon
from heavyball.numerics import broadcast_leaf
from heavyball.transforms import (
    Tempo,
    beta_debias,
    normuon_normalize,
    normuon_normalize_init,
)


def _tempo(count: int, age: int = 1) -> Tempo:
    return Tempo(
        step=torch.ones((), dtype=torch.long),
        age=torch.full((count,), age, dtype=torch.long),
        live=torch.ones(count, dtype=torch.bool),
        hyper=SimpleNamespace(
            beta2=torch.tensor(0.95, dtype=torch.float64),
            eps=torch.tensor(1e-8, dtype=torch.float64),
        ),
        refresh=False,
    )


def _old_normuon_formula(
    update: torch.Tensor,
    state: dict[str, torch.Tensor],
    tempo: Tempo,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_mean = update.square().mean(dim=-1, keepdim=True)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    moment2 = state["moment2"].square() * beta2 + v_mean * (1 - beta2)
    normalized = update * moment2.clamp_min(tempo.hyper.eps).rsqrt()
    scale = update.norm(dim=(-2, -1), keepdim=True) / normalized.norm(
        dim=(-2, -1), keepdim=True
    ).clamp_min(tempo.hyper.eps)
    return normalized * scale, moment2


def test_normuon_normalize_large_finite_fp64_stays_finite():
    update = torch.full((4, 4), 1e154, dtype=torch.float64)
    state = normuon_normalize_init(update)

    output, next_state, _ = normuon_normalize(update, None, None, state, _tempo(4))

    assert torch.isfinite(output).all()
    assert torch.equal(output, update)
    assert torch.isfinite(next_state["moment2"]).all()


def test_normuon_normalize_matches_old_formula_at_normal_range():
    torch.manual_seed(123)
    update = torch.randn(4, 4, dtype=torch.float64)
    state = normuon_normalize_init(update)
    state["moment2"].copy_(torch.rand_like(state["moment2"]) + 0.1)
    tempo = _tempo(4, age=3)
    expected, expected_moment2 = _old_normuon_formula(update, state, tempo)

    output, next_state, _ = normuon_normalize(update, None, None, state, tempo)

    torch.testing.assert_close(output, expected, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(
        next_state["moment2"].square(), expected_moment2, rtol=1e-14, atol=1e-15
    )


def test_normuon_empty_matrix_does_not_crash():
    # A zero-element matrix (e.g. shape (0, n)) is routed here as a 2-D leaf; the reductions must not
    # divide by zero over an empty axis. It should pass through unchanged.
    update = torch.zeros(3, 0, 4)
    state = normuon_normalize_init(torch.zeros(0, 4))
    output, next_state, _ = normuon_normalize(update, None, None, state, _tempo(3))
    assert output.numel() == 0
    assert torch.isfinite(output).all()
    assert torch.isfinite(next_state["moment2"]).all()


def test_normuon_optimizer_one_step_stays_finite():
    torch.manual_seed(321)
    model = torch.nn.Linear(4, 4)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 4)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = NorMuon(model.parameters(), lr=0.01)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
