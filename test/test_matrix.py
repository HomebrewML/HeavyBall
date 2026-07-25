"""Proofs for the slab-native matrix preconditioners."""

from dataclasses import replace

import torch

from heavyball.core import Engine
from heavyball.matrix import soap, soap_recipe
from heavyball.transforms import beta_debias


def _project(value: torch.Tensor, left: torch.Tensor, right: torch.Tensor, *, back: bool) -> torch.Tensor:
    if back:
        return torch.einsum("nab,nia,njb->nij", value, left, right)
    return torch.einsum("nij,nia,njb->nab", value, left, right)


def test_soap_refresh_transports():
    """Refreshes transport both moments into the new basis: the first by rotation (physical value
    retained), the second by Hadamard-square of the same basis change."""

    param = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer = Engine(
        [param],
        replace(soap_recipe, chain=(soap,)),
        lr=0.1,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.8,
        eps=1e-8,
        max_precond_dim=8,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    first = torch.tensor(((3.0, 0.0, 1.0, 0.0), (0.0, 2.0, 0.0, 1.0), (1.0, 0.0, 1.0, 2.0)))
    second = torch.tensor(((0.0, 1.0, 2.0, 0.0), (2.0, 1.0, 0.0, 3.0), (1.0, 3.0, 1.0, 0.0)))
    third = torch.tensor(((1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 3.0, 0.0), (2.0, 0.0, 1.0, 2.0)))

    initial_left, initial_right = state["Q_l"].clone(), state["Q_r"].clone()
    param.grad.copy_(first)
    optimizer.step(step_type="normal")
    first_left, first_right = state["Q_l"].clone(), state["Q_r"].clone()
    first_gg_left, first_gg_right = state["GG_l"].clone(), state["GG_r"].clone()
    first_avg, first_avg_sq = state["exp_avg"].clone(), state["exp_avg_sq"].clone()
    torch.testing.assert_close(first_left, initial_left, rtol=0, atol=0)
    torch.testing.assert_close(first_right, initial_right, rtol=0, atol=0)
    torch.testing.assert_close(first_gg_left, first.unsqueeze(0) @ first.unsqueeze(0).mT)
    torch.testing.assert_close(first_gg_right, first.unsqueeze(0).mT @ first.unsqueeze(0))

    param.grad.copy_(second)
    optimizer.step(step_type="refresh")
    refreshed_left, refreshed_right = state["Q_l"].clone(), state["Q_r"].clone()
    assert not torch.equal(refreshed_left, first_left)
    assert not torch.equal(refreshed_right, first_right)

    age = torch.full((1,), 2, dtype=torch.int64)
    hyper = optimizer.groups[0].hyper
    projected = _project(second.unsqueeze(0), first_left, first_right, back=False)
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    raw_avg = first_avg * beta1 + projected * (1 - beta1)
    raw_variance = first_avg_sq.square() * beta2 + projected.square() * (1 - beta2)
    physical_before_transport = _project(raw_avg, first_left, first_right, back=True)
    physical_after_transport = _project(state["exp_avg"], refreshed_left, refreshed_right, back=True)
    torch.testing.assert_close(physical_after_transport, physical_before_transport, rtol=2e-5, atol=2e-5)
    left_transition = torch.einsum("nAa,nAc->nac", first_left, refreshed_left).square()
    right_transition = torch.einsum("nBb,nBd->nbd", first_right, refreshed_right).square()
    transported_avg_sq = torch.einsum(
        "nab,nac,nbd->ncd", raw_variance, left_transition, right_transition
    ).sqrt()
    torch.testing.assert_close(state["exp_avg_sq"], transported_avg_sq, rtol=2e-5, atol=2e-5)
    assert not torch.equal(state["exp_avg"], raw_avg)
    assert not torch.equal(state["exp_avg_sq"], raw_variance.sqrt())

    shampoo_beta = beta_debias(hyper.shampoo_beta, age).reshape(1, 1, 1)
    expected_left = first_gg_left * shampoo_beta + (second.unsqueeze(0) @ second.unsqueeze(0).mT) * (1 - shampoo_beta)
    expected_right = first_gg_right * shampoo_beta + (second.unsqueeze(0).mT @ second.unsqueeze(0)) * (1 - shampoo_beta)
    torch.testing.assert_close(state["GG_l"], expected_left, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state["GG_r"], expected_right, rtol=2e-5, atol=2e-5)

    param.grad.copy_(third)
    optimizer.step(step_type="normal")
    torch.testing.assert_close(state["Q_l"], refreshed_left, rtol=0, atol=0)
    torch.testing.assert_close(state["Q_r"], refreshed_right, rtol=0, atol=0)
    assert not torch.equal(state["GG_l"], expected_left)
    assert not torch.equal(state["GG_r"], expected_right)


def test_soap_normal_and_refresh_are_stable_fullgraphs():
    """Normal preserves the basis, refresh changes it, and each host path compiles once."""

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
        optimizer = Engine(params, soap_recipe)
        state = optimizer.groups[0].states[0]
        initial_left = state["Q_l"].clone()
        gradient = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)

        optimizer.groups[0].grad_slab.copy_(gradient)
        optimizer.step(step_type="normal")
        torch.testing.assert_close(state["Q_l"], initial_left, rtol=0, atol=0)
        normal_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="normal")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == normal_graphs == 1

        optimizer.step(step_type="refresh")
        assert not torch.equal(state["Q_l"], initial_left)
        refresh_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="refresh")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == refresh_graphs == 2
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()
