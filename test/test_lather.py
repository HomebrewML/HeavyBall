"""Proofs for the slab-native LATHER port."""

from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.lather import lather
from heavyball.matrix import _project
from heavyball.transforms import beta_debias


def _eager_engine(params, **hyper) -> Engine:
    """Build the exact Engine path without compiling either step artifact."""

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, lather, **hyper)


def _orthogonal(dimension: int, *, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(1, dimension * dimension + 1, dtype=dtype).reshape(dimension, dimension)
    return torch.linalg.qr(values + torch.eye(dimension, dtype=dtype) * dimension).Q


def _reference_project(
    matrix: torch.Tensor,
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    back: bool,
) -> torch.Tensor:
    if left is not None:
        matrix = (left if back else left.mT) @ matrix
    if right is not None:
        matrix = matrix @ (right.mT if back else right)
    return matrix


@pytest.mark.parametrize(("shape", "max_size"), (((3, 4), 8), ((12, 5), 8)))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float64, 1e-12, 1e-12), (torch.float32, 2e-6, 2e-6)),
)
def test_lather_normal_step_is_adam_in_the_stored_basis(shape, max_size, dtype, rtol, atol):
    """Recompute projection, debiased Adam, and inverse projection from their equations."""

    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.1
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    optimizer = _eager_engine(
        [parameter],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_size_triangular=max_size,
        weight_decay=0.0,
    )
    group = optimizer.groups[0]
    state = group.states[0]
    left = None
    right = None
    if "Q_basis_0" in state:
        left = _orthogonal(shape[0], dtype=dtype).unsqueeze(0)
        state["Q_basis_0"].copy_(left)
    if "Q_basis_1" in state:
        right = _orthogonal(shape[1], dtype=dtype).unsqueeze(0)
        state["Q_basis_1"].copy_(right)
    if shape[0] > max_size:
        assert state["Q_0"].shape == (1, shape[0])
        assert "Q_basis_0" not in state

    old_avg = torch.linspace(-0.4, 0.6, parameter.numel(), dtype=dtype).reshape(1, *shape)
    old_rms = torch.linspace(0.3, 0.9, parameter.numel(), dtype=dtype).reshape(1, *shape)
    state["exp_avg"].copy_(old_avg)
    state["exp_avg_sq"].copy_(old_rms)
    group.age.fill_(1)
    q_state = {name: value.clone() for name, value in state.items() if name.startswith(("Q_", "running_"))}
    gradient = torch.linspace(-1.1, 1.3, parameter.numel(), dtype=dtype).reshape(shape)

    projected = _reference_project(gradient.unsqueeze(0), left, right, back=False)
    debiased_beta1 = beta1 / (1 + beta1)
    debiased_beta2 = beta2 / (1 + beta2)
    expected_avg = old_avg * debiased_beta1 + projected * (1 - debiased_beta1)
    expected_rms = (
        old_rms.square() * debiased_beta2
        + projected.square() * (1 - debiased_beta2)
    ).sqrt()
    projected_direction = expected_avg / expected_rms.clamp_min(eps**0.5)
    expected_direction = _reference_project(projected_direction, left, right, back=True)[0]

    before = parameter.detach().clone()
    parameter.grad.copy_(gradient)
    optimizer.step(step_type="normal")

    torch.testing.assert_close((before - parameter) / lr, expected_direction, rtol=rtol, atol=atol)
    torch.testing.assert_close(state["exp_avg"], expected_avg, rtol=rtol, atol=atol)
    torch.testing.assert_close(state["exp_avg_sq"], expected_rms, rtol=rtol, atol=atol)
    for name, value in q_state.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def test_lather_both_oversized_axes_use_linear_factor_storage():
    parameter = torch.nn.Parameter(torch.randn(12, 10))
    optimizer = _eager_engine([parameter], max_size_triangular=8)
    state = optimizer.groups[0].states[0]

    assert state["Q_0"].shape == (1, 12)
    assert state["Q_1"].shape == (1, 10)
    assert "Q_basis_0" not in state
    assert "Q_basis_1" not in state


def test_lather_refresh_transports():
    """Only refreshes change Q/bases, and the first moment keeps its physical value."""

    param = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer = _eager_engine(
        [param],
        lr=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        precond_lr=0.05,
        lower_bound_beta=0.9,
        dampening=1e-6,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    first = torch.tensor(((3.0, 0.0, 1.0, 0.0), (0.0, 2.0, 0.0, 1.0), (1.0, 0.0, 1.0, 2.0)))
    second = torch.tensor(((0.0, 1.0, 2.0, 0.0), (2.0, 1.0, 0.0, 3.0), (1.0, 3.0, 1.0, 0.0)))
    third = torch.tensor(((1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 3.0, 0.0), (2.0, 0.0, 1.0, 2.0)))
    q_names = (
        "Q_0",
        "Q_1",
        "Q_basis_0",
        "Q_basis_1",
        "running_lower_bound_0",
        "running_lower_bound_1",
    )
    initial = {name: state[name].clone() for name in q_names}

    param.grad.copy_(first)
    optimizer.step(step_type="normal")
    for name, value in initial.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    old_left, old_right = state["Q_basis_0"].clone(), state["Q_basis_1"].clone()
    old_avg, old_avg_sq = state["exp_avg"].clone(), state["exp_avg_sq"].clone()
    age = torch.full((1,), 2, dtype=torch.int64)
    hyper = optimizer.groups[0].hyper
    projected = _project(second.unsqueeze(0), old_left, old_right, back=False)
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    raw_avg = old_avg * beta1 + projected * (1 - beta1)
    raw_avg_sq = (old_avg_sq.square() * beta2 + projected.square() * (1 - beta2)).sqrt()
    physical_before_transport = _project(raw_avg, old_left, old_right, back=True)

    param.grad.copy_(second)
    torch.manual_seed(964)
    optimizer.step(step_type="refresh")
    refreshed = {name: state[name].clone() for name in q_names}
    assert not torch.equal(refreshed["Q_0"], initial["Q_0"])
    assert not torch.equal(refreshed["Q_1"], initial["Q_1"])
    assert not torch.equal(refreshed["Q_basis_0"], old_left)
    assert not torch.equal(refreshed["Q_basis_1"], old_right)

    physical_after_transport = _project(
        state["exp_avg"], refreshed["Q_basis_0"], refreshed["Q_basis_1"], back=True
    )
    torch.testing.assert_close(physical_after_transport, physical_before_transport, rtol=2e-5, atol=2e-5)
    left_transition = torch.einsum("nia,nic->nac", old_left, refreshed["Q_basis_0"]).square()
    right_transition = torch.einsum("njb,njd->nbd", old_right, refreshed["Q_basis_1"]).square()
    expected_sq = torch.einsum(
        "nab,nac,nbd->ncd",
        raw_avg_sq.square(),
        left_transition,
        right_transition,
    ).clamp_min(0).sqrt()
    torch.testing.assert_close(state["exp_avg_sq"], expected_sq, rtol=2e-5, atol=2e-5)
    assert not torch.equal(state["exp_avg"], raw_avg)
    assert not torch.equal(state["exp_avg_sq"], raw_avg_sq)

    param.grad.copy_(third)
    optimizer.step(step_type="normal")
    for name, value in refreshed.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)


def test_lather_normal_and_refresh_are_stable_fullgraphs():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
        optimizer = Engine(
            params,
            lather,
            lr=0.01,
            precond_lr=0.05,
            dampening=1e-6,
            weight_decay=0.0,
        )
        state = optimizer.groups[0].states[0]
        initial_basis = state["Q_basis_0"].clone()
        optimizer.groups[0].grad_slab.copy_(
            torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
        )

        optimizer.step(step_type="normal")
        torch.testing.assert_close(state["Q_basis_0"], initial_basis, rtol=0, atol=0)
        normal_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="normal")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == normal_graphs == 1

        optimizer.step(step_type="refresh")
        assert not torch.equal(state["Q_basis_0"], initial_basis)
        refresh_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="refresh")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == refresh_graphs == 2
        assert all(torch.isfinite(param).all() for param in params)
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()
