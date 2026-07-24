"""Behavioral coverage for preconditioner topology and validation branches."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.kron import kron, make_psgd_kron, psgd_kron_init
from heavyball.matrix import merged_matrix_shape, soap, soap_init, soap_recipe
from heavyball.psgd_pro import make_psgd_pro, qsgd
from heavyball.transforms import Tempo, beta_debias


def _eager_engine(params, recipe, **hyper) -> Engine:
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, recipe, **hyper)


def test_kron_both_diagonal_refresh_matches_closed_form():
    """Both oversized axes use the diagonal/diagonal refresh and apply paths."""

    rows, columns = 3, 4
    gradient = torch.arange(1, rows * columns + 1, dtype=torch.float64).reshape(rows, columns) / 10
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    optimizer = _eager_engine(
        [parameter],
        kron,
        lr=0.25,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=0.0,
        max_size_triangular=2,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    assert state["Q_0"].shape == (1, rows)
    assert state["Q_1"].shape == (1, columns)

    torch.manual_seed(123)
    probe = torch.randn(1, rows, columns, dtype=torch.float64)
    parameter.grad.copy_(gradient)
    with patch.object(Tempo, "randn_like", lambda _tempo, value: probe.to(value)):
        optimizer.step(step_type="refresh")

    hessian_vector = gradient.unsqueeze(0) + torch.finfo(gradient.dtype).eps * gradient.abs().unsqueeze(0) * probe
    term1_0 = hessian_vector.square().sum(dim=-1)
    term2_0 = probe.square().sum(dim=-1)
    expected_lower0 = (term1_0 + term2_0).amax(dim=-1)
    expected_q0 = 1 - 0.1 * (term1_0 - term2_0) / expected_lower0.unsqueeze(-1)
    term1_1 = hessian_vector.square().sum(dim=-2)
    term2_1 = probe.square().sum(dim=-2)
    expected_lower1 = (term1_1 + term2_1).amax(dim=-1)
    expected_q1 = 1 - 0.1 * (term1_1 - term2_1) / expected_lower1.unsqueeze(-1)
    expected_parameter = (
        -0.25
        * gradient
        * expected_q0.squeeze(0).square().unsqueeze(-1)
        * expected_q1.squeeze(0).square().unsqueeze(-2)
    )

    torch.testing.assert_close(state["Q_0"], expected_q0, rtol=0, atol=0)
    torch.testing.assert_close(state["Q_1"], expected_q1, rtol=0, atol=0)
    torch.testing.assert_close(state["running_lower_bound_0"], expected_lower0, rtol=0, atol=0)
    torch.testing.assert_close(state["running_lower_bound_1"], expected_lower1, rtol=0, atol=0)
    torch.testing.assert_close(parameter, expected_parameter, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("topology", ("triangular_diagonal", "diagonal_diagonal"))
def test_qsgd_oversized_topologies_apply_each_factor_once(topology):
    """QSGD applies a right-only diagonal factor and two diagonal factors exactly once."""

    gradient = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    limit = 3 if topology == "triangular_diagonal" else 2
    optimizer = _eager_engine(
        [parameter], qsgd, lr=0.25, max_size_triangular=limit, weight_decay=0.0
    )
    state = optimizer.groups[0].states[0]
    q1 = torch.tensor([[5.0, 6.0, 7.0, 8.0]], dtype=torch.float64)
    state["Q_1"].copy_(q1)
    if topology == "triangular_diagonal":
        q0 = torch.tensor(
            [[[2.0, 0.5, -0.25], [0.0, 3.0, 0.75], [0.0, 0.0, 4.0]]], dtype=torch.float64
        )
        state["Q_0"].copy_(q0)
        expected_update = (q0.squeeze(0) @ gradient) * q1
        assert state["Q_0"].ndim == 3
    else:
        q0 = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
        state["Q_0"].copy_(q0)
        expected_update = q0.mT * gradient * q1
        assert state["Q_0"].ndim == 2

    before_state = {name: value.clone() for name, value in state.items()}
    parameter.grad.copy_(gradient)
    optimizer.step(step_type="normal")

    torch.testing.assert_close(parameter, -0.25 * expected_update, rtol=0, atol=0)
    for name, expected in before_state.items():
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0)


def test_kron_initialization_and_factory_guards_report_exact_errors():
    with pytest.raises(ValueError, match="^kron requires nonempty merged 2D parameter leaves$"):
        psgd_kron_init(torch.zeros(0, 3))
    with pytest.raises(TypeError, match="^power_iterations must be a Python int$"):
        make_psgd_kron(2.0)


def test_psgd_pro_factory_guards_reject_bool_like_configuration():
    with pytest.raises(TypeError, match="^power_iterations must be a Python int$"):
        make_psgd_pro(True)
    with pytest.raises(TypeError, match="^sqrt must be a Python bool$"):
        make_psgd_pro(sqrt=1)


def test_matrix_shape_and_dimension_guards_report_exact_results():
    assert merged_matrix_shape((), 3) == ()
    with pytest.raises(
        ValueError,
        match="^leaf merges to >2D at max_precond_dim=3; N-factor preconditioning is a follow-up$",
    ):
        soap_init(torch.zeros(2, 4, 5), max_precond_dim=torch.tensor(3))
    with pytest.raises(
        ValueError,
        match="^leaf merges to 1D at max_precond_dim=3; matrix preconditioning requires 2D$",
    ):
        soap_init(torch.zeros(4), max_precond_dim=torch.tensor(3))


def test_soap_left_only_basis_refresh_transports_both_moments():
    """An oversized column axis leaves one basis, whose refresh transports both Adam moments."""

    parameter = torch.nn.Parameter(torch.zeros(2, 5, dtype=torch.float64))
    optimizer = _eager_engine(
        [parameter],
        replace(soap_recipe, chain=(soap,)),
        lr=0.1,
        beta1=0.9,
        beta2=0.8,
        shampoo_beta=0.7,
        eps=1e-8,
        max_precond_dim=2,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    assert set(state) == {
        "GG_l",
        "GG_l_scale",
        "Q_l",
        "exp_avg",
        "exp_avg_sq",
    }
    first = torch.tensor(
        [[3.0, 0.0, 1.0, 0.0, 2.0], [0.0, 2.0, 0.0, 1.0, 1.0]], dtype=torch.float64
    )
    second = torch.tensor(
        [[1.0, 2.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0, 3.0]], dtype=torch.float64
    )
    third = torch.tensor(
        [[0.0, 1.0, 3.0, 0.0, 2.0], [2.0, 1.0, 0.0, 2.0, 1.0]], dtype=torch.float64
    )

    parameter.grad.copy_(first)
    optimizer.step(step_type="normal")
    old_left = state["Q_l"].clone()
    old_avg = state["exp_avg"].clone()
    old_avg_sq = state["exp_avg_sq"].clone()
    torch.testing.assert_close(old_avg, first.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(old_avg_sq, first.square().unsqueeze(0), rtol=0, atol=0)

    parameter.grad.copy_(second)
    optimizer.step(step_type="refresh")
    new_left = state["Q_l"].clone()
    assert not torch.equal(new_left, old_left)

    age = torch.tensor([2], dtype=torch.int64)
    hyper = optimizer.groups[0].hyper
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    projected = old_left.mT @ second.unsqueeze(0)
    raw_avg = old_avg * beta1 + projected * (1 - beta1)
    raw_avg_sq = old_avg_sq * beta2 + projected.square() * (1 - beta2)
    change = new_left.mT @ old_left
    expected_avg = change @ raw_avg
    expected_avg_sq = change.square() @ raw_avg_sq

    torch.testing.assert_close(state["exp_avg"], expected_avg, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(state["exp_avg_sq"], expected_avg_sq, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(new_left @ state["exp_avg"], old_left @ raw_avg, rtol=1e-15, atol=1e-15)

    before = parameter.detach().clone()
    age = torch.tensor([3], dtype=torch.int64)
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    projected = new_left.mT @ third.unsqueeze(0)
    expected_avg = expected_avg * beta1 + projected * (1 - beta1)
    expected_avg_sq = expected_avg_sq * beta2 + projected.square() * (1 - beta2)
    projected_update = expected_avg / expected_avg_sq.sqrt().clamp_min(hyper.eps)
    expected_parameter = before - hyper.lr * (new_left @ projected_update).squeeze(0)

    parameter.grad.copy_(third)
    optimizer.step(step_type="normal")

    torch.testing.assert_close(parameter, expected_parameter, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(state["Q_l"], new_left, rtol=0, atol=0)
    torch.testing.assert_close(state["exp_avg"], expected_avg, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(state["exp_avg_sq"], expected_avg_sq, rtol=1e-15, atol=1e-15)
