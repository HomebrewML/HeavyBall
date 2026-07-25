import math
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.kl import (
    _factor_inverse,
    _heavy_factor_inverse,
    _heavy_kl_qr_basis,
    _kl_qr_basis,
)


def test_heavy_factor_inverse_is_thresholded_moore_penrose():
    eps_value = 1e-6
    stored_rms_values = (1e-6, math.sqrt(eps_value), 4.0, 9.0)
    stored_rms = torch.tensor(stored_rms_values, dtype=torch.float64)
    eps = torch.tensor(eps_value, dtype=torch.float64)
    threshold = math.sqrt(eps_value)

    expected_pseudoinverse = torch.tensor(
        [0.0 if value <= threshold else 1.0 / value for value in stored_rms_values],
        dtype=torch.float64,
    )
    expected_clamp = torch.tensor(
        [1.0 / max(value, threshold) for value in stored_rms_values],
        dtype=torch.float64,
    )

    heavy_inverse = _heavy_factor_inverse(stored_rms, eps)
    standard_inverse = _factor_inverse(stored_rms, eps)

    torch.testing.assert_close(heavy_inverse, expected_pseudoinverse, rtol=0, atol=1e-12)
    torch.testing.assert_close(standard_inverse, expected_clamp, rtol=0, atol=1e-12)
    assert heavy_inverse[2].item() == 0.25
    assert torch.equal(heavy_inverse.ne(standard_inverse), stored_rms <= threshold)


def test_heavy_kl_qr_basis_uses_source_order_and_preserves_subspace():
    generator = torch.Generator().manual_seed(0)
    raw_gram = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64)
    gram = raw_gram @ raw_gram.mT + 0.5 * torch.eye(5, dtype=torch.float64).unsqueeze(0)
    raw_basis = torch.randn(1, 5, 3, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(raw_basis).Q

    work = gram @ basis
    rayleigh_quotients = torch.einsum("nij,nij->nj", basis, work)
    expected_order = torch.argsort(rayleigh_quotients, dim=-1, descending=True)
    ordered_work = work.gather(-1, expected_order.unsqueeze(-2).expand_as(work))

    sorted_basis, order = _heavy_kl_qr_basis(gram, basis)
    standard_basis = _kl_qr_basis(gram, basis)

    assert torch.equal(order, expected_order)
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    torch.testing.assert_close(sorted_basis.mT @ sorted_basis, identity, rtol=0, atol=1e-12)
    torch.testing.assert_close(standard_basis.mT @ standard_basis, identity, rtol=0, atol=1e-12)
    expected_sorted = torch.linalg.qr(ordered_work).Q
    expected_standard = torch.linalg.qr(work).Q
    torch.testing.assert_close(
        sorted_basis @ sorted_basis.mT,
        expected_sorted @ expected_sorted.mT,
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        standard_basis @ standard_basis.mT,
        expected_standard @ expected_standard.mT,
        rtol=0,
        atol=1e-12,
    )


def _facade_trajectory(optimizer_class, initial, gradients, *, soap):
    torch.manual_seed(818)
    parameter = torch.nn.Parameter(initial.clone())
    parameter.grad = torch.zeros_like(parameter)
    hyperparameters = dict(
        lr=1e-2,
        beta1=0.5,
        beta2=0.5,
        eps=1e-4,
        weight_decay=0.0,
        preconditioner_update_probability=1.0,
        max_precond_dim=8,
        init_factor=1e-12,
    )
    if soap:
        hyperparameters["shampoo_beta"] = 0.5
    optimizer = optimizer_class([parameter], **hyperparameters)

    trajectory = []
    for gradient in gradients:
        parameter.grad.copy_(gradient)
        optimizer.step()
        trajectory.append(parameter.detach().clone())
    return trajectory


@pytest.mark.parametrize(
    ("heavy_optimizer", "standard_optimizer", "soap"),
    (
        (heavyball.HeavyKLSOAP, heavyball.KLSOAP, True),
        (heavyball.HeavyKLShampoo, heavyball.KLShampoo, False),
    ),
    ids=("soap", "shampoo"),
)
def test_heavy_kl_facade_diverges_on_rank_deficient_grams(heavy_optimizer, standard_optimizer, soap):
    initial = torch.linspace(-0.2, 0.2, 16, dtype=torch.float64).reshape(4, 4)
    left = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    right = torch.tensor([2.0, -1.0, 0.25, -0.5], dtype=torch.float64)
    rank_one_gradient = torch.outer(left, right)
    gradients = [scale * rank_one_gradient for scale in (1.0, -0.5, 2.0, 0.25, -1.5, 0.75)]
    assert torch.linalg.matrix_rank(rank_one_gradient) == 1

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        heavy_trajectory = _facade_trajectory(heavy_optimizer, initial, gradients, soap=soap)
        standard_trajectory = _facade_trajectory(standard_optimizer, initial, gradients, soap=soap)

    assert torch.isfinite(torch.stack(heavy_trajectory)).all()
    assert torch.isfinite(torch.stack(standard_trajectory)).all()
    max_abs_parameter_difference = (heavy_trajectory[-1] - standard_trajectory[-1]).abs().max()
    assert max_abs_parameter_difference > 1e-6
