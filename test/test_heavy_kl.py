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
    eigenvalue_values = (1e-12, eps_value, 4.0, 9.0)
    eigenvalues = torch.tensor(eigenvalue_values, dtype=torch.float64)
    eps = torch.tensor(eps_value, dtype=torch.float64)

    expected_pseudoinverse = torch.tensor(
        [0.0 if value <= eps_value else 1.0 / math.sqrt(value) for value in eigenvalue_values],
        dtype=torch.float64,
    )
    expected_clamp = torch.tensor(
        [1.0 / math.sqrt(max(value, eps_value)) for value in eigenvalue_values],
        dtype=torch.float64,
    )

    heavy_inverse = _heavy_factor_inverse(eigenvalues, eps)
    standard_inverse = _factor_inverse(eigenvalues, eps)

    torch.testing.assert_close(heavy_inverse, expected_pseudoinverse, rtol=0, atol=1e-12)
    torch.testing.assert_close(standard_inverse, expected_clamp, rtol=0, atol=1e-12)
    assert torch.equal(heavy_inverse.ne(standard_inverse), eigenvalues <= eps)


def test_heavy_kl_qr_basis_sorts_descending_rayleigh_quotients():
    generator = torch.Generator().manual_seed(0)
    raw_gram = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64)
    gram = raw_gram @ raw_gram.mT + 0.5 * torch.eye(5, dtype=torch.float64).unsqueeze(0)
    raw_basis = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(raw_basis).Q

    work = gram @ basis
    rayleigh_quotients = torch.einsum("nij,nij->nj", basis, work)
    expected_order = torch.argsort(rayleigh_quotients, dim=-1, descending=True)
    expected_sorted_basis = torch.linalg.qr(work.index_select(-1, expected_order[0])).Q
    expected_unsorted_basis = torch.linalg.qr(work).Q

    sorted_basis, order = _heavy_kl_qr_basis(gram, basis)
    standard_basis = _kl_qr_basis(gram, basis)

    assert torch.equal(order, expected_order)
    torch.testing.assert_close(sorted_basis, expected_sorted_basis, rtol=0, atol=1e-12)
    sorted_rayleigh_quotients = torch.einsum("nij,nij->nj", sorted_basis, gram @ sorted_basis)
    assert torch.all(sorted_rayleigh_quotients[:, :-1] >= sorted_rayleigh_quotients[:, 1:])
    torch.testing.assert_close(standard_basis, expected_unsorted_basis, rtol=0, atol=1e-12)
    assert not torch.allclose(sorted_basis, standard_basis, rtol=0, atol=1e-12)


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
