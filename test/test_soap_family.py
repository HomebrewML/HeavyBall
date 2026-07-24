from unittest.mock import patch

import pytest
import reference
import torch

from heavyball.core import build
from heavyball.matrix import soap_ademamix_recipe, soap_nadam_recipe, soap_recipe, solp_recipe

_BASE_HYPER = dict(
    lr=1e-2,
    beta1=0.9,
    beta2=0.99,
    eps=1e-8,
    weight_decay=0.05,
    shampoo_beta=0.9,
    max_precond_dim=8,
)
_STANDARD_INNER = ("lr", "beta1", "beta2", "eps", "weight_decay")
_VARIANTS = (
    pytest.param(soap_recipe, reference.adam, ("exp_avg",), {}, _STANDARD_INNER, id="soap"),
    pytest.param(solp_recipe, reference.laprop, ("exp_avg",), {}, _STANDARD_INNER, id="solp"),
    pytest.param(
        soap_nadam_recipe,
        reference.nadam,
        ("exp_avg",),
        {"momentum_decay": 4e-3},
        (*_STANDARD_INNER, "momentum_decay"),
        id="soap_nadam",
    ),
    pytest.param(
        soap_ademamix_recipe,
        reference.ademamix,
        ("exp_avg_fast", "exp_avg_slow"),
        {"beta3": 0.99, "alpha": 2.0, "beta3_warmup": 0.0, "alpha_warmup": 0.0},
        ("lr", "beta1", "beta2", "beta3", "alpha", "eps", "weight_decay"),
        id="soap_ademamix",
    ),
)


def _eager():
    return patch("heavyball.core.torch.compile", lambda function, **kwargs: function)


@pytest.mark.parametrize("recipe,inner,transported,extra_hyper,inner_keys", _VARIANTS)
def test_identity_basis_reduces_to_inner(recipe, inner, transported, extra_hyper, inner_keys):
    torch.manual_seed(0)
    init = torch.randn(3, 4, dtype=torch.float64)
    grads = [torch.randn(3, 4, dtype=torch.float64) for _ in range(8)]
    hyper = {**_BASE_HYPER, **extra_hyper}

    with _eager():
        param = torch.nn.Parameter(init.clone())
        optimizer = build([param], recipe, **hyper)
        for gradient in grads:
            param.grad.copy_(gradient)
            optimizer.step(step_type="normal")  # Without refresh, the eigenbasis stays identity.

    inner_hyper = {key: hyper[key] for key in inner_keys}
    torch.testing.assert_close(param.detach(), inner(init, grads, **inner_hyper), rtol=0, atol=1e-12)


def _physical(moment, left, right):
    return torch.einsum("ab,ia,jb->ij", moment, left, right)


@pytest.mark.parametrize("recipe,inner,transported,extra_hyper,inner_keys", _VARIANTS)
def test_refresh_preserves_physical_moments(recipe, inner, transported, extra_hyper, inner_keys):
    torch.manual_seed(1)
    init = torch.randn(3, 4, dtype=torch.float64)
    grads = [torch.randn(3, 4, dtype=torch.float64) for _ in range(5)]
    hyper = {**_BASE_HYPER, **extra_hyper}

    def run(refresh_last):
        with _eager():
            param = torch.nn.Parameter(init.clone())
            optimizer = build([param], recipe, **hyper)
            for index, gradient in enumerate(grads):
                param.grad.copy_(gradient)
                last = index == len(grads) - 1
                optimizer.step(step_type="refresh" if last and refresh_last else "normal")
            state = optimizer.groups[0].states[0]
            return {name: state[name][0].clone() for name in (*transported, "Q_l", "Q_r")}

    refreshed = run(refresh_last=True)
    held = run(refresh_last=False)
    for moment in transported:
        torch.testing.assert_close(
            _physical(refreshed[moment], refreshed["Q_l"], refreshed["Q_r"]),
            _physical(held[moment], held["Q_l"], held["Q_r"]),
            rtol=0,
            atol=1e-10,
        )


def _hadamard_square(second_moment, left_old, left_new, right_old, right_new):
    # The diagonal variance re-expressed in the new basis: each variance rotates as the squared change.
    left = (left_old.mT @ left_new).square()
    right = (right_old.mT @ right_new).square()
    return torch.einsum("ab,ac,bd->cd", second_moment, left, right)


@pytest.mark.parametrize("recipe,inner,transported,extra_hyper,inner_keys", _VARIANTS)
def test_refresh_transports_second_moment(recipe, inner, transported, extra_hyper, inner_keys):
    torch.manual_seed(2)
    init = torch.randn(3, 4, dtype=torch.float64)
    grads = [torch.randn(3, 4, dtype=torch.float64) for _ in range(6)]
    hyper = {**_BASE_HYPER, **extra_hyper}

    def run(refresh_last):
        # Both runs refresh at the penultimate step, so the final step's old basis is already non-trivial.
        with _eager():
            param = torch.nn.Parameter(init.clone())
            optimizer = build([param], recipe, **hyper)
            for index, gradient in enumerate(grads):
                param.grad.copy_(gradient)
                penultimate = index == len(grads) - 2
                last = index == len(grads) - 1
                refresh = penultimate or (last and refresh_last)
                optimizer.step(step_type="refresh" if refresh else "normal")
            state = optimizer.groups[0].states[0]
            return {name: state[name][0].clone() for name in ("exp_avg_sq", "Q_l", "Q_r")}

    refreshed = run(refresh_last=True)
    held = run(refresh_last=False)  # identical inner second moment, left in the old basis
    assert not torch.allclose(refreshed["Q_l"], held["Q_l"])  # the final refresh actually rotated the basis
    expected = _hadamard_square(held["exp_avg_sq"], held["Q_l"], refreshed["Q_l"], held["Q_r"], refreshed["Q_r"])
    torch.testing.assert_close(refreshed["exp_avg_sq"], expected, rtol=0, atol=1e-10)
