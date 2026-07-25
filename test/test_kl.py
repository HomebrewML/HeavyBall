"""Proofs for the slab-native KL-SOAP and KL-Shampoo ports."""

from unittest.mock import patch

import pytest
import torch

from heavyball.core import Engine
from heavyball.kl import kl_shampoo_recipe, kl_soap_recipe
from heavyball.matrix import _project


def test_kl_soap_matches_authors_raw_ema_and_seed_step():
    """KL-SOAP follows the authors' code (yorkerlin/KL-Methods, Lin et al. arXiv:2509.03378), not legacy:
    step 1 is SEED-ONLY (no parameter update) and every EMA is RAW beta, not debiased. For a seeded state
    and an identity gradient the eigenvalue is the RAW EMA 0.8*0.1 + 0.2*5 = 1.08 (the audit measured 2.82
    under the old beta_\u0064ebias) and the first Adam moments are the RAW (1-beta)*1 = 0.1 / 0.2 (were 1.0/1.0
    under debias)."""
    g = torch.eye(2, dtype=torch.float64)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        seed = torch.nn.Parameter(torch.zeros_like(g))
        seed_opt = Engine([seed], kl_soap_recipe, lr=0.1, beta1=0.9, beta2=0.8, eps=1e-8,
                          max_precond_dim=8, weight_decay=0.0, shampoo_beta=0.8)
        seed.grad.copy_(g)
        seed_opt.step(step_type="normal")
        assert torch.equal(seed.detach(), torch.zeros_like(g))  # seed-only step 1 skips the parameter

        param = torch.nn.Parameter(torch.zeros_like(g))
        optimizer = Engine([param], kl_soap_recipe, lr=0.1, beta1=0.9, beta2=0.8, eps=1e-8,
                           max_precond_dim=8, weight_decay=0.0, shampoo_beta=0.8)
        state = optimizer.groups[0].states[0]
        for key in ("Q_l", "Q_r"):
            state[key].copy_(torch.eye(2, dtype=torch.float64))
        for key in ("GG_l", "GG_r"):
            state[key].copy_(0.1 * torch.eye(2, dtype=torch.float64))
        state["eigenvalues_l"].fill_(0.1**0.5)
        state["eigenvalues_r"].fill_(0.1**0.5)
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
        optimizer.groups[0].age.fill_(2)  # authors' first real update (their step 2 == age 2 after seed)
        param.grad.copy_(g)
        optimizer.step(step_type="normal")
    raw_eigenvalue = 0.8 * 0.1 + (1 - 0.8) * 5.0  # opposite-factor-whitened estimate is 5; RAW EMA
    torch.testing.assert_close(
        state["eigenvalues_l"][0, 0].square(),
        torch.tensor(raw_eigenvalue, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(state["exp_avg"][0, 0, 0], torch.tensor((1 - 0.9) * 1.0, dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        state["exp_avg_sq"][0, 0, 0].square(),
        torch.tensor((1 - 0.8) * 1.0, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_kl_shampoo_seed_step_skips_param():
    """KL-Shampoo's step 1 is also seed-only: the parameter must not move on the first observed step."""
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        param = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
        before = param.detach().clone()
        optimizer = Engine([param], kl_shampoo_recipe, lr=0.1, beta1=0.9, beta2=0.8, eps=1e-8,
                           max_precond_dim=8, weight_decay=0.0)
        param.grad.copy_(torch.randn(4, 3, dtype=torch.float64))
        optimizer.step(step_type="normal")
    assert torch.equal(param.detach(), before)


def test_kl_soap_transports_second_moment():
    """A basis refresh rebases the prior second moment (Hadamard-square) before it blends new squares,
    independent of legacy (which is pre-fix and leaves it in the stale basis)."""

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(3)
        param = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
        hyper = dict(lr=0.1, beta1=0.9, beta2=0.8, eps=1e-8, max_precond_dim=8, weight_decay=0.0, shampoo_beta=0.7)
        optimizer = Engine([param], kl_soap_recipe, **hyper)
        state = optimizer.groups[0].states[0]
        gradients = [torch.randn(4, 3, dtype=torch.float64) for _ in range(5)]
        prior = old_left = old_right = None
        for index, gradient in enumerate(gradients):
            refresh = index in (1, 4)  # the final step refreshes from an already non-trivial basis
            if index == 4:
                prior = state["exp_avg_sq"].clone()
                old_left, old_right = state["Q_l"].clone(), state["Q_r"].clone()
            param.grad.copy_(gradient)
            optimizer.step(step_type="refresh" if refresh else "normal")

    new_left, new_right = state["Q_l"], state["Q_r"]
    assert not torch.allclose(old_left, new_left)
    # Hadamard-square transport, derived inline: each variance rotates by the squared basis change.
    left_change = (old_left.mT @ new_left).square()
    right_change = (old_right.mT @ new_right).square()
    rebased_prior = torch.einsum(
        "nab,nac,nbd->ncd", prior.square(), left_change, right_change
    )
    projected = _project(gradients[-1].reshape(1, 4, 3), new_left, new_right, back=False)
    beta2 = hyper["beta2"]  # KL uses a RAW second-moment EMA (Lin et al.), not a debiased beta
    expected = beta2 * rebased_prior + (1 - beta2) * projected.square()
    torch.testing.assert_close(
        state["exp_avg_sq"].square(), expected, rtol=0, atol=1e-12
    )


def test_kl_init_factor_initializes_eigenvalues():
    """KL recipes retain legacy's configurable, positive eigenvalue initializer."""

    for recipe in (kl_soap_recipe, kl_shampoo_recipe):
        parameter = torch.nn.Parameter(torch.zeros(3, 4))
        state = Engine([parameter], recipe, init_factor=0.25).groups[0].states[0]
        torch.testing.assert_close(
            state["eigenvalues_l"].square(),
            torch.full_like(state["eigenvalues_l"], 0.25),
        )
        torch.testing.assert_close(
            state["eigenvalues_r"].square(),
            torch.full_like(state["eigenvalues_r"], 0.25),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        Engine([torch.nn.Parameter(torch.zeros(3, 4))], kl_soap_recipe, init_factor=0.0)


@pytest.mark.parametrize(
    "recipe",
    (kl_soap_recipe, kl_shampoo_recipe),
    ids=("kl_soap", "kl_shampoo"),
)
def test_kl_normal_and_refresh_are_stable_fullgraphs(recipe):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
        optimizer = Engine(params, recipe)
        state = optimizer.groups[0].states[0]
        initial_left = state["Q_l"].clone()
        optimizer.groups[0].grad_slab.copy_(
            torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
        )

        optimizer.step(step_type="normal")
        optimizer.step(step_type="normal")
        torch.testing.assert_close(state["Q_l"], initial_left, rtol=0, atol=0)
        normal_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        assert normal_graphs == 1

        optimizer.step(step_type="refresh")
        assert not torch.equal(state["Q_l"], initial_left)
        refresh_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        optimizer.step(step_type="refresh")
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == refresh_graphs == 2
        assert all(torch.isfinite(param).all() for param in params)
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()
