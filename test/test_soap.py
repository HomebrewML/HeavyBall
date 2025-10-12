import pytest
import torch
import torch._dynamo

import heavyball.chainable as C
from heavyball import utils
from heavyball.chainable import SkipUpdate


def _assign_transform(fn):
    (transform,) = C.set_indices((fn,), retain=False)
    return transform


def _state_value(state, fn_name: str, label: str):
    prefix = f"{fn_name}_{label}_"
    for key, value in state.items():
        if key.startswith(prefix):
            return value
    raise KeyError(prefix)


def _run_initial_call(transform, state_fn, group, tensors, params):
    grads = [t.clone() for t in tensors]
    updates = [t.clone() for t in tensors]
    with pytest.raises(SkipUpdate):
        transform(state_fn, group, updates, grads, params)


def _make_state_fn():
    store = {}

    def state_fn(param):
        key = id(param)
        if key not in store:
            store[key] = {}
        return store[key]

    return state_fn


def _project(update, Q):
    return utils.project(utils.promote(update), Q, False)


def _project_back(tensor, Q):
    return utils.project(tensor, Q, True)


@torch._dynamo.disable
def _ademamix_reference(
    exp_avg_fast, exp_avg_slow, exp_avg_sq, grad, betas, step, eps, alpha, beta3_warmup, alpha_warmup
):
    return utils.ademamix_(
        exp_avg_fast, exp_avg_slow, exp_avg_sq, grad, betas, step, eps, alpha, beta3_warmup, alpha_warmup
    )


def test_scale_by_soap_matches_adam():
    torch.manual_seed(0)
    transform = _assign_transform(C.scale_by_soap)
    state_fn = _make_state_fn()
    params = [torch.randn(2, 2, dtype=torch.double)]
    group = {
        "max_precond_dim": 8,
        "precondition_1d": False,
        "step": 1,
        "eps": 1e-8,
        "shampoo_beta": 0.95,
        "is_preconditioning": True,
        "betas": (0.9, 0.999),
    }

    grad0 = torch.randn_like(params[0])
    _run_initial_call(transform, state_fn, group, [grad0], params)

    param_state = state_fn(params[0])
    exp_avg_before = _state_value(param_state, "scale_by_soap", "exp_avg").clone()
    exp_avg_sq_before = _state_value(param_state, "scale_by_soap", "exp_avg_sq").clone()
    Q_blocks = [_state_value(param_state, "scale_by_soap", "Q")]
    GG_before = [g.clone() for g in _state_value(param_state, "scale_by_soap", "GG")]

    group = {**group, "step": 2}
    grad1 = torch.randn_like(params[0])
    grads = [grad1]
    updates = [grad1.clone()]

    projected = [_project(u, q) for u, q in zip(updates, Q_blocks)]
    expected = utils.adam_(
        exp_avg_before,
        exp_avg_sq_before,
        projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    expected = [_project_back(p, q) for p, q in zip(expected, Q_blocks)]

    result = transform(state_fn, group, updates, grads, params)
    torch.testing.assert_close(result[0], expected[0])

    GG_after = _state_value(param_state, "scale_by_soap", "GG")
    assert any(not torch.allclose(a, b) for a, b in zip(GG_after, GG_before))


def test_scale_by_soap_laprop_matches_laprop():
    torch.manual_seed(1)
    transform = _assign_transform(C.scale_by_soap_laprop)
    state_fn = _make_state_fn()
    params = [torch.randn(2, 2, dtype=torch.double)]
    group = {
        "max_precond_dim": 8,
        "precondition_1d": False,
        "step": 1,
        "eps": 1e-8,
        "shampoo_beta": 0.95,
        "is_preconditioning": True,
        "betas": (0.9, 0.999),
    }

    grad0 = torch.randn_like(params[0])
    _run_initial_call(transform, state_fn, group, [grad0], params)

    param_state = state_fn(params[0])
    exp_avg_before = _state_value(param_state, "scale_by_soap_laprop", "exp_avg").clone()
    exp_avg_sq_before = _state_value(param_state, "scale_by_soap_laprop", "exp_avg_sq").clone()
    Q_blocks = [_state_value(param_state, "scale_by_soap_laprop", "Q")]

    group = {**group, "step": 2}
    grad1 = torch.randn_like(params[0])
    grads = [grad1]
    updates = [grad1.clone()]

    projected = [_project(u, q) for u, q in zip(updates, Q_blocks)]
    expected = utils.laprop_(
        exp_avg_before,
        exp_avg_sq_before,
        projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
    )
    expected = [_project_back(p, q) for p, q in zip(expected, Q_blocks)]

    result = transform(state_fn, group, updates, grads, params)
    torch.testing.assert_close(result[0], expected[0])


def test_scale_by_soap_ademamix_matches_reference():
    torch.manual_seed(2)
    transform = _assign_transform(C.scale_by_soap_ademamix)
    state_fn = _make_state_fn()
    params = [torch.randn(2, 2, dtype=torch.double)]
    group = {
        "max_precond_dim": 8,
        "precondition_1d": False,
        "step": 1,
        "eps": 1e-8,
        "shampoo_beta": 0.95,
        "is_preconditioning": True,
        "betas": (0.9, 0.999, 0.9999),
        "alpha": 2.0,
        "beta3_warmup": None,
        "alpha_warmup": None,
    }

    grad0 = torch.randn_like(params[0])
    _run_initial_call(transform, state_fn, group, [grad0], params)

    param_state = state_fn(params[0])
    exp_avg_fast_before = _state_value(param_state, "scale_by_soap_ademamix", "exp_avg_fast").clone()
    exp_avg_slow_before = _state_value(param_state, "scale_by_soap_ademamix", "exp_avg_slow").clone()
    exp_avg_sq_before = _state_value(param_state, "scale_by_soap_ademamix", "exp_avg_sq").clone()
    Q_blocks = [_state_value(param_state, "scale_by_soap_ademamix", "Q")]

    group = {**group, "step": 2}
    grad1 = torch.randn_like(params[0])
    grads = [grad1]
    updates = [grad1.clone()]

    projected = [_project(u, q) for u, q in zip(updates, Q_blocks)]
    expected = _ademamix_reference(
        exp_avg_fast_before,
        exp_avg_slow_before,
        exp_avg_sq_before,
        projected,
        group["betas"],
        group["step"] - 1,
        group["eps"],
        group["alpha"],
        group.get("beta3_warmup"),
        group.get("alpha_warmup"),
    )
    expected = [_project_back(p, q) for p, q in zip(expected, Q_blocks)]

    result = transform(state_fn, group, updates, grads, params)
    torch.testing.assert_close(result[0], expected[0])
