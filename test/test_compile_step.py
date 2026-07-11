import pytest
import torch

import heavyball
from heavyball.chainable import WarmupGuard, _walk_fns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTRA_KWARGS = {
    "PSGDKron": {"preconditioner_update_probability": 0.0},
    "QSGD": {"preconditioner_update_probability": 0.0},
    "PSGDLRA": {"preconditioner_update_probability": 0.0},
}

# Iterative inner ops (Newton-Schulz, eigendecomp) are inherently sensitive to FP op order;
# compile may fuse/reorder them differently than eager.
_LOOSE_COMPILE_TOL = {
    "Scion",
    "MuonAdamW",
    "SOAP",
    "KLShampoo",
}

_COMPILE_OPTS = (
    "AdamW",
    "NAdam",
    "ADOPT",
    "Scion",
    "MuonAdamW",
    "SFAdamW",
    "MSAMLaProp",
    "SOAP",
    "KLShampoo",
    "PSGDKron",
    "QSGD",
    "PSGDLRA",
    "AdEMAMix",
)

_COMPILE_CASES = tuple(pytest.param(name, {}, id=name) for name in _COMPILE_OPTS) + (
    pytest.param("AdamW", {"mars": True}, id="AdamW-mars"),
)


def _make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4),
    ).to(DEVICE)


def _run_steps(model, optimizer, n=2, seed=0xDEADBEEF):
    torch.manual_seed(seed)
    for _ in range(n):

        def closure():
            optimizer.zero_grad(set_to_none=True)
            data = torch.randn(4, 8, device=DEVICE)
            target = torch.randn(4, 4, device=DEVICE)
            loss = torch.nn.functional.mse_loss(model(data), target)
            loss.backward()
            return loss

        optimizer.step(closure)


def _assert_close(actual, expected, *, rtol, atol):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_close(a, b, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def _assert_muon_delta(actual, expected):
    actual_norm, expected_norm = actual.norm(), expected.norm()
    torch.testing.assert_close(actual_norm, expected_norm, rtol=0.08, atol=2e-5)
    if expected_norm > 0:
        assert torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0) >= 0.995


@pytest.mark.parametrize("opt_name,case_kwargs", _COMPILE_CASES)
def test_compile_step_matches_eager(opt_name, case_kwargs):
    opt_cls = getattr(heavyball, opt_name)
    kwargs = {**EXTRA_KWARGS.get(opt_name, {}), **case_kwargs}

    torch.manual_seed(0xDEADBEEF)
    model_ref = _make_model()
    model_test = _make_model()
    model_test.load_state_dict(model_ref.state_dict())
    initial = [p.detach().clone() for p in model_ref.parameters()]

    opt_ref = opt_cls(model_ref.parameters(), compile_step=False, **kwargs)
    opt_test = opt_cls(model_test.parameters(), compile_step=True, **kwargs)

    steps = 3 if opt_name in {"ADOPT", "SOAP", "KLShampoo"} else 2
    _run_steps(model_ref, opt_ref, steps)
    _run_steps(model_test, opt_test, steps)

    rtol, atol = (2e-2, 2e-5) if opt_name in _LOOSE_COMPILE_TOL else (2e-4, 2e-7)
    for p_ref, p_test, p0 in zip(model_ref.parameters(), model_test.parameters(), initial, strict=True):
        actual, expected = p_test - p0, p_ref - p0
        if opt_name == "MuonAdamW" and p_ref.ndim >= 2:
            _assert_muon_delta(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    if opt_name not in _LOOSE_COMPILE_TOL:
        _assert_close(opt_test.state_dict()["state"], opt_ref.state_dict()["state"], rtol=rtol, atol=atol)


def _max_warmup(opt):
    return max((len(ft.warmup_fns) for ft in _walk_fns(opt.fns) if isinstance(ft, WarmupGuard)), default=0)


@pytest.mark.parametrize("opt_name", _COMPILE_OPTS)
def test_needs_init_clears(opt_name):
    opt_cls = getattr(heavyball, opt_name)
    kwargs = dict(EXTRA_KWARGS.get(opt_name, {}))
    model = _make_model()
    opt = opt_cls(model.parameters(), **kwargs)
    n = _max_warmup(opt) + 1

    _run_steps(model, opt, n=n)

    for group in opt.param_groups:
        state = [opt.state_(p) for p in group["params"]]
        assert not opt._needs_init(state), (
            f"{opt_name}: _needs_init stuck True after {n} steps | compile_step will never engage"
        )
