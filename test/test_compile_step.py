import inspect

import pytest
import torch

import heavyball
from heavyball.chainable import ChainOpt, WarmupGuard, _walk_fns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTRA_KWARGS = {
    "AdamC": {"max_lr": 0.01},
}


def _optimizer_params():
    seen = set()
    params = []
    for name in heavyball.__all__:
        if not hasattr(heavyball, name):
            continue
        obj = getattr(heavyball, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, torch.optim.Optimizer):
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        if name == "SplitOpt":
            params.append(
                pytest.param(name, obj, id=name, marks=pytest.mark.skip(reason="SplitOpt requires dict specs"))
            )
            continue
        params.append(pytest.param(name, obj, id=name))
    return params


def _make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4),
    ).to(DEVICE)


def _run_steps(model, optimizer, n=5, seed=0xDEADBEEF):
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


@pytest.mark.parametrize("opt_name,opt_cls", _optimizer_params())
def test_compile_step_matches_eager(opt_name, opt_cls):
    """compile_step=True must produce identical parameters to compile_step=False."""
    sig = inspect.signature(opt_cls.__init__)
    if "compile_step" not in sig.parameters:
        pytest.skip("optimizer does not accept compile_step")

    kwargs = dict(EXTRA_KWARGS.get(opt_name, {}))

    torch.manual_seed(0xDEADBEEF)
    model_ref = _make_model()
    model_test = _make_model()
    model_test.load_state_dict(model_ref.state_dict())

    opt_ref = opt_cls(model_ref.parameters(), compile_step=False, **kwargs)
    opt_test = opt_cls(model_test.parameters(), compile_step=True, **kwargs)

    _run_steps(model_ref, opt_ref)
    _run_steps(model_test, opt_test)

    for p_ref, p_test in zip(model_ref.parameters(), model_test.parameters()):
        diff = (p_ref.data - p_test.data).abs().max().item()
        assert diff < 1e-4, f"compile_step diverged: max_diff={diff}"


def _max_warmup(opt):
    return max((len(ft.warmup_fns) for ft in _walk_fns(opt.fns) if isinstance(ft, WarmupGuard)), default=0)


@pytest.mark.parametrize("opt_name,opt_cls", _optimizer_params())
def test_needs_init_clears(opt_name, opt_cls):
    """_needs_init must become False after max_warmup + 1 steps for all ChainOpt optimizers.

    Catches bugs where Route-based or warmup_guard-based optimizers permanently
    force eager mode because different params accumulate different is_initialized
    sets that never individually cover _transform_ids.
    """
    if not issubclass(opt_cls, ChainOpt):
        pytest.skip("not a ChainOpt")

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
