import functools
import inspect

import pytest
import torch

import heavyball
from heavyball import chainable as C
from heavyball.utils import StatefulOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                pytest.param(
                    name, obj, id=name, marks=pytest.mark.skip(reason="SplitOpt requires dict specs, not raw params")
                )
            )
            continue
        if name == "SOAPNAdam":
            params.append(
                pytest.param(
                    name,
                    obj,
                    id=name,
                    marks=pytest.mark.xfail(reason="torch.compile Inductor AssertionError on CPU", strict=False),
                )
            )
            continue
        params.append(pytest.param(name, obj, id=name))
    return params


@pytest.mark.parametrize("opt_name,opt_cls", _optimizer_params())
def test_optimizer_runs_on_cpu(opt_name, opt_cls):
    torch.manual_seed(0xDEADBEEF)

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4),
    ).to(DEVICE)

    optimizer = opt_cls(model.parameters())

    initial = [param.detach().clone() for param in model.parameters()]

    def closure():
        optimizer.zero_grad(set_to_none=True)
        data = torch.randn(4, 8, device=DEVICE)
        target = torch.randn(4, 4, device=DEVICE)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        return loss

    for _ in range(5):
        optimizer.step(closure)

    updated = list(model.parameters())
    deltas = [torch.max(torch.abs(after - before)) for before, after in zip(initial, updated)]
    assert any(delta > 0 for delta in deltas)

    if isinstance(optimizer, StatefulOptimizer):
        # Ensure state dict round-trips without touching CUDA APIs.
        state_dict = optimizer.state_dict()
        clone = opt_cls(model.parameters())
        clone.load_state_dict(state_dict)
        assert clone.state_dict()["state"].keys() == state_dict["state"].keys()


def test_optimizer_keeps_constructor_compatibility_features():
    param = torch.nn.Parameter(torch.randn(4, 4, device=DEVICE))

    with pytest.warns(FutureWarning, match="renamed to 'multi_tensor'"):
        optimizer = heavyball.AdamW([param], foreach=True)
    assert optimizer.param_groups[0]["multi_tensor"] is True

    with pytest.raises(TypeError, match="Removed in HeavyBall"):
        heavyball.SOAP([param], normalize_grads=True)

    with pytest.raises(TypeError, match="unknown keyword arguments"):
        heavyball.AdamW([param], totally_fake=True)


def test_optimizer_accepts_explicit_orig_shapes():
    param = torch.nn.Parameter(torch.randn(4, 4, device=DEVICE))
    shapes = heavyball.capture_param_shapes([param])
    optimizer = heavyball.AdamW([param], orig_shapes=shapes)
    assert "orig_shapes" not in optimizer.param_groups[0]


def test_subclass_defaults_still_apply():
    class ScheduledSOAP(heavyball.SOAP):
        use_precond_schedule = True

    class DelayedPSGDKron(heavyball.PSGDKron):
        delayed = True
        exp_avg_input = False

    param = torch.nn.Parameter(torch.randn(4, 4, device=DEVICE))

    soap = ScheduledSOAP([param])
    assert "precondition_frequency" not in soap.param_groups[0]
    assert "precond_scheduler" not in soap.param_groups[0]

    psgd = DelayedPSGDKron([param])
    first_fn = psgd.fns[0]
    assert isinstance(first_fn, functools.partial)
    assert first_fn.func.get_fn() is C.scale_by_delayed_psgd.get_fn()
