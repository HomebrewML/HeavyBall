import inspect

import pytest
import torch

import heavyball
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
