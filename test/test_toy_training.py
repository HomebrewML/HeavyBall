import inspect
from collections.abc import Iterable

import pytest
import torch
import torch.nn.functional as F
from torch import optim

import heavyball

pytestmark = [
    pytest.mark.filterwarnings("ignore:CUDA initialization:"),
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
]


def _clone_tensors(tensors: Iterable[torch.Tensor]):
    return [tensor.detach().clone() for tensor in tensors]


def _flatten_tensors(tensors: Iterable[torch.Tensor]):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


EXTRA_KWARGS = {
    "ForeachAdamC": {"max_lr": 0.0025},
}


def _optimizer_params():
    params = []
    for name in sorted(dir(heavyball)):
        if name.startswith("_"):
            continue
        attr = getattr(heavyball, name)
        if not isinstance(attr, type) or not issubclass(attr, optim.Optimizer):
            continue
        if attr is optim.Optimizer:
            continue
        params.append(pytest.param(name, id=name))
    return params


@pytest.fixture(scope="module", params=_optimizer_params())
def toy_training_results(request):
    optimizer_name: str = request.param
    optimizer_cls = getattr(heavyball, optimizer_name)

    torch.manual_seed(0x172893)
    model = torch.nn.Linear(2, 1)
    param_list = list(model.parameters())
    data = torch.tensor([[0.5, -1.0], [1.5, 0.3], [-0.7, 0.9], [0.2, -0.4]], dtype=torch.float32)
    target = torch.tensor([[0.1], [1.3], [-0.6], [0.2]], dtype=torch.float32)

    init_params = _clone_tensors(param_list)
    previous_params = init_params
    params_history: list[list[torch.Tensor]] = []
    updates: list[list[torch.Tensor]] = []
    gradients: list[list[torch.Tensor]] = []

    sig = inspect.signature(optimizer_cls.__init__)
    kwargs = dict(EXTRA_KWARGS.get(optimizer_name, {}))
    if "foreach" in sig.parameters:
        kwargs["foreach"] = True

    if optimizer_name == "SAMWrapper":
        inner_kwargs = {}
        inner_sig = inspect.signature(heavyball.ForeachAdamW.__init__)
        if "foreach" in inner_sig.parameters:
            inner_kwargs["foreach"] = True
        inner_optimizer = heavyball.ForeachAdamW(param_list, **inner_kwargs)
        optimizer = optimizer_cls(param_list, wrapped_optimizer=inner_optimizer, **kwargs)
    else:
        optimizer = optimizer_cls(param_list, **kwargs)

    for _ in range(8):
        gradient_bucket: list[torch.Tensor] = []

        def closure():
            optimizer.zero_grad()
            prediction = model(data)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            if not gradient_bucket:
                gradient_bucket.extend(_clone_tensors([param.grad for param in param_list]))
            return loss

        optimizer.step(closure)

        if not gradient_bucket:
            gradient_bucket.extend(_clone_tensors([param.grad for param in param_list]))

        gradients.append(gradient_bucket)

        current_params = _clone_tensors(param_list)
        updates.append([curr - prev for curr, prev in zip(current_params, previous_params)])
        params_history.append(current_params)
        previous_params = current_params

    return {
        "name": optimizer_name,
        "initial": init_params,
        "updates": updates,
        "gradients": gradients,
        "params": params_history,
    }


def test_toy_updates_change(toy_training_results):
    updates = [_flatten_tensors(step) for step in toy_training_results["updates"]]
    first_update = updates[0]
    assert any(not torch.allclose(first_update, update, rtol=1e-6, atol=1e-8) for update in updates[1:]), (
        f"Updates did not change for {toy_training_results['name']}"
    )


def test_toy_gradients_change(toy_training_results):
    gradients = [_flatten_tensors(step) for step in toy_training_results["gradients"]]
    first_grad = gradients[0]
    assert any(not torch.allclose(first_grad, grad, rtol=1e-6, atol=1e-8) for grad in gradients[1:]), (
        f"Gradients did not change for {toy_training_results['name']}"
    )


def test_toy_parameters_change(toy_training_results):
    initial = _flatten_tensors(toy_training_results["initial"])
    final = _flatten_tensors(toy_training_results["params"][-1])
    assert not torch.allclose(initial, final, rtol=1e-6, atol=1e-8), (
        f"Parameters did not change for {toy_training_results['name']}"
    )
