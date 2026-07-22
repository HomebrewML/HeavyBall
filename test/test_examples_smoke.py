import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

import heavyball

EXAMPLE_MODULES = (
    "autoencoder",
    "branched_optimizer",
    "ddp_training",
    "ecc_bf16",
    "modify_functions",
)
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def eager_heavyball_steps():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _example(name):
    module_name = f"_heavyball_example_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _tiny_linear(*, dtype=torch.float32, bias=True):
    model = nn.Linear(4, 2, bias=bias, dtype=dtype)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model


def _run_two_steps(model, optimizer):
    params = list(model.parameters())
    initial = [param.detach().clone() for param in params]
    dtype = params[0].dtype
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype)
    targets = torch.ones(3, 2)

    for _ in range(2):

        def closure():
            loss = (model(inputs).float() - targets).square().mean()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        optimizer.zero_grad()
        assert torch.isfinite(loss)

    assert all(not torch.equal(before, param) for before, param in zip(initial, params, strict=True))
    assert all(torch.isfinite(param).all() for param in params)


@pytest.mark.parametrize("name", EXAMPLE_MODULES)
def test_example_imports_without_optional_training_dependencies(name):
    assert _example(name).__name__ == f"_heavyball_example_{name}"


def test_autoencoder_constructs_and_steps_psgdpro():
    example = _example("autoencoder")
    model = _tiny_linear(bias=False)
    optimizer = example.build_optimizer(model.parameters())

    assert isinstance(optimizer, heavyball.PSGDPro)
    _run_two_steps(model, optimizer)


def test_branched_optimizer_constructs_and_steps_both_routes():
    example = _example("branched_optimizer")
    model = _tiny_linear()
    optimizer = example.build_optimizer(model.parameters())

    recipes = [group.recipe for group in optimizer._engine.groups]
    assert any(recipe is example.BRANCHED_RECIPE.then for recipe in recipes)
    assert any(recipe is example.BRANCHED_RECIPE.otherwise for recipe in recipes)
    _run_two_steps(model, optimizer)


def test_ddp_example_constructs_facade_on_cpu():
    example = _example("ddp_training")
    model = _tiny_linear()
    optimizer = example.build_optimizer(model.parameters(), name="AdamW", lr=1e-3)

    assert isinstance(optimizer, heavyball.AdamW)


@pytest.mark.parametrize(
    "config_name",
    ("naive_fp32", "naive_bf16", "heavyball_fp32", "heavyball_bf16", "heavyball_bf16_state", "heavyball_ecc8_state"),
)
def test_precision_example_constructs_and_steps_each_configuration(config_name):
    example = _example("ecc_bf16")
    model = _tiny_linear(dtype=example.CONFIGS[config_name]["dtype"])
    optimizer = example.build_optimizer(config_name, model.parameters())

    _run_two_steps(model, optimizer)


def test_modify_functions_constructs_and_steps_composed_recipe():
    example = _example("modify_functions")
    model = _tiny_linear()
    optimizer = example.build_optimizer(model.parameters())

    recipes = [group.recipe for group in optimizer._engine.groups]
    assert any(recipe is example.ORTHOGONAL_SOAP for recipe in recipes)
    assert any(recipe is example.MODIFIED_SOAP.otherwise for recipe in recipes)
    _run_two_steps(model, optimizer)
