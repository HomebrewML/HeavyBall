"""Behavioral coverage for assigning parameter families through SplitOpt."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

import heavyball


@pytest.fixture(autouse=True)
def _eager_compile():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _model():
    return nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))


def _parameter_families(model):
    matrices = [parameter for parameter in model.parameters() if parameter.ndim == 2]
    vectors = [parameter for parameter in model.parameters() if parameter.ndim == 1]
    return matrices, vectors


def _split_optimizer(model):
    matrices, vectors = _parameter_families(model)
    optimizer = heavyball.SplitOpt([
        {"params": matrices, "optimizer": heavyball.Muon, "lr": 0.02},
        {"params": vectors, "optimizer": heavyball.AdamW, "lr": 1e-3},
    ])
    return optimizer, matrices, vectors


def _train(model, optimizer, inputs, targets, steps):
    for _ in range(steps):
        ((model(inputs) - targets) ** 2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)


def _assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_nested_equal(left, right)
        return
    assert actual == expected


def test_delegates_each_parameter_family_to_its_assigned_optimizer():
    torch.manual_seed(0)
    split_model = _model()
    standalone_model = copy.deepcopy(split_model)

    torch.manual_seed(1)
    split, split_matrices, split_vectors = _split_optimizer(split_model)
    torch.manual_seed(1)
    standalone_matrices, standalone_vectors = _parameter_families(standalone_model)
    standalone_muon = heavyball.Muon(standalone_matrices, lr=0.02)
    standalone_adamw = heavyball.AdamW(standalone_vectors, lr=1e-3)

    for step in range(6):
        for index, (split_parameter, standalone_parameter) in enumerate(
            zip(split_model.parameters(), standalone_model.parameters(), strict=True)
        ):
            gradient = torch.arange(
                split_parameter.numel(),
                dtype=split_parameter.dtype,
                device=split_parameter.device,
            ).reshape_as(split_parameter)
            gradient = gradient.mul(0.013 * (index + 1)).add(-0.2 + 0.037 * step)
            if (index + step) % 2:
                gradient = gradient.neg()
            split_parameter.grad.copy_(gradient)
            standalone_parameter.grad.copy_(gradient)

        split.step()
        standalone_muon.step()
        standalone_adamw.step()

        split.zero_grad(set_to_none=False)
        standalone_muon.zero_grad(set_to_none=False)
        standalone_adamw.zero_grad(set_to_none=False)
        for parameter in split_model.parameters():
            torch.testing.assert_close(
                parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0
            )

    for split_parameter, standalone_parameter in zip(
        split_matrices, standalone_matrices, strict=True
    ):
        torch.testing.assert_close(split_parameter, standalone_parameter, rtol=0, atol=0)
    for split_parameter, standalone_parameter in zip(
        split_vectors, standalone_vectors, strict=True
    ):
        torch.testing.assert_close(split_parameter, standalone_parameter, rtol=0, atol=0)


def test_checkpoint_resume_is_bit_identical():
    torch.manual_seed(2)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    uninterrupted = _model()
    uninterrupted_optimizer, _, _ = _split_optimizer(uninterrupted)

    _train(uninterrupted, uninterrupted_optimizer, inputs, targets, 4)
    model_checkpoint = {
        name: value.clone() for name, value in uninterrupted.state_dict().items()
    }
    optimizer_checkpoint = copy.deepcopy(uninterrupted_optimizer.state_dict())

    _train(uninterrupted, uninterrupted_optimizer, inputs, targets, 3)

    torch.manual_seed(987654321)
    resumed = _model()
    resumed_optimizer, _, _ = _split_optimizer(resumed)
    resumed.load_state_dict(model_checkpoint)
    resumed_optimizer.load_state_dict(optimizer_checkpoint)
    torch.manual_seed(123456789)
    _train(resumed, resumed_optimizer, inputs, targets, 3)

    for uninterrupted_parameter, resumed_parameter in zip(
        uninterrupted.parameters(), resumed.parameters(), strict=True
    ):
        torch.testing.assert_close(
            uninterrupted_parameter, resumed_parameter, rtol=0, atol=0
        )
    _assert_nested_equal(
        uninterrupted_optimizer.state_dict(),
        resumed_optimizer.state_dict(),
    )


def test_load_state_dict_rejects_optimizer_class_mismatch():
    optimizer, _, _ = _split_optimizer(_model())
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["classes"] = list(reversed(checkpoint["classes"]))

    with pytest.raises(ValueError):
        optimizer.load_state_dict(checkpoint)


def test_load_state_dict_rejects_wrong_optimizer_state_count():
    optimizer, _, _ = _split_optimizer(_model())
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["optimizers"].pop()

    with pytest.raises(ValueError):
        optimizer.load_state_dict(checkpoint)


def test_add_param_group_after_construction_raises_runtime_error():
    parameter = nn.Parameter(torch.ones(3))
    optimizer = heavyball.SplitOpt([
        {"params": [parameter], "optimizer": heavyball.AdamW, "lr": 1e-3}
    ])

    with pytest.raises(RuntimeError):
        optimizer.add_param_group({"params": [nn.Parameter(torch.zeros(3))]})


def test_empty_params_spec_is_skipped():
    parameter = nn.Parameter(torch.ones(3))
    optimizer = heavyball.SplitOpt([
        {"params": [], "optimizer": heavyball.Muon, "lr": 0.02},
        {"params": [parameter], "optimizer": heavyball.AdamW, "lr": 1e-3},
    ])

    assert [type(sub_optimizer) for sub_optimizer in optimizer.optimizers] == [heavyball.AdamW]


def test_parameter_in_two_specs_raises_value_error():
    parameter = nn.Parameter(torch.eye(2))

    with pytest.raises(ValueError):
        heavyball.SplitOpt([
            {"params": [parameter], "optimizer": heavyball.Muon, "lr": 0.02},
            {"params": [parameter], "optimizer": heavyball.AdamW, "lr": 1e-3},
        ])


def test_non_heavyball_optimizer_raises_type_error():
    parameter = nn.Parameter(torch.ones(3))

    with pytest.raises(TypeError):
        heavyball.SplitOpt([
            {"params": [parameter], "optimizer": torch.optim.SGD, "lr": 1e-3}
        ])


def test_zero_valid_specs_raises_value_error():
    with pytest.raises(ValueError):
        heavyball.SplitOpt([])
