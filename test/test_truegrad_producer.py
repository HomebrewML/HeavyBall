"""Regression tests for slab-backed TrueGrad observations."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import heavyball


def _eager_compile(function, **kwargs):
    del kwargs
    return function


def _per_sample_loss(weight, bias, sample):
    output = F.linear(sample, weight, bias)
    return (output.sin() + 0.1 * output.square()).sum()


def _batch_loss(output):
    return (output.sin() + 0.1 * output.square()).sum()


def _per_sample_sum_grad_squared(module, inputs, output_grad):
    named_parameters = tuple(module.named_parameters())
    totals = {
        name: torch.zeros_like(
            parameter,
            dtype=torch.float32 if parameter.dtype in (torch.float16, torch.bfloat16) else parameter.dtype,
        )
        for name, parameter in named_parameters
    }
    parameters = tuple(parameter for _, parameter in named_parameters)
    output = module(inputs)
    for sample_index, sample_output_grad in enumerate(output_grad):
        sample_grad_output = torch.zeros_like(output)
        sample_grad_output[sample_index] = sample_output_grad
        sample_grads = torch.autograd.grad(
            output,
            parameters,
            sample_grad_output,
            retain_graph=sample_index + 1 < len(output_grad),
        )
        for (name, _), sample_grad in zip(named_parameters, sample_grads, strict=True):
            totals[name].add_(sample_grad.to(totals[name].dtype).square())
    return totals


@pytest.mark.parametrize(
    "case",
    ["linear", "conv1d", "conv2d", "conv3d", "embedding", "layernorm", "groupnorm", "groupnorm_flat", "rmsnorm"],
)
def test_supported_truegrad_producers_match_per_sample_autograd(case):
    torch.manual_seed(20260725)
    if case == "linear":
        module, inputs = nn.Linear(4, 3), torch.randn(3, 2, 4)
    elif case == "conv1d":
        module, inputs = nn.Conv1d(2, 3, 3, padding=1), torch.randn(2, 2, 5)
    elif case == "conv2d":
        module = nn.Conv2d(4, 6, 3, padding=1, groups=2, padding_mode="reflect")
        inputs = torch.randn(2, 4, 4, 5)
    elif case == "conv3d":
        module, inputs = nn.Conv3d(2, 3, 3, padding=1), torch.randn(2, 2, 3, 4, 3)
    elif case == "embedding":
        module = nn.Embedding(7, 4, padding_idx=0, scale_grad_by_freq=True)
        inputs = torch.tensor([[0, 1, 1, 2], [1, 3, 0, 3], [4, 1, 4, 5]])
    elif case == "layernorm":
        module, inputs = nn.LayerNorm((2, 3)), torch.randn(3, 4, 2, 3)
    elif case == "groupnorm":
        module, inputs = nn.GroupNorm(2, 4), torch.randn(3, 4, 2, 3)
    elif case == "groupnorm_flat":
        module, inputs = nn.GroupNorm(2, 4), torch.randn(3, 4)
    else:
        if not hasattr(nn, "RMSNorm"):
            pytest.skip("RMSNorm is unavailable")
        module, inputs = nn.RMSNorm(4), torch.randn(3, 2, 4)

    reference = copy.deepcopy(module)
    with patch("heavyball.core.torch.compile", _eager_compile):
        heavyball.Engine(module.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(module)
    try:
        output = module(inputs)
        output_grad = torch.randn_like(output)
        output.backward(output_grad)
        expected = _per_sample_sum_grad_squared(reference, inputs, output_grad)

        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter.sum_grad_squared, expected[name], rtol=1e-5, atol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


def test_truegrad_step_fails_when_observation_was_not_produced():
    linear = nn.Linear(4, 3)
    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.HeavyBallOptimizer([linear.weight], heavyball.truegrad_adam)
    linear.weight.grad.normal_()

    with pytest.raises(ValueError, match="sum_grad_squared.*not produced"):
        optimizer.step()
    optimizer.produce(linear.weight, "sum_grad_squared", torch.ones_like(linear.weight))
    optimizer.step()


def test_linear_truegrad_producer_matches_vmap_reference_and_changes_update():
    torch.manual_seed(42)
    linear = nn.Linear(4, 3)
    plain_linear = nn.Linear(4, 3)
    plain_linear.load_state_dict(linear.state_dict())
    inputs = torch.randn(7, 4)

    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.HeavyBallOptimizer(
            linear.parameters(), heavyball.truegrad_adam, lr=0.05, beta1=0.9, beta2=0.99
        )
        plain_optimizer = heavyball.HeavyBallOptimizer(
            plain_linear.parameters(), heavyball.adamw, lr=0.05, beta1=0.9, beta2=0.99
        )
    handles = heavyball.register_truegrad(linear)

    initial_weight = linear.weight.detach().clone()
    initial_bias = linear.bias.detach().clone()
    optimizer.zero_grad()
    _batch_loss(linear(inputs)).backward()

    sample_grad = torch.func.grad(_per_sample_loss, argnums=(0, 1))
    weight_grads, bias_grads = torch.func.vmap(sample_grad, in_dims=(None, None, 0))(
        initial_weight, initial_bias, inputs
    )
    weight_reference = weight_grads.square().sum(dim=0)
    bias_reference = bias_grads.square().sum(dim=0)
    torch.testing.assert_close(linear.weight.sum_grad_squared, weight_reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(linear.bias.sum_grad_squared, bias_reference, rtol=1e-5, atol=1e-6)

    plain_optimizer.zero_grad()
    _batch_loss(plain_linear(inputs)).backward()
    torch.testing.assert_close(linear.weight.grad, plain_linear.weight.grad)
    optimizer.step()
    plain_optimizer.step()

    truegrad_update = initial_weight - linear.weight.detach()
    plain_update = initial_weight - plain_linear.weight.detach()
    assert not torch.allclose(truegrad_update, plain_update, rtol=1e-5, atol=1e-7)
    for handle in handles:
        handle.remove()


def test_tied_parameter_raises():
    torch.manual_seed(47)
    embedding = nn.Embedding(10, 8)
    linear = nn.Linear(8, 10, bias=False)
    linear.weight = embedding.weight
    model = nn.ModuleDict({"embedding": embedding, "linear": linear})

    with (
        patch("heavyball.core.torch.compile", _eager_compile),
        pytest.raises(ValueError, match="shared"),
    ):
        heavyball.register_truegrad(model)

