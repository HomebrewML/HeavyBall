"""Regression tests for slab-backed TrueGrad observations."""

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


def test_linear_truegrad_producer_preserves_observation_slab_binding():
    torch.manual_seed(43)
    linear = nn.Linear(4, 3)
    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.Engine(linear.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(linear)
    slab_views = {
        param: group.observations.sum_grad_squared[index]
        for group in optimizer.groups
        for index, param in enumerate(group.params)
    }

    optimizer.zero_grad()
    linear(torch.randn(5, 4)).square().sum().backward()

    for param in linear.parameters():
        assert param.sum_grad_squared.data_ptr() == slab_views[param].data_ptr()
        assert torch.count_nonzero(slab_views[param]) > 0
    for handle in handles:
        handle.remove()


def test_conv2d_truegrad_producer_matches_unfold_reference():
    torch.manual_seed(44)
    conv = nn.Conv2d(3, 5, 3, padding=1)
    inputs = torch.randn(4, 3, 6, 7)
    output_grads = []
    with patch("heavyball.core.torch.compile", _eager_compile):
        heavyball.HeavyBallOptimizer(conv.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(conv)

    output = conv(inputs)
    output.register_hook(lambda grad: output_grads.append(grad.detach()))
    _batch_loss(output).backward()

    output_grad = output_grads.pop()
    unfolded_input = F.unfold(
        inputs.square(),
        conv.kernel_size,
        dilation=conv.dilation,
        padding=conv.padding,
        stride=conv.stride,
    )
    flat_output_grad = output_grad.square().reshape(inputs.shape[0], conv.out_channels, -1)
    weight_reference = torch.einsum("bol,bkl->ok", flat_output_grad, unfolded_input).reshape(conv.weight.shape)
    bias_reference = output_grad.square().sum(dim=(0, 2, 3))
    torch.testing.assert_close(conv.weight.sum_grad_squared, weight_reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(conv.bias.sum_grad_squared, bias_reference, rtol=1e-6, atol=1e-6)
    for handle in handles:
        handle.remove()


def test_truegrad_step_runs_on_conv2d_and_changes_parameters():
    torch.manual_seed(45)
    conv = nn.Conv2d(3, 5, 3, padding=1)
    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.HeavyBallOptimizer([conv.weight, conv.bias], heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(conv)
    initial_params = [param.detach().clone() for param in conv.parameters()]

    optimizer.zero_grad()
    _batch_loss(conv(torch.randn(4, 3, 6, 7))).backward()
    optimizer.step()

    for param, initial_param in zip(conv.parameters(), initial_params, strict=True):
        assert not torch.equal(param, initial_param)
    for handle in handles:
        handle.remove()


def test_embedding_truegrad_producer_matches_vmap_reference():
    torch.manual_seed(46)
    embedding = nn.Embedding(10, 8)
    inputs = torch.randint(0, 10, (6, 4))
    output_grads = []
    with patch("heavyball.core.torch.compile", _eager_compile):
        heavyball.HeavyBallOptimizer(embedding.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(embedding)

    output = embedding(inputs)
    output.register_hook(lambda grad: output_grads.append(grad.detach()))
    _batch_loss(output).backward()

    output_grad = output_grads.pop()
    weight_reference = torch.zeros_like(embedding.weight)
    for sample_input, sample_output_grad in zip(inputs, output_grad, strict=True):
        for index, grad in zip(sample_input, sample_output_grad, strict=True):
            weight_reference[index] += grad.square()
    torch.testing.assert_close(embedding.weight.sum_grad_squared, weight_reference, rtol=1e-5, atol=1e-6)
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


def test_embedding_step_runs_and_updates():
    torch.manual_seed(48)
    embedding = nn.Embedding(10, 8)
    inputs = torch.randint(0, 10, (6, 4))
    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.HeavyBallOptimizer(embedding.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(embedding)
    initial_weight = embedding.weight.detach().clone()

    optimizer.zero_grad()
    _batch_loss(embedding(inputs)).backward()
    optimizer.step()

    assert not torch.equal(embedding.weight, initial_weight)
    for handle in handles:
        handle.remove()


def test_groupnorm_truegrad_producer_matches_reference():
    torch.manual_seed(49)
    gn = nn.GroupNorm(4, 16)
    x = torch.randn(8, 16, 3, 3)

    output_grads = []
    with patch("heavyball.core.torch.compile", _eager_compile):
        heavyball.HeavyBallOptimizer(gn.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(gn)

    output = gn(x)
    output.register_hook(lambda grad: output_grads.append(grad.detach()))
    _batch_loss(output).backward()
    output_grad = output_grads.pop()

    B, C, H, W = x.shape
    G = gn.num_groups
    x_grouped = x.detach().reshape(B, G, C // G, H, W)
    norm_dims = (2, 3, 4)
    mean = x_grouped.mean(norm_dims, keepdim=True)
    var = x_grouped.var(norm_dims, correction=0, keepdim=True)
    x_norm = ((x_grouped - mean) / (var + gn.eps).sqrt()).reshape(B, C, H, W)
    weight_ref = (output_grad * x_norm).square().sum(dim=(0, 2, 3))
    bias_ref = output_grad.square().sum(dim=(0, 2, 3))

    torch.testing.assert_close(gn.weight.sum_grad_squared, weight_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gn.bias.sum_grad_squared, bias_ref, rtol=1e-5, atol=1e-6)
    for handle in handles:
        handle.remove()


def test_groupnorm_step_runs_and_updates():
    torch.manual_seed(50)
    gn = nn.GroupNorm(4, 16)
    x = torch.randn(4, 16, 5, 5)
    with patch("heavyball.core.torch.compile", _eager_compile):
        optimizer = heavyball.HeavyBallOptimizer(gn.parameters(), heavyball.truegrad_adam)
    handles = heavyball.register_truegrad(gn)
    initial_weight = gn.weight.detach().clone()

    optimizer.zero_grad()
    _batch_loss(gn(x)).backward()
    optimizer.step()

    assert not torch.equal(gn.weight, initial_weight)
    for handle in handles:
        handle.remove()
