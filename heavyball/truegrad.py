"""TrueGrad observation producers for PyTorch modules."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.grad import conv1d_weight, conv2d_weight, conv3d_weight
from torch.utils.hooks import RemovableHandle

from .core import produce
from .numerics import _wide


def _sum_per_sample_squared(value: Tensor, parameter_ndim: int) -> Tensor:
    leading_ndim = value.ndim - parameter_ndim
    if leading_ndim == 0:
        return value.square()
    if leading_ndim > 1:
        value = value.sum(dim=tuple(range(1, leading_ndim)))
    return value.square().sum(dim=0)


def register_truegrad(module: nn.Module) -> tuple[RemovableHandle, ...]:
    """Register TrueGrad producers on supported modules below ``module``."""

    producer_modules: dict[int, str] = {}
    supported_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.LayerNorm, nn.GroupNorm)
    if hasattr(nn, "RMSNorm"):
        supported_types += (nn.RMSNorm,)
    for module_name, child in module.named_modules():
        if not isinstance(child, supported_types):
            continue
        current = module_name or "<root>"
        parameters = []
        if child.weight is not None:
            parameters.append(("weight", child.weight))
        bias = getattr(child, "bias", None)
        if bias is not None:
            parameters.append(("bias", bias))
        for name, param in parameters:
            existing = producer_modules.get(id(param))
            if existing is not None:
                raise ValueError(
                    f"Parameter {name} is shared between {existing} and {current}; "
                    "tied/shared parameters are not supported by register_truegrad() because independent "
                    "producers would make sum_grad_squared incorrect"
                )
            producer_modules[id(param)] = current

    handles = []
    for linear in (child for child in module.modules() if isinstance(child, nn.Linear)):
        captured = [None]

        def capture_input(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured) -> None:
            captured[0] = args[0].detach()

        @torch.no_grad()
        def produce_observations(
            linear: nn.Linear,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
            *,
            captured=captured,
        ) -> None:
            input = captured[0]
            captured[0] = None
            output_grad = grad_output[0].detach()
            batch_size = input.shape[0] if input.ndim > 1 else 1
            flat_input = _wide(input).reshape(batch_size, -1, input.shape[-1])
            flat_output_grad = _wide(output_grad).reshape(batch_size, -1, output_grad.shape[-1])
            if linear.weight.requires_grad:
                sample_weight_grads = torch.bmm(flat_output_grad.transpose(1, 2), flat_input)
                weight_sum_grad_squared = sample_weight_grads.square().sum(dim=0)
                produce(linear.weight, "sum_grad_squared", weight_sum_grad_squared)
            if linear.bias is not None and linear.bias.requires_grad:
                produce(linear.bias, "sum_grad_squared", flat_output_grad.sum(dim=1).square().sum(dim=0))

        handles.append(linear.register_forward_hook(capture_input))
        handles.append(linear.register_full_backward_hook(produce_observations))

    conv_types = (
        (nn.Conv1d, conv1d_weight),
        (nn.Conv2d, conv2d_weight),
        (nn.Conv3d, conv3d_weight),
    )
    for conv_type, conv_weight in conv_types:
        for conv in (child for child in module.modules() if isinstance(child, conv_type)):
            captured = [None]

            def capture_input(
                _module: nn.Module, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
            ) -> None:
                captured[0] = args[0].detach()

            @torch.no_grad()
            def produce_observations(
                conv: nn.Conv1d | nn.Conv2d | nn.Conv3d,
                _grad_input: tuple[Tensor | None, ...],
                grad_output: tuple[Tensor | None, ...],
                *,
                captured=captured,
                conv_weight=conv_weight,
            ) -> None:
                input = captured[0]
                captured[0] = None
                output_grad = grad_output[0].detach()
                if input.ndim == conv.weight.ndim - 1:
                    input = input.unsqueeze(0)
                    output_grad = output_grad.unsqueeze(0)
                input = _wide(input)
                output_grad = _wide(output_grad)
                padding = conv.padding
                if conv.padding_mode != "zeros":
                    input = F.pad(input, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
                    padding = 0
                if conv.weight.requires_grad:
                    weight_sum_grad_squared = torch.zeros_like(conv.weight, dtype=input.dtype)
                    for sample_input, sample_output_grad in zip(input, output_grad, strict=True):
                        sample_weight_grad = conv_weight(
                            sample_input.unsqueeze(0),
                            conv.weight.shape,
                            sample_output_grad.unsqueeze(0),
                            conv.stride,
                            padding,
                            conv.dilation,
                            conv.groups,
                        )
                        weight_sum_grad_squared.add_(sample_weight_grad.square())
                    produce(conv.weight, "sum_grad_squared", weight_sum_grad_squared)
                if conv.bias is not None and conv.bias.requires_grad:
                    spatial_dims = tuple(range(2, output_grad.ndim))
                    sample_bias_grads = output_grad.sum(dim=spatial_dims)
                    produce(conv.bias, "sum_grad_squared", sample_bias_grads.square().sum(dim=0))

            handles.append(conv.register_forward_hook(capture_input))
            handles.append(conv.register_full_backward_hook(produce_observations))

    for embedding in (child for child in module.modules() if isinstance(child, nn.Embedding)):
        captured = [None]

        def capture_input(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured) -> None:
            captured[0] = args[0].detach()

        @torch.no_grad()
        def produce_observations(
            embedding: nn.Embedding,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
            *,
            captured=captured,
        ) -> None:
            input = captured[0]
            captured[0] = None
            output_grad = _wide(grad_output[0].detach())
            if embedding.weight.requires_grad:
                batch_size = input.shape[0] if input.ndim > 0 else 1
                flat_input = input.reshape(batch_size, -1)
                flat_output_grad = output_grad.reshape(batch_size, -1, output_grad.shape[-1])
                weight_sum_grad_squared = torch.zeros_like(embedding.weight, dtype=output_grad.dtype)
                sample_weight_grad = torch.zeros_like(weight_sum_grad_squared)
                frequencies = None
                if embedding.scale_grad_by_freq:
                    frequency_input = input.reshape(-1)
                    if embedding.padding_idx is not None:
                        frequency_input = frequency_input[frequency_input != embedding.padding_idx]
                    frequencies = torch.bincount(
                        frequency_input,
                        minlength=embedding.num_embeddings,
                    )
                for sample_input, sample_output_grad in zip(flat_input, flat_output_grad, strict=True):
                    if embedding.padding_idx is not None:
                        valid = sample_input != embedding.padding_idx
                        sample_input = sample_input[valid]
                        sample_output_grad = sample_output_grad[valid]
                    if frequencies is not None:
                        sample_output_grad = sample_output_grad / frequencies[sample_input].unsqueeze(-1)
                    sample_weight_grad.zero_()
                    sample_weight_grad.index_add_(0, sample_input, sample_output_grad)
                    weight_sum_grad_squared.add_(sample_weight_grad.square())
                produce(embedding.weight, "sum_grad_squared", weight_sum_grad_squared)

        handles.append(embedding.register_forward_hook(capture_input))
        handles.append(embedding.register_full_backward_hook(produce_observations))

    for layernorm in (child for child in module.modules() if isinstance(child, nn.LayerNorm)):
        captured = [None]

        def capture_input(
            layernorm: nn.LayerNorm, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
        ) -> None:
            x = args[0].detach()
            captured[0] = F.layer_norm(x, layernorm.normalized_shape, None, None, layernorm.eps)

        @torch.no_grad()
        def produce_observations(
            layernorm: nn.LayerNorm,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
            *,
            captured=captured,
        ) -> None:
            x_norm = captured[0]
            captured[0] = None
            output_grad = grad_output[0].detach()
            weight = layernorm.weight
            if weight is not None and weight.requires_grad:
                weight_sum_grad_squared = _sum_per_sample_squared(_wide(output_grad) * _wide(x_norm), weight.ndim)
                produce(weight, "sum_grad_squared", weight_sum_grad_squared)
            bias = layernorm.bias
            if bias is not None and bias.requires_grad:
                bias_sum_grad_squared = _sum_per_sample_squared(_wide(output_grad), bias.ndim)
                produce(bias, "sum_grad_squared", bias_sum_grad_squared)

        handles.append(layernorm.register_forward_hook(capture_input))
        handles.append(layernorm.register_full_backward_hook(produce_observations))

    for groupnorm in (child for child in module.modules() if isinstance(child, nn.GroupNorm)):
        captured = [None]

        def capture_input(
            groupnorm: nn.GroupNorm, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
        ) -> None:
            x = args[0].detach()
            captured[0] = F.group_norm(x, groupnorm.num_groups, None, None, groupnorm.eps)

        @torch.no_grad()
        def produce_observations(
            groupnorm: nn.GroupNorm,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
            *,
            captured=captured,
        ) -> None:
            x_norm = captured[0]
            captured[0] = None
            output_grad = grad_output[0].detach()
            spatial_dims = tuple(range(2, x_norm.ndim))
            weight = groupnorm.weight
            if weight is not None and weight.requires_grad:
                sample_weight_grads = _wide(output_grad) * _wide(x_norm)
                if spatial_dims:
                    sample_weight_grads = sample_weight_grads.sum(dim=spatial_dims)
                weight_sum_grad_squared = sample_weight_grads.square().sum(dim=0)
                produce(weight, "sum_grad_squared", weight_sum_grad_squared)
            bias = groupnorm.bias
            if bias is not None and bias.requires_grad:
                sample_bias_grads = _wide(output_grad)
                if spatial_dims:
                    sample_bias_grads = sample_bias_grads.sum(dim=spatial_dims)
                bias_sum_grad_squared = sample_bias_grads.square().sum(dim=0)
                produce(bias, "sum_grad_squared", bias_sum_grad_squared)

        handles.append(groupnorm.register_forward_hook(capture_input))
        handles.append(groupnorm.register_full_backward_hook(produce_observations))

    if hasattr(nn, "RMSNorm"):
        for rmsnorm in (child for child in module.modules() if isinstance(child, nn.RMSNorm)):
            captured = [None]

            def capture_input(
                rmsnorm: nn.RMSNorm, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
            ) -> None:
                x = args[0].detach()
                captured[0] = F.rms_norm(x, rmsnorm.normalized_shape, None, rmsnorm.eps)

            @torch.no_grad()
            def produce_observations(
                rmsnorm: nn.RMSNorm,
                _grad_input: tuple[Tensor | None, ...],
                grad_output: tuple[Tensor | None, ...],
                *,
                captured=captured,
            ) -> None:
                x_norm = captured[0]
                captured[0] = None
                output_grad = grad_output[0].detach()
                weight = rmsnorm.weight
                if weight is not None and weight.requires_grad:
                    weight_sum_grad_squared = _sum_per_sample_squared(_wide(output_grad) * _wide(x_norm), weight.ndim)
                    produce(weight, "sum_grad_squared", weight_sum_grad_squared)

            handles.append(rmsnorm.register_forward_hook(capture_input))
            handles.append(rmsnorm.register_full_backward_hook(produce_observations))
    return tuple(handles)
