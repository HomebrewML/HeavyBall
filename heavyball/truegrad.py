"""TrueGrad observation producers for PyTorch modules."""

import torch
from torch import Tensor, nn
from torch.nn.grad import conv1d_weight, conv2d_weight, conv3d_weight
from torch.utils.hooks import RemovableHandle

from .core import produce
from .numerics import _wide


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
            flat_input = input.reshape(-1, input.shape[-1])
            flat_output_grad = output_grad.reshape(-1, output_grad.shape[-1])
            if linear.weight.requires_grad:
                # Widen before squaring: a finite fp16 gradient like 300 has a square (90000) that
                # overflows fp16 but is ordinary in the fp32 observation slab.
                weight_sum_grad_squared = torch.einsum(
                    "bo,bi->oi", _wide(flat_output_grad).square(), _wide(flat_input).square()
                )
                produce(linear.weight, "sum_grad_squared", weight_sum_grad_squared)
            if linear.bias is not None and linear.bias.requires_grad:
                produce(linear.bias, "sum_grad_squared", _wide(flat_output_grad).square().sum(dim=0))

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
                # The reference drops cross-spatial correlations for its per-position-independent Fisher.
                if conv.weight.requires_grad:
                    weight_sum_grad_squared = conv_weight(
                        _wide(input).square(),
                        conv.weight.shape,
                        _wide(output_grad).square(),
                        conv.stride,
                        conv.padding,
                        conv.dilation,
                        conv.groups,
                    )
                    produce(conv.weight, "sum_grad_squared", weight_sum_grad_squared)
                if conv.bias is not None and conv.bias.requires_grad:
                    spatial_dims = tuple(range(2, output_grad.ndim))
                    produce(conv.bias, "sum_grad_squared", _wide(output_grad).square().sum(dim=(0, *spatial_dims)))

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
                flat_output_grad = output_grad.reshape(-1, output_grad.shape[-1])
                flat_input = input.reshape(-1, 1).expand_as(flat_output_grad)
                weight_sum_grad_squared = torch.zeros_like(embedding.weight, dtype=output_grad.dtype)
                weight_sum_grad_squared.scatter_add_(0, flat_input, flat_output_grad.square())
                produce(embedding.weight, "sum_grad_squared", weight_sum_grad_squared)

        handles.append(embedding.register_forward_hook(capture_input))
        handles.append(embedding.register_full_backward_hook(produce_observations))

    for layernorm in (child for child in module.modules() if isinstance(child, nn.LayerNorm)):
        captured = [None]

        def capture_input(
            layernorm: nn.LayerNorm, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
        ) -> None:
            x = args[0].detach()
            normalized_dims = tuple(range(x.ndim - len(layernorm.normalized_shape), x.ndim))
            mean = x.mean(normalized_dims, keepdim=True)
            var = x.var(normalized_dims, correction=0, keepdim=True)
            x_norm = (x - mean) / (var + layernorm.eps).sqrt()
            captured[0] = x_norm

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
                batch_dims = tuple(range(x_norm.ndim - len(weight.shape)))
                weight_sum_grad_squared = (_wide(output_grad) * _wide(x_norm)).square().sum(dim=batch_dims)
                produce(weight, "sum_grad_squared", weight_sum_grad_squared)
            bias = layernorm.bias
            if bias is not None and bias.requires_grad:
                batch_dims = tuple(range(x_norm.ndim - len(bias.shape)))
                bias_sum_grad_squared = _wide(output_grad).square().sum(dim=batch_dims)
                produce(bias, "sum_grad_squared", bias_sum_grad_squared)

        handles.append(layernorm.register_forward_hook(capture_input))
        handles.append(layernorm.register_full_backward_hook(produce_observations))

    for groupnorm in (child for child in module.modules() if isinstance(child, nn.GroupNorm)):
        captured = [None]

        def capture_input(
            groupnorm: nn.GroupNorm, args: tuple[Tensor, ...], _output: Tensor, *, captured=captured
        ) -> None:
            x = args[0].detach()
            x_grouped = x.reshape(x.shape[0], groupnorm.num_groups, x.shape[1] // groupnorm.num_groups, *x.shape[2:])
            normalized_dims = tuple(range(2, x_grouped.ndim))
            mean = x_grouped.mean(normalized_dims, keepdim=True)
            var = x_grouped.var(normalized_dims, correction=0, keepdim=True)
            x_norm = ((x_grouped - mean) / (var + groupnorm.eps).sqrt()).reshape_as(x)
            captured[0] = x_norm

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
            batch_and_spatial = (0,) + tuple(range(2, x_norm.ndim))
            weight = groupnorm.weight
            if weight is not None and weight.requires_grad:
                weight_sum_grad_squared = (_wide(output_grad) * _wide(x_norm)).square().sum(dim=batch_and_spatial)
                produce(weight, "sum_grad_squared", weight_sum_grad_squared)
            bias = groupnorm.bias
            if bias is not None and bias.requires_grad:
                bias_sum_grad_squared = _wide(output_grad).square().sum(dim=batch_and_spatial)
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
                normalized_dims = tuple(range(x.ndim - len(rmsnorm.normalized_shape), x.ndim))
                rms = (x.square().mean(normalized_dims, keepdim=True) + rmsnorm.eps).sqrt()
                x_norm = x / rms
                captured[0] = x_norm

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
                    batch_dims = tuple(range(x_norm.ndim - len(weight.shape)))
                    weight_sum_grad_squared = (_wide(output_grad) * _wide(x_norm)).square().sum(dim=batch_dims)
                    produce(weight, "sum_grad_squared", weight_sum_grad_squared)

            handles.append(rmsnorm.register_forward_hook(capture_input))
            handles.append(rmsnorm.register_full_backward_hook(produce_observations))
    return tuple(handles)
