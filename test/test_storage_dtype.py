import pytest
import torch

import heavyball

_STEPS = 8
_FACADES = (heavyball.AdamW, heavyball.SOAP, heavyball.PSGDKron, heavyball.KLSOAP)


def _trajectory(facade, *, storage_dtype=...):
    initial = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(4, 4)
    generator = torch.Generator().manual_seed(731)
    gradients = [torch.randn(initial.shape, generator=generator) * 0.2 for _ in range(_STEPS)]
    parameter = torch.nn.Parameter(initial.clone())
    kwargs = {} if storage_dtype is ... else {"storage_dtype": storage_dtype}
    optimizer = facade([parameter], **kwargs)
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_(gradient)
        torch.manual_seed(9100 + step)
        optimizer.step()
    return parameter.detach().clone(), optimizer


def _state_slabs(optimizer):
    return [
        slab
        for engine in optimizer._engines
        for group in engine.groups
        for slots in (*group.states, group.commit_state)
        for slab in slots.values()
    ]


def test_default_is_bit_for_bit():
    for facade in (heavyball.AdamW, heavyball.SOAP):
        default, _ = _trajectory(facade)
        explicit_none, _ = _trajectory(facade, storage_dtype=None)
        assert torch.equal(default, explicit_none)


@pytest.mark.parametrize("facade", _FACADES)
def test_bf16_state_is_allocated_and_close(facade):
    bf16_parameter, bf16_optimizer = _trajectory(facade, storage_dtype=torch.bfloat16)
    slabs = _state_slabs(bf16_optimizer)
    assert slabs
    assert any(slab.dtype is torch.bfloat16 for slab in slabs)
    assert all(slab.dtype in (torch.bfloat16, torch.float64) for slab in slabs)
    assert torch.isfinite(bf16_parameter).all()

    fp32_parameter, _ = _trajectory(facade, storage_dtype=None)
    max_diff = (bf16_parameter - fp32_parameter).abs().max()
    relative_diff = max_diff / fp32_parameter.abs().max().clamp_min(torch.finfo(torch.float32).eps)
    assert relative_diff < 6e-2


def test_rejects_unsupported_dtype():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="storage_dtype must be None or torch.bfloat16"):
        heavyball.AdamW([parameter], storage_dtype=torch.float16)
