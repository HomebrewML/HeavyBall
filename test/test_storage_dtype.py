import pytest
import torch

import heavyball

_STEPS = 8
_FACADES = (heavyball.AdamW, heavyball.SOAP, heavyball.PSGDKron, heavyball.KLSOAP)
_ALL_FACADES = tuple(
    getattr(heavyball, name)
    for name in sorted(dir(heavyball))
    if isinstance(getattr(heavyball, name), type)
    and issubclass(getattr(heavyball, name), heavyball.HeavyBallOptimizer)
    and getattr(heavyball, name) is not heavyball.HeavyBallOptimizer
)


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


@pytest.mark.parametrize("facade", _ALL_FACADES, ids=lambda facade: facade.__name__)
def test_bf16_parameters_update(facade):
    if not torch.cuda.is_available():
        pytest.skip("GPU-only: iterates Muon, CPU inductor cpp_bmm bug")
    if facade.__name__.startswith("TrueGrad"):
        pytest.skip("TrueGrad facades require an external observation producer")
    initial = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(4, 4).bfloat16().cuda()
    generator = torch.Generator().manual_seed(731)
    gradients = [torch.randn(initial.shape, generator=generator).mul(0.2).bfloat16().cuda() for _ in range(_STEPS)]
    parameter = torch.nn.Parameter(initial.clone())
    initial_clone = parameter.detach().clone()
    optimizer = facade([parameter])
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_(gradient)
        torch.manual_seed(9100 + step)
        optimizer.step()

    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, initial_clone)


@pytest.mark.parametrize("facade", _FACADES)
def test_bf16_state_is_allocated_and_close(facade, capsys):
    bf16_parameter, bf16_optimizer = _trajectory(facade, storage_dtype=torch.bfloat16)
    slabs = _state_slabs(bf16_optimizer)
    assert slabs
    # bf16 storage narrows floating state to bf16, EXCEPT state deliberately kept at higher precision --
    # PSGD/LATHER keep their fp64 running_lower_bound stability scalars fp64 (see _narrowed_dtype).
    assert any(slab.dtype is torch.bfloat16 for slab in slabs)
    assert all(slab.dtype in (torch.bfloat16, torch.float64) for slab in slabs)
    assert torch.isfinite(bf16_parameter).all()

    fp32_parameter, _ = _trajectory(facade, storage_dtype=None)
    max_diff = (bf16_parameter - fp32_parameter).abs().max()
    relative_diff = max_diff / fp32_parameter.abs().max().clamp_min(torch.finfo(torch.float32).eps)
    with capsys.disabled():
        print(
            f"{facade.__name__} bf16-vs-fp32 max diff: {float(max_diff):.9e} "
            f"(relative {float(relative_diff):.9e})"
        )
    assert relative_diff < 6e-2


def test_bf16_state_saves_memory():
    _, fp32_optimizer = _trajectory(heavyball.AdamW, storage_dtype=None)
    _, bf16_optimizer = _trajectory(heavyball.AdamW, storage_dtype=torch.bfloat16)
    fp32_bytes = sum(slab.numel() * slab.element_size() for slab in _state_slabs(fp32_optimizer))
    bf16_bytes = sum(slab.numel() * slab.element_size() for slab in _state_slabs(bf16_optimizer))
    assert bf16_bytes * 2 == fp32_bytes


@pytest.mark.parametrize("facade", _ALL_FACADES, ids=lambda facade: facade.__name__)
def test_bf16_state_saves_memory_for_every_optimizer(facade):
    fp32_parameter = torch.nn.Parameter(torch.ones(8, 8))
    bf16_parameter = torch.nn.Parameter(torch.ones(8, 8))
    fp32_optimizer = facade([fp32_parameter], storage_dtype=None)
    bf16_optimizer = facade([bf16_parameter], storage_dtype=torch.bfloat16)
    fp32_bytes = sum(slab.numel() * slab.element_size() for slab in _state_slabs(fp32_optimizer))
    bf16_bytes = sum(slab.numel() * slab.element_size() for slab in _state_slabs(bf16_optimizer))
    if fp32_bytes == 0:
        pytest.skip("optimizer is stateless")
    assert 0.5 * fp32_bytes <= bf16_bytes < fp32_bytes


def test_rejects_unsupported_dtype():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="storage_dtype must be None or torch.bfloat16"):
        heavyball.AdamW([parameter], storage_dtype=torch.float16)
