import copy

import pytest
import torch

import heavyball

_STEPS = 8


def _trajectory(facade, *, storage_dtype=..., ecc=...):
    initial = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).reshape(4, 4)
    generator = torch.Generator().manual_seed(731)
    gradients = [torch.randn(initial.shape, generator=generator) * 0.2 for _ in range(_STEPS)]
    parameter = torch.nn.Parameter(initial.clone())
    kwargs = {}
    if storage_dtype is not ...:
        kwargs["storage_dtype"] = storage_dtype
    if ecc is not ...:
        kwargs["ecc"] = ecc
    torch.manual_seed(1234)
    optimizer = facade([parameter], **kwargs)
    for step, gradient in enumerate(gradients):
        parameter.grad.copy_(gradient)
        torch.manual_seed(9100 + step)
        optimizer.step()
    return parameter.detach().clone(), optimizer


def _state_and_corrections(optimizer):
    for engine in optimizer._engines:
        for group in engine.groups:
            yield from zip(group.states, group.state_corrections, strict=True)
            yield group.commit_state, group.commit_corrections


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


@pytest.mark.parametrize("facade", (heavyball.AdamW, heavyball.SOAP))
def test_ecc_much_closer_than_bf16(facade):
    fp32, _ = _trajectory(facade, storage_dtype=None)
    bf16, _ = _trajectory(facade, storage_dtype=torch.bfloat16)
    ecc8, _ = _trajectory(facade, ecc=8)
    ecc16, _ = _trajectory(facade, ecc=16)

    bf16_error = (bf16 - fp32).abs().max()
    ecc8_error = (ecc8 - fp32).abs().max()
    ecc16_error = (ecc16 - fp32).abs().max()
    assert ecc8_error < bf16_error / 8
    assert ecc16_error < bf16_error / 50


def test_ecc_masked_leaf_preserves_physical_state():
    parameter, optimizer = _trajectory(heavyball.AdamW, ecc=8)
    before_parameter = parameter.clone()
    before = [
        ({name: slab.clone() for name, slab in state.items()},
         {name: slab.clone() for name, slab in corrections.items()})
        for state, corrections in _state_and_corrections(optimizer)
    ]
    torch.manual_seed(441)
    optimizer._engine.step(observed=[False])
    assert torch.equal(optimizer._engine.params[0], before_parameter)
    for (state, corrections), (old_state, old_corrections) in zip(
        _state_and_corrections(optimizer), before, strict=True
    ):
        assert all(torch.equal(slab, old_state[name]) for name, slab in state.items())
        assert all(torch.equal(slab, old_corrections[name]) for name, slab in corrections.items())


def test_ecc_checkpoint_roundtrip():
    generator = torch.Generator().manual_seed(81)
    gradients = [torch.randn(4, 4, generator=generator) for _ in range(6)]
    source_parameter = torch.nn.Parameter(torch.randn(4, 4, generator=generator))
    source = heavyball.AdamW([source_parameter], ecc=8)
    for step, gradient in enumerate(gradients[:3]):
        source_parameter.grad.copy_(gradient)
        torch.manual_seed(600 + step)
        source.step()

    checkpoint = copy.deepcopy(source.state_dict())
    assert checkpoint["engines"][0]["ecc"] == 8
    assert checkpoint["engines"][0]["format"] == 4
    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    torch.manual_seed(987654321)
    target = heavyball.AdamW([target_parameter], ecc=8)
    target.load_state_dict(checkpoint)
    for source_pair, target_pair in zip(
        _state_and_corrections(source), _state_and_corrections(target), strict=True
    ):
        for source_slots, target_slots in zip(source_pair, target_pair, strict=True):
            assert source_slots.keys() == target_slots.keys()
            assert all(torch.equal(source_slots[name], target_slots[name]) for name in source_slots)

    torch.manual_seed(123456789)
    for gradient in gradients[3:]:
        source_parameter.grad.copy_(gradient)
        target_parameter.grad.copy_(gradient)
        source.step()
        target.step()
        assert torch.equal(source_parameter, target_parameter)
        _assert_nested_equal(source.state_dict(), target.state_dict())


def test_ecc_eval_swap_preserves_commit_state():
    torch.manual_seed(91)
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = heavyball.ScheduleFree([parameter], ecc=8)
    for step in range(3):
        parameter.grad.copy_(torch.randn_like(parameter))
        torch.manual_seed(720 + step)
        optimizer.step()
    optimizer.eval()
    optimizer.train()
    training_parameter = parameter.detach().clone()
    group = optimizer._engine.groups[0]
    old_state = {name: slab.clone() for name, slab in group.commit_state.items()}
    old_corrections = {name: slab.clone() for name, slab in group.commit_corrections.items()}

    torch.manual_seed(800)
    rng_state = torch.random.get_rng_state()
    optimizer.eval()
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.manual_seed(801)
    rng_state = torch.random.get_rng_state()
    optimizer.train()
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    assert torch.equal(parameter, training_parameter)
    assert all(torch.equal(slab, old_state[name]) for name, slab in group.commit_state.items())
    assert all(torch.equal(slab, old_corrections[name]) for name, slab in group.commit_corrections.items())


def test_ecc_rejects_incompatible_checkpoint_before_facade_mutation():
    source_parameter = torch.nn.Parameter(torch.ones(2, 2))
    source = heavyball.AdamW([source_parameter], ecc=8, lr=0.123)
    checkpoint = source.state_dict()

    target_parameter = torch.nn.Parameter(torch.ones(2, 2))
    target = heavyball.AdamW([target_parameter], ecc=16, lr=0.456)
    with pytest.raises(ValueError, match="ECC configuration"):
        target.load_state_dict(checkpoint)
    assert target.param_groups[0]["ecc"] == 16
    assert target.param_groups[0]["lr"] == 0.456
    assert target._engine.ecc is torch.int16


def test_ecc_rejects_mismatched_narrow_dtype():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = heavyball.AdamW([parameter], ecc=8)
    checkpoint = copy.deepcopy(optimizer._engine.state_dict())
    checkpoint["state"]["0"][0]["exp_avg"] = checkpoint["state"]["0"][0]["exp_avg"].float()
    with pytest.raises(ValueError, match="shape or dtype"):
        optimizer._engine.load_state_dict(checkpoint)


def test_ecc_checkpoint_requires_cadence_state():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = heavyball.SOAP([parameter], ecc=8)
    checkpoint = copy.deepcopy(optimizer._engine.state_dict())
    del checkpoint["cadence"]
    with pytest.raises(ValueError, match="cadence presence"):
        optimizer._engine.load_state_dict(checkpoint)


def test_rejects_bad_ecc():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="ecc must be None, 8, 16"):
        heavyball.AdamW([parameter], ecc=7)
    with pytest.raises(ValueError, match="ecc requires storage_dtype"):
        heavyball.AdamW([parameter], ecc=8, storage_dtype=torch.float16)


@pytest.mark.parametrize(
    ("ecc", "correction_dtype"),
    (("bf16+8", torch.int8), ("bf16+16", torch.int16)),
)
def test_legacy_ecc_names(ecc, correction_dtype):
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = heavyball.AdamW([parameter], ecc=ecc)
    assert optimizer._engine.ecc is correction_dtype
