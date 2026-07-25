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
    before = copy.deepcopy(optimizer.state_dict())
    torch.manual_seed(441)
    optimizer.step(observed=[False])
    assert torch.equal(parameter, before_parameter)
    after = optimizer.state_dict()
    for actual_engine, expected_engine in zip(
        after["engines"], before["engines"], strict=True
    ):
        actual_engine["step"] = expected_engine["step"]
    _assert_nested_equal(after, before)


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
    checkpoint = copy.deepcopy(optimizer.state_dict())

    torch.manual_seed(800)
    rng_state = torch.random.get_rng_state()
    optimizer.eval()
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.manual_seed(801)
    rng_state = torch.random.get_rng_state()
    optimizer.train()
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    assert torch.equal(parameter, training_parameter)
    _assert_nested_equal(optimizer.state_dict(), checkpoint)


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


def test_ecc_rejects_mismatched_narrow_dtype():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = heavyball.AdamW([parameter], ecc=8)
    checkpoint = copy.deepcopy(optimizer._engine.state_dict())
    checkpoint["state"]["0"][0]["exp_avg"] = checkpoint["state"]["0"][0]["exp_avg"].float()
    with pytest.raises(ValueError, match="shape or dtype"):
        optimizer._engine.load_state_dict(checkpoint)


def test_rejects_bad_ecc():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="ecc must be None, 8, 16"):
        heavyball.AdamW([parameter], ecc=7)
    with pytest.raises(ValueError, match="ecc requires storage_dtype"):
        heavyball.AdamW([parameter], ecc=8, storage_dtype=torch.float16)


@pytest.mark.parametrize(
    ("legacy", "numeric"),
    (("bf16+8", 8), ("bf16+16", 16)),
)
def test_legacy_ecc_names(legacy, numeric):
    legacy_parameter, _ = _trajectory(heavyball.AdamW, ecc=legacy)
    numeric_parameter, _ = _trajectory(heavyball.AdamW, ecc=numeric)
    assert torch.equal(legacy_parameter, numeric_parameter)
