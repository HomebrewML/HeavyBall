"""Behavioral coverage for Engine construction, ECC commit state, and checkpoints."""

import copy
import re
from collections import OrderedDict
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, Recipe, RefreshCadence, Route, adamw, msam_laprop, sgd
from heavyball.codecs import decode
from heavyball.core import PlainBinding, _bounded_cache_get_or_create
from heavyball.transforms import Tempo, sgd_commit


def _build(params, recipe=adamw, **kwargs):
    """Build eager Engine closures so coverage observes the real step implementation."""

    with patch("heavyball.core.torch.compile", lambda function, **_kwargs: function):
        return Engine(params, recipe, **kwargs)


def _parameter(value=(1.0, -2.0)):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))


def _engine_and_checkpoint(recipe=adamw, *, ecc=None, **kwargs):
    parameter = _parameter()
    engine = _build([parameter], recipe, param_keys=("p",), ecc=ecc, **kwargs)
    return parameter, engine, copy.deepcopy(engine.state_dict())


def _direct_commit(param, update, state, tempo):
    del state, tempo
    return param - update, {}


_plain_recipe = Recipe((), _direct_commit, {})
_cadence_recipe = Recipe((), _direct_commit, {"preconditioner_update_probability": 0.5})


def _passthrough(update, obs, param, state, tempo):
    del obs, param
    return update, state, tempo.live


def _varying_key_init(reference):
    name = "zero" if reference.flatten()[0].item() == 0 else "nonzero"
    return {name: torch.zeros_like(reference)}


def _varying_shape_init(reference):
    size = 1 if reference.flatten()[0].item() == 0 else 2
    return {"slot": reference.new_zeros(size)}


def _rollback_init(reference):
    if reference.numel() > 1:
        raise RuntimeError("deliberate initializer failure")
    return {}


def _key_transform(update, obs, param, state, tempo):
    return _passthrough(update, obs, param, state, tempo)


def _shape_transform(update, obs, param, state, tempo):
    return _passthrough(update, obs, param, state, tempo)


def _rollback_transform(update, obs, param, state, tempo):
    return _passthrough(update, obs, param, state, tempo)


_key_transform.init = _varying_key_init
_shape_transform.init = _varying_shape_init
_rollback_transform.init = _rollback_init


class _TaggedDeviceTensor(torch.Tensor):
    """CPU tensor subclass exposing distinct valid indexed CPU device keys."""

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)

    @property
    def device(self):
        return torch.device(self._device_tag)


def _tagged_parameter(index):
    parameter = torch.nn.Parameter(_TaggedDeviceTensor(torch.ones(1)))
    parameter._device_tag = f"cpu:{index}"
    return parameter


def _exact(message):
    return f"^{re.escape(message)}$"


def test_bounded_ordered_cache_refreshes_hits_for_lru_eviction(monkeypatch):
    import heavyball.core as core

    cache = OrderedDict((key, key.upper()) for key in ("a", "b"))
    monkeypatch.setattr(core, "_COMPILE_CACHE_MAX_SIZE", 2)

    assert _bounded_cache_get_or_create(cache, "a", lambda: "new") == "A"
    _bounded_cache_get_or_create(cache, "c", lambda: "C")

    assert tuple(cache) == ("a", "c")


@pytest.mark.parametrize("ecc", (None, 16))
def test_fp32_adamw_and_ecc16_skip_philox_rounding_noise(ecc):
    parameter = _parameter()

    with patch.object(
        Tempo,
        "random_like",
        side_effect=AssertionError("unnecessary Philox noise"),
    ):
        engine = _build([parameter], adamw, ecc=ecc)
        parameter.grad.fill_(1)
        engine.step()


def test_default_observation_path_reuses_all_true_masks_and_binding_validation():
    parameter = _parameter()
    engine = _build([parameter], adamw)
    original_validate = PlainBinding.validate
    validations = 0

    def counted_validate(binding, param_row, grad_row):
        nonlocal validations
        validations += 1
        return original_validate(binding, param_row, grad_row)

    parameter.grad.fill_(1)
    with patch.object(PlainBinding, "validate", counted_validate):
        engine.step()
        assert validations == 0
        assert engine.groups[0].observed_cache is None

        engine.step(observed=[False])
        assert validations == 1
        assert not engine.groups[0].observed.any()

        engine.step()
        assert validations == 1
        assert engine.groups[0].observed.all()
        assert engine.groups[0].observed_cache is None

        parameter.data = parameter.data.clone()
        with pytest.raises(ValueError, match="weights.*no longer slab-bound"):
            engine.step()
        assert validations == 2


def test_refresh_cadence_rejects_an_invalid_scheduled_probability():
    cadence = RefreshCadence(lambda step: 1.5 if step == 1 else 0.5)

    with pytest.raises(
        ValueError,
        match=_exact("preconditioner_update_probability must be a number in [0, 1]"),
    ):
        cadence.next_step_type()

    assert cadence.step == 1


def test_tensor_hyperparameter_must_contain_one_value():
    with pytest.raises(
        ValueError,
        match=_exact("hyperparameters must be 0-d tensors or Python scalars"),
    ):
        _build([_parameter()], adamw, lr=torch.tensor([0.1, 0.2]))


def test_one_element_tensor_hyperparameter_is_normalized_to_a_scalar_cell():
    engine = _build([_parameter()], adamw, lr=torch.tensor([0.125]))

    assert engine.hyper.lr.ndim == 0
    assert float(engine.hyper.lr) == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("params", "kwargs", "error_type", "message"),
    (
        (
            [_parameter()],
            {"param_keys": ()},
            ValueError,
            "param_keys must contain one key for every supplied parameter",
        ),
        ([_parameter()], {"param_keys": (1,)}, TypeError, "param_keys must contain strings"),
        (
            [torch.nn.Parameter(torch.ones(1), requires_grad=False)],
            {},
            ValueError,
            "Engine requires at least one trainable parameter",
        ),
        (
            [torch.nn.Parameter(torch.ones(1, dtype=torch.complex64))],
            {},
            TypeError,
            "Engine supports floating-point parameters only",
        ),
    ),
    ids=("key-count", "key-type", "no-trainable-parameter", "complex-parameter"),
)
def test_engine_rejects_invalid_parameter_inputs(params, kwargs, error_type, message):
    with pytest.raises(error_type, match=_exact(message)):
        _build(params, adamw, **kwargs)


def test_engine_rejects_a_duplicate_parameter():
    parameter = _parameter()
    with pytest.raises(
        ValueError, match=_exact("a parameter may appear only once in an Engine")
    ):
        _build([parameter, parameter], adamw)


def test_engine_rejects_duplicate_parameter_keys():
    with pytest.raises(ValueError, match=_exact("param_keys must be unique")):
        _build([_parameter(), _parameter()], adamw, param_keys=("p", "p"))


def test_route_predicate_must_return_a_host_bool():
    route = Route(lambda _info: torch.tensor(True), adamw, sgd)
    with pytest.raises(TypeError, match=_exact("Route predicates must return a host bool")):
        _build([_parameter()], route)


def test_engine_requires_a_recipe_or_route():
    with pytest.raises(TypeError, match=_exact("Engine requires a Recipe or Route")):
        _build([_parameter()], object())


def test_routed_recipes_must_agree_on_global_clipping():
    clipped = replace(sgd, clip_global_norm=1.0)
    unclipped = replace(sgd, clip_global_norm=None)
    route = Route(lambda info: info.ndim == 1, clipped, unclipped)
    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1, 1))]

    with pytest.raises(
        ValueError, match=_exact("routed recipes must agree on clip_global_norm")
    ):
        _build(params, route)


def test_global_clipping_rejects_parameters_on_distinct_devices():
    params = [_tagged_parameter(0), _tagged_parameter(1)]
    with pytest.raises(
        ValueError, match=_exact("global clipping requires parameters on one device")
    ):
        _build(params, _plain_recipe, clip_global_norm=1.0)


def test_storage_dtype_string_allocates_bfloat16_state():
    engine = _build([_parameter()], adamw, storage_dtype="torch.bfloat16")
    floating_slots = [
        value
        for group in engine.groups
        for slots in group.states
        for value in slots.values()
        if value.is_floating_point()
    ]

    assert engine.storage_dtype is torch.bfloat16
    assert {value.dtype for value in floating_slots} == {torch.bfloat16}
    state = engine.groups[0].states[0]
    assert state["exp_avg"].dtype is torch.bfloat16
    assert state["exp_avg_sq"].dtype is torch.bfloat16


def test_constructor_rollback_removes_new_observation_bindings():
    recipe = Recipe(
        (_rollback_transform,),
        sgd_commit,
        {"lr": 0.1, "weight_decay": 0.0},
        observations=("probe",),
    )
    params = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(2))]
    original_data_ptrs = [param.data_ptr() for param in params]
    original_grads = [torch.full_like(param, index + 1.0) for index, param in enumerate(params)]
    for param, grad in zip(params, original_grads, strict=True):
        param.grad = grad

    with pytest.raises(RuntimeError, match=_exact("deliberate initializer failure")):
        _build(params, recipe)

    assert [param.data_ptr() for param in params] == original_data_ptrs
    assert all(param.grad is grad for param, grad in zip(params, original_grads, strict=True))
    assert all(not hasattr(param, "probe") for param in params)
    assert all(not hasattr(param, "_heavyball_observation_binding") for param in params)


def test_transform_initializer_keys_must_match_within_a_bucket():
    recipe = Recipe(
        (_key_transform,), sgd_commit, {"lr": 0.1, "weight_decay": 0.0}
    )
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.ones(2))]

    with pytest.raises(
        ValueError,
        match=_exact("transform initializer returned incompatible state keys within a bucket"),
    ):
        _build(params, recipe)


def test_transform_initializer_shapes_must_match_within_a_bucket():
    recipe = Recipe(
        (_shape_transform,), sgd_commit, {"lr": 0.1, "weight_decay": 0.0}
    )
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.ones(2))]

    with pytest.raises(
        ValueError,
        match=_exact("transform initializer returned incompatible state shape or dtype within a bucket"),
    ):
        _build(params, recipe)


def test_global_clip_combines_norms_from_multiple_buckets():
    vector = torch.nn.Parameter(torch.zeros(1))
    matrix = torch.nn.Parameter(torch.zeros(1, 1))
    engine = _build([vector, matrix], _plain_recipe, clip_global_norm=5.0)
    vector.grad.fill_(3.0)
    matrix.grad.fill_(4.0)

    engine.step()

    scale = torch.tensor(5.0) / (torch.tensor(25.0).sqrt() + 1e-6)
    torch.testing.assert_close(vector, torch.full_like(vector, -3.0 * scale), rtol=0, atol=0)
    torch.testing.assert_close(matrix, torch.full_like(matrix, -4.0 * scale), rtol=0, atol=0)


def test_ecc_commit_step_and_eval_swap_update_logical_state():
    torch.manual_seed(120)
    parameter = _parameter()
    engine = _build(
        [parameter],
        msam_laprop,
        ecc=8,
        lr=0.1,
        beta1=0.0,
        beta2=0.0,
        eps=1e-8,
        weight_decay=0.0,
        sam_step_size=0.2,
    )
    group = engine.groups[0]
    parameter.grad.copy_(torch.tensor([3.0, -4.0]))

    torch.manual_seed(121)
    engine.step()

    decoded_z = decode(
        group.commit_state["z"], group.commit_corrections["z"], torch.int8
    )
    decoded_exp_avg = decode(
        group.commit_state["exp_avg"], group.commit_corrections["exp_avg"], torch.int8
    )
    torch.testing.assert_close(
        decoded_z, torch.tensor([[0.9, -1.9]]), rtol=0, atol=4e-5
    )
    torch.testing.assert_close(
        decoded_exp_avg, torch.tensor([[1.0, -1.0]]), rtol=0, atol=4e-5
    )
    training_parameter = parameter.detach().clone()
    z_storage = group.commit_state["z"].clone()
    z_correction = group.commit_corrections["z"].clone()

    torch.manual_seed(122)
    engine.eval()

    torch.testing.assert_close(parameter, decoded_z[0], rtol=0, atol=0)
    assert torch.equal(group.commit_state["z"], z_storage)
    assert torch.equal(group.commit_corrections["z"], z_correction)
    decoded_saved = decode(
        group.commit_state["saved"], group.commit_corrections["saved"], torch.int8
    )
    torch.testing.assert_close(decoded_saved[0], training_parameter, rtol=0, atol=4e-5)

    engine.train()

    torch.testing.assert_close(parameter, decoded_saved[0], rtol=0, atol=0)
    assert torch.equal(group.commit_state["z"], z_storage)
    assert torch.equal(group.commit_corrections["z"], z_correction)


def test_train_mode_must_be_boolean():
    engine = _build([_parameter()], _plain_recipe)
    with pytest.raises(ValueError, match=_exact("training mode is expected to be boolean")):
        engine.train(1)
    assert engine._train_mode is True


def test_multi_device_step_counters_use_device_keys():
    params = [_tagged_parameter(0), _tagged_parameter(1)]
    engine = _build(params, _plain_recipe, param_keys=("left", "right"))

    checkpoint = engine.state_dict()

    assert checkpoint["step"] == {"cpu:0": 1, "cpu:1": 1}


@pytest.mark.parametrize(
    ("probability", "message"),
    (
        ("often", "cadence probability must be a number"),
        (1.5, "cadence probability must be in [0, 1]"),
    ),
)
def test_state_dict_rejects_invalid_cadence_probability(probability, message):
    engine = _build(
        [_parameter()],
        _cadence_recipe,
        preconditioner_update_probability=probability,
    )

    with pytest.raises(ValueError, match=_exact(message)):
        engine.state_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("format", 1, "expected a format-2, format-3, or format-4 Engine state dict"),
        ("train_mode", 1, "checkpoint train mode must be a bool"),
        ("age", {}, "checkpoint ages do not match this Engine"),
        ("state", {}, "checkpoint parameter keys do not match this Engine"),
        ("hyper", {}, "checkpoint hyperparameters do not match this Engine"),
        ("step", 1, "checkpoint step counters do not match this Engine"),
    ),
    ids=("format", "train-mode", "ages", "parameter-keys", "hyperparameters", "steps"),
)
def test_load_rejects_invalid_top_level_checkpoint_fields(field, value, message):
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint[field] = value

    with pytest.raises(ValueError, match=_exact(message)):
        engine.load_state_dict(checkpoint)


def test_load_rejects_a_non_mapping_checkpoint():
    engine = _build([_parameter()], adamw, param_keys=("p",))
    with pytest.raises(
        ValueError,
        match=_exact("expected a format-2, format-3, or format-4 Engine state dict"),
    ):
        engine.load_state_dict([])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (None, [], "checkpoint cadence state does not match this Engine"),
        ("probability", "often", "checkpoint cadence probability must be a number"),
        ("probability", 1.5, "checkpoint cadence probability must be in [0, 1]"),
        ("step", -1, "checkpoint cadence step must be a non-negative int"),
        ("cumulative", "many", "checkpoint cadence accumulators must be numbers"),
    ),
    ids=("mapping", "probability-type", "probability-range", "step", "accumulators"),
)
def test_load_rejects_invalid_cadence_state(field, value, message):
    _, engine, checkpoint = _engine_and_checkpoint(_cadence_recipe)
    if field is None:
        checkpoint["cadence"] = value
    else:
        checkpoint["cadence"][field] = value

    with pytest.raises(ValueError, match=_exact(message)):
        engine.load_state_dict(checkpoint)


def test_load_rejects_an_age_with_the_wrong_shape():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["age"]["p"] = torch.ones(1)

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint age has an incompatible shape for parameter 'p'"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_non_mapping_transform_state():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["state"]["p"] = []

    with pytest.raises(
        ValueError, match=_exact("checkpoint transforms do not match parameter 'p'")
    ):
        engine.load_state_dict(checkpoint)


def test_load_accepts_legacy_state_without_an_empty_commit_bucket():
    source_parameter, source, checkpoint = _engine_and_checkpoint()
    source_parameter.grad.copy_(torch.tensor([0.25, -0.5]))
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())
    del checkpoint["state"]["p"]["commit"]
    target = _build([_parameter((9.0, 8.0))], adamw, param_keys=("p",))

    target.load_state_dict(checkpoint)

    for name, value in checkpoint["state"]["p"][0].items():
        assert torch.equal(target.groups[0].states[0][name][0], value)
    assert target.groups[0].age.item() == 1


def test_load_rejects_unexpected_transform_indices():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["state"]["p"][5] = {}

    with pytest.raises(
        ValueError, match=_exact("checkpoint transforms do not match parameter 'p'")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_transform_slots():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["state"]["p"][0] = {}

    with pytest.raises(
        ValueError, match=_exact("checkpoint slots do not match parameter 'p'")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_commit_slots():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["state"]["p"]["commit"] = {"unexpected": torch.zeros(())}

    with pytest.raises(
        ValueError, match=_exact("checkpoint commit slots do not match parameter 'p'")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_incompatible_commit_slot_tensor():
    _, engine, checkpoint = _engine_and_checkpoint(msam_laprop, ecc=8)
    checkpoint["state"]["p"]["commit"]["z"] = torch.zeros(3, dtype=torch.bfloat16)

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint commit slot 'z' has an incompatible shape or dtype"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_ecc_format_without_ecc_metadata():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    del checkpoint["ecc"]

    with pytest.raises(
        ValueError, match=_exact("checkpoint format does not match its ECC configuration")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_a_different_ecc_width():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    checkpoint["ecc"] = 16

    with pytest.raises(
        ValueError, match=_exact("checkpoint ECC configuration does not match this Engine")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_correction_parameter_keys():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    checkpoint["corrections"] = {}

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint correction parameter keys do not match this Engine"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_correction_transform_indices():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    checkpoint["corrections"]["p"] = {}

    with pytest.raises(
        ValueError, match=_exact("checkpoint corrections do not match parameter 'p'")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_correction_slots():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    checkpoint["corrections"]["p"][0] = {}

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint correction slots do not match parameter 'p'"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_an_incompatible_transform_correction():
    _, engine, checkpoint = _engine_and_checkpoint(ecc=8)
    checkpoint["corrections"]["p"][0]["exp_avg"] = torch.zeros(3, dtype=torch.int8)

    with pytest.raises(
        ValueError, match=_exact("checkpoint correction slot 'exp_avg' is incompatible")
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_mismatched_commit_correction_slots():
    _, engine, checkpoint = _engine_and_checkpoint(msam_laprop, ecc=8)
    checkpoint["corrections"]["p"]["commit"] = {}

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint commit corrections do not match parameter 'p'"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_an_incompatible_commit_correction():
    _, engine, checkpoint = _engine_and_checkpoint(msam_laprop, ecc=8)
    checkpoint["corrections"]["p"]["commit"]["z"] = torch.zeros(2, dtype=torch.int16)

    with pytest.raises(
        ValueError, match=_exact("checkpoint commit correction slot 'z' is incompatible")
    ):
        engine.load_state_dict(checkpoint)


def test_ecc_commit_state_checkpoint_loads_physical_values_exactly():
    source_parameter, source, _ = _engine_and_checkpoint(msam_laprop, ecc=8)
    source_parameter.grad.copy_(torch.tensor([0.5, -0.25]))
    torch.manual_seed(310)
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())
    target = _build([_parameter((8.0, 9.0))], msam_laprop, param_keys=("p",), ecc=8)

    target.load_state_dict(checkpoint)

    source_group = source.groups[0]
    target_group = target.groups[0]
    for name in source_group.commit_state:
        assert torch.equal(target_group.commit_state[name], source_group.commit_state[name])
        assert torch.equal(
            target_group.commit_corrections[name], source_group.commit_corrections[name]
        )
    assert target.step_count.item() == source.step_count.item()
    assert target_group.age.item() == source_group.age.item()


def test_load_rejects_mismatched_hyperparameter_names():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["hyper"][(0, 0)]["unexpected"] = torch.tensor(1.0)

    with pytest.raises(
        ValueError,
        match=_exact("checkpoint hyperparameter names do not match this Engine"),
    ):
        engine.load_state_dict(checkpoint)


def test_load_rejects_a_non_scalar_hyperparameter_tensor():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["hyper"][(0, 0)]["lr"] = torch.ones(1)

    with pytest.raises(
        ValueError, match=_exact("checkpoint hyperparameter 'lr' must be scalar")
    ):
        engine.load_state_dict(checkpoint)


def test_load_accepts_numeric_hyperparameter_and_device_step_values():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["hyper"][(0, 0)]["lr"] = 0.125
    checkpoint["step"] = {"cpu": 7}

    engine.load_state_dict(checkpoint)

    assert float(engine.hyper.lr) == pytest.approx(0.125)
    assert engine.step_count.item() == 7


def test_load_rejects_unexpected_step_counter_keys():
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["step"] = {"other": 1}

    with pytest.raises(
        ValueError, match=_exact("checkpoint step counters do not match this Engine")
    ):
        engine.load_state_dict(checkpoint)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (float("inf"), "checkpoint fill value must be finite"),
        (-1, "checkpoint fill value must be a non-negative integer for a counter"),
        (2**100, "checkpoint fill value is out of range for torch.int64"),
    ),
    ids=("non-finite", "negative", "out-of-range"),
)
def test_load_rejects_invalid_step_counter_values(value, message):
    _, engine, checkpoint = _engine_and_checkpoint()
    checkpoint["step"]["global"] = value

    with pytest.raises(ValueError, match=_exact(message)):
        engine.load_state_dict(checkpoint)
