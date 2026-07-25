"""Regression coverage for the facade hardening requested for the 4.0 release."""

import copy
import math
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.transforms import WHOLE


@pytest.fixture(autouse=True)
def _eager_compile():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


@pytest.mark.parametrize(
    ("facade", "name", "value"),
    (
        (heavyball.AdamW, "lr", math.nan),
        (heavyball.AdamW, "beta1", 1.5),
        (heavyball.AdamW, "beta2", -0.5),
        (heavyball.AdamW, "eps", -1.0),
        (heavyball.AdamW, "weight_decay", -0.1),
        (heavyball.AdamC, "max_lr", -0.1),
        (heavyball.MARSAdamW, "mars_gamma", math.inf),
    ),
)
def test_constructor_rejects_invalid_hyperparameter_domains(facade, name, value):
    parameter = torch.nn.Parameter(torch.ones(2))

    with pytest.raises((TypeError, ValueError)) as raised:
        facade([parameter], **{name: value})

    message = str(raised.value)
    assert facade.__name__ in message
    assert repr(name) in message
    assert repr(value) in message
    assert parameter.grad is None
    assert torch.equal(parameter, torch.ones_like(parameter))


def test_param_group_hyper_validation_precedes_engine_or_parameter_mutation():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = heavyball.AdamW([parameter], lr=0.1)
    parameter.grad.fill_(1)
    engine_lr = optimizer._engine.groups[0].hyper.lr.clone()
    before = parameter.detach().clone()
    optimizer.param_groups[0]["lr"] = math.nan

    with pytest.raises(ValueError, match=r"AdamW.*'lr'=nan.*finite"):
        optimizer.step()

    assert torch.equal(optimizer._engine.groups[0].hyper.lr, engine_lr)
    assert torch.equal(parameter, before)


def test_inherited_max_lr_sentinel_is_only_valid_during_construction():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = heavyball.AdamC([parameter], lr=0.1)
    parameter.grad.fill_(1)
    before = parameter.detach().clone()
    optimizer.param_groups[0]["max_lr"] = None

    with pytest.raises(TypeError, match=r"AdamC.*'max_lr'=None.*real scalar"):
        optimizer.step()

    assert torch.equal(parameter, before)


def test_splitopt_publishes_live_child_groups_for_schedulers():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    optimizer = heavyball.SplitOpt(
        [
            {"params": [first], "optimizer": heavyball.AdamW, "lr": 0.01},
            {"params": [second], "optimizer": heavyball.SGD, "lr": 0.02},
        ]
    )

    assert optimizer.param_groups == [
        optimizer.optimizers[0].param_groups[0],
        optimizer.optimizers[1].param_groups[0],
    ]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    first.grad.fill_(1)
    second.grad.fill_(1)
    optimizer.step(observed=(False, False))
    scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == [0.005, 0.01]
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.005
    assert optimizer.optimizers[1].param_groups[0]["lr"] == 0.01


def test_splitopt_delegates_modes_and_partitions_observed_mapping():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    optimizer = heavyball.SplitOpt(
        [
            {"params": [first], "optimizer": heavyball.SFAdamW, "lr": 0.01},
            {"params": [second], "optimizer": heavyball.SGD, "lr": 0.02},
        ]
    )

    assert optimizer.eval() is optimizer
    assert all(not child._engine._train_mode for child in optimizer.optimizers)
    assert optimizer.train() is optimizer
    assert all(child._engine._train_mode for child in optimizer.optimizers)

    first.grad.fill_(1)
    second.grad.fill_(1)
    optimizer.step(observed={first: True, second: False})

    assert not torch.equal(first, torch.zeros_like(first))
    assert torch.equal(second, torch.zeros_like(second))


def test_psgd_lra_is_whole_scoped_for_builtin_and_public_factory():
    assert heavyball.psgd_lra.distributed_scope == WHOLE
    assert heavyball.make_psgd_lra(rank=3).distributed_scope == WHOLE


@pytest.mark.parametrize(
    "facade",
    (
        heavyball.TrueGradAdam,
        heavyball.TrueGradRMSprop,
        heavyball.TrueGradLaProp,
        heavyball.TrueGradNAdam,
    ),
)
def test_observation_bearing_facades_reject_fsdp2_before_binding(facade):
    with pytest.raises(
        ValueError,
        match=rf"{facade.__name__}\.fsdp2.*observation-bearing.*not supported",
    ):
        facade.fsdp2(torch.nn.Linear(2, 2))


def test_rebound_ddp_style_gradient_names_required_setting():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = heavyball.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)

    with pytest.raises(ValueError, match=r"gradient_as_bucket_view=False"):
        optimizer.step()


def test_registry_reports_declarative_contract_and_lifecycle_metadata():
    canonical_names = heavyball.list_optimizers()
    all_names = heavyball.list_optimizers(include_aliases=True)
    assert canonical_names == sorted(canonical_names)
    assert set(all_names) - set(canonical_names) == {"ScheduleFree", "WhitenAdamW"}

    schedule_free = heavyball.describe("ScheduleFree")
    assert schedule_free["name"] == "ScheduleFree"
    assert schedule_free["canonical_name"] == "SFAdamW"
    assert schedule_free["aliases"] == ["ScheduleFree"]
    assert schedule_free["recipe"] == heavyball.describe("SFAdamW")["recipe"]

    assert heavyball.describe("SOAP")["recipe"] == {
        "when": "matrix_route",
        "then": {
            "transforms": ["soap"],
            "commit": "sgd_commit",
        },
        "otherwise": {
            "transforms": ["adam"],
            "commit": "sgd_commit",
        },
    }

    scion = heavyball.describe("Scion")
    assert "reinitializes" in scion["lifecycle"]
    assert "first observed step only" in heavyball.describe("ADOPT")["lifecycle"]
    assert "bootstraps" in heavyball.describe("SUDSAdamW")["lifecycle"]
    assert heavyball.describe("AdamW")["distributed_modes"] == {
        "single_process": True,
        "DDP": True,
        "FSDP2": True,
    }
    assert heavyball.describe("TrueGradAdam")["distributed_modes"]["FSDP2"] is False
    assert "gradient_as_bucket_view=False" in " ".join(
        heavyball.describe("AdamW")["distributed_limitations"]
    )


def test_estimate_state_bytes_counts_storage_and_ecc_slabs_without_mutation():
    parameters = [
        torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        torch.nn.Parameter(torch.arange(3, dtype=torch.float32)),
    ]
    before = [parameter.detach().clone() for parameter in parameters]
    total_elements = sum(parameter.numel() for parameter in parameters)

    assert heavyball.estimate_state_bytes(parameters, heavyball.SGD) == 0
    assert heavyball.estimate_state_bytes(parameters, "AdamW") == total_elements * 8
    assert (
        heavyball.estimate_state_bytes(
            parameters,
            "AdamW",
            storage_dtype=torch.bfloat16,
        )
        == total_elements * 4
    )
    assert (
        heavyball.estimate_state_bytes(
            parameters,
            "AdamW",
            storage_dtype=torch.bfloat16,
            ecc=8,
        )
        == total_elements * 6
    )
    assert (
        heavyball.estimate_state_bytes(
            parameters,
            "AdamW",
            storage_dtype=torch.bfloat16,
            ecc=16,
        )
        == total_elements * 8
    )
    for parameter, original in zip(parameters, before, strict=True):
        assert torch.equal(parameter, original)
        assert parameter.grad is None


def _dcp_payload(optimizer):
    engine = optimizer._engine
    cadence = None
    if engine._cadence is not None:
        cadence = {
            "probability": engine._cadence.probability,
            "step": engine._cadence.step,
            "cumulative": engine._cadence.cumulative,
            "compensation": engine._cadence.compensation,
        }
    leaf_indices = {
        engine._param_keys[id(param)]: int(group.leaf_indices[index])
        for group in engine.groups
        for index, param in enumerate(group.params)
    }
    engine_metadata = {
        "schema": "test",
        "train_mode": engine._train_mode,
        "cadence": cadence,
        "rng": {"seed": engine._rng_seed, "leaf_indices": leaf_indices},
        "param_init_pending": dict(engine._deferred_param_init_pending),
    }
    param_groups = tuple(
        {
            "param_count": len(group["params"]),
            "values": {name: value for name, value in group.items() if name != "params"},
        }
        for group in optimizer.param_groups
    )
    return {
        "metadata": {
            "format": 1,
            "param_groups": param_groups,
            "engines": (engine_metadata,),
        },
        "engines": {
            "0": {
                "rng_seeds": {
                    str(index): value.detach().clone()
                    for index, value in enumerate(engine._rng_seeds.values())
                },
                "groups": {
                    str(index): {
                        "age": group.age.detach().clone(),
                        "leaf_indices": group.leaf_indices.detach().clone(),
                    }
                    for index, group in enumerate(engine.groups)
                },
                "steps": {
                    str(index): value.detach().clone()
                    for index, value in enumerate(engine._steps)
                },
            }
        },
    }


def _assert_rejected_dcp_does_not_copy(
    monkeypatch,
    optimizer,
    mutate_checkpoint,
    expected_error,
):
    state_dict = _dcp_payload(optimizer)
    saved_metadata = copy.deepcopy(state_dict["metadata"])
    mutate_checkpoint(saved_metadata, state_dict)
    target = torch.tensor(7.0)
    staged = torch.tensor(-3.0)

    monkeypatch.setattr(
        optimizer,
        "_dcp_state_dict",
        lambda *, staging: (state_dict, [(target, staged)]),
    )
    import torch.distributed.checkpoint as dcp

    def fake_load(loaded, *, checkpoint_id):
        del checkpoint_id
        loaded["metadata"] = saved_metadata

    monkeypatch.setattr(dcp, "load", fake_load)
    with pytest.raises((TypeError, ValueError), match=expected_error):
        optimizer.dcp_load("unused")
    assert target.item() == 7.0


@pytest.mark.parametrize("invalid", (math.nan, "not-a-number"))
def test_dcp_rejects_bad_public_hyper_before_tensor_copy(monkeypatch, invalid):
    optimizer = heavyball.AdamW([torch.nn.Parameter(torch.ones(2))])

    def mutate(metadata, _state_dict):
        metadata["param_groups"][0]["values"]["lr"] = invalid

    _assert_rejected_dcp_does_not_copy(
        monkeypatch,
        optimizer,
        mutate,
        r"AdamW.*'lr'",
    )


def test_dcp_rejects_unresolved_max_lr_before_tensor_copy(monkeypatch):
    optimizer = heavyball.AdamC([torch.nn.Parameter(torch.ones(2))], lr=0.1)

    def mutate(metadata, _state_dict):
        metadata["param_groups"][0]["values"]["max_lr"] = None

    _assert_rejected_dcp_does_not_copy(
        monkeypatch,
        optimizer,
        mutate,
        r"AdamC.*'max_lr'=None",
    )


@pytest.mark.parametrize("field", ("cumulative", "compensation"))
def test_dcp_rejects_nonfinite_cadence_before_tensor_copy(monkeypatch, field):
    optimizer = heavyball.PSGDLRA(
        [torch.nn.Parameter(torch.ones(2))],
        preconditioner_update_probability=0.5,
    )

    def mutate(metadata, _state_dict):
        metadata["engines"][0]["cadence"][field] = math.nan

    _assert_rejected_dcp_does_not_copy(
        monkeypatch,
        optimizer,
        mutate,
        "cadence state",
    )


@pytest.mark.parametrize("invalid_age", (-1, 1))
def test_dcp_rejects_negative_or_step_unbounded_age_before_copy(
    monkeypatch,
    invalid_age,
):
    optimizer = heavyball.AdamW([torch.nn.Parameter(torch.ones(2))])

    def mutate(_metadata, state_dict):
        state_dict["engines"]["0"]["groups"]["0"]["age"].fill_(invalid_age)

    _assert_rejected_dcp_does_not_copy(
        monkeypatch,
        optimizer,
        mutate,
        "ages must be non-negative and smaller than their step counter",
    )
