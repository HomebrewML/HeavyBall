"""Proofs for HeavyBall 4.0's slab-native optimizer core."""

import inspect
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import heavyball
import reference
from heavyball import (
    Engine, ParamInfo, Recipe, Route, SAM, adam, adamw, ademamix, adopt, build, caution, cautious_adamw,
    laprop, laprop_ortho, laprop_transform, lion, mars, mars_adamw, momentum, muon, muon_commit, muon_laprop,
    nadam, ortho_laprop, orthogonalize, orthograd, orthograd_adamw, rmsprop, sgd, sgd_commit, shampoo,
    shampoo_adamw, sign_laprop, signsgd, soap, soap_adamw, truegrad_adam, unscaled_adamw, whiten_adamw, whitening,
)
from heavyball.transforms import Tempo

VALUES = dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03)


def _copy_grads(params, grads):
    for param, grad in zip(params, grads, strict=True):
        param.grad.copy_(grad)


def _fresh_params(values):
    return [torch.nn.Parameter(value.detach().clone()) for value in values]


_PORTED_ELEMENTWISE_CASES = (
    ("rmsprop", rmsprop, dict(lr=1.0, beta2=0.99, eps=1e-6, weight_decay=0.03), 30),
    ("lion", lion, dict(lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.03), 31),
    ("laprop", laprop, dict(lr=1.0, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03), 32),
    (
        "nadam",
        nadam,
        dict(lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, momentum_decay=4e-3, weight_decay=0.03),
        33,
    ),
    (
        "ademamix",
        ademamix,
        dict(
            lr=1.0,
            beta1=0.9,
            beta2=0.999,
            beta3=0.9999,
            eps=1e-8,
            alpha=2.0,
            beta3_warmup=8,
            alpha_warmup=6,
            weight_decay=0.03,
        ),
        34,
    ),
    (
        "unscaled_adamw",
        unscaled_adamw,
        dict(lr=1.0, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        35,
    ),
    ("signsgd", signsgd, dict(lr=0.1, weight_decay=0.03), 36),
    (
        "mars_adamw",
        mars_adamw,
        dict(lr=1.0, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03, mars_gamma=0.0025),
        37,
    ),
    (
        "orthograd_adamw",
        orthograd_adamw,
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        38,
    ),
    (
        "cautious_adamw",
        cautious_adamw,
        # Legacy applies decoupled decay outside caution; zero decay isolates their shared modifier math.
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
        39,
    ),
    (
        "ortho_laprop",
        ortho_laprop,
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        40,
    ),
    (
        "laprop_ortho",
        laprop_ortho,
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        41,
    ),
    (
        "sign_laprop",
        sign_laprop,
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        42,
    ),
    (
        "muon_laprop",
        muon_laprop,
        dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        43,
    ),
)


@pytest.mark.parametrize(
    ("name", "recipe", "values", "seed"),
    _PORTED_ELEMENTWISE_CASES,
    ids=[case[0] for case in _PORTED_ELEMENTWISE_CASES],
)
def test_elementwise_port_compiles_and_runs_finite(name, recipe, values, seed):
    torch._dynamo.reset()
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(3, 2)) for _ in range(2)]
    optimizer = build(params, recipe, **values)
    assert optimizer.compiled_step is optimizer.compiled_steps["normal"]
    assert all(value.shape[0] == len(params) for state in optimizer.groups[0].states for value in state.values())
    for param in params:
        param.grad.normal_()
    optimizer.step()
    assert all(torch.isfinite(param).all() for param in params), name
    torch._dynamo.reset()


def test_ademamix_warmups_follow_leaf_age():
    torch._dynamo.reset()
    torch.manual_seed(37)
    values = dict(
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        eps=1e-8,
        alpha=2.0,
        beta3_warmup=8,
        alpha_warmup=6,
        weight_decay=0.0,
    )
    leader = torch.nn.Parameter(torch.randn(3, 2))
    late = torch.nn.Parameter(torch.randn(3, 2))
    fresh = torch.nn.Parameter(late.detach().clone())
    optimizer = build([leader, late], ademamix, **values)
    fresh_optimizer = build([fresh], ademamix, **values)

    for _ in range(4):
        leader.grad.normal_()
        late.grad.normal_()
        optimizer.step(observed=[True, False])

    for _ in range(3):
        gradient = torch.randn_like(late)
        leader.grad.normal_()
        late.grad.copy_(gradient)
        fresh.grad.copy_(gradient)
        optimizer.step(observed=[True, True])
        fresh_optimizer.step()

    assert torch.equal(optimizer.groups[0].age[1], fresh_optimizer.groups[0].age[0])
    for name, value in optimizer.state[late].items():
        assert torch.equal(value, fresh_optimizer.state[fresh][name]), name
    assert torch.equal(late, fresh)
    torch._dynamo.reset()


def _copy_keyed_grads(params, keys, grads):
    for param, key in zip(params, keys, strict=True):
        param.grad.copy_(grads[key])


def _empty_init(ref_leaf):
    del ref_leaf
    return {}


def _orthogonalized(update):
    output, _, _ = orthogonalize(
        update,
        None,
        None,
        {},
        Tempo(
            torch.ones((), dtype=torch.long, device=update.device),
            torch.ones(update.shape[0], dtype=torch.int64, device=update.device),
            torch.ones(update.shape[0], dtype=torch.bool, device=update.device),
            SimpleNamespace(),
            False,
        ),
    )
    return output


def test_orthograd_boundary_cases():
    param = torch.tensor(((0.0, 0.0, 0.0, 0.0), (1.0, -2.0, 3.0, -4.0), (1.0, -2.0, 3.0, -4.0)))
    update = torch.tensor(((1.0, -2.0, 3.0, -4.0), (0.0, 0.0, 0.0, 0.0), (2.0, -4.0, 6.0, -8.0)))
    eps = torch.tensor(1e-8)
    result, _, _ = orthograd(
        update,
        None,
        param,
        {},
        Tempo(
            torch.ones((), dtype=torch.long),
            torch.ones(param.shape[0], dtype=torch.int64),
            torch.ones(param.shape[0], dtype=torch.bool),
            SimpleNamespace(eps=eps),
            False,
        ),
    )
    # Zero param leaves the update untouched; a zero or param-parallel update projects to nothing.
    torch.testing.assert_close(result[0], update[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(result[1], torch.zeros_like(result[1]), rtol=0, atol=1e-6)
    torch.testing.assert_close(result[2], torch.zeros_like(result[2]), rtol=0, atol=1e-6)


def scale2(update, obs, param, state, tempo):
    del obs, param, state
    return update * 2, {}, tempo.live


scale2.init = _empty_init


def add_one(update, obs, param, state, tempo):
    del obs, param, state
    return update + torch.ones_like(update), {}, tempo.live


add_one.init = _empty_init


def _counter_init(ref_leaf):
    return {"counter": torch.zeros((), dtype=torch.long, device=ref_leaf.device)}


def dummy_stateful(update, obs, param, state, tempo):
    del obs, param
    return update, {"counter": state["counter"] + 1}, tempo.live


dummy_stateful.init = _counter_init


def _counter_commit_init(ref_leaf):
    return {"count": torch.zeros((), dtype=torch.long, device=ref_leaf.device)}


def _counter_commit(param, update, state, tempo):
    return param - tempo.hyper.lr * update, {"count": state["count"] + 1}


_counter_commit.init = _counter_commit_init


def _swap_state_init(ref_leaf):
    return {
        "a": torch.tensor(0.0, device=ref_leaf.device),
        "b": torch.tensor(1.0, device=ref_leaf.device),
    }


def _swap_transform_state(update, obs, param, state, tempo):
    del obs, param
    return update, {"a": state["b"], "b": state["a"]}, tempo.live


_swap_transform_state.init = _swap_state_init


def _swap_commit_state(param, update, state, tempo):
    del update, tempo
    return param, {"a": state["b"], "b": state["a"]}


_swap_commit_state.init = _swap_state_init


def _snapshot_init(ref_leaf):
    return {
        "snapshot": ref_leaf.detach().clone(),
        "saw_requires_grad": torch.tensor(ref_leaf.requires_grad, device=ref_leaf.device),
    }


def _keep_snapshot(update, obs, param, state, tempo):
    del obs, param
    return update, {
        "snapshot": state["snapshot"],
        "saw_requires_grad": state["saw_requires_grad"],
    }, tempo.live


_keep_snapshot.init = _snapshot_init


def use_stat(update, obs, param, state, tempo):
    del update, param, state
    return obs.my_stat, {}, tempo.live


use_stat.init = _empty_init


def use_grad_squared(update, obs, param, state, tempo):
    del update, param, state
    return obs.grad * obs.grad, {}, tempo.live


use_grad_squared.init = _empty_init


def use_absent_observation(update, obs, param, state, tempo):
    del update, param, state
    return obs.absent, {}, tempo.live


use_absent_observation.init = _empty_init


@pytest.mark.parametrize("recipe", (sgd, adopt), ids=("sgd", "adopt"))
def test_sgd_commit_weight_decay_is_settable_and_zero_is_a_noop(recipe):
    initial = torch.tensor((2.0, -4.0))
    decayed_param = torch.nn.Parameter(initial.clone())
    decayed = build([decayed_param], recipe, lr=0.25, weight_decay=0.1)
    decayed_param.grad.zero_()
    decayed.step()
    decayed_param.grad.zero_()
    decayed.step()
    assert torch.all(decayed_param.abs() < initial.abs())

    default_param = torch.nn.Parameter(initial.clone())
    zero_param = torch.nn.Parameter(initial.clone())
    default = build([default_param], recipe, lr=0.25)
    zero = build([zero_param], recipe, lr=0.25, weight_decay=0.0)
    for gradient in (torch.tensor((0.5, -0.25)), torch.tensor((-0.75, 1.0))):
        default_param.grad.copy_(gradient)
        zero_param.grad.copy_(gradient)
        default.step()
        zero.step()

    assert torch.equal(default_param, zero_param)
    for default_group, zero_group in zip(default.groups, zero.groups, strict=True):
        for name in ("param_slab", "grad_slab", "observed", "age", "step"):
            assert torch.equal(getattr(default_group, name), getattr(zero_group, name))
        for default_slots, zero_slots in zip(default_group.states, zero_group.states, strict=True):
            for name in default_slots:
                assert torch.equal(default_slots[name], zero_slots[name])
        for name in default_group.commit_state:
            assert torch.equal(default_group.commit_state[name], zero_group.commit_state[name])
        for name in vars(default_group.hyper):
            assert torch.equal(getattr(default_group.hyper, name), getattr(zero_group.hyper, name))


def test_composition_consumes_update():
    torch._dynamo.reset()
    values = {**VALUES, "eps": 1.0, "weight_decay": 0.0}
    base_param = torch.nn.Parameter(torch.zeros(1))
    scaled_param = torch.nn.Parameter(torch.zeros(1))
    base = Engine([base_param], replace(adamw, chain=(adam,)), **values)
    scaled = Engine([scaled_param], replace(adamw, chain=(scale2, adam)), **values)

    base_param.grad.fill_(0.5)
    scaled_param.grad.fill_(0.5)
    base.step()
    scaled.step()

    assert not torch.equal(base_param, scaled_param)
    torch._dynamo.reset()


def test_caution_composes_after_adam_and_masks_disagreeing_entries():
    torch._dynamo.reset()
    values = dict(lr=1.0, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0)
    plain_param = torch.nn.Parameter(torch.zeros(4))
    cautious_param = torch.nn.Parameter(torch.zeros(4))
    plain = build([plain_param], replace(adamw, chain=(adam,)), **values)
    cautious = build([cautious_param], cautious_adamw, **values)
    assert cautious.groups[0].recipe.chain == (adam, caution)

    first_grad = torch.tensor((1.0, -1.0, 1.0, -1.0))
    second_grad = torch.tensor((-0.01, -2.0, 2.0, 0.01))
    plain_param.grad.copy_(first_grad)
    cautious_param.grad.copy_(first_grad)
    plain.step()
    cautious.step()

    plain_param.grad.copy_(second_grad)
    cautious_param.grad.copy_(second_grad)
    plain_before = plain_param.detach().clone()
    cautious_before = cautious_param.detach().clone()
    plain.step()
    cautious.step()

    plain_update = (plain_before - plain_param) / values["lr"]
    cautious_update = (cautious_before - cautious_param) / values["lr"]
    aligned = ((second_grad > 0) & (plain_update > 0)) | ((second_grad < 0) & (plain_update < 0))
    scale = aligned.numel() / aligned.sum().clamp_min(1)
    expected = torch.where(aligned, plain_update, torch.zeros_like(plain_update)) * scale

    assert torch.equal(aligned, torch.tensor((False, True, True, False)))
    assert torch.equal(cautious_update[~aligned], torch.zeros_like(cautious_update[~aligned]))
    torch.testing.assert_close(cautious_update, expected, rtol=1e-6, atol=1e-6)
    torch._dynamo.reset()


def test_whiten_preconditions_update_stream():
    gradient = torch.tensor(((2.0, 0.5), (0.25, 3.0)))
    base_param = torch.nn.Parameter(torch.zeros_like(gradient))
    transformed_param = torch.nn.Parameter(torch.zeros_like(gradient))
    values = dict(lr=0.1, eps=1e-6)
    base = Engine([base_param], whitening, **values)
    transformed = Engine([transformed_param], replace(whitening, chain=(add_one, *whitening.chain)), **values)
    base_before = base_param.detach().clone()
    transformed_before = transformed_param.detach().clone()

    base_param.grad.copy_(gradient)
    transformed_param.grad.copy_(gradient)
    base.step(step_type="refresh")
    transformed.step(step_type="refresh")

    base_group = base.groups[0]
    transformed_group = transformed.groups[0]
    torch.testing.assert_close(base_group.states[0]["GG"], transformed_group.states[1]["GG"])
    torch.testing.assert_close(base_group.states[0]["Q"], transformed_group.states[1]["Q"])
    assert not torch.equal(base_param, transformed_param)
    factor = transformed_group.states[1]["Q"]
    transformed_update = gradient + torch.ones_like(gradient)
    if bool((transformed_group.states[1]["GG_scale"] < 0).all()):
        expected = factor @ (factor.mT @ transformed_update)
    else:
        expected = factor @ transformed_update
    torch.testing.assert_close((transformed_before - transformed_param) / values["lr"], expected[0])
    base_factor = base_group.states[0]["Q"]
    if bool((base_group.states[0]["GG_scale"] < 0).all()):
        base_expected = base_factor @ (base_factor.mT @ gradient)
    else:
        base_expected = base_factor @ gradient
    torch.testing.assert_close(
        (base_before - base_param) / values["lr"], base_expected[0]
    )


def test_failed_whitening_construction_preserves_parameter_bindings():
    param = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    original_pointer = param.data_ptr()
    original_grad = param.grad

    with (
        patch("heavyball.core.torch.compile", lambda function, **kwargs: function),
        pytest.raises(ValueError, match="whiten requires square"),
    ):
        Engine([param], whitening)

    assert param.data_ptr() == original_pointer
    assert param.grad is original_grad


def test_later_bucket_init_failure_restores_all_external_bindings():
    observed = torch.nn.Parameter(torch.zeros(1))
    rejected = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    params = (observed, rejected)
    original_pointers = tuple(param.data_ptr() for param in params)
    original_grads = tuple(torch.ones_like(param) for param in params)
    for param, grad in zip(params, original_grads, strict=True):
        param.grad = grad
    original_binding = object()
    original_observation = object()
    observed._heavyball_observation_binding = original_binding
    observed.sum_grad_squared = original_observation
    recipe = Route(lambda info: info.shape == (1,), truegrad_adam, whitening)

    with (
        patch("heavyball.core.torch.compile", lambda function, **kwargs: function),
        pytest.raises(ValueError, match="whiten requires square"),
    ):
        Engine(params, recipe)

    assert tuple(param.data_ptr() for param in params) == original_pointers
    assert all(param.grad is grad for param, grad in zip(params, original_grads, strict=True))
    assert observed._heavyball_observation_binding is original_binding
    assert observed.sum_grad_squared is original_observation


def test_late_construction_failure_restores_all_external_bindings():
    param = torch.nn.Parameter(torch.zeros(1))
    original_pointer = param.data_ptr()
    original_grad = torch.ones_like(param)
    param.grad = original_grad
    original_binding = object()
    original_observation = object()
    param._heavyball_observation_binding = original_binding
    param.sum_grad_squared = original_observation

    with (
        patch("heavyball.core.torch.compile", side_effect=RuntimeError("compile failed")),
        pytest.raises(RuntimeError, match="compile failed"),
    ):
        Engine([param], truegrad_adam)

    assert param.data_ptr() == original_pointer
    assert param.grad is original_grad
    assert param._heavyball_observation_binding is original_binding
    assert param.sum_grad_squared is original_observation


def test_lazy_downstream_state_gated():
    torch._dynamo.reset()
    param = torch.nn.Parameter(torch.zeros(1))
    recipe = replace(adopt, chain=(*adopt.chain, dummy_stateful))
    optimizer = Engine([param], recipe, **{key: value for key, value in VALUES.items() if key != "weight_decay"})
    counter = optimizer.groups[0].states[1]["counter"]

    param.grad.fill_(1)
    optimizer.step()
    assert torch.equal(counter, torch.zeros_like(counter))

    optimizer.step()
    assert torch.equal(counter, torch.ones_like(counter))

    optimizer.step()
    assert torch.equal(counter, torch.full_like(counter, 2))
    torch._dynamo.reset()


def test_stateful_commit_advances_its_slab_without_changing_sgd_math():
    stateful_param = torch.nn.Parameter(torch.tensor((2.0, -1.0)))
    plain_param = torch.nn.Parameter(stateful_param.detach().clone())
    stateful_recipe = Recipe(chain=(), commit=_counter_commit, defaults=dict(lr=0.25))
    stateful = Engine([stateful_param], stateful_recipe)
    plain = Engine([plain_param], sgd, lr=0.25)
    counter = stateful.groups[0].commit_state["count"]

    gradient = torch.tensor((3.0, -4.0))
    stateful_param.grad.copy_(gradient)
    plain_param.grad.copy_(gradient)
    stateful.step()
    plain.step()

    assert torch.equal(counter, torch.ones_like(counter))
    torch.testing.assert_close(stateful_param, plain_param, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
@pytest.mark.parametrize("state_owner", ("transform", "commit"))
def test_step_materializes_aliased_state_candidates(compiled, state_owner):
    param = torch.nn.Parameter(torch.zeros(1))
    if state_owner == "transform":
        recipe = Recipe(
            chain=(_swap_transform_state,),
            commit=sgd_commit,
            defaults=dict(lr=0.0, weight_decay=0.0),
        )
    else:
        recipe = Recipe(chain=(), commit=_swap_commit_state, defaults={})

    if compiled:
        optimizer = Engine([param], recipe)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine([param], recipe)
    param.grad.zero_()
    optimizer.step()

    slots = optimizer.groups[0].states[0] if state_owner == "transform" else optimizer.groups[0].commit_state
    assert (slots["a"].item(), slots["b"].item()) == (1.0, 0.0)


def test_transform_init_uses_each_leaf_value():
    first = torch.nn.Parameter(torch.tensor((1.0, 2.0)))
    second = torch.nn.Parameter(torch.tensor((7.0, 9.0)))
    recipe = Recipe(
        chain=(_keep_snapshot,),
        commit=sgd_commit,
        defaults=dict(lr=0.0, weight_decay=0.0),
    )

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([first, second], recipe)

    assert torch.equal(optimizer.state[first]["snapshot"], torch.tensor((1.0, 2.0)))
    assert torch.equal(optimizer.state[second]["snapshot"], torch.tensor((7.0, 9.0)))
    assert optimizer.state[first]["saw_requires_grad"].item()
    assert optimizer.state[second]["saw_requires_grad"].item()


def test_compiled_graph_clean(tmp_path):
    source = """
import torch
from heavyball import Engine, adamw

params = [torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.randn(3))]
optimizer = Engine(params, adamw, clip_global_norm=1.0)
for param in params:
    param.grad.normal_()
optimizer.step()
"""
    env = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).parents[1], env=env,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Output code:" in output
    assert output.count("AOT ID:") == 1
    assert "stack" not in output
    assert "_local_scalar_dense" not in output
    assert ".item(" not in output
    assert "synchronize" not in output


def _refresh_output_code(tmp_path, step_type, recipe="whitening", shape=(3, 3)):
    source = f"""
import torch
from heavyball import Engine, {recipe}

params = [torch.nn.Parameter(torch.randn({shape[0]}, {shape[1]})) for _ in range(2)]
optimizer = Engine(params, {recipe})
optimizer.groups[0].grad_slab.normal_()
optimizer.step(step_type={step_type!r})
"""
    env = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / f"{recipe}_{step_type}"))
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).parents[1], env=env,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    paths = re.findall(r"Output code written to: (.*\.py)", output)
    assert paths, output
    return Path(paths[-1]).read_text()


def test_refresh_artifact_has_eigh_normal_does_not(tmp_path):
    normal_code = _refresh_output_code(tmp_path, "normal")
    refresh_code = _refresh_output_code(tmp_path, "refresh")

    assert "eigh" not in normal_code.lower()
    assert "eigh" in refresh_code.lower()
    for code in (normal_code, refresh_code):
        assert "stack" not in code
        assert "_local_scalar_dense" not in code
        assert ".item(" not in code


def test_host_dispatch_updates_Q_on_refresh_only():
    params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)]
    optimizer = Engine(params, whitening, lr=1e-2, eps=1e-6)
    group = optimizer.groups[0]
    gradient = torch.diag(torch.tensor((2.0, 3.0, 4.0)))
    group.grad_slab.copy_(gradient)
    expected_gg = torch.zeros_like(group.states[0]["GG"])
    initial_q = group.states[0]["Q"].detach().clone()

    optimizer.step(step_type="normal")
    expected_gg += group.grad_slab @ group.grad_slab.mT
    torch.testing.assert_close(group.states[0]["Q"], initial_q, rtol=0, atol=0)
    torch.testing.assert_close(group.states[0]["GG"], expected_gg)

    optimizer.step(step_type="refresh")
    expected_gg += group.grad_slab @ group.grad_slab.mT
    refreshed_q = group.states[0]["Q"].detach().clone()
    assert not torch.equal(refreshed_q, initial_q)
    torch.testing.assert_close(group.states[0]["GG"], expected_gg)

    optimizer.step(step_type="normal")
    expected_gg += group.grad_slab @ group.grad_slab.mT
    torch.testing.assert_close(group.states[0]["Q"], refreshed_q, rtol=0, atol=0)
    torch.testing.assert_close(group.states[0]["GG"], expected_gg)


def test_shampoo_nonuniform_state_shapes():
    params = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(3)]
    state = Engine(params, shampoo).groups[0].states[0]

    assert {name: tuple(value.shape) for name, value in state.items()} == {
        "GG_l": (3, 3, 3),
        "GG_l_scale": (3,),
        "GG_r": (3, 5, 5),
        "GG_r_scale": (3,),
        "L": (3, 3, 3),
        "R": (3, 5, 5),
    }
    torch.testing.assert_close(state["L"], torch.eye(3).expand_as(state["L"]), rtol=0, atol=0)
    torch.testing.assert_close(state["R"], torch.eye(5).expand_as(state["R"]), rtol=0, atol=0)
    torch.testing.assert_close(state["GG_l"], torch.zeros_like(state["GG_l"]), rtol=0, atol=0)
    torch.testing.assert_close(state["GG_r"], torch.zeros_like(state["GG_r"]), rtol=0, atol=0)


def test_shampoo_refresh_only_eigh(tmp_path):
    torch._dynamo.reset()
    normal_code = _refresh_output_code(tmp_path, "normal", "shampoo", (3, 5))
    refresh_code = _refresh_output_code(tmp_path, "refresh", "shampoo", (3, 5))

    assert "eigh" not in normal_code.lower()
    assert "eigh" in refresh_code.lower()
    for code in (normal_code, refresh_code):
        assert "stack" not in code
        assert "_local_scalar_dense" not in code
        assert ".item(" not in code

    optimizer = Engine([torch.nn.Parameter(torch.randn(3, 5)) for _ in range(2)], shampoo, lr=1e-2, eps=1e-6)
    state = optimizer.groups[0].states[0]
    gradient = torch.arange(1, 31, dtype=torch.float32).reshape(2, 3, 5)
    optimizer.groups[0].grad_slab.copy_(gradient)
    gram_left, gram_right = gradient @ gradient.mT, gradient.mT @ gradient
    initial_l, initial_r = state["L"].detach().clone(), state["R"].detach().clone()

    optimizer.step(step_type="normal")
    torch.testing.assert_close(state["GG_l"], gram_left)
    torch.testing.assert_close(state["GG_r"], gram_right)
    assert torch.equal(state["L"], initial_l)
    assert torch.equal(state["R"], initial_r)

    optimizer.step(step_type="refresh")
    torch.testing.assert_close(state["GG_l"], gram_left * 2)
    torch.testing.assert_close(state["GG_r"], gram_right * 2)
    refreshed_l, refreshed_r = state["L"].detach().clone(), state["R"].detach().clone()
    assert not torch.equal(refreshed_l, initial_l)
    assert not torch.equal(refreshed_r, initial_r)

    optimizer.step(step_type="normal")
    torch.testing.assert_close(state["GG_l"], gram_left * 3)
    torch.testing.assert_close(state["GG_r"], gram_right * 3)
    assert torch.equal(state["L"], refreshed_l)
    assert torch.equal(state["R"], refreshed_r)
    torch._dynamo.reset()


def test_shampoo_preconditions_update_stream():
    torch._dynamo.reset()
    gradient = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5)
    base_param = torch.nn.Parameter(torch.zeros_like(gradient))
    transformed_param = torch.nn.Parameter(torch.zeros_like(gradient))
    values = dict(lr=0.1, eps=0.1)
    base = Engine([base_param], shampoo, **values)
    transformed = Engine([transformed_param], replace(shampoo, chain=(add_one, *shampoo.chain)), **values)
    transformed_before = transformed_param.detach().clone()

    base_param.grad.copy_(gradient)
    transformed_param.grad.copy_(gradient)
    base.step(step_type="refresh")
    transformed.step(step_type="refresh")

    base_state = base.groups[0].states[0]
    transformed_state = transformed.groups[0].states[1]
    base_update = gradient.unsqueeze(0)
    transformed_update = base_update + torch.ones_like(base_update)
    torch.testing.assert_close(
        base_state["GG_l"], base_update @ base_update.mT
    )
    torch.testing.assert_close(
        base_state["GG_r"], base_update.mT @ base_update
    )
    torch.testing.assert_close(
        transformed_state["GG_l"],
        transformed_update @ transformed_update.mT,
    )
    torch.testing.assert_close(
        transformed_state["GG_r"],
        transformed_update.mT @ transformed_update,
    )
    assert not torch.equal(base_state["GG_l"], transformed_state["GG_l"])
    assert not torch.equal(base_state["GG_r"], transformed_state["GG_r"])
    assert not torch.equal(base_param, transformed_param)
    if bool((transformed_state["GG_l_scale"] < 0).all()):
        expected = transformed_state["L"] @ (
            transformed_state["L"].mT @ transformed_update
        )
    else:
        expected = transformed_state["L"] @ transformed_update
    if bool((transformed_state["GG_r_scale"] < 0).all()):
        expected = (expected @ transformed_state["R"]) @ transformed_state["R"].mT
    else:
        expected = expected @ transformed_state["R"]
    torch.testing.assert_close((transformed_before - transformed_param) / values["lr"], expected[0])
    torch._dynamo.reset()


def test_shampoo_runs_finite():
    torch._dynamo.reset()
    torch.manual_seed(14)
    matrices = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(2)]
    vectors = [torch.nn.Parameter(torch.randn(5)) for _ in range(2)]
    params = [*matrices, *vectors]
    optimizer = build(params, shampoo_adamw, **{**VALUES, "eps": 1e-4})

    assert all({"GG_l", "GG_r", "L", "R"} <= optimizer.state[param].keys() for param in matrices)
    assert all({"exp_avg", "exp_avg_sq"} <= optimizer.state[param].keys() for param in vectors)
    for step in range(4):
        _copy_grads(params, [torch.randn_like(param) for param in params])
        optimizer.step(step_type="refresh" if step % 2 else "normal")
    assert all(torch.isfinite(param).all() for param in params)
    torch._dynamo.reset()


def test_soap_runs_finite():
    torch._dynamo.reset()
    torch.manual_seed(15)
    matrices = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(2)]
    vectors = [torch.nn.Parameter(torch.randn(5)) for _ in range(2)]
    params = [*matrices, *vectors]
    optimizer = build(params, soap_adamw, **{**VALUES, "eps": 1e-4})

    assert any(group.recipe is soap for group in optimizer.groups)
    assert all(
        {"GG_l", "GG_r", "Q_l", "Q_r", "exp_avg", "exp_avg_sq"} <= optimizer.state[param].keys()
        for param in matrices
    )
    assert all({"exp_avg", "exp_avg_sq"} <= optimizer.state[param].keys() for param in vectors)
    for step in range(4):
        _copy_grads(params, [torch.randn_like(param) for param in params])
        optimizer.step(step_type="refresh" if step % 2 else "normal")
    assert all(torch.isfinite(param).all() for param in params)
    torch._dynamo.reset()


def test_precond_runs_and_is_finite():
    torch.manual_seed(8)
    square = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)]
    other = [torch.nn.Parameter(torch.randn(2, 3)), torch.nn.Parameter(torch.randn(3))]
    params = [*square, *other]
    optimizer = build(params, whiten_adamw, **VALUES)

    assert all({"Q", "GG"} <= optimizer.state[param].keys() for param in square)
    assert all({"exp_avg", "exp_avg_sq"} <= optimizer.state[param].keys() for param in other)
    for step in range(4):
        _copy_grads(params, [torch.randn_like(param) for param in params])
        optimizer.step(step_type="refresh" if step % 2 else "normal")
    assert all(torch.isfinite(param).all() for param in params)


def test_both_variants_fullgraph():
    source = inspect.getsource(Engine._compile_step)
    assert 'torch.compile(whole_step, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")' in source

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    optimizer = Engine([torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)], whitening)
    optimizer.groups[0].grad_slab.normal_()
    optimizer.step(step_type="normal")
    optimizer.groups[0].grad_slab.normal_()
    optimizer.step(step_type="refresh")

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 2


def test_whole_step_has_no_scalar_gate():
    optimizer = Engine([torch.nn.Parameter(torch.randn(2, 2))], adamw)
    whole_step = optimizer.compiled_step.__wrapped__
    plans = next(
        cell.cell_contents
        for cell in whole_step.__closure__
        if isinstance(cell.cell_contents, tuple)
        and cell.cell_contents
        and isinstance(cell.cell_contents[0], tuple)
    )

    assert all(
        not (isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.ndim == 0)
        for plan in plans
        for value in plan
    )


def test_routing():
    torch.manual_seed(5)
    matrices = [torch.nn.Parameter(torch.randn(2, 3)) for _ in range(2)]
    vector = torch.nn.Parameter(torch.randn(3))
    params = [*matrices, vector]
    initials = [param.detach().clone() for param in matrices]
    seen = []

    def matrix_recipe(info):
        assert isinstance(info, ParamInfo)
        seen.append((info.shape, info.ndim, info.dtype))
        return info.ndim == 2

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    optimizer = build(params, Route(matrix_recipe, adamw, sgd), **VALUES)
    grads = [torch.randn_like(param) for param in params]
    vector_before = vector.detach().clone()
    _copy_grads(params, grads)
    optimizer.step()

    assert len(optimizer.groups) == 2
    assert any(group.recipe is adamw for group in optimizer.groups)
    assert any(group.recipe is sgd for group in optimizer.groups)
    assert seen == [(tuple(param.shape), param.ndim, param.dtype) for param in params]
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1

    for param, initial, grad in zip(matrices, initials, grads[:2], strict=True):
        expected = reference.adam(initial.double(), [grad.double()], **VALUES)
        torch.testing.assert_close(param.double(), expected, rtol=0, atol=1e-5)
    expected_vector = vector_before * (1 - VALUES["lr"] * VALUES["weight_decay"]) - VALUES["lr"] * grads[2]
    torch.testing.assert_close(vector, expected_vector, rtol=0, atol=1e-5)


def test_hyperparameter_values_must_be_numeric():
    param = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(TypeError, match="numbers or 0-d tensors"):
        Engine([param], adamw, lr="fast")
    with pytest.raises(ValueError, match="0-d tensors or Python scalars"):
        Engine([param], adamw, lr=[1.0, 2.0])


def test_hyperparameter_overrides_are_declared_by_recipes():
    param = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="beta_1"):
        Engine([param], adamw, beta_1=0.5)

    matrix_recipe = Recipe(chain=(), commit=sgd_commit, defaults=dict(lr=1.0, matrix_only=2.0))
    vector_recipe = Recipe(chain=(), commit=sgd_commit, defaults=dict(lr=3.0, vector_only=4.0))
    params = [torch.nn.Parameter(torch.zeros(1, 1)), torch.nn.Parameter(torch.zeros(1))]
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(
            params,
            Route(lambda info: info.ndim == 2, matrix_recipe, vector_recipe),
            lr=0.25,
        )

    matrix_group = next(group for group in optimizer.groups if group.recipe is matrix_recipe)
    vector_group = next(group for group in optimizer.groups if group.recipe is vector_recipe)
    assert torch.equal(matrix_group.hyper.lr, torch.tensor(0.25))
    assert torch.equal(vector_group.hyper.lr, torch.tensor(0.25))
    assert not hasattr(matrix_group.hyper, "vector_only")
    assert not hasattr(vector_group.hyper, "matrix_only")


def test_global_clip():
    torch.manual_seed(6)
    params = [torch.nn.Parameter(torch.randn(2, 3)), torch.nn.Parameter(torch.randn(3))]
    before = [param.detach().clone() for param in params]
    grads = [torch.randn_like(param) for param in params]
    max_norm, lr = 0.2, 0.1
    optimizer = Engine(params, replace(sgd, clip_global_norm=max_norm), lr=lr)
    _copy_grads(params, grads)
    optimizer.step()

    total = sum(grad.square().sum() for grad in grads)
    scale = (torch.tensor(max_norm) / (total.sqrt() + 1e-6)).clamp(max=1.0)
    for param, initial, update in zip(params, before, grads, strict=True):
        torch.testing.assert_close((initial - param) / lr, update * scale, rtol=0, atol=1e-5)


def test_global_clip_excludes_weight_decay():
    param = torch.nn.Parameter(torch.tensor((10.0, -20.0), dtype=torch.float64))
    before = param.detach().clone()
    grad = torch.tensor((3.0, 4.0), dtype=torch.float64)
    max_norm, lr, weight_decay = 0.5, 0.1, 0.2
    recipe = replace(adamw, chain=(), clip_global_norm=max_norm)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], recipe, lr=lr, weight_decay=weight_decay)
    param.grad.copy_(grad)
    optimizer.step()

    scale = (torch.tensor(max_norm, dtype=grad.dtype) / (grad.square().sum().sqrt() + 1e-6)).clamp(max=1.0)
    expected = before * (1 - lr * weight_decay) - lr * grad * scale
    torch.testing.assert_close(param, expected, rtol=0, atol=1e-12)


def test_unobserved_param_frozen():
    torch.manual_seed(13)
    params = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(3)]
    optimizer = Engine(params, adamw, **VALUES)
    before = [param.detach().clone() for param in params]
    frozen_state = {name: value.detach().clone() for name, value in optimizer.state[params[1]].items()}

    _copy_grads(params, [torch.randn_like(param) for param in params])
    optimizer.step(observed=[True, False, True])

    assert not torch.equal(params[0], before[0])
    assert torch.equal(params[1], before[1])
    assert not torch.equal(params[2], before[2])
    for name, value in optimizer.state[params[1]].items():
        assert torch.equal(value, frozen_state[name])

    before_unobserved = params[1].detach().clone()
    _copy_grads(params, [torch.randn_like(param) for param in params])
    optimizer.step(observed=[True, True, True])
    assert not torch.equal(params[1], before_unobserved)
    assert any(torch.count_nonzero(value) for value in optimizer.state[params[1]].values())


def test_unobserved_excluded_from_global_clip():
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    max_norm, lr = 2.0, 0.1
    optimizer = Engine(params, replace(sgd, clip_global_norm=max_norm), lr=lr)
    before = [param.detach().clone() for param in params]
    observed_grad = torch.tensor((3.0, 4.0))
    params[0].grad.copy_(observed_grad)
    params[1].grad.fill_(float("inf"))

    optimizer.step(observed=[True, False])

    scale = (torch.tensor(max_norm) / (observed_grad.square().sum().sqrt() + 1e-6)).clamp(max=1.0)
    torch.testing.assert_close((before[0] - params[0]) / lr, observed_grad * scale, rtol=0, atol=1e-5)
    assert torch.equal(params[1], before[1])
    assert torch.isfinite(params[0]).all()


def test_adopt_seed_excluded_from_global_clip():
    values = dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8)
    recipe = replace(adopt, clip_global_norm=0.05)
    seed = torch.nn.Parameter(torch.zeros(1))
    live = torch.nn.Parameter(torch.zeros(1))
    reference = torch.nn.Parameter(torch.zeros(1))

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([seed, live], recipe, **values)
        reference_optimizer = Engine([reference], recipe, **values)

    seed.grad.zero_()
    live.grad.fill_(1)
    reference.grad.fill_(1)
    optimizer.step(observed=[False, True])
    reference_optimizer.step()

    seed_before = seed.detach().clone()
    seed.grad.fill_(1e10)
    live.grad.fill_(1)
    reference.grad.fill_(1)
    optimizer.step()
    reference_optimizer.step()

    assert torch.equal(seed, seed_before)
    torch.testing.assert_close(live, reference, rtol=0, atol=0)


def test_global_clip_keeps_large_finite_gradients_finite():
    param = torch.nn.Parameter(torch.zeros(1))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], replace(sgd, clip_global_norm=1.0), lr=1.0)

    param.grad.fill_(2e19)
    optimizer.step()

    torch.testing.assert_close(param, torch.tensor((-1.0,)), rtol=0, atol=0)


def test_grads_slab_backed():
    layer = torch.nn.Linear(4, 3)
    params = list(layer.parameters())
    optimizer = Engine(params, adamw)
    loss = layer(torch.randn(5, 4)).square().sum()
    loss.backward()
    for param in params:
        group, index = next(
            (group, index)
            for group in optimizer.groups
            for index, leaf in enumerate(group.params)
            if leaf is param
        )
        assert param.grad.data_ptr() == group.grad_slab[index].data_ptr()
        assert torch.count_nonzero(group.grad_slab[index]) > 0
    before = [param.detach().clone() for param in params]
    optimizer.step()
    assert any(not torch.equal(param, old) for param, old in zip(params, before, strict=True))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_runtime_speed(capsys):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (512, 512) if device.type == "cuda" else (128, 128)
    params = [torch.nn.Parameter(torch.randn(shape, device=device)) for _ in range(64)]
    optimizer = Engine(params, adamw, lr=1e-3)
    group = optimizer.groups[0]
    for _ in range(3):
        group.grad_slab.normal_()
        optimizer.step()
    samples = []
    for _ in range(50):
        group.grad_slab.normal_()
        _sync()
        start = time.perf_counter()
        optimizer.step()
        _sync()
        samples.append(time.perf_counter() - start)
    median_ms = statistics.median(samples) * 1e3
    with capsys.disabled():
        print(f"opt compiled-step median: {median_ms:.3f} ms")
    assert median_ms > 0
    assert max(samples) < statistics.median(samples) * 20 + 0.01


def test_open_observation_no_core_edit():
    custom_param = torch.nn.Parameter(torch.zeros(2))
    squared_param = torch.nn.Parameter(torch.zeros(2))
    custom_recipe = Recipe(
        chain=(use_stat,), commit=sgd_commit, defaults=dict(lr=1.0, weight_decay=0.0), observations=("my_stat",)
    )
    squared_recipe = Recipe(chain=(use_grad_squared,), commit=sgd_commit, defaults=dict(lr=1.0, weight_decay=0.0))
    custom = Engine([custom_param], custom_recipe)
    squared = Engine([squared_param], squared_recipe)
    custom_group = custom.groups[0]

    custom_param.grad.fill_(2)
    custom.produce(custom_param, "my_stat", torch.full_like(custom_param, 9))
    squared_param.grad.fill_(2)
    assert custom_param.my_stat.data_ptr() == custom_group.observations.my_stat[0].data_ptr()

    custom.step()
    squared.step()

    custom_update = -custom_param.detach()
    squared_update = -squared_param.detach()
    assert torch.isfinite(custom_update).all()
    torch.testing.assert_close(custom_update, torch.full_like(custom_update, 9.0))
    torch.testing.assert_close(squared_update, torch.full_like(squared_update, 4.0))
    assert not torch.equal(custom_update, squared_update)

    custom.zero_grad()
    assert torch.equal(custom_param.grad, torch.zeros_like(custom_param.grad))
    assert torch.equal(custom_param.my_stat, torch.zeros_like(custom_param.my_stat))


def test_truegrad_observation_produce_preserves_slab_binding_and_is_required():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
    optimizer = Engine(params, truegrad_adam)
    group = optimizer.groups[0]
    with pytest.raises(ValueError, match="not a declared observation"):
        optimizer.produce(params[0], "wrong_name", torch.ones_like(params[0]))
    with pytest.raises(ValueError, match="does not match bound shape"):
        optimizer.produce(params[0], "sum_grad_squared", torch.ones(1))
    for index, param in enumerate(params):
        param.grad.normal_()
        slab_view = group.observations.sum_grad_squared[index]
        optimizer.produce(param, "sum_grad_squared", torch.full_like(param, 2.0))
        optimizer.produce(param, "sum_grad_squared", torch.ones_like(param))
        assert param.sum_grad_squared.data_ptr() == slab_view.data_ptr()
        torch.testing.assert_close(slab_view, torch.full_like(slab_view, 3.0))
    optimizer.step()
    optimizer.zero_grad()
    for param in params:
        param.grad.normal_()
    with pytest.raises(ValueError, match="sum_grad_squared.*not produced"):
        optimizer.step()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_unobserved_leaf_does_not_require_observation_producer():
    active = torch.nn.Parameter(torch.zeros(1))
    inactive = torch.nn.Parameter(torch.zeros(1))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([active, inactive], truegrad_adam)
    inactive_before = inactive.detach().clone()
    inactive_state = {name: value.detach().clone() for name, value in optimizer.state[inactive].items()}
    active.grad.fill_(1.0)
    inactive.grad.fill_(1.0)
    optimizer.produce(active, "sum_grad_squared", torch.ones_like(active))

    optimizer.step(observed=[True, False])

    assert not torch.equal(active, torch.zeros_like(active))
    assert torch.equal(inactive, inactive_before)
    for name, value in optimizer.state[inactive].items():
        assert torch.equal(value, inactive_state[name])


def test_bf16_observation_slabs_are_fp32():
    param = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    optimizer = Engine([param], truegrad_adam)
    observation = optimizer.groups[0].observations.sum_grad_squared

    assert observation.dtype == torch.float32
    assert param.sum_grad_squared.dtype == torch.float32
    assert param.sum_grad_squared.data_ptr() == observation[0].data_ptr()


def test_undeclared_observation_errors():
    param = torch.nn.Parameter(torch.zeros(2))
    recipe = Recipe(chain=(use_absent_observation,), commit=sgd_commit, defaults=dict(lr=1.0))
    optimizer = Engine([param], recipe)
    param.grad.fill_(1)

    with pytest.raises(Exception, match="absent"):
        optimizer.step()


def test_bf16_stochastic_rounding():
    torch.manual_seed(3)
    param = torch.nn.Parameter(torch.ones(256, dtype=torch.bfloat16))
    optimizer = Engine([param], adamw, lr=1e-4)
    before = param.detach().clone()
    optimizer.groups[0].grad_slab.fill_(1)
    for _ in range(32):
        optimizer.step()
    assert torch.any(param != before)


def test_lr_mutation_no_recompile():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    params = [torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.randn(3))]
    optimizer = Engine(params, adamw)
    for param in params:
        param.grad.normal_()
    optimizer.step()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    optimizer.hyper.lr.fill_(1e-3)
    for param in params:
        param.grad.normal_()
    optimizer.step()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_resume_equivalence_adamw():
    torch.manual_seed(9)
    initial = [torch.randn(2, 3), torch.randn(3)]
    grads = [[torch.randn_like(value) for value in initial] for _ in range(10)]
    keys = ("weight", "bias")
    params_a, params_c = _fresh_params(initial), _fresh_params(initial)
    optimizer_a = build(params_a, adamw, param_keys=keys, **VALUES)
    optimizer_c = build(params_c, adamw, param_keys=keys, **VALUES)

    for grad in grads[:5]:
        _copy_grads(params_a, grad)
        optimizer_a.step()
        _copy_grads(params_c, grad)
        optimizer_c.step()

    saved_params = _fresh_params(params_a)
    state_dict = optimizer_a.state_dict()
    optimizer_b = build(saved_params, adamw, param_keys=keys, **VALUES)
    optimizer_b.load_state_dict(state_dict)

    for grad in grads[5:]:
        _copy_grads(params_a, grad)
        _copy_grads(saved_params, grad)
        _copy_grads(params_c, grad)
        optimizer_a.step()
        optimizer_b.step()
        optimizer_c.step()

    for param_a, param_b, param_c in zip(params_a, saved_params, params_c, strict=True):
        assert torch.equal(param_a, param_b)
        assert torch.equal(param_a, param_c)


def test_resume_equivalence_whitening():
    torch.manual_seed(10)
    initial = [torch.randn(3, 3), torch.randn(2, 3)]
    grads = [[torch.randn_like(value) for value in initial] for _ in range(10)]
    keys = ("square", "rectangular")
    params_a, params_c = _fresh_params(initial), _fresh_params(initial)
    optimizer_a = build(params_a, whiten_adamw, param_keys=keys, **VALUES)
    optimizer_c = build(params_c, whiten_adamw, param_keys=keys, **VALUES)

    for step, grad in enumerate(grads[:5]):
        step_type = "refresh" if step == 2 else "normal"
        _copy_grads(params_a, grad)
        optimizer_a.step(step_type=step_type)
        _copy_grads(params_c, grad)
        optimizer_c.step(step_type=step_type)

    saved_params = _fresh_params(params_a)
    state_dict = optimizer_a.state_dict()
    optimizer_b = build(saved_params, whiten_adamw, param_keys=keys, **VALUES)
    optimizer_b.load_state_dict(state_dict)
    for slot in ("Q", "GG"):
        assert torch.equal(optimizer_a.state[params_a[0]][slot], optimizer_b.state[saved_params[0]][slot])
    assert torch.equal(optimizer_a.step_count, optimizer_b.step_count)

    for step, grad in enumerate(grads[5:], start=5):
        step_type = "refresh" if step == 7 else "normal"
        _copy_grads(params_a, grad)
        _copy_grads(saved_params, grad)
        _copy_grads(params_c, grad)
        optimizer_a.step(step_type=step_type)
        optimizer_b.step(step_type=step_type)
        optimizer_c.step(step_type=step_type)

    for param_a, param_b, param_c in zip(params_a, saved_params, params_c, strict=True):
        assert torch.equal(param_a, param_b)
        assert torch.equal(param_a, param_c)


def test_state_dict_is_logical():
    torch.manual_seed(11)
    initial = [torch.randn(2, 2) for _ in range(3)]
    keys = ("first", "second", "third")
    grads = [{key: torch.randn_like(value) for key, value in zip(keys, initial, strict=True)} for _ in range(6)]
    params_a = _fresh_params(initial)
    optimizer_a = build(params_a, adamw, param_keys=keys, **VALUES)

    for grad in grads[:3]:
        _copy_keyed_grads(params_a, keys, grad)
        optimizer_a.step()

    state_dict = optimizer_a.state_dict()
    assert set(state_dict) == {
        "format",
        "train_mode",
        "step",
        "age",
        "hyper",
        "state",
        "fingerprint",
        "param_init_pending",
    }
    assert state_dict["format"] == 3
    assert state_dict["train_mode"] is True
    assert set(state_dict["age"]) == set(keys)
    assert torch.equal(state_dict["age"]["first"], torch.tensor(3, dtype=torch.int64))
    assert set(state_dict["state"]) == set(keys)
    assert all(isinstance(key, str) for key in state_dict["state"])
    assert set(state_dict["state"]["first"]) == {0, "commit"}
    assert set(state_dict["state"]["first"][0]) == {"exp_avg", "exp_avg_sq"}
    assert state_dict["state"]["first"]["commit"] == {}
    assert state_dict["state"]["first"][0]["exp_avg"].shape == initial[0].shape
    assert state_dict["state"]["first"][0]["exp_avg"].device.type == "cpu"
    assert state_dict["fingerprint"]["first"] == {
        "chain": (("adam", None),),
        "commit": "sgd_commit",
        "observations": (),
        "clip_global_norm": None,
    }

    shuffled = (2, 0, 1)
    shuffled_keys = tuple(keys[index] for index in shuffled)
    params_b = _fresh_params([params_a[index] for index in shuffled])
    optimizer_b = build(params_b, adamw, param_keys=shuffled_keys, **VALUES)
    optimizer_b.load_state_dict(state_dict)

    for grad in grads[3:]:
        _copy_keyed_grads(params_a, keys, grad)
        _copy_keyed_grads(params_b, shuffled_keys, grad)
        optimizer_a.step()
        optimizer_b.step()

    values_a = dict(zip(keys, params_a, strict=True))
    values_b = dict(zip(shuffled_keys, params_b, strict=True))
    for key in keys:
        assert torch.equal(values_a[key], values_b[key])


def test_recipe_fingerprint_loads_matching_recipe_and_rejects_mismatch():
    source_recipe = Recipe(chain=(scale2,), commit=sgd_commit, defaults=dict(lr=0.25, weight_decay=0.0))
    mismatch_recipe = Recipe(chain=(add_one,), commit=sgd_commit, defaults=dict(lr=0.25, weight_decay=0.0))
    source_param = torch.nn.Parameter(torch.ones(2))
    matching_param = torch.nn.Parameter(torch.ones(2))
    mismatch_param = torch.nn.Parameter(torch.ones(2))

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        source = Engine([source_param], source_recipe, param_keys=("weight",))
        matching = Engine([matching_param], source_recipe, param_keys=("weight",), lr=0.75)
        mismatch = Engine([mismatch_param], mismatch_recipe, param_keys=("weight",), lr=0.75)

    source_param.grad.fill_(1)
    source.step()
    checkpoint = source.state_dict()
    matching.load_state_dict(checkpoint)
    assert torch.equal(matching.groups[0].age, source.groups[0].age)
    assert torch.equal(matching.hyper.lr, source.hyper.lr)

    mismatch_age = mismatch.groups[0].age.detach().clone()
    mismatch_lr = mismatch.hyper.lr.detach().clone()
    with pytest.raises(ValueError, match="recipe fingerprint") as error:
        mismatch.load_state_dict(checkpoint)

    assert "saved" in str(error.value)
    assert "current" in str(error.value)
    assert torch.equal(mismatch.groups[0].age, mismatch_age)
    assert torch.equal(mismatch.hyper.lr, mismatch_lr)


def test_load_state_dict_rejects_late_error_without_mutating_ages():
    source_param = torch.nn.Parameter(torch.ones(2))
    source = build([source_param], adamw, param_keys=("weight",), **VALUES)
    source_param.grad.fill_(1)
    source.step()
    checkpoint = source.state_dict()
    del checkpoint["state"]["weight"][0]["exp_avg"]

    target_param = torch.nn.Parameter(torch.ones(2))
    target = build([target_param], adamw, param_keys=("weight",), **VALUES)
    ages_before = [group.age.detach().clone() for group in target.groups]
    with pytest.raises(ValueError, match="checkpoint slots"):
        target.load_state_dict(checkpoint)

    for group, age_before in zip(target.groups, ages_before, strict=True):
        assert torch.equal(group.age, age_before)


def test_load_preserves_step_and_hyper():
    torch.manual_seed(12)
    params_a = [torch.nn.Parameter(torch.randn(2, 2))]
    optimizer_a = build(params_a, sgd, param_keys=("weight",), lr=0.125)
    for _ in range(3):
        params_a[0].grad.normal_()
        optimizer_a.step()
    optimizer_a.hyper.lr.fill_(0.03125)
    state_dict = optimizer_a.state_dict()

    params_b = _fresh_params(params_a)
    optimizer_b = build(params_b, sgd, param_keys=("weight",), lr=0.9)
    optimizer_b.load_state_dict(state_dict)
    assert state_dict["step"] == {"global": 4}
    assert torch.equal(optimizer_a.step_count, optimizer_b.step_count)
    assert torch.equal(optimizer_a.hyper.lr, optimizer_b.hyper.lr)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    grad = torch.ones_like(params_b[0])
    before = params_b[0].detach().clone()
    _copy_grads(params_b, [grad])
    optimizer_b.step()
    assert torch.equal(params_b[0], before - 0.03125 * grad)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    optimizer_b.hyper.lr.fill_(0.0625)
    before = params_b[0].detach().clone()
    _copy_grads(params_b, [grad])
    optimizer_b.step()
    assert torch.equal(params_b[0], before - 0.0625 * grad)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_muon_routing():
    torch.manual_seed(17)
    matrices = [torch.nn.Parameter(torch.randn(7, 4)) for _ in range(2)]
    vector = torch.nn.Parameter(torch.randn(4))
    params = [*matrices, vector]

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    optimizer = build(params, muon, **VALUES)

    matrix_groups = [group for group in optimizer.groups if group.recipe.chain == (momentum, orthogonalize)]
    adam_groups = [group for group in optimizer.groups if group.recipe is adamw]
    assert len(matrix_groups) == 1
    assert len(adam_groups) == 1
    assert all(set(optimizer.state[param]) == {"exp_avg"} for param in matrices)
    assert set(optimizer.state[vector]) == {"exp_avg", "exp_avg_sq"}

    before = [param.detach().clone() for param in params]
    _copy_grads(params, [torch.randn_like(param) for param in params])
    optimizer.step()

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
    assert all(not torch.equal(param, initial) for param, initial in zip(params, before, strict=True))
    assert all(torch.isfinite(param).all() for param in params)
    torch._dynamo.reset()


def test_muon_laprop_routing():
    torch.manual_seed(44)
    matrices = [torch.nn.Parameter(torch.randn(7, 4)) for _ in range(2)]
    vector = torch.nn.Parameter(torch.randn(4))
    params = [*matrices, vector]

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    optimizer = build(params, muon_laprop, **VALUES)

    matrix_groups = [group for group in optimizer.groups if group.recipe.chain == (laprop_transform, orthogonalize)]
    adam_groups = [group for group in optimizer.groups if group.recipe is adamw]
    assert len(matrix_groups) == 1
    assert matrix_groups[0].recipe.commit is muon_commit
    assert len(adam_groups) == 1
    assert all(set(optimizer.state[param]) == {"exp_avg", "exp_avg_sq"} for param in params)

    _copy_grads(params, [torch.randn_like(param) for param in params])
    optimizer.step()

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
    assert all(torch.isfinite(param).all() for param in params)
    torch._dynamo.reset()


def test_orthogonalize_semi_orthogonal(capsys):
    torch.manual_seed(18)
    update = torch.randn(1, 4, 7)
    orthogonal = _orthogonalized(update)
    identity = torch.eye(update.shape[-2], dtype=orthogonal.dtype)
    error = (orthogonal @ orthogonal.mT - identity).abs().max()

    with capsys.disabled():
        print(f"muon semi-orthogonality max error: {float(error):.9e}")
    assert error < 0.3


def test_newtonschulz_fp32_tracks_fp64_and_is_semi_orthogonal(capsys):
    torch.manual_seed(19)
    slab = torch.randn(3, 7, 4)
    fp32_output = _orthogonalized(slab)
    fp64_output = _orthogonalized(slab.double())
    error = (fp32_output.double() - fp64_output).abs().max()
    identity = torch.eye(slab.shape[-1], dtype=torch.float64)
    deviation = max((leaf.mT @ leaf - identity).abs().max() for leaf in fp64_output)
    with capsys.disabled():
        print(f"newton-schulz fp32-vs-fp64 max error: {float(error):.3e} semi-ortho dev: {float(deviation):.3e}")
    assert error < 5e-2
    assert deviation < 0.3


def test_orthogonalize_matches_svd_polar_factor(capsys):
    """The Newton-Schulz map must converge to the SVD polar factor U @ V^T -- the orthogonalization OF
    the input, not merely SOME semi-orthogonal matrix. This is an algorithm-independent oracle (any
    correct orthogonalization has this target, whatever its coefficient schedule), so it pins the
    input->output relationship that semi-orthogonality (UU^T~I) and fp32-vs-fp64 self-comparison do not:
    a fixed-output or coefficient-corrupted map that stays semi-orthogonal would still fail here."""
    torch.manual_seed(20)
    worst = 0.0
    for shape in ((4, 7), (7, 4), (16, 16), (32, 8)):
        slab = torch.randn(2, *shape, dtype=torch.float64)
        output = _orthogonalized(slab)
        for leaf_out, leaf_in in zip(output, slab):
            u, _, vh = torch.linalg.svd(leaf_in, full_matrices=False)
            worst = max(worst, (leaf_out - u @ vh).abs().max().item())
    with capsys.disabled():
        print(f"muon polar-factor (U@V^T) max error: {worst:.3e}")
    assert worst < 5e-2  # 5 modded-nanogpt NS steps converge to the polar factor to ~1e-2


def test_muon_fullgraph_clean(tmp_path):
    source = """
import torch
from heavyball import Engine, muon

params = [torch.nn.Parameter(torch.randn(7, 4)) for _ in range(2)]
params.append(torch.nn.Parameter(torch.randn(4)))
optimizer = Engine(params, muon)
for param in params:
    param.grad.normal_()
optimizer.step()
"""
    env = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).parents[1], env=env,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Output code:" in output
    assert output.count("AOT ID:") == 1
    assert "stack" not in output
    assert "_local_scalar_dense" not in output
    assert ".item(" not in output
    assert "synchronize" not in output

    banned = subprocess.run(
        ["grep", "-rnE", r"_foreach|vmap|torch\.cond|dynamic=True|torch\.stack|enabled|\bamp\b", "heavyball/"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True,
    )
    assert banned.returncode == 1, banned.stdout + banned.stderr


def test_sam_two_evaluations():
    params = [torch.nn.Parameter(torch.tensor((3.0, 4.0))), torch.nn.Parameter(torch.tensor((12.0,)))]
    rho = 0.25
    eps = 1e-12
    base = Engine(params, sgd, lr=0.0)
    optimizer = SAM(base, rho=rho, eps=eps)
    seen = []
    gradients = []

    def closure():
        base.zero_grad()
        seen.append(tuple(param.detach().clone() for param in params))
        loss = sum(param.square().sum() for param in params)
        loss.backward()
        gradients.append(tuple(param.grad.detach().clone() for param in params))
        return loss

    loss = optimizer.step(closure)

    assert len(seen) == 2
    assert len(gradients) == 2
    squared = [gradient.square().sum() for gradient in gradients[0]]
    global_grad_norm = squared[0]
    for value in squared[1:]:
        global_grad_norm = global_grad_norm + value
    global_grad_norm = global_grad_norm.sqrt()
    for first, second, gradient in zip(seen[0], seen[1], gradients[0], strict=True):
        perturbation = rho * gradient / (global_grad_norm + eps)
        assert not torch.equal(first, second)
        torch.testing.assert_close(second, first + perturbation, rtol=0, atol=1e-6)
    torch.testing.assert_close(loss, sum(value.square().sum() for value in seen[0]))


def test_sam_restores_before_update_uses_only_perturbed_gradient():
    param = torch.nn.Parameter(torch.tensor((3.0, 4.0)))
    rho = 0.5
    eps = 1e-12
    lr = 0.1
    base = Engine([param], sgd, lr=lr)
    optimizer = SAM(base, rho=rho, eps=eps)
    original = param.detach().clone()
    seen = []
    gradients = []

    def closure():
        seen.append(param.detach().clone())
        loss = 0.5 * param.square().sum()
        loss.backward()
        gradients.append(param.grad.detach().clone())
        return loss

    param.grad.zero_()
    base.step(observed=[False])
    optimizer.step(closure)

    assert len(seen) == 2
    assert len(gradients) == 2
    squared = [gradient.square().sum() for gradient in (gradients[0],)]
    global_grad_norm = squared[0]
    perturbation = rho * gradients[0] / (global_grad_norm.sqrt() + eps)
    torch.testing.assert_close(seen[1], seen[0] + perturbation, rtol=0, atol=1e-6)
    torch.testing.assert_close(gradients[1], original + perturbation, rtol=0, atol=1e-6)
    expected = original - lr * (original + perturbation)
    assert torch.equal(base.groups[0].observed, torch.ones_like(base.groups[0].observed))
    torch.testing.assert_close(param, expected, rtol=0, atol=1e-5)


def test_sam_perturb_keeps_large_finite_gradients_finite():
    param = torch.nn.Parameter(torch.zeros(1))
    with patch("torch.compile", lambda function, **kwargs: function):
        optimizer = SAM(Engine([param], sgd, lr=0.0), rho=0.05)

    param.grad.fill_(2e19)
    optimizer.compiled_perturb()

    torch.testing.assert_close(param, torch.tensor((0.05,)), rtol=0, atol=1e-7)


def test_sam_restores_when_second_closure_raises():
    param = torch.nn.Parameter(torch.tensor((3.0, 4.0)))
    with patch("torch.compile", lambda function, **kwargs: function):
        base = Engine([param], sgd, lr=0.1)
        optimizer = SAM(base, rho=0.5)
    original = param.detach().clone()
    step = base.step_count.detach().clone()
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second closure failed")
        loss = 0.5 * param.square().sum()
        loss.backward()
        return loss

    with pytest.raises(RuntimeError, match="second closure failed"):
        optimizer.step(closure)

    assert calls == 2
    assert torch.equal(param, original)
    assert torch.equal(base.step_count, step)


def test_sam_perturb_clears_grad_and_observation_slabs():
    param = torch.nn.Parameter(torch.ones(1))
    recipe = Recipe(chain=(), commit=sgd_commit, defaults=dict(lr=0.0, weight_decay=0.0), observations=("stat",))
    with patch("torch.compile", lambda function, **kwargs: function):
        base = Engine([param], recipe)
        optimizer = SAM(base)
    calls = 0

    def closure():
        nonlocal calls
        if calls == 1:
            assert torch.equal(param.grad, torch.zeros_like(param.grad))
            assert torch.equal(param.stat, torch.zeros_like(param.stat))
        loss = param.sum()
        loss.backward()
        base.produce(param, "stat", torch.full_like(param, 7))
        calls += 1
        return loss

    optimizer.step(closure)

    assert calls == 2


def test_sam_rejects_observation_missing_from_second_closure():
    param = torch.nn.Parameter(torch.ones(1))
    recipe = Recipe(
        chain=(use_stat,), commit=sgd_commit,
        defaults=dict(lr=1.0, weight_decay=0.0), observations=("my_stat",),
    )
    with patch("torch.compile", lambda function, **kwargs: function):
        base = Engine([param], recipe)
        optimizer = SAM(base, rho=0.1)
    original = param.detach().clone()
    calls = 0

    def closure():
        nonlocal calls
        loss = param.sum()
        loss.backward()
        if calls == 0:
            base.produce(param, "my_stat", torch.full_like(param, 7.0))
        calls += 1
        return loss

    base.zero_grad()
    with pytest.raises(ValueError, match="my_stat.*not produced"):
        optimizer.step(closure)

    assert calls == 2
    assert torch.equal(param, original)
    assert not getattr(param, "_heavyball_observation_binding").produced


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sam_restores_low_precision_params(dtype):
    param = torch.nn.Parameter(torch.tensor((0.1,), dtype=dtype))
    base = Engine([param], sgd, lr=0.0)
    optimizer = SAM(base, rho=0.05)
    original = param.detach().clone()

    def closure():
        base.zero_grad()
        param.grad.fill_(1)
        return param.sum()

    optimizer.step(closure)

    assert torch.equal(param, original)


def test_sam_reduces_loss():
    torch.manual_seed(21)
    model = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    inputs = torch.linspace(-1, 1, 16).unsqueeze(-1)
    targets = 1.5 * inputs - 0.25
    base = Engine(model.parameters(), adamw, lr=0.03)
    optimizer = SAM(base, rho=0.05)

    def closure():
        base.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        return loss

    start = torch.nn.functional.mse_loss(model(inputs), targets)
    for _ in range(12):
        optimizer.step(closure)
    finish = torch.nn.functional.mse_loss(model(inputs), targets)

    assert all(torch.isfinite(param).all() for param in model.parameters())
    assert finish < start


def test_sam_phases_fullgraph_clean(tmp_path):
    source = """
import torch
from heavyball import Engine, SAM, adamw

params = [torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.randn(3))]
base = Engine(params, adamw)
optimizer = SAM(base)
for param in params:
    param.grad.normal_()
optimizer.compiled_perturb()
for param in params:
    param.grad.normal_()
optimizer.compiled_restore()
"""
    env = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).parents[1], env=env,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("AOT ID:") == 2
    paths = re.findall(r"Output code written to: (.*\.py)", output)
    assert paths, output
    for path in {Path(path) for path in paths}:
        code = path.read_text()
        assert "stack" not in code
        assert "_local_scalar_dense" not in code
        assert ".item(" not in code

    phases = inspect.getsource(SAM._compile_perturb) + inspect.getsource(SAM._compile_restore)
    assert phases.count('torch.compile(whole_step, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")') == 2
    assert "base_step" not in phases
    assert "compiled_step" not in phases
    banned = subprocess.run(
        ["grep", "-rnE", r"_foreach|vmap|torch\.cond|dynamic=True|torch\.stack|enabled|\bamp\b", "heavyball/"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True,
    )
    assert banned.returncode == 1, banned.stdout + banned.stderr


def test_sam_composes_with_any_base():
    param = torch.nn.Parameter(torch.tensor((2.0, -1.0)))
    base = Engine([param], adopt, lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8)
    optimizer = SAM(base, rho=0.05)
    before = param.detach().clone()

    def closure():
        base.zero_grad()
        loss = param.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.step(closure)

    assert torch.equal(base.groups[0].states[0]["seen"], torch.ones(1, dtype=torch.bool))
    assert torch.isfinite(param).all()
    assert not torch.equal(param, before)


def test_age_equals_step_when_dense():
    torch._dynamo.reset()
    for recipe in (adamw, adopt):
        params = [torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(3))]
        optimizer = Engine(params, recipe, **{name: VALUES[name] for name in recipe.defaults})

        for value in range(1, 5):
            current_step = optimizer.step_count.detach().clone()
            for param in params:
                param.grad.fill_(value)
            optimizer.step()
            for group in optimizer.groups:
                assert torch.equal(group.age, torch.ones_like(group.age) * current_step)
    torch._dynamo.reset()


def test_late_observed_leaf_starts_at_age_one():
    torch._dynamo.reset()
    values = dict(lr=0.125, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0)
    params = [
        torch.nn.Parameter(torch.tensor((1.5, -2.0))),
        torch.nn.Parameter(torch.tensor((-0.25, 0.75))),
    ]
    optimizer = Engine(params, adamw, **values)
    group = optimizer.groups[0]
    assert len(optimizer.groups) == 1
    assert group.age.shape == (2,)

    for value in range(1, 5):
        params[0].grad.fill_(value)
        params[1].grad.fill_(-value)
        optimizer.step(observed=[True, False])
        assert torch.equal(group.age[1], torch.zeros_like(group.age[1]))

    late_before = params[1].detach().clone()
    gradient = torch.tensor((0.25, -0.5))
    fresh = torch.nn.Parameter(late_before.clone())
    fresh_optimizer = Engine([fresh], adamw, **values)
    params[0].grad.fill_(0.75)
    params[1].grad.copy_(gradient)
    fresh.grad.copy_(gradient)

    optimizer.step(observed=[True, True])
    fresh_optimizer.step()

    assert torch.equal(group.age[1], torch.ones_like(group.age[1]))
    torch.testing.assert_close(late_before - params[1], late_before - fresh, rtol=0, atol=1e-6)
    torch._dynamo.reset()


def test_age_persists_across_steps():
    torch._dynamo.reset()
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = Engine([param], sgd, lr=0.1)
    group = optimizer.groups[0]

    for observed in (True, False, True, False, True):
        param.grad.fill_(1)
        optimizer.step(observed=[observed])

    assert torch.equal(group.age, torch.tensor((3,), dtype=torch.int64))
    torch._dynamo.reset()
