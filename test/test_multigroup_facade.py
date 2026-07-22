"""Multi-param-group behavior for the torch.optim facade."""

from unittest.mock import call, patch

import pytest
import torch
from torch import nn

import heavyball


@pytest.fixture(autouse=True)
def _eager_compile():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def test_param_groups_match_separate_optimizers():
    initial_a = torch.tensor([1.0, -2.0, 0.5])
    initial_b = torch.tensor([-1.5, 3.0])
    gradient_a = torch.tensor([0.5, -0.25, 1.5])
    gradient_b = torch.tensor([-0.75, 0.125])

    multi_a = nn.Parameter(initial_a.clone())
    multi_b = nn.Parameter(initial_b.clone())
    multi = heavyball.AdamW(
        [
            {"params": [multi_a], "lr": 0.1},
            {"params": [multi_b], "lr": 0.01},
        ]
    )

    separate_a = nn.Parameter(initial_a.clone())
    separate_b = nn.Parameter(initial_b.clone())
    optimizer_a = heavyball.AdamW([separate_a], lr=0.1)
    optimizer_b = heavyball.AdamW([separate_b], lr=0.01)

    multi_a.grad.copy_(gradient_a)
    multi_b.grad.copy_(gradient_b)
    separate_a.grad.copy_(gradient_a)
    separate_b.grad.copy_(gradient_b)
    multi.step()
    optimizer_a.step()
    optimizer_b.step()

    assert torch.equal(multi_a, separate_a)
    assert torch.equal(multi_b, separate_b)


def test_clip_global_norm_reduces_across_all_param_groups():
    param_a = nn.Parameter(torch.tensor(0.0))
    param_b = nn.Parameter(torch.tensor(0.0))
    optimizer = heavyball.SGD(
        [
            {"params": [param_a], "lr": 1.0},
            {"params": [param_b], "lr": 1.0},
        ],
        clip_global_norm=2.5,
    )

    param_a.grad.fill_(3.0)
    param_b.grad.fill_(4.0)
    optimizer.step()

    updates = torch.stack((-param_a.detach(), -param_b.detach()))
    torch.testing.assert_close(updates, torch.tensor((1.5, 2.0)), rtol=0, atol=1e-6)
    assert not torch.allclose(updates, torch.tensor((2.5, 2.5)))
    assert len(optimizer._engines) == 1


def test_param_group_learning_rates_remain_independent():
    param_a = nn.Parameter(torch.tensor(0.0))
    param_b = nn.Parameter(torch.tensor(0.0))
    optimizer = heavyball.SGD(
        [
            {"params": [param_a], "lr": 0.1},
            {"params": [param_b], "lr": 0.5},
        ]
    )

    param_a.grad.fill_(1.0)
    param_b.grad.fill_(1.0)
    optimizer.step()

    torch.testing.assert_close(-param_a, torch.tensor(0.1))
    torch.testing.assert_close(-param_b, torch.tensor(0.5))


def test_routed_soap_uses_each_param_groups_shampoo_beta():
    gradient_stream = (
        torch.tensor(((1.0, 2.0), (3.0, 4.0))),
        torch.tensor(((0.5, -1.0), (2.0, -0.25))),
    )
    multi_params = [nn.Parameter(torch.eye(2)), nn.Parameter(torch.eye(2))]
    separate_params = [nn.Parameter(torch.eye(2)), nn.Parameter(torch.eye(2))]
    multi = heavyball.SOAP(
        [
            {"params": [multi_params[0]], "shampoo_beta": 0.2},
            {"params": [multi_params[1]], "shampoo_beta": 0.7},
        ],
        weight_decay=0.0,
        preconditioner_update_probability=0.0,
    )
    separate = [
        heavyball.SOAP(
            [param],
            shampoo_beta=shampoo_beta,
            weight_decay=0.0,
            preconditioner_update_probability=0.0,
        )
        for param, shampoo_beta in zip(separate_params, (0.0, 0.9), strict=True)
    ]
    multi.param_groups[0]["shampoo_beta"] = 0.0
    multi.param_groups[1]["shampoo_beta"] = 0.9

    for gradient in gradient_stream:
        for param in (*multi_params, *separate_params):
            param.grad.copy_(gradient)
        multi.step()
        for optimizer in separate:
            optimizer.step()

    multi_grams = [multi._engine.state[param]["GG_l"] for param in multi_params]
    separate_grams = [optimizer._engine.state[param]["GG_l"] for optimizer, param in zip(
        separate, separate_params, strict=True
    )]
    assert all(torch.equal(actual, expected) for actual, expected in zip(
        multi_grams, separate_grams, strict=True
    ))
    assert not torch.equal(*multi_grams)


@pytest.mark.parametrize(
    ("name", "first", "second"),
    (
        ("clip_global_norm", 1.0, 2.0),
        ("storage_dtype", None, torch.bfloat16),
        ("ecc", 8, 16),
    ),
)
def test_conflicting_engine_wide_param_group_values_raise(name, first, second):
    groups = [
        {"params": [nn.Parameter(torch.tensor(0.0))], name: first},
        {"params": [nn.Parameter(torch.tensor(0.0))], name: second},
    ]

    with pytest.raises(ValueError, match=rf"optimizer-wide values for '{name}'"):
        heavyball.SGD(groups)


def test_conflicting_cadence_probabilities_raise():
    groups = [
        {"params": [nn.Parameter(torch.eye(2))], "preconditioner_update_probability": 0.25},
        {"params": [nn.Parameter(torch.eye(2))], "preconditioner_update_probability": 0.75},
    ]

    with pytest.raises(
        ValueError,
        match="optimizer-wide values for 'preconditioner_update_probability'",
    ):
        heavyball.SOAP(groups)


def test_static_hypers_preserve_adamw_trajectory():
    torch.manual_seed(24)
    initial = torch.randn(5)
    gradients = [torch.randn_like(initial) for _ in range(4)]
    param_a = nn.Parameter(initial.clone())
    param_b = nn.Parameter(initial.clone())
    optimizer_a = heavyball.AdamW([param_a], lr=0.03, weight_decay=0.01)
    optimizer_b = heavyball.AdamW([param_b], lr=0.03, weight_decay=0.01)

    for gradient in gradients:
        param_a.grad.copy_(gradient)
        param_b.grad.copy_(gradient)
        optimizer_a.step()
        optimizer_b.step()
        assert torch.equal(param_a, param_b)


def test_scheduler_style_lr_mutation_updates_only_owning_engine():
    initial = torch.tensor([1.0, -1.0])
    gradient = torch.tensor([0.25, -0.5])
    param = nn.Parameter(initial.clone())
    other = nn.Parameter(torch.tensor([2.0]))
    other_initial = other.detach().clone()
    optimizer = heavyball.SGD(
        [
            {"params": [param], "lr": 0.1},
            {"params": [other], "lr": 0.01},
        ]
    )

    optimizer.param_groups[0]["lr"] = 0.2
    param.grad.copy_(gradient)
    other.grad.fill_(1.0)
    optimizer.step()

    expected_new_lr = initial - 0.2 * gradient
    expected_old_lr = initial - 0.1 * gradient
    assert torch.equal(param, expected_new_lr)
    assert not torch.equal(param, expected_old_lr)
    assert torch.equal(other, other_initial - 0.01)

    optimizer.param_groups[0]["lr"] = 0.05
    before_param = param.detach().clone()
    before_other = other.detach().clone()
    param.grad.copy_(gradient)
    other.grad.fill_(1.0)
    optimizer.step()

    assert torch.equal(param, before_param - 0.05 * gradient)
    assert torch.equal(other, before_other - 0.01)


def test_unchanged_hypers_are_not_resynced():
    param = nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = heavyball.AdamW([param], lr=0.1)
    engine = optimizer._engines[0]

    with patch.object(engine, "set_hyper", wraps=engine.set_hyper) as set_hyper:
        for _ in range(3):
            param.grad.fill_(1.0)
            optimizer.step()

        assert set_hyper.call_count == 0

        optimizer.param_groups[0]["lr"] = 0.05
        param.grad.fill_(1.0)
        optimizer.step()

    assert set_hyper.call_args_list == [call("lr", 0.05, group_id=0)]


def test_add_param_group_rebuilds_single_engine_and_updates_parameter():
    first = nn.Parameter(torch.tensor([1.0]))
    added = nn.Parameter(torch.tensor([2.0]))
    optimizer = heavyball.SGD([first], lr=0.1)

    first.grad.fill_(0.5)
    optimizer.step()
    old_engine = optimizer._engine
    old_age = old_engine.groups[0].age.clone()
    old_step = old_engine.step_count.clone()
    optimizer.add_param_group({"params": [added], "lr": 0.05})

    assert len(optimizer._engines) == 1
    assert optimizer._engine is not old_engine
    assert {id(param) for param in optimizer._engine.params} == {id(first), id(added)}
    first_group = next(group for group in optimizer._engine.groups if any(param is first for param in group.params))
    added_group = next(group for group in optimizer._engine.groups if any(param is added for param in group.params))
    assert torch.equal(first_group.age, old_age)
    assert torch.equal(optimizer._engine.step_count, old_step)
    assert first_group.leaf_indices.item() == 0
    assert added_group.leaf_indices.item() == 1
    first.grad.fill_(1.0)
    added.grad.fill_(2.0)
    optimizer.step()
    assert torch.equal(added, torch.tensor([1.9]))


def test_multigroup_state_dict_roundtrip():
    torch.manual_seed(23)
    param_a = nn.Parameter(torch.randn(3))
    param_b = nn.Parameter(torch.randn(2))
    optimizer_a = heavyball.AdamW(
        [
            {"params": [param_a], "lr": 0.1},
            {"params": [param_b], "lr": 0.01},
        ]
    )

    for _ in range(3):
        param_a.grad.copy_(torch.randn_like(param_a))
        param_b.grad.copy_(torch.randn_like(param_b))
        optimizer_a.step()

    checkpoint = optimizer_a.state_dict()
    assert isinstance(checkpoint["engines"], list)
    assert len(checkpoint["engines"]) == 1
    assert {key[0] for key in checkpoint["engines"][0]["hyper"]} == {0, 1}

    restored_a = nn.Parameter(param_a.detach().clone())
    restored_b = nn.Parameter(param_b.detach().clone())
    optimizer_b = heavyball.AdamW(
        [
            {"params": [restored_a], "lr": 0.9},
            {"params": [restored_b], "lr": 0.8},
        ]
    )
    optimizer_b.load_state_dict(checkpoint)

    gradient_a = torch.randn_like(param_a)
    gradient_b = torch.randn_like(param_b)
    param_a.grad.copy_(gradient_a)
    param_b.grad.copy_(gradient_b)
    restored_a.grad.copy_(gradient_a)
    restored_b.grad.copy_(gradient_b)
    optimizer_a.step()
    optimizer_b.step()

    assert torch.equal(param_a, restored_a)
    assert torch.equal(param_b, restored_b)
