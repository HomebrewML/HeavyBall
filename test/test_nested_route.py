from unittest.mock import patch

import torch

from heavyball import Recipe, Route, adamw, kron, psgd_pro, sgd
from heavyball.core import build, fsdp2_recipe_scope_supported
from heavyball.matrix import matrix_route
from heavyball.optim import HeavyBallOptimizer


def _group_for(engine, param):
    return next(group for group in engine.groups if any(candidate is param for candidate in group.params))


@patch("torch.compile", new=lambda function, **kwargs: function)
def test_nested_route_assigns_each_parameter_to_the_priority_leaf():
    conv = torch.nn.Conv2d(3, 4, 3)
    linear = torch.nn.Linear(7, 5)
    params = (conv.weight, linear.weight, linear.bias)
    route = Route(
        lambda info: info.ndim >= 3,
        kron,
        Route(matrix_route, psgd_pro, adamw),
    )

    optimizer = HeavyBallOptimizer(params, route)
    engine = optimizer._engine

    conv_group = _group_for(engine, conv.weight)
    linear_group = _group_for(engine, linear.weight)
    bias_group = _group_for(engine, linear.bias)

    assert conv_group.recipe is kron
    assert {"Q_0", "Q_1"} <= conv_group.states[0].keys()
    assert linear_group.recipe is psgd_pro
    assert {"Q_0", "Q_1"} <= linear_group.states[0].keys()
    assert bias_group.recipe is adamw
    assert "exp_avg" in bias_group.states[0]


@patch("torch.compile", new=lambda function, **kwargs: function)
def test_flat_route_remains_supported():
    linear = torch.nn.Linear(3, 2)
    engine = build((linear.weight, linear.bias), Route(matrix_route, psgd_pro, adamw))

    assert _group_for(engine, linear.weight).recipe is psgd_pro
    assert _group_for(engine, linear.bias).recipe is adamw


def test_fsdp2_recipe_scope_support_checks_nested_route_leaves():
    def unsupported_transform(update, obs, param, state, tempo):
        return update, state, tempo.live

    unsupported_transform.distributed_shard_separable = False
    unsupported = Recipe(chain=(unsupported_transform,), commit=sgd.commit, defaults=sgd.defaults)
    supported_route = Route(lambda info: True, adamw, Route(lambda info: True, sgd, adamw))
    unsupported_route = Route(lambda info: True, adamw, Route(lambda info: True, sgd, unsupported))

    assert fsdp2_recipe_scope_supported(supported_route) is True
    assert fsdp2_recipe_scope_supported(unsupported_route) is False
