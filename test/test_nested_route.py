from unittest.mock import patch

import torch

from heavyball import Recipe, Route, adamw, sgd
from heavyball.core import fsdp2_recipe_scope_supported
from heavyball.matrix import matrix_route
from heavyball.optim import HeavyBallOptimizer


@patch("torch.compile", new=lambda function, **kwargs: function)
def test_nested_route_assigns_each_parameter_to_the_priority_leaf():
    conv = torch.nn.Conv2d(3, 4, 3)
    linear = torch.nn.Linear(7, 5)
    params = (conv.weight, linear.weight, linear.bias)

    def scaled_recipe(scale):
        def transform(update, obs, param, state, tempo):
            del obs, param, state
            return update * scale, {}, tempo.live

        def init(reference):
            del reference
            return {}

        transform.init = init
        return Recipe((transform,), sgd.commit, sgd.defaults)

    route = Route(
        lambda info: info.ndim >= 3,
        scaled_recipe(3),
        Route(matrix_route, scaled_recipe(2), scaled_recipe(1)),
    )
    before = [param.detach().clone() for param in params]
    gradients = [torch.ones_like(param) for param in params]

    optimizer = HeavyBallOptimizer(params, route, lr=0.1, weight_decay=0.0)
    for param, gradient in zip(params, gradients, strict=True):
        param.grad.copy_(gradient)
    optimizer.step()

    for param, initial, gradient, scale in zip(
        params, before, gradients, (3, 2, 1), strict=True
    ):
        torch.testing.assert_close(param, initial - 0.1 * scale * gradient)


def test_fsdp2_recipe_scope_support_checks_nested_route_leaves():
    def unsupported_transform(update, obs, param, state, tempo):
        return update, state, tempo.live

    unsupported_transform.distributed_shard_separable = False
    unsupported = Recipe(chain=(unsupported_transform,), commit=sgd.commit, defaults=sgd.defaults)
    supported_route = Route(lambda info: True, adamw, Route(lambda info: True, sgd, adamw))
    unsupported_route = Route(lambda info: True, adamw, Route(lambda info: True, sgd, unsupported))

    assert fsdp2_recipe_scope_supported(supported_route) is True
    assert fsdp2_recipe_scope_supported(unsupported_route) is False
