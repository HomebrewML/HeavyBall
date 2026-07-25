"""Family-wide invariant for mergeable convolution matrix routing."""

from unittest.mock import patch

import pytest
import torch

import heavyball

_CONV_SHAPE = (4, 3, 3, 3)


_ROUTED_MATRIX_ADAMW = (
    ("soap_adamw", heavyball.soap_adamw, heavyball.soap),
    ("shampoo_adamw", heavyball.shampoo_adamw, heavyball.shampoo),
    ("kl_soap_adamw", heavyball.kl_soap_adamw, heavyball.kl_soap),
    ("kl_shampoo_adamw", heavyball.kl_shampoo_adamw, heavyball.kl_shampoo),
    ("kron_adamw", heavyball.kron_adamw, heavyball.kron),
    ("psgd_pro_adamw", heavyball.psgd_pro_adamw, heavyball.psgd_pro),
    ("qsgd_adamw", heavyball.qsgd_adamw, heavyball.qsgd),
    ("lather_adamw", heavyball.lather_adamw, heavyball.lather),
)


def _trajectory(recipe):
    generator = torch.Generator().manual_seed(43)
    initial = torch.randn(_CONV_SHAPE, generator=generator, dtype=torch.float64)
    gradients = [torch.randn(_CONV_SHAPE, generator=generator, dtype=torch.float64) for _ in range(3)]
    parameter = torch.nn.Parameter(initial)
    torch.manual_seed(71)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build([parameter], recipe, lr=1e-3, weight_decay=0.0)
    for index, gradient in enumerate(gradients):
        parameter.grad.copy_(gradient)
        optimizer.step(step_type="refresh" if index in (0, 2) else "normal")
    return parameter.detach()


@pytest.mark.parametrize(
    ("name", "route", "matrix_recipe"),
    _ROUTED_MATRIX_ADAMW,
    ids=[case[0] for case in _ROUTED_MATRIX_ADAMW],
)
def test_mergeable_conv_matches_the_matrix_branch(name, route, matrix_recipe):
    torch.testing.assert_close(
        _trajectory(route),
        _trajectory(matrix_recipe),
        rtol=0,
        atol=0,
        msg=f"{name} did not select its matrix branch",
    )
