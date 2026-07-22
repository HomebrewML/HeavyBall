"""Family-wide invariant for mergeable convolution matrix routing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.matrix import matrix_route, merged_matrix_shape

_CONV_SHAPE = (4, 3, 3, 3)
_MERGED_SHAPE = (4, 27)
_BARE_ADAM_STATE = {"exp_avg", "exp_avg_sq"}


def _build_conv_state(route: heavyball.Route) -> dict[str, torch.Tensor]:
    parameter = torch.nn.Parameter(torch.zeros(_CONV_SHAPE))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build([parameter], route, lr=1e-3)
    return optimizer.groups[0].states[0]


def _stored_matrix_shape(state: dict[str, torch.Tensor]) -> tuple[int, ...]:
    if "exp_avg" in state:
        return tuple(state["exp_avg"].shape[1:])

    # Factor-only recipes do not persist the update; their square-factor widths
    # are the two axes of the merged matrix on which the transform operates.
    factor_widths = {
        value.shape[-1]
        for value in state.values()
        if value.ndim >= 3 and value.shape[-2] == value.shape[-1]
    }
    return tuple(sorted(factor_widths))


_ROUTED_MATRIX_ADAMW = (
    ("soap_adamw", heavyball.soap_adamw, "Q_l"),
    ("shampoo_adamw", heavyball.shampoo_adamw, "L"),
    ("kl_soap_adamw", heavyball.kl_soap_adamw, "eigenvalues_l"),
    ("kl_shampoo_adamw", heavyball.kl_shampoo_adamw, "eigenvalues_l"),
    ("kron_adamw", heavyball.kron_adamw, "Q_0"),
    ("psgd_pro_adamw", heavyball.psgd_pro_adamw, "Q_0"),
    ("qsgd_adamw", heavyball.qsgd_adamw, "Q_0"),
    ("lather_adamw", heavyball.lather_adamw, "Q_basis_0"),
)


@pytest.mark.parametrize(
    ("name", "route", "preconditioner_key"),
    _ROUTED_MATRIX_ADAMW,
    ids=[case[0] for case in _ROUTED_MATRIX_ADAMW],
)
def test_mergeable_conv_uses_matrix_preconditioner(name, route, preconditioner_key):
    state = _build_conv_state(route)

    assert preconditioner_key in state, (
        f"{name} silently substituted bare Adam for a convolution that merges to 2D; "
        f"state keys were {set(state)}"
    )
    assert set(state) != _BARE_ADAM_STATE
    assert _stored_matrix_shape(state) == _MERGED_SHAPE


def test_matrix_route_boundary_distinguishes_preconditioning_from_fallthrough():
    mergeable = SimpleNamespace(shape=_CONV_SHAPE, ndim=len(_CONV_SHAPE))
    n_factor = SimpleNamespace(shape=(128, 1000, 3, 3), ndim=4)

    assert merged_matrix_shape(mergeable.shape, 2048) == _MERGED_SHAPE
    assert len(merged_matrix_shape(n_factor.shape, 2048)) > 2
    assert matrix_route(mergeable) is True
    assert matrix_route(n_factor) is False


def test_mergeable_conv_state_is_not_bare_adam():
    name, route, preconditioner_key = _ROUTED_MATRIX_ADAMW[0]
    state = _build_conv_state(route)

    assert preconditioner_key in state and set(state) != _BARE_ADAM_STATE, (
        f"{name} must expose preconditioner state for a merge-to-2D convolution; "
        "an ndim == 2 route would fall through to bare Adam here"
    )
