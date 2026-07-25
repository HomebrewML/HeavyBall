"""Independent equation and property oracles for Aurora Algorithm 3.

Aurora is applied only to tall matrices. Its inner loop carries the newly computed polar direction
``X_k`` and a geometrically damped row-scale ``D_k``; wide and square matrices retain Muon.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import ParamInfo, aurora, muon
from heavyball.transforms import Tempo, balanced_orthogonalize, orthogonalize


def _tempo(count: int) -> Tempo:
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(
            beta2=torch.tensor(0.5, dtype=torch.float64),
            eps=torch.tensor(1e-8, dtype=torch.float64),
        ),
        False,
    )


def _direction(transform, x):
    return transform(x.clone(), None, None, {}, _tempo(x.shape[0]))[0]


def _row_leverage(factor):
    tall = factor if factor.shape[-2] >= factor.shape[-1] else factor.mT
    return (tall**2).sum(dim=-1)


@pytest.mark.parametrize("shape", [(8, 4), (16, 3)])
def test_balanced_orthogonalize_is_a_polar_factor(shape):
    torch.manual_seed(0)
    g = torch.randn(1, *shape, dtype=torch.float64)
    factor = _direction(balanced_orthogonalize, g)[0]
    tall = factor if factor.shape[-2] >= factor.shape[-1] else factor.mT
    singular = torch.linalg.svdvals(tall)
    assert singular.min() > 0.95
    assert singular.max() < 1.05


@pytest.mark.parametrize("shape", [(8, 4), (16, 3)])
def test_balanced_orthogonalize_uniformizes_row_leverage(shape):
    torch.manual_seed(1)
    tall = torch.randn(1, *shape, dtype=torch.float64)
    tall[0, :2] *= 40.0
    g = tall
    target = shape[1] / shape[0]
    plain = _row_leverage(_direction(orthogonalize, g)[0])
    balanced = _row_leverage(_direction(balanced_orthogonalize, g)[0])
    assert plain.std() > 0.15
    assert balanced.std() < plain.std()
    assert balanced.mean().item() == pytest.approx(target, abs=0.05)


@pytest.mark.parametrize("shape", [(5, 5), (4, 8)])
def test_balanced_orthogonalize_leaves_non_tall_matrices_to_plain_muon(shape):
    torch.manual_seed(2)
    g = torch.randn(1, *shape, dtype=torch.float64)
    torch.testing.assert_close(
        _direction(balanced_orthogonalize, g),
        _direction(orthogonalize, g),
        rtol=0,
        atol=0,
    )


def test_aurora_matches_algorithm_three_with_exact_polar_oracle():
    torch.manual_seed(7)
    momentum = torch.randn(1, 8, 3, dtype=torch.float64)

    def exact_polar(update, obs, param, state, tempo):
        del obs, param, state
        u, _, vh = torch.linalg.svd(update, full_matrices=False)
        return u @ vh, {}, tempo.live

    direction = momentum / momentum.norm(dim=(-2, -1), keepdim=True)
    row_scale = torch.ones_like(direction[..., :1])
    for _ in range(2):
        row_norm = direction.norm(dim=-1, keepdim=True)
        row_scale = row_scale.mul(row_norm).sqrt()
        direction = exact_polar(
            (3 / 8) ** 0.5 * direction / row_scale,
            None,
            None,
            {},
            _tempo(1),
        )[0]

    with patch("heavyball.transforms.orthogonalize", exact_polar):
        actual = _direction(balanced_orthogonalize, momentum)
    torch.testing.assert_close(actual, direction, rtol=1e-14, atol=1e-14)


def test_aurora_routes_only_tall_matrices_and_uses_muon_otherwise():
    assert aurora.when(ParamInfo(torch.empty(8, 4)))
    assert not aurora.when(ParamInfo(torch.empty(4, 8)))
    assert not aurora.when(ParamInfo(torch.empty(4, 4)))
    assert aurora.otherwise.then is muon.then

