"""Independent property oracle for NorMuon's row-wise raw second moment and RMS-0.2 scaling.

Verifies NorMuon's defining behavior against neither the upstream source nor the shipped transform:
the orthogonalized direction is divided by its per-output-neuron (row) RMS -- reducing the second
moment over the input axis for tall and wide leaves alike, per NorMuon (arXiv:2510.05491) Alg. 1 --
then rescaled to Frobenius norm ``0.2 * sqrt(m * n)``. Property oracle calibrated independently
against the paper.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball.transforms import Tempo, normuon_normalize, orthogonalize


def _tempo(count: int) -> Tempo:
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(beta2=torch.tensor(0.95, dtype=torch.float64), eps=torch.tensor(1e-8, dtype=torch.float64)),
        False,
    )


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_normuon_equalizes_axis_rms_and_scales_to_point_two(shape):
    torch.manual_seed(0)
    m = torch.randn(1, *shape, dtype=torch.float64)
    orth = orthogonalize(m.clone(), None, None, {}, _tempo(1))[0]
    state = normuon_normalize.init(m[0])
    update = normuon_normalize(orth.clone(), None, None, state, _tempo(1))[0]
    assert update.square().mean().sqrt().item() == pytest.approx(0.2, rel=1e-6)
    reduce_dim = -1
    before = orth.square().mean(dim=reduce_dim).sqrt().std()
    after = update.square().mean(dim=reduce_dim).sqrt().std()
    assert after.item() < 0.2 * before.item()


def _trajectory(beta1, lr):
    parameter = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float64))
    gradients = (
        torch.tensor(
            ((1.0, -2.0, 3.0), (-4.0, 5.0, -6.0), (2.0, 7.0, -1.0), (8.0, -3.0, 4.0)),
            dtype=torch.float64,
        ),
        torch.tensor(
            ((-3.0, 1.0, 2.0), (6.0, -5.0, 4.0), (7.0, -2.0, 3.0), (-1.0, 8.0, -4.0)),
            dtype=torch.float64,
        ),
    )
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.NorMuon(
            [parameter],
            lr=lr,
            beta1=beta1,
            weight_decay=0.0,
        )
    values = []
    for gradient in gradients:
        parameter.grad.copy_(gradient)
        optimizer.step()
        values.append(parameter.detach().clone())
    return values


def test_normuon_public_route_uses_momentum_and_plain_rms_point_two_updates():
    momentum = _trajectory(beta1=0.9, lr=0.01)
    without_momentum = _trajectory(beta1=0.0, lr=0.01)
    doubled_lr = _trajectory(beta1=0.9, lr=0.02)

    first_update_rms = momentum[0].square().mean().sqrt() / 0.01
    assert first_update_rms.item() == pytest.approx(0.2, rel=1e-6)
    assert not torch.allclose(momentum[1], without_momentum[1])
    for doubled, ordinary in zip(doubled_lr, momentum, strict=True):
        torch.testing.assert_close(doubled, ordinary * 2, rtol=0, atol=1e-12)
