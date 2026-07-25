"""Independent property oracle for NorMuon's row-wise raw second moment and RMS-0.2 scaling.

Verifies NorMuon's defining behavior against neither the upstream source nor the shipped transform:
the orthogonalized direction is divided by its per-output-neuron (row) RMS -- reducing the second
moment over the input axis for tall and wide leaves alike, per NorMuon (arXiv:2510.05491) Alg. 1 --
then rescaled to Frobenius norm ``0.2 * sqrt(m * n)``. Property oracle calibrated independently
against the paper.
"""

from types import SimpleNamespace

import pytest
import torch

from heavyball import normuon
from heavyball.transforms import (
    Tempo,
    first_moment,
    normuon_normalize,
    orthogonalize,
    sgd_commit,
)


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


def test_normuon_recipe_uses_ema_and_plain_sgd_commit():
    matrix = normuon.then
    assert matrix.chain == (first_moment, orthogonalize, normuon_normalize)
    assert matrix.commit is sgd_commit

