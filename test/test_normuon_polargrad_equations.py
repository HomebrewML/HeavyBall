"""Independent FP64 equation oracles for the two matrix optimizers a review flagged, grounded in the
published algorithms (not heavyball's own code):

  PolarGrad  (arXiv:2505.21799): update = polar_factor(G) * nuclear_norm(G), with <U_p, G> = tr(Sigma).
  NorMuon    (arXiv:2510.05491, Alg. 1): after orthogonalization, normalize by a per-output-neuron (ROW)
             second moment -- reduce O*O across COLUMNS (the `in` axis), a length-`out` vector, applied
             uniformly regardless of whether the matrix is tall or wide.

The polar factor and nuclear norm come from an FP64 SVD, so a bug shared between the transform and a naive
reference cannot satisfy these.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

with patch("heavyball.core.torch.compile", lambda f, **k: f):
    import heavyball  # noqa: F401
from heavyball.transforms import (
    Tempo,
    normuon_normalize,
    normuon_normalize_init,
    polargrad_direction,
)


def _tempo(count):
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(beta2=torch.tensor(0.95, dtype=torch.float64), eps=torch.tensor(1e-7, dtype=torch.float64)),
        False,
    )


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _svd_polar(g):
    w, s, vh = torch.linalg.svd(g, full_matrices=False)
    return w @ vh, s


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (6, 6), (16, 3)])
def test_polargrad_is_polar_factor_times_nuclear_norm(shape):
    torch.manual_seed(0)
    g = torch.randn(1, *shape, dtype=torch.float64)
    got = polargrad_direction(g.clone(), None, None, {}, _tempo(1))[0][0]
    u, s = _svd_polar(g[0])
    ref = u * s.sum()
    assert _cos(got, ref) > 0.999          # direction is the polar factor
    assert (got - ref).norm() / ref.norm() < 0.05  # magnitude is the nuclear norm (Newton-Schulz tol)


def _perrow_normuon(dm, eps=1e-7):
    # NorMuon Alg. 1: reduce O*O across columns (the `in` axis) -> per output-neuron (row) vector,
    # normalize, then restore the Frobenius norm.
    v = dm.square().mean(dim=-1, keepdim=True)
    normed = dm * v.clamp_min(eps).rsqrt()
    return normed * (dm.norm() / normed.norm().clamp_min(eps))


@pytest.mark.parametrize("shape", [(8, 4), (16, 3), (4, 8), (3, 16)])
def test_normuon_normalizes_per_output_neuron(shape):
    # A direction with one output neuron (row) inflated. NorMuon must divide each ROW by its own RMS
    # (balancing output neurons) for BOTH tall and wide matrices. heavyball's shorter-axis reduction
    # normalizes per input feature on wide matrices, so it fails to balance the neurons there.
    torch.manual_seed(0)
    d = torch.randn(1, *shape, dtype=torch.float64)
    d[0, 0] *= 5.0
    state = {"moment2": normuon_normalize_init(d)["moment2"]}
    got = normuon_normalize(d.clone(), None, None, state, _tempo(1))[0][0]
    ref = _perrow_normuon(d[0])
    assert _cos(got, ref) > 0.999, f"{shape}: normuon axis diverges from per-output-neuron (cos={_cos(got, ref):.4f})"
