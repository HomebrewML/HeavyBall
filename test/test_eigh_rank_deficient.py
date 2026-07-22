"""The eigh-based preconditioner roots must stay finite on degenerate Gram matrices.

Shampoo's ``_inverse_fourth_root`` and whitening's ``_matrix_inv_sqrt`` feed a Gram to
``torch.linalg.eigh``. An early Gram is rank-deficient (it starts at zeros and accumulates
rank <= batch << dim), and on some such matrices LAPACK's eigh SILENTLY returns NaN, which
propagates into the weights and later raises a _LinAlgError -- heavyball.Shampoo crashed on
~1/3 of seeds within 3 steps of a real MNIST autoencoder. The fix regularizes the Gram
(symmetrize + eps on the diagonal, the pattern already used in lather.py) before eigh.

This guards the regularization's finiteness property across degenerate Grams. It is NOT a
red-green reproduction of the original crash: eigh only NaNs on the specific gram values that
arise in real training (when eigh SUCCEEDS the regularized and raw paths agree, since eps is
already clamped onto the eigenvalues), so the crash cannot be reproduced by a compact synthetic
Gram. The crash fix itself is validated end-to-end on the real reproducer (Shampoo on MNIST,
6/6 seeds finite after the fix, was 4/6).
"""
import pytest
import torch

from heavyball.matrix import _inverse_fourth_root
from heavyball.transforms import _matrix_inv_sqrt


def _rank_deficient_gram(dimension: int, rank: int) -> torch.Tensor:
    torch.manual_seed(dimension * 100 + rank)
    factor = torch.randn(1, dimension, rank)  # factor @ factor^T has rank <= r < d
    return factor @ factor.mT


@pytest.mark.parametrize("inverse_root", (_inverse_fourth_root, _matrix_inv_sqrt))
@pytest.mark.parametrize(
    "gram",
    (
        torch.zeros(1, 24, 24),        # rank 0: the extreme degenerate case
        _rank_deficient_gram(24, 1),   # rank 1
        _rank_deficient_gram(48, 5),   # low rank
        _rank_deficient_gram(64, 60),  # high rank, still deficient (the real-crash regime)
    ),
)
def test_inverse_root_is_finite_on_degenerate_gram(inverse_root, gram):
    # Finiteness is the guaranteed property: the degenerate subspace is scaled by eps^-1/2..-1/4, which
    # amplifies roundoff too much for a meaningful symmetry tolerance, but NaN (the bug) must never occur.
    assert torch.isfinite(inverse_root(gram, torch.tensor(1e-8))).all()
