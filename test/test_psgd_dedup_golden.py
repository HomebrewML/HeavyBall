import torch

from heavyball.psgd_pro import _precondition_nfactor


def test_nfactor_mixed_factor_application_matches_dense_axis_products():
    generator = torch.Generator().manual_seed(20260723)
    update = torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
    triangular0 = torch.triu(
        torch.randn(1, 2, 2, generator=generator, dtype=torch.float64)
    )
    diagonal1 = torch.rand(1, 3, generator=generator, dtype=torch.float64) + 0.5
    triangular2 = torch.triu(
        torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
    )
    factors = [triangular0, diagonal1, triangular2]

    expected = update[0]
    dense_factors = [
        triangular0[0].mT @ triangular0[0],
        torch.diag(diagonal1[0].square()),
        triangular2[0].mT @ triangular2[0],
    ]
    for axis, factor in enumerate(dense_factors):
        expected = torch.tensordot(factor, expected, dims=([1], [axis])).movedim(0, axis)

    torch.testing.assert_close(
        _precondition_nfactor(update, factors)[0],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
