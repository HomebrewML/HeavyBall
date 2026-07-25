"""Preconditioner robustness on pathological-but-reachable gradients, and per-leaf isolation when a
gradient's square overflows fp32.

``test_feature_matrix`` exercises well-conditioned transformer weights; this covers the untested
surface where the gradient makes the accumulated covariance singular or degenerate: rank-one (singular
covariance), all-zero, identical-every-step (rank-one over time), and 1e-30 (underflowing) gradients.
Every matrix/preconditioner optimizer must keep parameters finite across these -- measured over enough
steps to force at least one preconditioner refresh.

The second test complements ``test_nan_isolation`` (which injects a NaN into one leaf's gradient): a
finite gradient whose square overflows fp32 drives the SOAP/Shampoo/PSGD covariance to ``inf`` and the
owning leaf to NaN -- allowed, exactly as an injected NaN is -- but it must not corrupt a slab-mate that
shares the same shape bucket.
"""

import pytest
import torch

import heavyball

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compile-first GPU path")

MATRIX_OPTS = ["SOAP", "Shampoo", "KLSOAP", "SOAPNAdam", "PSGDKron", "PSGDPro", "QSGD", "LATHER"]
SHAPE = (32, 24)


@pytest.mark.parametrize("name", MATRIX_OPTS)
def test_reachable_pathological_gradients_stay_finite(name):
    g = torch.Generator(device="cuda").manual_seed(0)
    rows, columns = SHAPE
    left = torch.randn(rows, 1, generator=g, device="cuda")
    right = torch.randn(1, columns, generator=g, device="cuda")
    constant = torch.randn(SHAPE, generator=g, device="cuda")
    patterns = {
        "rank_one": lambda _: left @ right,
        "all_zero": lambda _: torch.zeros(SHAPE, device="cuda"),
        "constant": lambda _: constant,
        "underflow": lambda _: 1e-30 * constant,
    }
    for pattern, make_grad in patterns.items():
        weight = torch.nn.Parameter(torch.randn(SHAPE, generator=g, device="cuda"))
        opt = getattr(heavyball, name)([weight], lr=1e-3)
        for step in range(20):  # exceed any preconditioner-refresh cadence
            (weight * make_grad(step)).sum().backward()
            opt.step()
            opt.zero_grad()
        assert torch.isfinite(weight).all(), f"{name} produced non-finite params on {pattern} gradients"


@pytest.mark.parametrize("name", MATRIX_OPTS)
def test_overflowing_gradient_isolated_from_slabmate(name):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.nn.Parameter(torch.randn(SHAPE, generator=g, device="cuda"))
    b = torch.nn.Parameter(torch.randn(SHAPE, generator=g, device="cuda"))  # same shape -> same slab bucket
    opt = getattr(heavyball, name)([a, b], lr=1e-3)
    for _ in range(8):
        grad_a = 1e25 * torch.randn(SHAPE, generator=g, device="cuda")  # square overflows fp32
        grad_b = torch.randn(SHAPE, generator=g, device="cuda")
        ((a * grad_a).sum() + (b * grad_b).sum()).backward()
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(b).all(), f"{name}: an overflowing gradient on one leaf corrupted its slab-mate"
