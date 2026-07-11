import pytest
import torch

from heavyball.utils import max_singular_value, max_singular_value_exact


def hilbert_matrix(n):
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(0)
    return 1.0 / (i + j - 1)


def _make_matrix(shape, cond=10, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    m, n = shape
    r = min(m, n)
    q_left, _ = torch.linalg.qr(torch.randn(m, r, dtype=torch.float32))
    q_right, _ = torch.linalg.qr(torch.randn(n, r, dtype=torch.float32))
    exponents = torch.linspace(0, -1, r, dtype=torch.float32)
    spectrum = cond**exponents
    diag = torch.diag(spectrum)
    return (q_left @ diag @ q_right.T).contiguous().to(dtype)


def assert_close(x, y, atol: None | float = None, rtol: None | float = None):
    torch.testing.assert_close(x.double(), y.double(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape,cond,dtype,rtol",
    [
        ((32, 32), 1e10, torch.float64, 1e-13),
        ((32, 32), 1e4, torch.float32, 1e-5),
        ((4, 4), 1e10, torch.float32, 1e-5),
        ((32, 32), 10, torch.bfloat16, 5e-4),
        ((4, 4), 1, torch.bfloat16, 5e-4),
    ],
)
def test_max_singular_value(shape, cond, dtype, rtol):
    A = _make_matrix(shape, cond=cond, dtype=dtype)
    approx = max_singular_value(A)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, rtol=rtol, atol=0)
    assert_close(max_singular_value_exact(A), exact, rtol=1e-5 if dtype != torch.float64 else 1e-13, atol=0)


def test_max_singular_value_rank_deficient():
    A = torch.randn(32, 32)
    A[:, -1] = 0.0
    approx = max_singular_value(A)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, atol=0, rtol=5e-5)


@pytest.mark.parametrize("shape", ((4, 4), (32, 32)))
def test_max_singular_value_ill_conditioned(shape):
    A = hilbert_matrix(shape[0])
    approx = max_singular_value(A)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, atol=0, rtol=1e-13)
