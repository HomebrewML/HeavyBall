import pytest
import torch
from torch._dynamo import config

from heavyball.utils import _max_singular_value_ndim, max_singular_value, min_singular_value

config.cache_size_limit = 2**20
config.accumulated_cache_size_limit = 2**20


def hilbert_matrix(n):
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(0)
    return 1.0 / (i + j - 1).cuda()


def _make_matrix(shape, cond=10, dtype=torch.float32, symmetric=False, seed=0):
    torch.manual_seed(seed)
    m, n = shape
    r = min(m, n)
    q_left, _ = torch.linalg.qr(torch.randn(m, r, dtype=torch.float32))
    q_right, _ = torch.linalg.qr(torch.randn(n, r, dtype=torch.float32))
    exponents = torch.linspace(0, -1, r, dtype=torch.float32)
    spectrum = cond**exponents
    diag = torch.diag(spectrum)
    if symmetric:
        if m != n:
            raise ValueError("symmetric=True requires a square matrix")
        return (q_left @ diag @ q_left.T).contiguous().to(dtype).cuda()
    return (q_left @ diag @ q_right.T).contiguous().to(dtype).cuda()


def assert_close(x, y, atol: None | float = None, rtol: None | float = None):
    torch.testing.assert_close(x.double(), y.double(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", ((4, 4), (32, 32), (128, 128), (10, 5), (5, 10)))
@pytest.mark.parametrize("cond", (1, 10, 1e4, 1e10, 1e18, 1e30, 1e300))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("power_iter", (0, 5, 20))
def test_max_singular_value(shape, cond, dtype, power_iter):
    rtol = 1 / (power_iter + 1) * (0.1 if power_iter else 0.01)
    A = _make_matrix(shape, cond=cond, dtype=dtype)
    approx = max_singular_value(A, power_iter=power_iter)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("shape", ((4, 4), (32, 32), (128, 128)))
@pytest.mark.parametrize("cond", (1, 10, 1e4, 1e10, 1e18, 1e30, 1e300))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("power_iter", (0, 5, 20))
def test_min_singular_value(shape, cond, dtype, power_iter):
    rtol = 1 / (power_iter + 1) * (1 if dtype == torch.bfloat16 else 0.1)
    A = _make_matrix(shape, cond=cond, dtype=dtype, symmetric=True)
    approx = min_singular_value(A, power_iter=power_iter)
    exact = torch.linalg.svdvals(A.double()).min()
    if exact.abs() < 1e-8:
        assert_close(approx, exact, atol=1e-6)
    else:
        assert_close(approx, exact, rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("shape", ((3, 4, 5), (16, 32, 64), (16, 16, 512)))
def test_max_singular_value_ndim(shape, bound: float = 2):
    torch.manual_seed(0)
    A = torch.randn(shape).cuda()
    approx = _max_singular_value_ndim(A, power_iter=2)
    exact = torch.linalg.svdvals(A.double()).max()
    assert (approx.double() > exact.double()).item()
    assert (exact.double() * bound > approx.double()).item()


@pytest.mark.parametrize("shape", ((32, 32), (128, 128), (512, 512)))
def test_max_singular_value_rank_deficient(shape):
    A = torch.randn(shape).cuda()
    A[:, -1] = 0.0
    approx = max_singular_value(A, power_iter=10)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, atol=1e-6, rtol=0.1)


@pytest.mark.parametrize("shape", ((4, 4), (32, 32), (128, 128), (512, 512)))
def test_max_singular_value_ill_conditioned(shape):
    A = hilbert_matrix(shape[0])
    approx = max_singular_value(A, power_iter=10)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, atol=1e-6, rtol=0.1)
