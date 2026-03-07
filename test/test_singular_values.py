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


# Pareto-optimal smoke tests: known-good (dtype, power_iter, shape, cond) combos.
# Full sweep lives in benchmarks/bench_singular_values.py.


@pytest.mark.parametrize(
    "shape,cond,dtype,power_iter,rtol",
    [
        # fp64 with power iterations is the gold standard
        ((128, 128), 1e10, torch.float64, 20, 0.005),
        ((32, 32), 1e4, torch.float64, 5, 0.02),
        # fp32 with moderate power iterations
        ((128, 128), 10, torch.float32, 20, 0.02),
        ((32, 32), 1e4, torch.float32, 5, 0.02),
        ((4, 4), 1e10, torch.float32, 5, 0.02),
        # bf16 with power iterations
        ((32, 32), 10, torch.bfloat16, 5, 0.1),
        ((4, 4), 1, torch.bfloat16, 5, 0.1),
    ],
)
def test_max_singular_value(shape, cond, dtype, power_iter, rtol):
    A = _make_matrix(shape, cond=cond, dtype=dtype)
    approx = max_singular_value(A, power_iter=power_iter)
    exact = torch.linalg.svdvals(A.double()).max()
    assert_close(approx, exact, rtol=rtol, atol=1e-5)


@pytest.mark.parametrize(
    "shape,cond,dtype,power_iter,rtol",
    [
        ((32, 32), 1, torch.float64, 20, 0.01),
        ((32, 32), 10, torch.float64, 5, 0.1),
        ((4, 4), 1e4, torch.float64, 5, 0.1),
        ((32, 32), 1, torch.float32, 20, 0.01),
        ((4, 4), 10, torch.float32, 5, 0.1),
    ],
)
def test_min_singular_value(shape, cond, dtype, power_iter, rtol):
    A = _make_matrix(shape, cond=cond, dtype=dtype, symmetric=True)
    approx = min_singular_value(A, power_iter=power_iter)
    exact = torch.linalg.svdvals(A.double()).min()
    if exact.abs() < 1e-8:
        assert_close(approx, exact, atol=1e-6)
    else:
        assert_close(approx, exact, rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("shape", ((3, 4, 5),))
def test_max_singular_value_ndim(shape, bound: float = 2):
    torch.manual_seed(0x172893)
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
