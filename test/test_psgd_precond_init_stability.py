import sys

import numpy as np
import pytest
import torch
from torch._dynamo import config

from heavyball.utils import _lse_mean, divided_root, mean_root, stable_exp

config.cache_size_limit = sys.maxsize
config.accumulated_cache_size_limit = sys.maxsize


def np_mean_pow_root(z_np: np.ndarray, pow_val: float, eps_val: float):
    z_np = np.abs(z_np).clip(min=eps_val) ** pow_val
    mean_pow = np.mean(z_np) ** (1 / pow_val / 2)
    return mean_pow


def _get_tensor(numel, dtype, maxval: float = 10):
    x = torch.randn(numel, dtype=dtype)  # * torch.arange(numel, dtype=dtype)
    x /= x.abs().max()
    x *= maxval
    x = x.clone()
    return x, x.numpy().astype(np.float128)


tolerance = {
    np.float64: {"rtol": 1e-10, "atol": 1e-12},
    np.float32: {"rtol": 1e-5, "atol": 1e-6},
    np.float16: {"rtol": 1e-2, "atol": 1e-3},
}


def _isclose(x, y):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if not np.isfinite(x).all():
        assert ~(np.isfinite(x) ^ np.isfinite(y)).any()  # all are nan together
    if np.isfinite(x).any():
        for k, v in tolerance.items():  # numpy doesn't support indexing
            if x.dtype == k:
                tol = v
                break
        else:
            raise ValueError(f"dtype {x.dtype} not supported")
        assert np.allclose(x[np.isfinite(x)], y[np.isfinite(y)], **tol)


@pytest.mark.parametrize("x_val", list(range(-10, 10)))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_stable_exp_scalar(x_val, dtype):
    x = torch.tensor(x_val, dtype=dtype)
    result = stable_exp(x)
    expected = np.exp(x_val) if x_val <= 0 else 1 / np.exp(-x_val)
    _isclose(result.to(x.dtype), expected)


@pytest.mark.parametrize("numel", [1, 4, 64, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_stable_exp_tensor(numel, dtype):
    x, x_np = _get_tensor(numel, dtype)
    result = stable_exp(x)
    expected = np.exp(x_np)
    _isclose(result.to(x.dtype), expected)


@pytest.mark.parametrize("numel", [1, 4, 64, 512])
@pytest.mark.parametrize("pow_val", [1, 3, 7, 15])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_lse_mean(numel, pow_val, dtype):
    x, x_np = _get_tensor(numel, dtype)
    result = _lse_mean(x, pow_val, 1e-20)
    expected = np.log(np.mean(np.abs(x_np) ** pow_val)) / pow_val / 2
    _isclose(result.to(x.dtype), expected)


@pytest.mark.parametrize("numel", [1, 4, 64, 512])
@pytest.mark.parametrize("pow_val", [1, 3, 7, 15])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_mean_root(numel, pow_val, dtype):
    x, x_np = _get_tensor(numel, dtype)
    result = mean_root(x, pow_val)
    expected = 1 / (np.mean(np.abs(x_np) ** pow_val) ** (1 / pow_val / 2))
    _isclose(result.to(x.dtype), expected)


@pytest.mark.parametrize("numel", [1, 4, 64, 512])
@pytest.mark.parametrize("pow0_val", [1, 5, 15])
@pytest.mark.parametrize("pow1_val", [1, 5, 15])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_divided_root(numel, pow0_val, pow1_val, dtype):
    x, x_np = _get_tensor(numel, dtype)
    y, y_np = _get_tensor(numel, dtype)
    result = divided_root(x, y, pow0_val, pow1_val)
    expected = np_mean_pow_root(x_np, pow0_val, 1e-12) / np_mean_pow_root(y_np, pow1_val, 1e-12)
    _isclose(result.to(x.dtype), expected)
