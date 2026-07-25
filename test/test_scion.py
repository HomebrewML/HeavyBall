"""Parity and compile proofs for the slab-native Scion port."""

import math
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, Scion, scion, scion_lmo, scion_param_init, sgd
from heavyball.transforms import Tempo


def _tempo(count: int, dtype: torch.dtype, *, scale: float = 1.25) -> Tempo:
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(
            eps=torch.tensor(1e-8, dtype=dtype),
            scale=torch.tensor(scale, dtype=dtype),
        ),
        False,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float64, 1e-12, 1e-12), (torch.float32, 1e-6, 1e-6)),
)
def test_scion_lmo_matches_closed_form_for_each_leaf_rank(dtype, rtol, atol):
    """Recompute vector and spectral LMO branches from norm and SVD definitions."""

    for shape in ((5, 3), (4, 3, 2, 3), (7,)):
        torch.manual_seed(202)
        update = torch.randn(3, *shape, dtype=dtype)
        actual, _, _ = scion_lmo(update.clone(), None, None, {}, _tempo(update.shape[0], dtype))
        if len(shape) == 1:
            expected = update / torch.linalg.vector_norm(update, dim=1, keepdim=True)
            expected = expected * (1.25 * math.sqrt(shape[0]))
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            continue

        flat = update.reshape(update.shape[0], shape[0], -1)
        u, _, vh = torch.linalg.svd(flat, full_matrices=False)
        polar = u @ vh
        if len(shape) >= 3:
            spatial = math.prod(shape[2:])
            factor = 1.25 * math.sqrt(shape[0] / shape[1]) / spatial
        else:
            factor = 1.25 * math.sqrt(shape[0] / shape[1])
        expected = (polar * factor).reshape_as(actual)
        relative_error = (actual - expected).norm() / expected.norm()
        assert relative_error < 0.05


def test_scion_applies_lmo_to_debiased_first_moment():
    parameter = torch.nn.Parameter(torch.tensor((9.0, -7.0), dtype=torch.float64))
    beta1 = 0.9
    lr = 0.1
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Scion([parameter], lr=lr, beta1=beta1, eps=1e-8, scale=1.0, weight_decay=0.0)
    bootstrap = torch.tensor((-3.0, 4.0), dtype=torch.float64)
    gradients = (
        torch.tensor((10.0, 0.0), dtype=torch.float64),
        torch.tensor((0.0, 1.0), dtype=torch.float64),
    )
    for gradient in (bootstrap, *gradients):
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()

    debiased_beta = beta1 * (1 - beta1) / (1 - beta1 ** 2)
    averaged = debiased_beta * gradients[0] + (1 - debiased_beta) * gradients[1]
    expected = lr * averaged / torch.linalg.vector_norm(averaged) * math.sqrt(averaged.numel())
    momentum_free = lr * gradients[1] / torch.linalg.vector_norm(gradients[1]) * math.sqrt(gradients[1].numel())
    actual = before - parameter

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert not torch.allclose(actual, momentum_free, rtol=0, atol=1e-12)
    assert "exp_avg" in optimizer._engine.groups[0].states[0]


def _reference_scion_init(value: torch.Tensor, *, seed: int, scale: float) -> torch.Tensor:
    """Directly apply seeded orthogonal slices and the Scion fan scaling."""

    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    if value.ndim < 2:
        return torch.zeros_like(value)
    expected = value.double().clone()
    for spatial_index in product(*(range(size) for size in expected.shape[2:])):
        torch.nn.init.orthogonal_(
            expected[(slice(None), slice(None), *spatial_index)],
            generator=generator,
        )
    spatial = math.prod(expected.shape[2:])
    expected.mul_(math.sqrt(expected.shape[0] / expected.shape[1]) / max(spatial, 1))
    return expected.to(value.dtype) * scale


@pytest.mark.parametrize("shape", ((5, 3), (4, 3, 2, 3), (7,)))
def test_scion_param_init_follows_seeded_orthogonal_slice_recipe(shape):
    """The initializer follows the independently recomputed PyTorch orthogonal recipe."""

    seed = 17
    actual = torch.randn(*shape, dtype=torch.float64)
    expected = _reference_scion_init(actual, seed=seed, scale=1.0)
    scion_param_init(actual, seed=seed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _scion_trajectory(dtype: torch.dtype, grad_sequence, *, compiled: bool):
    params = [
        torch.nn.Parameter(torch.zeros(7, 4, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(4, 3, 2, 2, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(9, dtype=dtype)),
    ]
    if compiled:
        optimizer = Engine(params, scion, lr=0.02, eps=1e-8, scale=1.0)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine(params, scion, lr=0.02, eps=1e-8, scale=1.0)
    for gradients in grad_sequence:
        for param, gradient in zip(params, gradients, strict=True):
            param.grad.copy_(gradient.to(dtype))
        optimizer.step()
    return [param.detach().clone() for param in params]


def test_scion_fp64_accuracy():
    """Compiled fp32 stays within the pinned fp64 trajectory budget."""

    torch._dynamo.reset()
    torch.manual_seed(401)
    shapes = ((7, 4), (4, 3, 2, 2), (9,))
    gradients = [[torch.randn(*shape, dtype=torch.float64) for shape in shapes] for _ in range(30)]
    truth = _scion_trajectory(torch.float64, gradients, compiled=False)
    torch.manual_seed(402)
    actual = _scion_trajectory(torch.float32, gradients, compiled=True)
    error = max(
        (result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True)
    )
    assert error <= 2e-3
    torch._dynamo.reset()


def test_scion_param_init_is_deferred_and_uses_global_parameter_order():
    """Construction is non-mutating; first step initializes by supplied parameter order."""

    params = [
        torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64), requires_grad=False),
        torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64)),
        torch.nn.Parameter(torch.randn(5, dtype=torch.float64)),
        torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64)),
    ]
    before = [param.detach().clone() for param in params]
    init_scale = 1.75
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(params, scion, scale=init_scale, lr=0.0)

    for param, initial in zip(params, before, strict=True):
        torch.testing.assert_close(param, initial, rtol=0, atol=0)
    for param in params[1:]:
        param.grad.zero_()
    optimizer.step()

    torch.testing.assert_close(params[0], before[0], rtol=0, atol=0)
    for seed, (param, initial) in enumerate(zip(params[1:], before[1:], strict=True), start=1):
        expected = _reference_scion_init(initial, seed=seed, scale=init_scale)
        assert not torch.equal(param, initial)
        torch.testing.assert_close(param, expected, rtol=0, atol=0)

    untouched = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
    before_untouched = untouched.detach().clone()
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        Engine([untouched], sgd)
    torch.testing.assert_close(untouched, before_untouched, rtol=0, atol=0)


def test_scion_executes_one_stable_fullgraph_after_deferred_init():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [
            torch.nn.Parameter(torch.randn(5, 3)),
            torch.nn.Parameter(torch.randn(4, 3, 2, 3)),
            torch.nn.Parameter(torch.randn(7)),
        ]
        optimizer = Engine(params, scion)
        gradients = [torch.linspace(-1, 1, param.numel()).reshape_as(param) for param in params]

        for param, gradient in zip(params, gradients, strict=True):
            param.grad.copy_(gradient)
        optimizer.step()  # deferred seeded initialization
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 0

        for _ in range(2):
            for param, gradient in zip(params, gradients, strict=True):
                param.grad.copy_(gradient)
            optimizer.step()

        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
        assert all(torch.isfinite(param).all() for param in params)
    finally:
        torch._dynamo.reset()
@pytest.mark.parametrize(("rows", "cols"), ((4, 6), (6, 4), (5, 5)))
def test_scion_lmo_is_the_scaled_spectral_polar(rows, cols):
    """Scion's defining spectral-norm LMO (arXiv 2502.07529): the matrix update is the norm-ball
    linear-minimization-oracle output, scale * sqrt(fan_out/fan_in) * polar(M), where polar(M) = U @ Vh
    from M = U diag(S) Vh. So a wide input spectrum is driven to near-constant singular values (a
    semi-orthogonal matrix). Checked against torch.linalg.svd's polar factor and the closed-form scale;
    the shipped Newton-Schulz orthogonalizer approximates the polar in 5 steps, hence the 5% band."""

    torch.manual_seed(1)
    left = torch.linalg.qr(torch.randn(rows, rows, dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(cols, cols, dtype=torch.float64))[0]
    rank = min(rows, cols)
    spectrum = torch.linspace(1.0, 30.0, rank, dtype=torch.float64)  # condition 30: wide, still NS-5 tractable
    update = ((left[:, :rank] * spectrum) @ right[:, :rank].T).unsqueeze(0)  # [1, rows, cols] matrix leaf
    u, _, vh = torch.linalg.svd(update[0], full_matrices=False)
    polar = u @ vh

    for scale in (0.5, 1.0, 2.0):
        actual, _, _ = scion_lmo(update.clone(), None, None, {}, _tempo(1, torch.float64, scale=scale))
        expected = scale * math.sqrt(rows / cols) * polar
        singular = torch.linalg.svdvals(actual[0])
        assert singular.max() / singular.min() < 1.15  # the wide input spectrum is norm-constrained to ~constant
        assert (actual[0] - expected).norm() / actual[0].norm() < 0.05  # the scaled polar, vs an independent SVD
