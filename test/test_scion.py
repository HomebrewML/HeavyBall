"""Parity and compile proofs for the slab-native Scion port."""

import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, Scion, scion, scion_lmo, scion_param_init, sgd
from heavyball.transforms import Tempo


@contextmanager
def _legacy_eager():
    import heavyball_legacy.utils as legacy

    previous = legacy.compile_mode
    legacy.compile_mode = None
    try:
        yield legacy
    finally:
        legacy.compile_mode = previous


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
def test_scion_lmo_matches_legacy(dtype, rtol, atol):
    """The batched dispatch must match legacy's per-leaf auto LMO."""

    for shape in ((5, 3), (4, 3, 2, 3), (7,)):
        torch.manual_seed(202)
        update = torch.randn(3, *shape, dtype=dtype)
        expected = update.clone()
        with _legacy_eager() as legacy:
            torch.manual_seed(301)
            legacy.scion_auto_lmo_(
                [expected[index] for index in range(expected.shape[0])],
                torch.tensor(1.25, dtype=dtype),
                torch.tensor(1e-8, dtype=dtype),
            )
        torch.manual_seed(301)
        actual, _, _ = scion_lmo(update.clone(), None, None, {}, _tempo(update.shape[0], dtype))
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_scion_applies_lmo_to_debiased_first_moment():
    parameter = torch.nn.Parameter(torch.tensor((9.0, -7.0), dtype=torch.float64))
    beta1 = 0.9
    lr = 0.1
    optimizer = Scion([parameter], lr=lr, beta1=beta1, eps=1e-8, scale=1.0, weight_decay=0.0)
    gradients = (
        torch.tensor((10.0, 0.0), dtype=torch.float64),
        torch.tensor((0.0, 1.0), dtype=torch.float64),
    )
    for gradient in gradients:
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


@pytest.mark.parametrize("shape", ((5, 3), (4, 3, 2, 3), (7,)))
def test_scion_param_init_matches_legacy(shape):
    """The build hook's per-leaf initializer has legacy's seeded result."""

    seed = 17
    expected = torch.randn(*shape, dtype=torch.float64)
    actual = expected.clone()
    with _legacy_eager() as legacy:
        legacy.scion_auto_init_param_(expected, torch.tensor(1.0, dtype=torch.float64), seed=seed)
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


def test_scion_param_init_applied_at_build():
    """Build-time initialization uses global parameter order, not slab bucket order."""

    params = [
        torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64), requires_grad=False),
        torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64)),
        torch.nn.Parameter(torch.randn(5, dtype=torch.float64)),
        torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64)),
    ]
    before = [param.detach().clone() for param in params]
    init_scale = 1.75
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        Engine(params, scion, scale=init_scale)

    torch.testing.assert_close(params[0], before[0], rtol=0, atol=0)
    for seed, (param, initial) in enumerate(zip(params[1:], before[1:], strict=True), start=1):
        expected = initial.clone()
        with _legacy_eager() as legacy:
            legacy.scion_auto_init_param_(expected, torch.tensor(init_scale, dtype=torch.float64), seed=seed)
        assert not torch.equal(param, initial)
        torch.testing.assert_close(param, expected, rtol=0, atol=0)

    untouched = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
    before_untouched = untouched.detach().clone()
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        Engine([untouched], sgd)
    torch.testing.assert_close(untouched, before_untouched, rtol=0, atol=0)


def test_scion_fullgraph_clean(tmp_path):
    source = """
import torch
from heavyball import Engine, scion

params = [
    torch.nn.Parameter(torch.randn(5, 3)),
    torch.nn.Parameter(torch.randn(4, 3, 2, 3)),
    torch.nn.Parameter(torch.randn(7)),
]
optimizer = Engine(params, scion)
for param in params:
    param.grad.normal_()
optimizer.step()
"""
    # max-autotune's cpp-gemm autotuning can crash this subprocess under full-suite CPU load; the
    # graph-structure invariant asserted below is autotune-independent, so retry a transient nonzero
    # exit (surfaced) and assert on a clean compile. A real Scion regression crashes every attempt.
    for attempt in range(3):
        env = dict(
            os.environ,
            TORCH_LOGS="output_code",
            TORCHINDUCTOR_FX_GRAPH_CACHE="0",
            TORCHINDUCTOR_CACHE_DIR=str(tmp_path / f"inductor-{attempt}"),
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break
        print(f"scion fullgraph subprocess crashed on attempt {attempt + 1}/3, retrying:\n{result.stdout + result.stderr}")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Output code:" in output
    assert output.count("AOT ID:") == 1
    for path in {Path(value) for value in re.findall(r"Output code written to: (.*\.py)", output)}:
        code = path.read_text()
        assert "stack" not in code
        assert "_local_scalar_dense" not in code
        assert ".item(" not in code

    banned = subprocess.run(
        [
            "grep",
            "-rnE",
            r"_foreach|vmap|torch\.cond|while_loop|dynamic=True|torch\.stack|\.item\(|autocast",
            "heavyball/scion.py",
            "heavyball/core.py",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert banned.returncode == 1, banned.stdout + banned.stderr
    imports = subprocess.run(
        ["grep", "-rn", r"heavyball_legacy.utils\|heavyball_legacy.chainable", "heavyball/scion.py", "heavyball/core.py"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert imports.returncode == 1, imports.stdout + imports.stderr


@pytest.mark.parametrize(("rows", "cols"), ((4, 6), (6, 4), (5, 5)))
def test_scion_lmo_is_the_scaled_spectral_polar(rows, cols):
    """Scion's defining spectral-norm LMO (arXiv 2502.07529): the matrix update is the norm-ball
    linear-minimization-oracle output, scale * sqrt(fan_out/fan_in) * polar(M), where polar(M) = U @ Vh
    from M = U diag(S) Vh. So a wide input spectrum is driven to near-constant singular values (a
    semi-orthogonal matrix). Checked against torch.linalg.svd's polar factor and the closed-form scale --
    independent of legacy AND of the shipped Newton-Schulz orthogonalize, which only approximates the
    polar in 5 steps (hence the 5% band). test_scion_lmo_matches_legacy pins this branch only against a
    second HeavyBall implementation, so a bug shared by both would pass it; this cannot."""

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
