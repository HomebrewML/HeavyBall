"""Direct legacy parity and numerical proofs for the slab-native SUDS port."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import heavyball
import heavyball_legacy
from heavyball import Engine, suds, suds_adamw
from heavyball.suds import eigvecs_product_rank1, oja_update, stable_l2_normalize


@contextmanager
def _legacy_eager():
    import heavyball_legacy.utils as legacy

    previous = legacy.compile_mode
    legacy.compile_mode = None
    try:
        yield
    finally:
        legacy.compile_mode = previous


def _copy_grads(params, gradients) -> None:
    for parameter, gradient in zip(params, gradients, strict=True):
        parameter.grad.copy_(gradient)


@pytest.mark.xfail(
    reason="slab-native SUDS now transports Adam moments across Householder basis changes; "
    "legacy omits this transport, so parity is intentionally broken (see test_suds_transport.py)",
    strict=True,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol", "storage_dtype"),
    (
        (torch.float64, 1e-10, 1e-10, "float64"),
        (torch.float32, 2e-5, 2e-5, "float32"),
    ),
)
def test_suds_matches_legacy(dtype, rtol, atol, storage_dtype):
    """Every parameter delta directly matches legacy ``SUDSAdamW`` step by step."""

    torch._dynamo.reset()
    torch.manual_seed(101)
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        precond_lr=0.13,
    )
    initial = [torch.randn(3, 4, dtype=dtype), torch.randn(3, 4, dtype=dtype), torch.randn((), dtype=dtype)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(9)]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    legacy_params = [torch.nn.Parameter(value.clone()) for value in initial]
    try:
        optimizer = Engine(params, suds_adamw, **values)
        with _legacy_eager():
            legacy_optimizer = heavyball_legacy.SUDSAdamW(
                legacy_params,
                lr=values["lr"],
                betas=(values["beta1"], values["beta2"]),
                eps=values["eps"],
                weight_decay=values["weight_decay"],
                precond_lr=values["precond_lr"],
                storage_dtype=storage_dtype,
                compile_step=False,
            )
            for step, step_gradients in enumerate(gradients, start=1):
                before = [parameter.detach().clone() for parameter in params]
                legacy_before = [parameter.detach().clone() for parameter in legacy_params]
                _copy_grads(params, step_gradients)
                for parameter, gradient in zip(legacy_params, step_gradients, strict=True):
                    parameter.grad = gradient.clone()

                optimizer.step()
                legacy_optimizer.step()

                for index, (parameter, legacy_parameter, prior, legacy_prior) in enumerate(
                    zip(params, legacy_params, before, legacy_before, strict=True)
                ):
                    torch.testing.assert_close(
                        prior - parameter,
                        legacy_prior - legacy_parameter,
                        rtol=rtol,
                        atol=atol,
                        msg=f"step {step}, parameter delta {index}",
                    )
                    torch.testing.assert_close(
                        parameter,
                        legacy_parameter,
                        rtol=rtol,
                        atol=atol,
                        msg=f"step {step}, parameter {index}",
                    )
    finally:
        torch._dynamo.reset()


def _trajectory(dtype: torch.dtype, gradients, *, compiled: bool):
    values = dict(lr=0.017, beta1=0.87, beta2=0.97, eps=1e-8, weight_decay=0.031, precond_lr=0.13)
    params = [
        torch.nn.Parameter(torch.zeros(11, 7, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(4, dtype=dtype)),
        torch.nn.Parameter(torch.zeros((), dtype=dtype)),
    ]
    if compiled:
        optimizer = Engine(params, suds_adamw, **values)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine(params, suds_adamw, **values)
    for step_gradients in gradients:
        _copy_grads(params, [gradient.to(dtype) for gradient in step_gradients])
        optimizer.step()
    return [parameter.detach().clone() for parameter in params]


def test_suds_fp64_accuracy():
    """The compiled fp32 trajectory remains within the pinned fp64 SUDS budget."""

    torch._dynamo.reset()
    torch.manual_seed(102)
    gradients = [
        [
            torch.randn(11, 7, dtype=torch.float64),
            torch.randn(4, dtype=torch.float64),
            torch.randn((), dtype=torch.float64),
        ]
        for _ in range(80)
    ]
    try:
        truth = _trajectory(torch.float64, gradients, compiled=False)
        actual = _trajectory(torch.float32, gradients, compiled=True)
        error = max((result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True))
        assert error <= 3e-5
    finally:
        torch._dynamo.reset()


def test_suds_householder_rotation():
    """The batched reflector matches an explicit Householder matrix and round-trips."""

    torch.manual_seed(103)
    gradient = torch.randn(3, 9, dtype=torch.float64)
    direction = torch.randn_like(gradient)
    rotated, w = eigvecs_product_rank1(gradient, direction)
    round_trip, cached_w = eigvecs_product_rank1(rotated, direction, w)
    reflector = torch.eye(gradient.shape[-1], dtype=gradient.dtype).expand(gradient.shape[0], -1, -1)
    reflector = reflector - 2 * w.unsqueeze(-1) * w.unsqueeze(-2)
    reference = torch.einsum("ni,nij->nj", gradient, reflector)

    torch.testing.assert_close(rotated, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(round_trip, gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cached_w, w, rtol=0, atol=0)
    expected_direction = stable_l2_normalize(direction, dim=-1, eps=1e-12)
    first_column = reflector[..., 0]
    torch.testing.assert_close(first_column, expected_direction, rtol=1e-12, atol=1e-12)


def _compiled_code(tmp_path: Path) -> str:
    source = """
import torch
from heavyball import Engine, suds_adamw

params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
optimizer = Engine(params, suds_adamw, lr=0.01, precond_lr=0.05, weight_decay=0.0)
for _ in range(3):
    for parameter in params:
        parameter.grad.normal_()
    optimizer.step()
"""
    environment = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    paths = [Path(value) for value in re.findall(r"Output code written to: (.*\.py)", output)]
    assert paths, output
    return paths[-1].read_text()


def test_suds_fullgraph_clean(tmp_path):
    """SUDS compiles as a scalar-free full graph and has no legacy dependency."""

    artifact = _compiled_code(tmp_path).lower()
    for forbidden in (".item(", "while_loop", "torch.cond", "cond", "torch.stack", "stack", "_local_scalar_dense"):
        assert forbidden not in artifact

    source = Path(suds.__module__.replace(".", "/") + ".py")
    if not source.exists():
        source = Path(__file__).parents[1] / "heavyball" / "suds.py"
    text = source.read_text()
    assert not re.search(r"_foreach|vmap|torch\.cond|while_loop|dynamic=True|torch\.stack|\.item\(|autocast", text)
    assert "heavyball_legacy.utils" not in text
    assert "heavyball_legacy.chainable" not in text


def test_suds_oja_update_converges_to_the_top_eigenvector():
    """SUDS learns its rank-1 Fisher direction by an Oja power-iteration step (oja_update advances
    state['fisher_approx'] every step). Its defining property: fed a gradient stream whose covariance has a
    dominant direction u (cov = I + 20 uu^T, eigen-gap 21:1), the iterate converges to +-u. Checked against
    a hand-constructed top eigenvector -- independent of legacy AND of the shipped step. The isotropic
    control (no dominant direction) must NOT converge to u, so the alignment is the learned signal, not an
    artifact. test_suds_matches_legacy pins this learning only against a second HeavyBall implementation."""

    dimension = 8
    torch.manual_seed(0)
    target = torch.randn(dimension, dtype=torch.float64)
    target = target / torch.linalg.vector_norm(target)
    root = torch.eye(dimension, dtype=torch.float64) + (21.0 ** 0.5 - 1.0) * torch.outer(target, target)

    def mean_alignment(seed, sample):
        torch.manual_seed(seed)
        iterate = torch.randn(1, dimension, dtype=torch.float64)
        alignments = []
        for step in range(4000):
            iterate = oja_update(iterate, sample(), lr=0.02)
            if step >= 3000:
                alignments.append((iterate[0] @ target).abs() / torch.linalg.vector_norm(iterate[0]))
        return torch.stack(alignments).mean()

    dominant = mean_alignment(1, lambda: (root @ torch.randn(dimension, dtype=torch.float64)).unsqueeze(0))
    isotropic = mean_alignment(2, lambda: torch.randn(1, dimension, dtype=torch.float64))
    assert dominant > 0.9  # the iterate learns the dominant covariance direction
    assert isotropic < 0.6  # with no dominant direction it does not converge to target
