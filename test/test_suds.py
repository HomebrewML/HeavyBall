"""Direct legacy parity and numerical proofs for the slab-native SUDS port."""

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, suds, suds_adamw
from heavyball.suds import eigvecs_product_rank1, oja_update, stable_l2_normalize


def _copy_grads(params, gradients) -> None:
    for parameter, gradient in zip(params, gradients, strict=True):
        parameter.grad.copy_(gradient)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-11, 1e-11),
        (torch.float32, 2e-5, 2e-5),
    ),
)
@pytest.mark.parametrize("shape", ((3, 4), (4,), ()))
def test_suds_bootstrap_and_second_step_follow_householder_adam_math(dtype, rtol, atol, shape):
    """Recompute the bootstrap, Householder direction, and Oja state without an optimizer oracle."""

    torch.manual_seed(101 + len(shape))
    lr, eps, precond_lr = 0.017, 1e-8, 0.13
    parameter = torch.nn.Parameter(torch.randn(shape, dtype=dtype))
    initial = parameter.detach().clone()
    first_gradient = torch.randn_like(parameter)
    second_gradient = torch.randn_like(parameter)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(
            [parameter],
            suds_adamw,
            lr=lr,
            beta1=0.87,
            beta2=0.97,
            eps=eps,
            weight_decay=0.0,
            precond_lr=precond_lr,
        )
    state = optimizer.groups[0].states[0]

    parameter.grad.copy_(first_gradient)
    optimizer.step()
    torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
    fisher = first_gradient.reshape(1, -1)
    fisher = fisher / torch.linalg.vector_norm(fisher, dim=1, keepdim=True)
    torch.testing.assert_close(state["fisher_approx"].reshape(1, -1), fisher, rtol=rtol, atol=atol)
    assert state["seen"].all()
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0

    e1 = torch.zeros_like(fisher)
    e1[:, 0] = 1
    w = e1 - fisher
    w_norm = torch.linalg.vector_norm(w, dim=1, keepdim=True)
    w = torch.where(w_norm >= 1e-12, w / w_norm.clamp_min(1e-12), torch.zeros_like(w))
    identity = torch.eye(fisher.shape[1], dtype=dtype).unsqueeze(0)
    reflector = identity - 2 * w.unsqueeze(-1) * w.unsqueeze(-2)
    flat_gradient = second_gradient.reshape(1, -1)
    rotated = torch.bmm(flat_gradient.unsqueeze(1), reflector).squeeze(1)
    adam_direction = rotated / rotated.abs().clamp_min(eps**0.5)
    expected_direction = torch.bmm(adam_direction.unsqueeze(1), reflector).squeeze(1).reshape(shape)
    projection = (flat_gradient * fisher).sum(dim=1, keepdim=True)
    next_fisher = fisher + precond_lr * projection * (flat_gradient - projection * fisher)
    next_fisher = next_fisher / torch.linalg.vector_norm(next_fisher, dim=1, keepdim=True)

    before = parameter.detach().clone()
    parameter.grad.copy_(second_gradient)
    optimizer.step()

    torch.testing.assert_close((before - parameter) / lr, expected_direction, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        state["fisher_approx"].reshape(1, -1),
        next_fisher,
        rtol=rtol,
        atol=atol,
    )


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
    """SUDS compiles as a scalar-free full graph."""

    artifact = _compiled_code(tmp_path).lower()
    for forbidden in (".item(", "while_loop", "torch.cond", "cond", "torch.stack", "stack", "_local_scalar_dense"):
        assert forbidden not in artifact

    source = Path(suds.__module__.replace(".", "/") + ".py")
    if not source.exists():
        source = Path(__file__).parents[1] / "heavyball" / "suds.py"
    text = source.read_text()
    assert not re.search(r"_foreach|vmap|torch\.cond|while_loop|dynamic=True|torch\.stack|\.item\(|autocast", text)


def test_suds_oja_update_converges_to_the_top_eigenvector():
    """SUDS learns its rank-1 Fisher direction by an Oja power-iteration step (oja_update advances
    state['fisher_approx'] every step). Its defining property: fed a gradient stream whose covariance has a
    dominant direction u (cov = I + 20 uu^T, eigen-gap 21:1), the iterate converges to +-u. Checked against
    a hand-constructed top eigenvector. The isotropic control (no dominant direction) must NOT converge
    to u, so the alignment is the learned signal, not an artifact."""

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
