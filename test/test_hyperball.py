"""Parity and constraint proofs for the slab-native HyperBall port."""

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import heavyball
import reference
from heavyball import Engine, hyperball, hyperball_adamw, hyperball_commit


def _copy_grads(params, gradients):
    for param, gradient in zip(params, gradients, strict=True):
        param.grad.copy_(gradient)


def _initial_norm(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(-1)
    scale = flat.abs().amax()
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    norm = torch.linalg.vector_norm(flat / safe_scale)
    return torch.cat((scale.reshape(1), norm.reshape(1)))


def _assert_ball_radius(param: torch.Tensor, init_norm: torch.Tensor) -> None:
    actual = torch.linalg.vector_norm(param.detach())
    radius = init_norm[0] * init_norm[1]
    torch.testing.assert_close(actual, radius, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-12, 1e-12),
        (torch.float32, 3e-6, 3e-6),
    ),
)
@pytest.mark.parametrize("caution", (False, True))
def test_hyperball_matches_pure_reference(dtype, rtol, atol, caution):
    """Every HyperBall step, including the first norm capture, matches the paper recurrence."""

    torch._dynamo.reset()
    torch.manual_seed(91)
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        caution=caution,
        cautious_weight_decay=True,
    )
    initial = [torch.randn(3, 2, dtype=dtype), torch.randn(3, 2, dtype=dtype)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(9)]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    history = [[] for _ in initial]
    try:
        optimizer = Engine(params, hyperball_adamw, **values)
        assert hyperball.commit is hyperball_commit
        state = optimizer.groups[0].commit_state
        assert torch.equal(state["seen"], torch.zeros_like(state["seen"]))
        for step, step_gradients in enumerate(gradients, start=1):
            _copy_grads(params, step_gradients)
            for index, gradient in enumerate(step_gradients):
                history[index].append(gradient)
            optimizer.step()
            for index, param in enumerate(params):
                expected = reference.hyperball(initial[index], history[index], **values)
                torch.testing.assert_close(
                    param.detach(),
                    expected,
                    rtol=rtol,
                    atol=atol,
                    msg=f"step {step} param {index}",
                )
            if step == 1:
                assert torch.equal(state["seen"], torch.ones_like(state["seen"]))
                for index, original in enumerate(initial):
                    torch.testing.assert_close(
                        state["init_norm"][index],
                        _initial_norm(original),
                        rtol=rtol,
                        atol=atol,
                    )
    finally:
        torch._dynamo.reset()


def test_hyperball_adamw_routes_vector_to_adamw_and_matrix_to_hyperball():
    matrix_initial = torch.tensor(((3.0, -4.0), (1.5, 2.0)), dtype=torch.float64)
    vector_initial = torch.tensor((3.0, 4.0), dtype=torch.float64)
    matrix = torch.nn.Parameter(matrix_initial.clone())
    vector = torch.nn.Parameter(vector_initial.clone())
    adamw_vector = torch.nn.Parameter(vector_initial.clone())
    values = dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0)
    optimizer = heavyball.HyperBallAdamW([matrix, vector], **values)
    adamw_optimizer = heavyball.build([adamw_vector], heavyball.adamw, **values)

    matrix.grad.copy_(torch.tensor(((0.5, -1.0), (0.25, 0.75)), dtype=torch.float64))
    vector_gradient = torch.tensor((1.0, 1.0), dtype=torch.float64)
    vector.grad.copy_(vector_gradient)
    adamw_vector.grad.copy_(vector_gradient)
    optimizer.step()
    adamw_optimizer.step()

    torch.testing.assert_close(vector, adamw_vector, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        torch.linalg.vector_norm(matrix),
        torch.linalg.vector_norm(matrix_initial),
        rtol=2e-12,
        atol=2e-12,
    )


def _hyperball_trajectory(dtype: torch.dtype, initial, gradients, *, compiled: bool):
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        caution=True,
        cautious_weight_decay=True,
    )
    params = [torch.nn.Parameter(value.to(dtype).clone()) for value in initial]
    if compiled:
        optimizer = Engine(params, hyperball_adamw, **values)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine(params, hyperball_adamw, **values)
    for step_gradients in gradients:
        _copy_grads(params, [gradient.to(dtype) for gradient in step_gradients])
        optimizer.step()
    return [param.detach().clone() for param in params]


def test_hyperball_fp64_accuracy():
    """The compiled fp32 HyperBall trajectory remains within its fp64 budget."""

    torch._dynamo.reset()
    torch.manual_seed(92)
    initial = [torch.randn(11, 7, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(80)]
    try:
        truth = _hyperball_trajectory(torch.float64, initial, gradients, compiled=False)
        actual = _hyperball_trajectory(torch.float32, initial, gradients, compiled=True)
        error = max(
            (result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True)
        )
        assert error <= 3e-5
    finally:
        torch._dynamo.reset()


def test_hyperball_constrains_norm():
    """Each leaf captures its radius once and is projected to it after every update."""

    torch._dynamo.reset()
    initial = [
        torch.tensor(((3.0, -4.0), (1.5, 2.0)), dtype=torch.float64),
        torch.tensor(((-2.0, 5.0), (4.0, -1.0)), dtype=torch.float64),
    ]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    values = dict(lr=0.1, beta2=0.99, eps=1e-8, weight_decay=0.03, caution=True, cautious_weight_decay=True)
    try:
        optimizer = Engine(params, hyperball_adamw, **values)
        state = optimizer.groups[0].commit_state
        _copy_grads(
            params,
            (
                torch.tensor(((0.5, -1.0), (0.25, 0.75)), dtype=torch.float64),
                torch.tensor(((-0.5, 1.0), (-0.25, -0.75)), dtype=torch.float64),
            ),
        )
        optimizer.step(observed=(True, False))
        assert torch.equal(state["seen"], torch.tensor((True, False), device=state["seen"].device))
        torch.testing.assert_close(state["init_norm"][0], _initial_norm(initial[0]), rtol=0, atol=0)
        assert torch.equal(state["init_norm"][1], torch.zeros_like(state["init_norm"][1]))
        _assert_ball_radius(params[0], state["init_norm"][0])

        first_init_norm = state["init_norm"].detach().clone()
        _copy_grads(
            params,
            (
                torch.tensor(((-0.25, 0.5), (0.75, -1.0)), dtype=torch.float64),
                torch.tensor(((0.25, -0.5), (0.75, 1.0)), dtype=torch.float64),
            ),
        )
        optimizer.step(observed=(False, True))
        assert torch.equal(state["seen"], torch.ones_like(state["seen"]))
        torch.testing.assert_close(state["init_norm"][0], first_init_norm[0], rtol=0, atol=0)
        torch.testing.assert_close(state["init_norm"][1], _initial_norm(initial[1]), rtol=0, atol=0)
        _assert_ball_radius(params[0], state["init_norm"][0])
        _assert_ball_radius(params[1], state["init_norm"][1])

        latched_init_norm = state["init_norm"].detach().clone()
        _copy_grads(
            params,
            (
                torch.tensor(((1.0, 0.5), (-0.75, 0.25)), dtype=torch.float64),
                torch.tensor(((-1.0, 0.5), (0.75, -0.25)), dtype=torch.float64),
            ),
        )
        optimizer.step()
        assert torch.equal(state["seen"], torch.ones_like(state["seen"]))
        assert torch.equal(state["init_norm"], latched_init_norm)
        for index, param in enumerate(params):
            _assert_ball_radius(param, state["init_norm"][index])
    finally:
        torch._dynamo.reset()


def test_hyperball_fullgraph_clean(tmp_path):
    """The compiled HyperBall step is scalar-free and uses no dynamic-control-flow helpers."""

    source = """
import torch
from heavyball import Engine, hyperball_adamw

params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
optimizer = Engine(params, hyperball_adamw, caution=True, weight_decay=0.03, cautious_weight_decay=True)
for _ in range(3):
    for param in params:
        param.grad.normal_()
    optimizer.step()
"""
    env = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    paths = {Path(value) for value in re.findall(r"Output code written to: (.*\.py)", output)}
    assert paths, output
    for path in paths:
        code = path.read_text()
        for forbidden in (".item(", "while_loop", "cond", "stack", "_local_scalar_dense"):
            assert forbidden not in code, f"{forbidden} in {path}"
