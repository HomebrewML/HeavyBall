"""Proofs for the slab-native matrix preconditioners."""

import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from heavyball.core import Engine
from heavyball.matrix import soap, soap_recipe
from heavyball.transforms import beta_debias


def _project(value: torch.Tensor, left: torch.Tensor, right: torch.Tensor, *, back: bool) -> torch.Tensor:
    if back:
        return torch.einsum("nab,nia,njb->nij", value, left, right)
    return torch.einsum("nij,nia,njb->nab", value, left, right)


def _matrix_recipe():
    return replace(soap_recipe, chain=(soap,))


def _object_step(optimizer: Engine, *, step_type: str, compile: bool) -> None:
    """Use the engine's exact object step eagerly for the fp64 truth path."""

    if compile:
        optimizer.step(step_type=step_type)
        return
    for group in optimizer.groups:
        group.observed.fill_(True)
    eager = getattr(optimizer.compiled_steps[step_type], "__wrapped__", None)
    if eager is None:
        optimizer.step(step_type=step_type)
        return
    eager()
    for step in optimizer._steps:
        step.add_(1)


def _run_trajectory(
    dtype: torch.dtype, *, compile: bool, initial: torch.Tensor, gradients: list[torch.Tensor]
) -> list[torch.Tensor]:
    params = [torch.nn.Parameter(initial.to(dtype).clone()) for _ in range(2)]
    optimizer = Engine(
        params,
        _matrix_recipe(),
        lr=0.03,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.8,
        eps=1e-8,
        max_precond_dim=8,
        weight_decay=0.0,
    )
    for step, gradient in enumerate(gradients, start=1):
        for param in params:
            param.grad.copy_(gradient.to(dtype))
        _object_step(optimizer, step_type="refresh" if step % 3 == 0 else "normal", compile=compile)
    return [param.detach().clone() for param in params]


def test_soap_fp64_accuracy(capsys):
    """The fp32 object trajectory stays close to the fp64 object trajectory."""

    torch.manual_seed(31)
    initial = torch.randn(3, 4, dtype=torch.float64)
    gradients = [torch.randn_like(initial) for _ in range(8)]
    truth = _run_trajectory(torch.float64, compile=False, initial=initial, gradients=gradients)
    actual = _run_trajectory(torch.float32, compile=True, initial=initial, gradients=gradients)
    error = max((result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True))
    with capsys.disabled():
        print(f"soap fp64 max error: {float(error):.9e}")
    assert error <= 3e-5


def test_soap_refresh_transports():
    """Refreshes transport both moments into the new basis: the first by rotation (physical value
    retained), the second by Hadamard-square of the same basis change."""

    param = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer = Engine(
        [param],
        _matrix_recipe(),
        lr=0.1,
        beta1=0.9,
        beta2=0.95,
        shampoo_beta=0.8,
        eps=1e-8,
        max_precond_dim=8,
        weight_decay=0.0,
    )
    state = optimizer.groups[0].states[0]
    first = torch.tensor(((3.0, 0.0, 1.0, 0.0), (0.0, 2.0, 0.0, 1.0), (1.0, 0.0, 1.0, 2.0)))
    second = torch.tensor(((0.0, 1.0, 2.0, 0.0), (2.0, 1.0, 0.0, 3.0), (1.0, 3.0, 1.0, 0.0)))
    third = torch.tensor(((1.0, 2.0, 0.0, 1.0), (0.0, 1.0, 3.0, 0.0), (2.0, 0.0, 1.0, 2.0)))

    initial_left, initial_right = state["Q_l"].clone(), state["Q_r"].clone()
    param.grad.copy_(first)
    optimizer.step(step_type="normal")
    first_left, first_right = state["Q_l"].clone(), state["Q_r"].clone()
    first_gg_left, first_gg_right = state["GG_l"].clone(), state["GG_r"].clone()
    first_avg, first_avg_sq = state["exp_avg"].clone(), state["exp_avg_sq"].clone()
    torch.testing.assert_close(first_left, initial_left, rtol=0, atol=0)
    torch.testing.assert_close(first_right, initial_right, rtol=0, atol=0)
    torch.testing.assert_close(first_gg_left, first.unsqueeze(0) @ first.unsqueeze(0).mT)
    torch.testing.assert_close(first_gg_right, first.unsqueeze(0).mT @ first.unsqueeze(0))

    param.grad.copy_(second)
    optimizer.step(step_type="refresh")
    refreshed_left, refreshed_right = state["Q_l"].clone(), state["Q_r"].clone()
    assert not torch.equal(refreshed_left, first_left)
    assert not torch.equal(refreshed_right, first_right)

    age = torch.full((1,), 2, dtype=torch.int64)
    hyper = optimizer.groups[0].hyper
    projected = _project(second.unsqueeze(0), first_left, first_right, back=False)
    beta1 = beta_debias(hyper.beta1, age).reshape(1, 1, 1)
    beta2 = beta_debias(hyper.beta2, age).reshape(1, 1, 1)
    raw_avg = first_avg * beta1 + projected * (1 - beta1)
    raw_variance = first_avg_sq.square() * beta2 + projected.square() * (1 - beta2)
    physical_before_transport = _project(raw_avg, first_left, first_right, back=True)
    physical_after_transport = _project(state["exp_avg"], refreshed_left, refreshed_right, back=True)
    torch.testing.assert_close(physical_after_transport, physical_before_transport, rtol=2e-5, atol=2e-5)
    left_transition = torch.einsum("nAa,nAc->nac", first_left, refreshed_left).square()
    right_transition = torch.einsum("nBb,nBd->nbd", first_right, refreshed_right).square()
    transported_avg_sq = torch.einsum(
        "nab,nac,nbd->ncd", raw_variance, left_transition, right_transition
    ).sqrt()
    torch.testing.assert_close(state["exp_avg_sq"], transported_avg_sq, rtol=2e-5, atol=2e-5)
    assert not torch.equal(state["exp_avg"], raw_avg)
    assert not torch.equal(state["exp_avg_sq"], raw_variance.sqrt())

    shampoo_beta = beta_debias(hyper.shampoo_beta, age).reshape(1, 1, 1)
    expected_left = first_gg_left * shampoo_beta + (second.unsqueeze(0) @ second.unsqueeze(0).mT) * (1 - shampoo_beta)
    expected_right = first_gg_right * shampoo_beta + (second.unsqueeze(0).mT @ second.unsqueeze(0)) * (1 - shampoo_beta)
    torch.testing.assert_close(state["GG_l"], expected_left, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(state["GG_r"], expected_right, rtol=2e-5, atol=2e-5)

    param.grad.copy_(third)
    optimizer.step(step_type="normal")
    torch.testing.assert_close(state["Q_l"], refreshed_left, rtol=0, atol=0)
    torch.testing.assert_close(state["Q_r"], refreshed_right, rtol=0, atol=0)
    assert not torch.equal(state["GG_l"], expected_left)
    assert not torch.equal(state["GG_r"], expected_right)


def _compiled_code(tmp_path: Path, step_type: str) -> str:
    source = f"""
import torch
from heavyball.core import Engine
from heavyball.matrix import soap_recipe

params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
optimizer = Engine(params, soap_recipe)
optimizer.groups[0].grad_slab.normal_()
optimizer.step(step_type={step_type!r})
"""
    environment = dict(os.environ, TORCH_LOGS="output_code", TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / step_type))
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    paths = re.findall(r"Output code written to: (.*\.py)", output)
    assert paths, output
    return Path(paths[-1]).read_text()


def test_soap_fullgraph_clean(tmp_path):
    """Normal SOAP is QR-free; the host-selected refresh graph contains QR."""

    normal = _compiled_code(tmp_path, "normal").lower()
    refresh = _compiled_code(tmp_path, "refresh").lower()
    assert "linalg_qr" not in normal
    assert "linalg_qr" in refresh
    for artifact in (normal, refresh):
        assert "torch.stack" not in artifact
        assert "_local_scalar_dense" not in artifact
        assert ".item(" not in artifact

    source = Path(soap.__module__.replace(".", "/") + ".py")
    if not source.exists():
        source = Path(__file__).parents[1] / "heavyball" / "matrix.py"
    text = source.read_text()
    assert not re.search(r"_foreach|vmap|torch\.cond|dynamic=True|torch\.stack|\.item\(|\benabled\b|\bamp\b", text)
