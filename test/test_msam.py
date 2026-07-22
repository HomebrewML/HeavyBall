"""Parity, accuracy, and lifecycle proofs for the slab-native MSAMLaProp port."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import heavyball
import heavyball_legacy
from heavyball import Engine, msam_laprop


@contextmanager
def _legacy_eager():
    import heavyball_legacy.utils as legacy

    previous = legacy.compile_mode
    legacy.compile_mode = None
    try:
        yield
    finally:
        legacy.compile_mode = previous


def _copy_grads(params, gradients):
    for param, gradient in zip(params, gradients, strict=True):
        param.grad.copy_(gradient)


def _assert_params_close(actual, expected, *, rtol, atol, stage):
    for index, (result, reference) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol, msg=f"{stage}, parameter {index}")


def _cautious_msam_recipe():
    return replace(
        msam_laprop,
        defaults={**msam_laprop.defaults, "caution": 0.0, "cautious_weight_decay": 0.0},
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol", "storage_dtype"),
    (
        (torch.float64, 1e-12, 1e-12, "float64"),
        (torch.float32, 2e-6, 2e-6, "float32"),
    ),
)
@pytest.mark.parametrize("caution", (False, True))
def test_msam_matches_legacy(dtype, rtol, atol, storage_dtype, caution):
    """Every MSAM step and eval/train swap matches legacy, including caution."""

    if dtype is torch.float64 and caution:
        pytest.skip(
            "4.0 computes the caution rescale in the parameter's fp64 compute dtype while legacy uses "
            "fp32, so exact parity cannot hold. The fp64 rescale is verified through cautious_adamw's "
            "reference; the caution integration is covered by the fp32 case here."
        )
    torch._dynamo.reset()
    torch.manual_seed(81)
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        sam_step_size=0.13,
        caution=caution,
        cautious_weight_decay=True,
    )
    initial = [torch.randn(3, 2, dtype=dtype), torch.randn(3, 2, dtype=dtype)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(9)]
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    legacy_params = [torch.nn.Parameter(value.clone()) for value in initial]
    try:
        optimizer = Engine(params, _cautious_msam_recipe(), **values)
        with _legacy_eager():
            legacy_optimizer = heavyball_legacy.MSAMLaProp(
                legacy_params,
                lr=values["lr"],
                betas=(values["beta1"], values["beta2"]),
                eps=values["eps"],
                weight_decay=values["weight_decay"],
                sam_step_size=values["sam_step_size"],
                caution=values["caution"],
                cautious_weight_decay=values["cautious_weight_decay"],
                storage_dtype=storage_dtype,
                compile_step=False,
            )
            for step, step_gradients in enumerate(gradients, start=1):
                _copy_grads(params, step_gradients)
                for param, gradient in zip(legacy_params, step_gradients, strict=True):
                    param.grad = gradient.clone()
                optimizer.step()
                legacy_optimizer.step()
                _assert_params_close(params, legacy_params, rtol=rtol, atol=atol, stage=f"step {step}")
                if step == 4:
                    optimizer.eval()
                    legacy_optimizer.eval()
                    _assert_params_close(params, legacy_params, rtol=rtol, atol=atol, stage="eval")
                    optimizer.train()
                    legacy_optimizer.train()
                    _assert_params_close(params, legacy_params, rtol=rtol, atol=atol, stage="train")
    finally:
        torch._dynamo.reset()


def _msam_trajectory(dtype: torch.dtype, gradients, *, compiled: bool):
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        sam_step_size=0.13,
        caution=True,
        cautious_weight_decay=True,
    )
    params = [
        torch.nn.Parameter(torch.zeros(11, 7, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(4, dtype=dtype)),
    ]
    if compiled:
        optimizer = Engine(params, _cautious_msam_recipe(), **values)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine(params, _cautious_msam_recipe(), **values)
    for step, step_gradients in enumerate(gradients, start=1):
        _copy_grads(params, [gradient.to(dtype) for gradient in step_gradients])
        optimizer.step()
        if step == 31:
            optimizer.eval()
            optimizer.train()
    return [param.detach().clone() for param in params]


def test_msam_fp64_accuracy():
    """The compiled fp32 MSAM trajectory stays within its pinned fp64 budget."""

    torch._dynamo.reset()
    torch.manual_seed(82)
    shapes = ((11, 7), (4,))
    gradients = [[torch.randn(*shape, dtype=torch.float64) for shape in shapes] for _ in range(80)]
    try:
        truth = _msam_trajectory(torch.float64, gradients, compiled=False)
        actual = _msam_trajectory(torch.float32, gradients, compiled=True)
        error = max(
            (result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True)
        )
        assert error <= 3e-5
    finally:
        torch._dynamo.reset()


def test_msam_eval_swap_exact_with_caution():
    """A caution-filtered perturbed parameter round-trips exactly through eval mode."""

    torch._dynamo.reset()
    param = torch.nn.Parameter(torch.tensor((1.0, -2.0), dtype=torch.float64))
    values = dict(
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.1,
        sam_step_size=0.1,
        caution=True,
        cautious_weight_decay=True,
    )
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], _cautious_msam_recipe(), **values)
    for gradient in (
        torch.tensor((1.0, 1.0), dtype=torch.float64),
        torch.tensor((-0.1, 1.0), dtype=torch.float64),
    ):
        param.grad.copy_(gradient)
        optimizer.step()

    state = optimizer.groups[0].commit_state
    perturbed_param = param.detach().clone()
    z = state["z"][0].detach().clone()
    exp_avg = state["exp_avg"][0].detach().clone()
    rederived = z - exp_avg / torch.linalg.vector_norm(exp_avg) * values["sam_step_size"]
    assert not torch.equal(perturbed_param, rederived)

    optimizer.eval()
    assert torch.equal(param, z)  # eval shows the unperturbed master
    assert torch.equal(state["z"][0], z)  # the master is untouched, not exchanged into the parameter
    assert torch.equal(state["saved"][0], perturbed_param)  # the exact perturbed iterate is saved

    optimizer.train()
    assert torch.equal(param, perturbed_param)  # train restores the exact perturbed iterate
    assert torch.equal(state["z"][0], z)  # the master is still untouched
    torch._dynamo.reset()


def test_msam_master_is_fp32_and_swap_exact_for_low_precision_params():
    """MSAM's master z accumulates in fp32 for an fp16 parameter (matching the fp32-parameter run), and
    its eval/train swap restores the exact perturbed iterate instead of quantizing the master."""

    def master(dtype):
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            param = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
            optimizer = Engine(
                [param], _cautious_msam_recipe(), lr=1e-4, beta1=0.0, beta2=0.0, eps=1e-8,
                weight_decay=0.0, sam_step_size=0.0,
            )
            for _ in range(16):
                param.grad.fill_(1)
                optimizer.step()
            return optimizer.groups[0].commit_state["z"]

    z16, z32 = master(torch.float16), master(torch.float32)
    assert z16.dtype == torch.float32
    torch.testing.assert_close(z16, z32, rtol=0, atol=0)
    assert (z16 - 1.0).abs().max() > 1e-3

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        torch.manual_seed(3)
        param = torch.nn.Parameter(torch.randn(16, dtype=torch.float16))
        optimizer = Engine(
            [param], _cautious_msam_recipe(), lr=0.03, beta1=0.9, beta2=0.99, eps=1e-8,
            weight_decay=0.0, sam_step_size=0.1,
        )
        for _ in range(4):
            param.grad.copy_(torch.randn_like(param))
            optimizer.step()
        before = param.detach().clone()
        optimizer.eval()
        optimizer.train()
        assert torch.equal(param, before)  # the fp16 perturbed iterate is restored bit-for-bit


def test_msam_lifecycle_fullgraph_clean(tmp_path):
    """MSAM's step and literal lifecycle swaps remain scalar-free fullgraph artifacts."""

    source = """
import torch
from dataclasses import replace
from heavyball import Engine, msam_laprop

params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
recipe = replace(msam_laprop, defaults={**msam_laprop.defaults, "caution": 0.0})
optimizer = Engine(params, recipe, caution=True)
for _ in range(3):
    for param in params:
        param.grad.normal_()
    optimizer.step()
optimizer.eval()
optimizer.train()
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
    assert len(paths) >= 2, output
    for path in paths:
        code = path.read_text()
        for forbidden in (".item(", "while_loop", "cond", "stack", "_local_scalar_dense"):
            assert forbidden not in code, f"{forbidden} in {path}"


@pytest.mark.parametrize("radius", (0.05, 0.1, 0.2))
def test_msam_perturbs_the_master_by_exactly_the_sam_radius(radius):
    """MSAM's defining SAM geometry: the training iterate sits at a fixed distance sam_step_size from the
    master iterate z (along the normalized first-moment direction), so the forward/backward pass sees a
    perturbed point. Verified through the public eval/train swap -- eval exposes the master z, train restores
    the perturbed iterate -- so ||param_train - z|| = sam_step_size and scales linearly with it. Independent
    of legacy AND of the shipped commit; test_msam_matches_legacy pins the step only against a second
    HeavyBall implementation, and the radius formula otherwise sits unasserted in the eval-swap test."""

    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], _cautious_msam_recipe(), lr=0.1, sam_step_size=radius, weight_decay=0.0)
    for _ in range(5):
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
    train_iterate = param.detach().clone()
    optimizer.eval()
    master = param.detach().clone()
    optimizer.train()
    assert torch.equal(param.detach(), train_iterate)  # the swap restores the perturbed iterate exactly
    torch.testing.assert_close(
        (train_iterate - master).norm(), torch.tensor(radius, dtype=torch.float64), rtol=1e-6, atol=1e-9
    )
