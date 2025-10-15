import os
import random
import warnings
from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn
from utils import (
    _global_l2_norm,
    _global_rms_norm,
    _local_l2_norm,
    _local_rms_norm,
)

import heavyball
from heavyball.utils import (
    _compilable_global_l2norm_clip_,
    _compilable_global_rmsnorm_clip_,
    _compilable_l2_clip_,
    _compilable_rmsnorm_clip_,
    caution,
    disable_caution_scaling,
    fused_hook,
    hook_optimizer_into_model,
    line_to_triu,
    mars_correction,
    merge_group,
    orthogonalize_grad_to_param,
    precond_update_prob_schedule,
    psgd_should_update,
    sam_step,
    stochastic_add_,
    stochastic_add_divide_,
    stochastic_divide_with_eps_,
    stochastic_multiply_,
    triu_to_line,
    warn_once,
)

# Ensure Torch dynamo stays disabled on CI runners without GPU support.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
heavyball.utils.compile_mode = None


def _make_batch(seed: int = 0, batch: int = 8):
    torch.manual_seed(seed)
    data = torch.randn(batch, 4)
    target = torch.randn(batch, 2)
    return data, target


def _clone_parameters(model: nn.Module):
    return [param.detach().clone() for param in model.parameters()]


def _parameter_drift(model: nn.Module, reference: list[Tensor]) -> float:
    diffs = [current.detach() - ref for current, ref in zip(model.parameters(), reference, strict=True)]
    flat = torch.cat([diff.reshape(-1) for diff in diffs])
    return flat.norm().item()


def test_hook_optimizer_into_model_matches_sgd():
    # Baseline optimizer run.
    baseline = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    hooked = deepcopy(baseline)
    data, target = _make_batch(seed=1)

    manual_opt = heavyball.SGD(baseline.parameters(), lr=0.05)
    for _ in range(3):
        manual_opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(baseline(data), target)
        loss.backward()
        manual_opt.step()

    reference = _clone_parameters(baseline)

    optimizers = hook_optimizer_into_model(hooked, heavyball.SGD, lr=0.05)
    assert len(optimizers) == sum(param.requires_grad for param in hooked.parameters())

    for _ in range(3):
        loss = torch.nn.functional.mse_loss(hooked(data), target)
        loss.backward()

    for param, ref in zip(hooked.parameters(), reference, strict=True):
        assert torch.allclose(param, ref, atol=1e-6)


def test_fused_hook_updates_parameters_without_manual_step():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    initial = _clone_parameters(model)
    data, target = _make_batch(seed=2, batch=6)

    fused_optimizer = fused_hook(model.parameters(), heavyball.SGD, lr=0.05)

    # Users should not call step manually.
    with pytest.warns(UserWarning):
        fused_optimizer.step()

    for _ in range(2):
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()

    drift = _parameter_drift(model, initial)
    assert drift > 0.0
    for param in model.parameters():
        assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param), atol=0, rtol=0)


def test_sam_step_accumulates_and_zeros_gradients():
    params = [torch.nn.Parameter(torch.full((3,), 1.0)), torch.nn.Parameter(torch.arange(3.0))]

    for i, param in enumerate(params):
        param.requires_grad_(True)
        param.grad = torch.full_like(param, 0.5 * (i + 1))

    originals = [param.detach().clone() for param in params]
    returned = sam_step(params, ball_size=0.1, adaptive=False)

    for clone, original in zip(returned, originals, strict=True):
        assert torch.allclose(clone, original)

    for param, original, scale in zip(params, originals, (0.5, 1.0), strict=True):
        expected = original + scale * 0.1
        assert torch.allclose(param.detach(), expected, atol=1e-6)
        assert torch.allclose(param.grad, torch.zeros_like(param.grad), atol=0, rtol=0)


@pytest.mark.parametrize(
    "clip_fn,metric",
    [
        (_compilable_l2_clip_, _local_l2_norm),
        (_compilable_rmsnorm_clip_, _local_rms_norm),
    ],
)
def test_local_clip_functions_limit_each_tensor_norm(clip_fn, metric):
    tensors = [torch.full((6,), 5.0), torch.full((6,), -3.0)]
    clip_fn(tensors, clip_at=1.5)
    for tensor in tensors:
        assert metric(tensor) <= 1.5 * (1 + 1e-3)


@pytest.mark.parametrize(
    "clip_fn,metric",
    [
        (
            _compilable_global_l2norm_clip_,
            _global_l2_norm,
        ),
        (
            _compilable_global_rmsnorm_clip_,
            _global_rms_norm,
        ),
    ],
)
def test_global_clip_functions_limit_group_norm(clip_fn, metric):
    tensors = [torch.full((4,), 4.0), torch.full((4,), -2.0)]
    clip_fn(tensors, clip_at=0.75)
    assert metric(tensors) <= 0.75 * (1 + 1e-3)


def test_triu_line_roundtrip_on_cpu():
    tensors = [
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
    ]
    packed = triu_to_line(tensors)
    restored = line_to_triu(packed, symmetric_output=True)
    for original, rebuilt in zip(tensors, restored, strict=True):
        assert torch.allclose(rebuilt, torch.triu(original) + torch.triu(original, diagonal=1).T)


def test_warn_once_only_emits_single_warning(monkeypatch):
    storage = set()
    monkeypatch.setattr(heavyball.utils, "_warned", storage)

    with pytest.warns(UserWarning):
        warn_once("run once")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warn_once("run once")
    assert captured == []


def test_psgd_should_update_accumulates_probability():
    group = {"stochastic_schedule": False}
    outcomes = [psgd_should_update(group, 0.4) for _ in range(4)]
    assert outcomes[:2] == [False, False]
    assert outcomes[2] is True
    assert outcomes[3] in (False, True)
    assert group["cumulative_prob_prob_step"] == 4


def test_psgd_should_update_stochastic_schedule_uses_rng():
    rng = random.Random(123)
    group = {"stochastic_schedule": True}
    calls = [psgd_should_update(group, 0.5, rng=rng) for _ in range(5)]
    rng = random.Random(123)
    expected = [rng.random() < 0.5 for _ in range(5)]
    assert calls == expected


def test_stochastic_math_helpers_match_expected_results(n=1024):
    torch.manual_seed(0x172893)
    a = torch.arange(n).float()
    b = 1 + torch.arange(n).float() ** 2

    c = a.clone()
    stochastic_add_(c, b, alpha=0.5)
    assert torch.allclose(c.float(), a + b * 0.5)

    c = a.clone()
    stochastic_multiply_(c, b)
    assert torch.allclose(c.float(), a * b)

    c = a.clone()
    stochastic_add_divide_(c, b, alpha=1.0, divisor=2.0)
    assert torch.allclose(c.float(), (a + b * 1) / 2)

    orig = heavyball.utils.default_division_backend
    try:
        heavyball.utils.atan2_scale = 1024
        for backend in heavyball.utils.DivisionBackend:
            heavyball.utils.default_division_backend = backend
            c = a.clone()
            stochastic_divide_with_eps_(c, b)
            assert torch.allclose(c.float(), a / b), f"Backend {backend} failed"
    finally:
        heavyball.utils.default_division_backend = orig


def test_stochastic_math_accuracy():
    torch.manual_seed(0x172893)
    items = 8
    steps = 2048
    increments = torch.full((items,), 1e-3, dtype=torch.float32)

    baseline = torch.zeros(items, dtype=torch.bfloat16)
    stochastic = torch.zeros(items, dtype=torch.bfloat16)
    ground_truth = torch.zeros(items, dtype=torch.float64)

    for _ in range(steps):
        baseline.add_(increments)
        ground_truth.add_(increments)
        stochastic_add_(stochastic, increments)

    baseline_error = torch.abs(baseline.float() - ground_truth.float()).mean().item()
    stochastic_error = torch.abs(stochastic.float() - ground_truth.float()).mean().item()

    assert baseline_error > 1.0
    assert stochastic_error < 0.2
    assert stochastic_error < baseline_error * 0.2

    baseline_bias = abs(baseline.float().mean().item() - ground_truth.float().mean().item())
    stochastic_bias = abs(stochastic.float().mean().item() - ground_truth.float().mean().item())
    assert stochastic_bias < baseline_bias


def test_disable_caution_scaling_toggles_behavior():
    grad = torch.tensor([1.0, -1.0])
    update = torch.tensor([1.0, 1.0])
    original = heavyball.utils._compilable_cautioning
    try:
        scaled = caution(grad, update.clone())
        assert torch.allclose(scaled, torch.tensor([2.0, 0.0]))

        disable_caution_scaling()
        unscaled = caution(grad, update.clone())
        assert torch.allclose(unscaled, torch.tensor([1.0, 0.0]))
    finally:
        heavyball.utils._compilable_cautioning = original


def test_precond_update_prob_schedule_basic_decay():
    schedule = precond_update_prob_schedule(max_prob=1.0, min_prob=0.25, decay=0.5, flat_start=2)
    values = [schedule(step) for step in range(1, 6)]
    assert values[:2] == [1.0, 1.0]
    assert values[2] == pytest.approx(0.5)
    assert values[3:] == [0.25, 0.25]


def test_merge_group_merges_only_when_enabled():
    tensor = torch.ones(2, 3, 2, 2)
    disabled = merge_group({"merge_dims": False}, tensor)
    assert isinstance(disabled, tuple)
    assert torch.equal(disabled[0], tensor)

    enabled = merge_group({"merge_dims": True, "max_precond_dim": 4}, tensor)
    assert isinstance(enabled, list)
    assert enabled[0].shape == torch.Size([2, 3, 4])


def test_orthogonalize_grad_to_param_outputs_orthogonal_grad():
    weight = torch.torch.tensor([3.0, 4.0])
    grad = torch.torch.tensor([1.0, 2.0])
    orthogonalize_grad_to_param([weight], [grad], eps=1e-6, graft=False)
    assert torch.allclose((weight * grad).sum(), torch.tensor(0.0), atol=1e-6)


def test_mars_correction_updates_old_gradient_copy():
    g = [torch.torch.tensor([1.0, 2.0])]
    old = [torch.zeros(2)]
    mars_correction(g, old, beta1=0.9, gamma=0.2)
    assert torch.allclose(old[0], torch.torch.tensor([1.0, 2.0]))
