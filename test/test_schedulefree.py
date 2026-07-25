"""Parity, lifecycle, and compile proofs for the slab-native SFAdamW port."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import reference
import torch

import heavyball
from heavyball import Engine, adamw, sf_adamw


def _copy_grads(params, gradients):
    for param, gradient in zip(params, gradients, strict=True):
        param.grad.copy_(gradient)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-12, 1e-12),
        (torch.float32, 2e-6, 2e-6),
    ),
)
def test_sf_matches_pure_reference(dtype, rtol, atol):
    """Every schedule-free step and eval/train swap matches a pure Defazio schedule-free AdamW
    reference (facebookresearch/schedule_free), independent of the legacy oracle: a bug replicated
    from legacy into 4.0 would pass a legacy-parity test but is caught against the paper recurrence."""

    torch._dynamo.reset()
    torch.manual_seed(71)
    values = dict(lr=0.017, beta1=0.87, beta2=0.97, eps=1e-8, weight_decay=0.031, weight_lr_power=2.0, r=0.5)
    initial = [torch.randn(3, 2, dtype=dtype), torch.randn(3, 2, dtype=dtype)]
    gradients = [[torch.randn_like(value) for value in initial] for _ in range(9)]
    for step, step_gradients in enumerate(gradients, start=1):
        for gradient in step_gradients:
            gradient[0, 0] = (-1) ** step * step * 1e-6
    params = [torch.nn.Parameter(value.clone()) for value in initial]
    history = [[] for _ in initial]
    try:
        optimizer = Engine(params, sf_adamw, **values)
        for index, param in enumerate(params):
            torch.testing.assert_close(optimizer.groups[0].commit_state["z"][index], param, rtol=0, atol=0)
        for step, step_gradients in enumerate(gradients, start=1):
            _copy_grads(params, step_gradients)
            for index, gradient in enumerate(step_gradients):
                history[index].append(gradient)
            optimizer.step()
            for index, param in enumerate(params):
                expected_y, _ = reference.schedule_free_adamw(initial[index], history[index], **values)
                torch.testing.assert_close(param.detach(), expected_y, rtol=rtol, atol=atol, msg=f"step {step} param {index}")
            if step == 4:
                optimizer.eval()
                for index, param in enumerate(params):
                    y, z = reference.schedule_free_adamw(initial[index], history[index], **values)
                    expected_eval = reference.schedule_free_eval(y, z, values["beta1"])
                    torch.testing.assert_close(param.detach(), expected_eval, rtol=rtol, atol=atol, msg=f"eval param {index}")
                optimizer.train()
                for index, param in enumerate(params):
                    y, _ = reference.schedule_free_adamw(initial[index], history[index], **values)
                    torch.testing.assert_close(param.detach(), y, rtol=rtol, atol=atol, msg=f"train param {index}")
    finally:
        torch._dynamo.reset()


def _sf_trajectory(dtype: torch.dtype, gradients, *, compiled: bool):
    values = dict(
        lr=0.017,
        beta1=0.87,
        beta2=0.97,
        eps=1e-8,
        weight_decay=0.031,
        weight_lr_power=2.0,
        r=0.5,
    )
    params = [
        torch.nn.Parameter(torch.zeros(11, 7, dtype=dtype)),
        torch.nn.Parameter(torch.zeros(4, dtype=dtype)),
    ]
    if compiled:
        optimizer = Engine(params, sf_adamw, **values)
    else:
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            optimizer = Engine(params, sf_adamw, **values)
    for step_gradients in gradients:
        _copy_grads(params, [gradient.to(dtype) for gradient in step_gradients])
        optimizer.step()
    return [param.detach().clone() for param in params]


def test_sf_fp64_accuracy():
    """The compiled fp32 schedule-free trajectory stays near its fp64 truth run."""

    torch._dynamo.reset()
    torch.manual_seed(72)
    shapes = ((11, 7), (4,))
    gradients = [[torch.randn(*shape, dtype=torch.float64) for shape in shapes] for _ in range(80)]
    try:
        truth = _sf_trajectory(torch.float64, gradients, compiled=False)
        actual = _sf_trajectory(torch.float32, gradients, compiled=True)
        error = max(
            (result.double() - expected).abs().max() for result, expected in zip(actual, truth, strict=True)
        )
        assert error <= 2e-5
    finally:
        torch._dynamo.reset()


def test_sf_master_iterate_is_fp32_for_low_precision_params():
    """The schedule-free averaging master z must accumulate in fp32 even for an fp16 parameter, so
    sub-ULP per-step increments are not discarded. A narrow z would stay pinned at 1.0; the fp16-parameter
    master must match the fp32-parameter master exactly."""

    def master(dtype):
        with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
            param = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
            optimizer = heavyball.ScheduleFree([param], lr=1e-4, beta1=0.0, beta2=0.0, eps=1e-8, weight_decay=0.0)
            for _ in range(16):
                param.grad.fill_(1)
                optimizer.step()
        return optimizer._engine.groups[0].commit_state["z"]

    z16, z32 = master(torch.float16), master(torch.float32)
    assert z16.dtype == torch.float32
    torch.testing.assert_close(z16, z32, rtol=0, atol=0)
    assert (z16 - 1.0).abs().max() > 1e-3  # the master actually accumulated rather than rounding to a noop


def test_sf_eval_swap_roundtrip():
    """Eval follows the schedule-free representation swap and train restores the prior iterate."""

    torch._dynamo.reset()
    param = torch.nn.Parameter(torch.tensor((1.0, -2.0), dtype=torch.float64))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], sf_adamw, lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8)
    for gradient in (
        torch.tensor((0.5, -0.25), dtype=torch.float64),
        torch.tensor((-0.75, 0.5), dtype=torch.float64),
        torch.tensor((0.25, 1.0), dtype=torch.float64),
    ):
        param.grad.copy_(gradient)
        optimizer.step()
    training_iterate = param.detach().clone()
    z = optimizer.groups[0].commit_state["z"][0].detach().clone()

    optimizer.eval()
    evaluation_iterate = param.detach().clone()
    assert not torch.equal(evaluation_iterate, training_iterate)
    expected_evaluation = training_iterate + (z - training_iterate) * (1 - 1 / 0.9)
    torch.testing.assert_close(evaluation_iterate, expected_evaluation, rtol=1e-12, atol=1e-12)

    optimizer.train()
    torch.testing.assert_close(param, training_iterate, rtol=1e-12, atol=1e-12)
    torch._dynamo.reset()


def test_sf_zero_beta1_eval_swap_is_noop():
    """With beta1=0 there is no averaging (the eval/train interpolation weights degenerate), so both
    representation swaps leave the parameter unchanged -- provable from the algorithm, no legacy needed."""

    param = torch.nn.Parameter(torch.tensor((1.0, -2.0), dtype=torch.float64))
    values = dict(lr=0.1, beta1=0.0, beta2=0.99, eps=1e-8)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], sf_adamw, **values)
    for gradient in (
        torch.tensor((0.5, -0.25), dtype=torch.float64),
        torch.tensor((-0.75, 0.5), dtype=torch.float64),
    ):
        param.grad.copy_(gradient)
        optimizer.step()
    training_iterate = param.detach().clone()
    optimizer.eval()
    torch.testing.assert_close(param, training_iterate, rtol=0, atol=0)
    optimizer.train()
    torch.testing.assert_close(param, training_iterate, rtol=0, atol=0)


@pytest.mark.parametrize("cautious_weight_decay", (False, True))
def test_sf_caution_and_decay_match_direct_recurrence(cautious_weight_decay):
    """Independently recompute decay-before-caution and schedule-free averaging."""

    values = dict(
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.1,
        caution=True,
        cautious_weight_decay=cautious_weight_decay,
    )
    weight_lr_power = 2.0
    params = [
        torch.nn.Parameter(torch.tensor((100.0, -100.0), dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor((-2.0, 3.0), dtype=torch.float64)),
    ]
    states = [
        {
            "param": param.detach().clone(),
            "z": param.detach().clone(),
            "variance": torch.zeros_like(param),
            "weight_sum": torch.zeros((), dtype=param.dtype),
            "lr_max": torch.zeros((), dtype=param.dtype),
        }
        for param in params
    ]
    recipe = replace(
        sf_adamw,
        defaults={**sf_adamw.defaults, "caution": 0.0, "cautious_weight_decay": 0.0},
    )
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(params, recipe, **values)
    for step, step_gradients in enumerate(
        (
            (torch.tensor((-0.01, 0.01), dtype=torch.float64), torch.tensor((0.5, -0.5), dtype=torch.float64)),
            (torch.tensor((0.2, -0.3), dtype=torch.float64), torch.tensor((-0.4, 0.6), dtype=torch.float64)),
        ),
        start=1,
    ):
        for state, gradient in zip(states, step_gradients, strict=True):
            state["variance"].mul_(values["beta2"]).addcmul_(
                gradient,
                gradient,
                value=1 - values["beta2"],
            )
            rms = (state["variance"] / (1 - values["beta2"] ** step)).sqrt()
            update = gradient / rms.clamp_min(values["eps"] ** 0.5)
            if cautious_weight_decay:
                decay_aligned = (
                    ((state["param"] > 0) & (update > 0))
                    | ((state["param"] < 0) & (update < 0))
                )
                decay_param = torch.where(decay_aligned, state["param"], torch.zeros_like(state["param"]))
            else:
                decay_param = state["param"]
            update = update + decay_param * values["weight_decay"]
            aligned = ((gradient > 0) & (update > 0)) | ((gradient < 0) & (update < 0))
            caution_scale = update.numel() / aligned.sum().clamp_min(1).to(update.dtype)
            update = torch.where(aligned, update, torch.zeros_like(update)) * caution_scale

            state["lr_max"] = torch.maximum(
                state["lr_max"],
                state["lr_max"].new_tensor(abs(values["lr"])),
            )
            weight = state["lr_max"].pow(weight_lr_power)
            state["weight_sum"] = state["weight_sum"] + weight
            ckp1 = weight / state["weight_sum"]
            state["param"] = torch.lerp(state["param"], state["z"], ckp1)
            state["param"] = state["param"] + update * (
                values["lr"] * (values["beta1"] * (1 - ckp1)) - values["lr"]
            )
            state["z"] = state["z"] - update * values["lr"]

        _copy_grads(params, step_gradients)
        optimizer.step()
        torch.testing.assert_close(
            torch.stack([param.detach() for param in params]),
            torch.stack([state["param"] for state in states]),
            rtol=1e-12,
            atol=1e-12,
        )
        commit_state = optimizer.groups[0].commit_state
        torch.testing.assert_close(commit_state["z"], torch.stack([state["z"] for state in states]), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            commit_state["weight_sum"],
            torch.stack([state["weight_sum"] for state in states]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_sf_train_mode_checkpointed():
    """An eval-mode checkpoint resumes in eval mode; old checkpoints default to train mode."""

    torch._dynamo.reset()
    values = dict(lr=0.03, beta1=0.9, beta2=0.99, eps=1e-8)
    param = torch.nn.Parameter(torch.tensor((1.0, -2.0)))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], sf_adamw, **values)
    for gradient in (torch.tensor((0.2, -0.4)), torch.tensor((-0.3, 0.5)), torch.tensor((0.1, 0.7))):
        param.grad.copy_(gradient)
        optimizer.step()
    optimizer.eval()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    assert checkpoint["train_mode"] is False

    resumed_param = torch.nn.Parameter(param.detach().clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        resumed = Engine([resumed_param], sf_adamw, **values)
    resumed.load_state_dict(checkpoint)
    assert resumed._train_mode is False
    evaluation_iterate = resumed_param.detach().clone()
    assert resumed.eval() is resumed
    torch.testing.assert_close(resumed_param, evaluation_iterate, rtol=0, atol=0)

    optimizer.train()
    resumed.train()
    torch.testing.assert_close(resumed_param, param, rtol=0, atol=0)

    old_checkpoint = copy.deepcopy(checkpoint)
    del old_checkpoint["train_mode"]
    old_param = torch.nn.Parameter(evaluation_iterate.clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        old = Engine([old_param], sf_adamw, **values)
    old.load_state_dict(old_checkpoint)
    assert old._train_mode is True
    torch._dynamo.reset()


def test_sf_lr_max_tracks_peak_lr():
    """lr_max in state tracks the peak lr across steps, preventing ckp1 instability at the
    warmup-to-constant transition (facebookresearch/schedule_free uses lr_max, not lr, for
    the per-step weight)."""

    param = torch.nn.Parameter(torch.tensor((1.0, -2.0), dtype=torch.float64))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], sf_adamw, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_lr_power=2.0)
    # Warmup: lr from 1e-4 to 1e-3
    for step in range(5):
        lr = 1e-4 + (1e-3 - 1e-4) * (step + 1) / 5
        optimizer.set_hyper("lr", lr)
        param.grad.copy_(torch.tensor((0.1, -0.05), dtype=torch.float64))
        optimizer.step()
    lr_max_after_warmup = optimizer.groups[0].commit_state["lr_max"][0].item()
    assert abs(lr_max_after_warmup - 1e-3) < 1e-12, f"lr_max should be 1e-3, got {lr_max_after_warmup}"
    # Lower lr (cosine decay): lr_max stays at peak
    optimizer.set_hyper("lr", 5e-4)
    param.grad.copy_(torch.tensor((0.1, -0.05), dtype=torch.float64))
    optimizer.step()
    lr_max_after_decay = optimizer.groups[0].commit_state["lr_max"][0].item()
    assert abs(lr_max_after_decay - 1e-3) < 1e-12, f"lr_max should still be 1e-3, got {lr_max_after_decay}"


def test_non_schedulefree_lifecycle_is_parameter_noop():
    """Ordinary commits only record the host mode; they do not swap parameters."""

    param = torch.nn.Parameter(torch.tensor((1.0, -2.0)))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], adamw)
    before = param.detach().clone()
    assert optimizer.eval() is optimizer
    assert optimizer._train_mode is False
    torch.testing.assert_close(param, before, rtol=0, atol=0)
    assert optimizer.train() is optimizer
    assert optimizer._train_mode is True
    torch.testing.assert_close(param, before, rtol=0, atol=0)


def test_sf_step_and_lifecycle_swaps_are_stable_fullgraphs():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
        optimizer = Engine(params, sf_adamw)
        for step in range(3):
            for index, param in enumerate(params):
                param.grad.copy_(
                    torch.linspace(-1, 1, param.numel()).reshape_as(param) * (step + index + 1)
                )
            optimizer.step()
            if step == 0:
                step_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        assert step_graphs == 1
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == step_graphs

        training = [param.detach().clone() for param in params]
        optimizer.eval()
        evaluation = [param.detach().clone() for param in params]
        optimizer.train()
        for param, expected in zip(params, training, strict=True):
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
        assert any(not torch.equal(train, eval_) for train, eval_ in zip(training, evaluation, strict=True))
        lifecycle_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]

        optimizer.eval()
        optimizer.train()
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == lifecycle_graphs == 3
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()
