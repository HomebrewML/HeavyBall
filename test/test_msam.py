"""Parity, accuracy, and lifecycle proofs for the slab-native MSAMLaProp port."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, msam_laprop


def _copy_grads(params, gradients):
    for param, gradient in zip(params, gradients, strict=True):
        param.grad.copy_(gradient)


def _cautious_msam_recipe():
    return replace(
        msam_laprop,
        defaults={**msam_laprop.defaults, "caution": 0.0, "cautious_weight_decay": 0.0},
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-12, 1e-12),
        (torch.float32, 2e-6, 2e-6),
    ),
)
@pytest.mark.parametrize("caution", (False, True))
def test_msam_matches_direct_rmsprop_momentum_sam_recurrence(dtype, rtol, atol, caution):
    """Recompute RMS normalization, caution, master decay, and SAM perturbation directly."""

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
    states = [
        {
            "z": value.clone(),
            "raw_momentum": torch.zeros_like(value),
            "raw_variance": torch.zeros_like(value),
        }
        for value in initial
    ]
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine(params, _cautious_msam_recipe(), **values)

    for step, step_gradients in enumerate(gradients, start=1):
        expected_params = []
        expected_momenta = []
        for state, gradient in zip(states, step_gradients, strict=True):
            state["raw_variance"].mul_(values["beta2"]).addcmul_(
                gradient,
                gradient,
                value=1 - values["beta2"],
            )
            rms = (state["raw_variance"] / (1 - values["beta2"] ** step)).sqrt()
            normalized = gradient / rms.clamp_min(values["eps"] ** 0.5)
            state["raw_momentum"].mul_(values["beta1"]).add_(
                normalized,
                alpha=1 - values["beta1"],
            )
            momentum = state["raw_momentum"] / (1 - values["beta1"] ** step)
            filtered = momentum
            if caution:
                aligned = ((gradient > 0) & (momentum > 0)) | ((gradient < 0) & (momentum < 0))
                scale = momentum.numel() / aligned.sum().clamp_min(1).to(dtype)
                filtered = torch.where(aligned, momentum, torch.zeros_like(momentum)) * scale
            decay = torch.where(
                ((state["z"] > 0) & (filtered > 0)) | ((state["z"] < 0) & (filtered < 0)),
                torch.as_tensor(values["weight_decay"], dtype=dtype),
                torch.zeros((), dtype=dtype),
            )
            state["z"] = state["z"] * (1 - decay * values["lr"]) - filtered * values["lr"]
            norm = torch.linalg.vector_norm(filtered)
            direction = filtered / torch.where(norm != 0, norm, torch.ones_like(norm))
            expected_params.append(state["z"] - direction * values["sam_step_size"])
            expected_momenta.append(momentum)

        _copy_grads(params, step_gradients)
        optimizer.step()
        for index, (param, expected) in enumerate(zip(params, expected_params, strict=True)):
            torch.testing.assert_close(
                param,
                expected,
                rtol=rtol,
                atol=atol,
                msg=f"step {step}, parameter {index}",
            )
        commit_state = optimizer.groups[0].commit_state
        torch.testing.assert_close(commit_state["z"], torch.stack([state["z"] for state in states]), rtol=rtol, atol=atol)
        torch.testing.assert_close(commit_state["exp_avg"], torch.stack(expected_momenta), rtol=rtol, atol=atol)
        if step == 4:
            optimizer.eval()
            torch.testing.assert_close(
                torch.stack([param.detach() for param in params]),
                torch.stack([state["z"] for state in states]),
                rtol=rtol,
                atol=atol,
            )
            optimizer.train()
            torch.testing.assert_close(
                torch.stack([param.detach() for param in params]),
                torch.stack(expected_params),
                rtol=rtol,
                atol=atol,
            )


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


def test_msam_step_and_lifecycle_swaps_are_stable_fullgraphs():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
        optimizer = Engine(params, _cautious_msam_recipe(), caution=True)
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
        master = [param.detach().clone() for param in params]
        optimizer.train()
        for param, expected in zip(params, training, strict=True):
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
        assert any(not torch.equal(train, base) for train, base in zip(training, master, strict=True))
        lifecycle_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]

        optimizer.eval()
        optimizer.train()
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == lifecycle_graphs == 3
        assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("radius", (0.05, 0.1, 0.2))
def test_msam_perturbs_the_master_by_exactly_the_sam_radius(radius):
    """MSAM's defining SAM geometry: the training iterate sits at a fixed distance sam_step_size from the
    master iterate z (along the normalized first-moment direction), so the forward/backward pass sees a
    perturbed point. Verified through the public eval/train swap -- eval exposes the master z, train restores
    the perturbed iterate -- so ||param_train - z|| = sam_step_size and scales linearly with it."""

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
