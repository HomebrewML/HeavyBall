"""Independent fp64 equation oracles for three exported first-order optimizers.

AdamC follows Defazio's corrected-weight-decay rule (arXiv:2506.02285). SignLaProp starts from
LaProp's normalize-before-momentum equation (arXiv:2002.04839) and applies the public facade's
per-leaf L2 sign graft. UnscaledAdamW's defining public contract is normalize before momentum and
then restore the current second-moment scale.

The oracles use raw, uncorrected NumPy moments and closed-form bias correction. They therefore do
not reproduce HeavyBall's bias-corrected state recurrence or call any HeavyBall recipe/transform.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

import heavyball


@pytest.fixture(autouse=True)
def _eager():
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        yield


_INITIAL = (
    np.array([1.25, -0.75, 0.125], dtype=np.float64),
    np.array([[0.6, -1.1], [2.0, -0.3]], dtype=np.float64),
)

# The small second coordinate in each leaf keeps sqrt(v_hat) below eps, distinguishing the endorsed
# clamp floor from an additive epsilon while the other coordinates exercise the ordinary RMS branch.
_GRADIENTS = (
    (
        np.array([0.4, 2.0e-8, -0.7], dtype=np.float64),
        np.array([[0.3, -3.0e-8], [0.9, -0.2]], dtype=np.float64),
    ),
    (
        np.array([-0.25, -5.0e-8, 0.15], dtype=np.float64),
        np.array([[-0.4, 6.0e-8], [0.2, 0.5]], dtype=np.float64),
    ),
    (
        np.array([0.8, 4.0e-8, -0.35], dtype=np.float64),
        np.array([[0.7, -2.0e-8], [-0.6, 0.25]], dtype=np.float64),
    ),
    (
        np.array([-0.1, -7.0e-8, 0.55], dtype=np.float64),
        np.array([[0.15, 9.0e-8], [0.45, -0.8]], dtype=np.float64),
    ),
    (
        np.array([0.3, 3.0e-8, -0.6], dtype=np.float64),
        np.array([[-0.5, -4.0e-8], [0.35, 0.1]], dtype=np.float64),
    ),
    (
        np.array([-0.45, 8.0e-8, 0.2], dtype=np.float64),
        np.array([[0.55, 5.0e-8], [-0.25, 0.65]], dtype=np.float64),
    ),
)


def _adamc_from_equations(parameters, gradients, states, step, *, lr, beta1, beta2, eps, weight_decay, max_lr):
    """Adam's raw moments plus Defazio's lr/max_lr corrected decoupled decay."""

    updated = []
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        first = states[index].get("raw_first", np.zeros_like(parameter))
        second = states[index].get("raw_second", np.zeros_like(parameter))
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * np.square(gradient)
        states[index]["raw_first"] = first
        states[index]["raw_second"] = second

        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        adam_direction = first_hat / np.maximum(np.sqrt(second_hat), eps)

        corrected_decay = weight_decay * (lr / max_lr)
        updated.append(parameter - lr * adam_direction - lr * corrected_decay * parameter)
    return updated


def _unscaled_adamw_from_equations(
    parameters, gradients, states, step, *, lr, beta1, beta2, eps, weight_decay
):
    """Average variance-normalized gradients, then restore the current RMS scale."""

    updated = []
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        second = states[index].get("raw_second", np.zeros_like(parameter))
        second = beta2 * second + (1.0 - beta2) * np.square(gradient)
        states[index]["raw_second"] = second
        second_hat = second / (1.0 - beta2**step)
        current_scale = np.maximum(np.sqrt(second_hat), eps)

        normalized_gradient = gradient / current_scale
        normalized_momentum = states[index].get("raw_normalized_momentum", np.zeros_like(parameter))
        normalized_momentum = beta1 * normalized_momentum + (1.0 - beta1) * normalized_gradient
        states[index]["raw_normalized_momentum"] = normalized_momentum
        normalized_momentum_hat = normalized_momentum / (1.0 - beta1**step)

        unscaled_direction = normalized_momentum_hat * current_scale
        updated.append(parameter - lr * unscaled_direction - lr * weight_decay * parameter)
    return updated


def _sign_laprop_from_equations(
    parameters, gradients, states, step, *, lr, beta1, beta2, eps, weight_decay
):
    """LaProp's normalized-gradient momentum followed by an exact per-leaf L2 sign graft."""

    updated = []
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        second = states[index].get("raw_second", np.zeros_like(parameter))
        second = beta2 * second + (1.0 - beta2) * np.square(gradient)
        states[index]["raw_second"] = second
        second_hat = second / (1.0 - beta2**step)
        denominator = np.maximum(np.sqrt(second_hat), eps)

        normalized_gradient = gradient / denominator
        first = states[index].get("raw_first", np.zeros_like(parameter))
        first = beta1 * first + (1.0 - beta1) * normalized_gradient
        states[index]["raw_first"] = first
        laprop_direction = first / (1.0 - beta1**step)

        nonzero = np.count_nonzero(laprop_direction)
        if nonzero:
            grafted_sign = np.sign(laprop_direction) * np.sqrt(
                np.sum(np.square(laprop_direction)) / nonzero
            )
        else:
            grafted_sign = np.zeros_like(laprop_direction)
        updated.append(parameter - lr * grafted_sign - lr * weight_decay * parameter)
    return updated


def _assert_public_trajectory(optimizer_class, oracle, hyper, *, learning_rates=None):
    parameters = [torch.nn.Parameter(torch.from_numpy(value.copy())) for value in _INITIAL]
    expected = [value.copy() for value in _INITIAL]
    states = [{} for _ in _INITIAL]
    optimizer = optimizer_class(parameters, **hyper)

    if learning_rates is None:
        learning_rates = (hyper["lr"],) * len(_GRADIENTS)

    for step, (gradients, lr) in enumerate(zip(_GRADIENTS, learning_rates, strict=True), start=1):
        optimizer.param_groups[0]["lr"] = lr
        step_hyper = dict(hyper, lr=lr)
        expected = oracle(expected, gradients, states, step, **step_hyper)

        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad.copy_(torch.from_numpy(gradient))
        optimizer.step()

        for index, (parameter, target) in enumerate(zip(parameters, expected, strict=True)):
            torch.testing.assert_close(
                parameter.detach(),
                torch.from_numpy(target),
                rtol=0.0,
                atol=1e-12,
                msg=lambda message, step=step, index=index: f"step {step}, parameter {index}: {message}",
            )


def test_adamc_matches_corrected_weight_decay_equation():
    hyper = dict(
        lr=0.011,
        beta1=0.73,
        beta2=0.91,
        eps=1e-4,
        weight_decay=0.17,
        max_lr=0.02,
    )
    learning_rates = (0.011, 0.008, 0.005, 0.0025, 0.001, 0.0005)
    _assert_public_trajectory(
        heavyball.AdamC,
        _adamc_from_equations,
        hyper,
        learning_rates=learning_rates,
    )


def test_unscaled_adamw_matches_normalize_momentum_then_restore_scale_equation():
    hyper = dict(lr=0.007, beta1=0.73, beta2=0.91, eps=1e-4, weight_decay=0.17)
    _assert_public_trajectory(heavyball.UnscaledAdamW, _unscaled_adamw_from_equations, hyper)


def test_sign_laprop_matches_normalize_before_momentum_then_sign_graft_equation():
    hyper = dict(lr=0.007, beta1=0.73, beta2=0.91, eps=1e-4, weight_decay=0.17)
    _assert_public_trajectory(heavyball.SignLaProp, _sign_laprop_from_equations, hyper)
