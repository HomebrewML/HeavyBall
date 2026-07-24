"""Behavioral contracts for every default optimizer facade.

These tests intentionally inspect parameter deltas, not implementation state.  In
particular, the whitening check has a red control below: constructing ``Whitening``
with ``preconditioner_update_probability=0.0`` leaves the identity basis frozen and
makes the same whitening-property assertion fail.  That is the silent no-op this
file is meant to catch.
"""

from functools import cache
from typing import NamedTuple

import pytest
import torch

from heavyball import optim


_ANISOTROPIC_GRADIENT = torch.tensor(
    ((9.0, 4.0, -1.0), (6.0, -3.0, -0.5), (3.0, -1.0, 2.0))
)
_SECOND_GRADIENT = torch.tensor(
    ((-2.0, 8.0, 3.0), (1.0, -0.5, 6.0), (-4.0, 2.0, -1.0))
)
_THIRD_GRADIENT = torch.tensor(
    ((1.0, -7.0, 2.0), (-3.0, 5.0, -4.0), (8.0, -0.25, 3.0))
)
_INITIAL_PARAMETER = torch.tensor(
    ((1.0, 2.0, -1.0), (-2.0, 1.0, 0.5), (0.25, -1.0, 3.0))
)

_CONSTANT_GRADIENTS = (_ANISOTROPIC_GRADIENT,) * 5
_VARYING_GRADIENTS = (
    _ANISOTROPIC_GRADIENT,
    _SECOND_GRADIENT,
    _THIRD_GRADIENT,
    _ANISOTROPIC_GRADIENT,
)
_DEPRECATED_FACADE_ALIASES = frozenset(("ScheduleFree", "WhitenAdamW"))


class _Run(NamedTuple):
    updates: tuple[torch.Tensor, ...]
    parameters: tuple[torch.Tensor, ...]
    hyper: dict[str, object]


def _build_default(optimizer_name: str, parameter: torch.nn.Parameter):
    # Engine's many optimizer closures share one whole_step code object.  Clear
    # Dynamo's eight-entry guard cache between facades while retaining the real
    # fullgraph compiled path for every step.
    torch._dynamo.reset()
    return getattr(optim, optimizer_name)([parameter])


def _facade_classes() -> tuple[type[optim.HeavyBallOptimizer], ...]:
    return tuple(
        value
        for name in optim.__all__
        if name not in _DEPRECATED_FACADE_ALIASES
        and isinstance((value := getattr(optim, name)), type)
        and issubclass(value, optim.HeavyBallOptimizer)
        and value is not optim.HeavyBallOptimizer
    )


@cache
def _run_default(optimizer_name: str, scenario: str) -> _Run:
    """Build one facade with no hyperparameter overrides and collect real deltas."""

    if scenario == "constant_zero":
        initial = torch.zeros_like(_INITIAL_PARAMETER)
        gradients = _CONSTANT_GRADIENTS
        reset = torch.zeros_like(initial)
    elif scenario == "varying_zero":
        initial = torch.zeros_like(_INITIAL_PARAMETER)
        gradients = _VARYING_GRADIENTS
        reset = torch.zeros_like(initial)
    elif scenario == "natural":
        initial = _INITIAL_PARAMETER
        gradients = (_ANISOTROPIC_GRADIENT, _SECOND_GRADIENT)
        reset = None
    else:  # pragma: no cover - a test-author error, not an optimizer behavior
        raise ValueError(f"unknown scenario {scenario!r}")

    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = _build_default(optimizer_name, parameter)
    updates = []
    parameters = []
    for gradient in gradients:
        if reset is not None:
            parameter.detach().copy_(reset)
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()
        updates.append(before - parameter.detach())
        parameters.append(parameter.detach().clone())
    return _Run(tuple(updates), tuple(parameters), dict(optimizer.param_groups[0]))


@cache
def _run_frozen_preconditioner(optimizer_name: str) -> _Run:
    """Negative-control trajectory with the full-matrix basis held at identity."""

    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(torch.zeros_like(_INITIAL_PARAMETER))
    torch._dynamo.reset()
    optimizer = getattr(optim, optimizer_name)(
        [parameter], preconditioner_update_probability=0.0
    )
    updates = []
    parameters = []
    for gradient in _CONSTANT_GRADIENTS:
        parameter.detach().zero_()
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()
        updates.append(before - parameter.detach())
        parameters.append(parameter.detach().clone())
    return _Run(tuple(updates), tuple(parameters), dict(optimizer.param_groups[0]))


def _assert_finite_nonzero(optimizer_name: str, run: _Run) -> None:
    stacked = torch.stack(run.updates)
    assert torch.isfinite(stacked).all(), f"{optimizer_name}: non-finite update {stacked}"
    largest = stacked.flatten(start_dim=1).norm(dim=1).max().item()
    assert largest > 1e-8, f"{optimizer_name}: every default update was zero (max norm={largest:.3g})"


def _lr(run: _Run) -> float:
    return float(run.hyper["lr"])


def _unit(update: torch.Tensor) -> torch.Tensor:
    return update / update.norm()


def _gram_defect(value: torch.Tensor) -> float:
    gram = value @ value.mT
    scaled = gram / (gram.trace() / gram.shape[0])
    return torch.linalg.matrix_norm(scaled - torch.eye(gram.shape[0])).item()


def _assert_cross_axis_preconditioning(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    """A refreshed full-matrix factor must not reduce to its identity-basis update."""

    run = _run_default(optimizer_class.__name__, "constant_zero")
    frozen = _run_frozen_preconditioner(optimizer_class.__name__)
    _assert_finite_nonzero(optimizer_class.__name__, run)
    distance = (_unit(run.updates[-1]) - _unit(frozen.updates[-1])).norm().item()
    assert distance > 0.1, (
        f"{optimizer_class.__name__}: refreshed update still matches the identity-basis "
        f"diagonal/raw direction (unit-direction distance={distance:.6g}, required > 0.1)"
    )


def _assert_muon(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    singular_values = torch.linalg.svdvals(run.updates[-1])
    relative_spread = ((singular_values.max() - singular_values.min()) / singular_values.mean()).item()
    input_singular_values = torch.linalg.svdvals(_ANISOTROPIC_GRADIENT)
    input_spread = (
        (input_singular_values.max() - input_singular_values.min()) / input_singular_values.mean()
    ).item()
    assert relative_spread < 0.1 and relative_spread < input_spread / 5, (
        f"{optimizer_class.__name__}: update is not semi-orthogonal; singular_values="
        f"{singular_values.tolist()}, relative_spread={relative_spread:.6g}, "
        f"input_relative_spread={input_spread:.6g}"
    )


def _is_clearly_whiter(update: torch.Tensor) -> tuple[bool, float, float]:
    input_defect = _gram_defect(_ANISOTROPIC_GRADIENT)
    update_defect = _gram_defect(update)
    return update_defect < input_defect / 10, update_defect, input_defect


def _assert_whitening(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    applied, update_defect, input_defect = _is_clearly_whiter(run.updates[-1])
    assert applied, (
        f"{optimizer_class.__name__}: refreshed update was not clearly whiter; "
        f"Gram defect update={update_defect:.6g}, input={input_defect:.6g}, "
        f"required update < input/10={input_defect / 10:.6g}"
    )


def _assert_scion(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    direction = run.updates[-1] / _lr(run)
    spectral_norm = torch.linalg.matrix_norm(direction, ord=2).item()
    scale = float(run.hyper["scale"])
    bound = 1.1 * scale
    raw_spectral_norm = torch.linalg.matrix_norm(_ANISOTROPIC_GRADIENT, ord=2).item()
    assert spectral_norm <= bound and raw_spectral_norm > 5 * bound, (
        f"Scion: spectral LMO bound missing; update spectral norm={spectral_norm:.6g}, "
        f"bound={bound:.6g}, raw-gradient spectral norm={raw_spectral_norm:.6g}"
    )


def _assert_adamw(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    sgd_run = _run_default("SGD", "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    relative_difference = ((run.updates[0] - sgd_run.updates[0]).norm() / sgd_run.updates[0].norm()).item()
    assert relative_difference > 0.5, (
        f"AdamW: first update substituted plain SGD (relative difference={relative_difference:.6g})"
    )


def _assert_mars(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "varying_zero")
    adam_run = _run_default("AdamW", "varying_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    correction = (run.updates[1] / _lr(run) - adam_run.updates[1] / _lr(adam_run)).norm().item()
    assert correction > 0.01, f"MARSAdamW: prior-gradient correction missing (direction delta={correction:.6g})"


def _run_orthograd(optimizer_name: str) -> _Run:
    orthogonal_gradient = _SECOND_GRADIENT - (
        (_SECOND_GRADIENT * _INITIAL_PARAMETER).sum() / _INITIAL_PARAMETER.square().sum()
    ) * _INITIAL_PARAMETER
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_name, parameter)
    updates = []
    parameters = []
    for gradient in (_INITIAL_PARAMETER, orthogonal_gradient):
        parameter.detach().copy_(_INITIAL_PARAMETER)
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()
        updates.append(before - parameter.detach())
        parameters.append(parameter.detach().clone())
    return _Run(tuple(updates), tuple(parameters), dict(optimizer.param_groups[0]))


def _assert_orthograd_first(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_orthograd(optimizer_class.__name__)
    _assert_finite_nonzero(optimizer_class.__name__, run)
    parallel_norm = run.updates[0].norm().item()
    assert parallel_norm < 1e-8, (
        f"{optimizer_class.__name__}: gradient parallel to the parameter survived OrthoGrad "
        f"(update norm={parallel_norm:.6g})"
    )


def _assert_cautious(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    conflicting = _ANISOTROPIC_GRADIENT.clone()
    conflicting[0].mul_(-0.05)
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(torch.zeros_like(_INITIAL_PARAMETER))
    optimizer = _build_default(optimizer_class.__name__, parameter)
    updates = []
    for gradient in (_ANISOTROPIC_GRADIENT, conflicting):
        parameter.detach().zero_()
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()
        updates.append(before - parameter.detach())
    run = _Run(tuple(updates), (), dict(optimizer.param_groups[0]))
    _assert_finite_nonzero(optimizer_class.__name__, run)
    rejected_norm = run.updates[-1][0].norm().item()
    retained_norm = run.updates[-1][1:].norm().item()
    assert rejected_norm < 1e-8 and retained_norm > 1e-4, (
        f"CautiousAdamW: conflicting row was not selectively rejected; "
        f"rejected norm={rejected_norm:.6g}, retained norm={retained_norm:.6g}"
    )


def _assert_hyperball(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_class.__name__, parameter)
    initial = parameter.detach().clone()
    parameter.grad.copy_(_ANISOTROPIC_GRADIENT)
    optimizer.step()
    update = initial - parameter.detach()
    run = _Run((update,), (parameter.detach().clone(),), dict(optimizer.param_groups[0]))
    _assert_finite_nonzero(optimizer_class.__name__, run)
    norm_error = abs(parameter.norm().item() - initial.norm().item())
    assert norm_error < 1e-5, f"HyperBallAdamW: sphere projection missing (norm error={norm_error:.6g})"


def _assert_rms_or_sign(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    magnitudes = (run.updates[0] / _lr(run)).abs()
    spread = (magnitudes.max() - magnitudes.min()).item()
    mean_error = abs(magnitudes.mean().item() - 1.0)
    assert spread < 1e-5 and mean_error < 1e-5, (
        f"{optimizer_class.__name__}: first update is not sign/RMS-normalized; "
        f"magnitude spread={spread:.6g}, mean error from one={mean_error:.6g}"
    )


def _assert_laprop(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "varying_zero")
    adam_run = _run_default("AdamW", "varying_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    distance = (run.updates[-1] / _lr(run) - adam_run.updates[-1] / _lr(adam_run)).norm().item()
    assert distance > 0.1, f"LaProp: normalize-before-momentum path missing (direction delta={distance:.6g})"


def _assert_laprop_ortho(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_class.__name__, parameter)
    before = parameter.detach().clone()
    parameter.grad.copy_(_ANISOTROPIC_GRADIENT)
    optimizer.step()
    update = before - parameter.detach()
    run = _Run((update,), (parameter.detach().clone(),), dict(optimizer.param_groups[0]))
    _assert_finite_nonzero(optimizer_class.__name__, run)
    relative_dot = abs((update * before).sum().item()) / (update.norm() * before.norm()).item()
    assert relative_dot < 1e-4, (
        f"LaPropOrtho: final update is not orthogonal to the parameter (relative dot={relative_dot:.6g})"
    )


def _assert_sign_laprop(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "varying_zero")
    laprop_run = _run_default("LaProp", "varying_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    magnitudes = (run.updates[1] / _lr(run)).abs()
    laprop_magnitudes = (laprop_run.updates[1] / _lr(laprop_run)).abs()
    spread = (magnitudes.max() - magnitudes.min()).item()
    reference_spread = (laprop_magnitudes.max() - laprop_magnitudes.min()).item()
    assert spread < 1e-5 and reference_spread > 0.1, (
        f"SignLaProp: sign graft missing; magnitude spread={spread:.6g}, "
        f"ungrafted LaProp spread={reference_spread:.6g}"
    )


def _assert_nadam(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    magnitudes = (run.updates[0] / _lr(run)).abs()
    spread = (magnitudes.max() - magnitudes.min()).item()
    mean = magnitudes.mean().item()
    assert spread < 1e-5 and mean > 1.02, (
        f"NAdam: Nesterov look-ahead weight missing; normalized magnitude mean={mean:.6g}, spread={spread:.6g}"
    )


def _assert_ademamix(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    magnitudes = (run.updates[0] / _lr(run)).abs()
    spread = (magnitudes.max() - magnitudes.min()).item()
    mean = magnitudes.mean().item()
    assert spread < 1e-5 and mean > 1.0001, (
        f"AdEMAMix: slow EMA contribution missing; normalized magnitude mean={mean:.7g}, spread={spread:.6g}"
    )


def _assert_adopt(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "varying_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    seed_norm = run.updates[0].norm().item()
    next_norm = run.updates[1].norm().item()
    assert seed_norm < 1e-8 and next_norm > 1e-4, (
        f"ADOPT: expected seed-only transition then a live update; first norm={seed_norm:.6g}, "
        f"second norm={next_norm:.6g}"
    )


def _assert_unscaled_adam(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    sgd_run = _run_default("SGD", "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    difference = (run.updates[0] / _lr(run) - sgd_run.updates[0] / _lr(sgd_run)).norm().item()
    assert difference < 1e-5, f"UnscaledAdamW: first unscaled step is not raw-gradient-like (delta={difference:.6g})"


def _assert_sgd(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "constant_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    error = (run.updates[0] / _lr(run) - _ANISOTROPIC_GRADIENT).norm().item()
    assert error < 1e-5, f"SGD: update is not the raw gradient direction (error={error:.6g})"


def _run_eval_swap(optimizer_name: str) -> tuple[_Run, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_name, parameter)
    updates = []
    parameters = []
    for gradient in (_ANISOTROPIC_GRADIENT, _SECOND_GRADIENT):
        before = parameter.detach().clone()
        parameter.grad.copy_(gradient)
        optimizer.step()
        updates.append(before - parameter.detach())
        parameters.append(parameter.detach().clone())
    train_parameter = parameter.detach().clone()
    optimizer.eval()
    eval_parameter = parameter.detach().clone()
    optimizer.train()
    restored_parameter = parameter.detach().clone()
    run = _Run(tuple(updates), tuple(parameters), dict(optimizer.param_groups[0]))
    return run, train_parameter, eval_parameter, restored_parameter


def _assert_schedule_free(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run, train_parameter, eval_parameter, restored_parameter = _run_eval_swap(optimizer_class.__name__)
    _assert_finite_nonzero(optimizer_class.__name__, run)
    representation_gap = (train_parameter - eval_parameter).norm().item()
    round_trip_error = (train_parameter - restored_parameter).abs().max().item()
    assert representation_gap > 1e-5 and round_trip_error < 1e-7, (
        f"{optimizer_class.__name__}: train/eval iterate swap missing; "
        f"representation gap={representation_gap:.6g}, "
        f"round-trip error={round_trip_error:.6g}"
    )


def _assert_msam(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run, train_parameter, eval_parameter, restored_parameter = _run_eval_swap(optimizer_class.__name__)
    _assert_finite_nonzero(optimizer_class.__name__, run)
    perturbation = (train_parameter - eval_parameter).norm().item()
    sam_step_size = float(run.hyper["sam_step_size"])
    round_trip_error = (train_parameter - restored_parameter).abs().max().item()
    assert abs(perturbation - sam_step_size) < 1e-5 and round_trip_error < 1e-7, (
        f"MSAM: SAM perturbation/eval swap missing; perturbation={perturbation:.6g}, "
        f"sam_step_size={sam_step_size:.6g}, round-trip error={round_trip_error:.6g}"
    )


def _assert_suds(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    run = _run_default(optimizer_class.__name__, "varying_zero")
    adam_run = _run_default("AdamW", "varying_zero")
    _assert_finite_nonzero(optimizer_class.__name__, run)
    distance = (run.updates[-1] / _lr(run) - adam_run.updates[-1] / _lr(adam_run)).norm().item()
    assert run.updates[0].norm().item() < 1e-8 and distance > 0.5, (
        f"SUDSAdamW: rank-one Fisher rotation missing; seed update norm={run.updates[0].norm().item():.6g}, "
        f"Adam direction delta={distance:.6g}"
    )


def _assert_generic_training(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    """Future facades remain covered until they receive a name-specific rule."""

    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_class.__name__, parameter)
    initial_loss = parameter.square().sum().item()
    updates = []
    parameters = []
    for _ in range(5):
        before = parameter.detach().clone()
        parameter.grad.copy_(parameter.detach())
        optimizer.step()
        updates.append(before - parameter.detach())
        parameters.append(parameter.detach().clone())
    run = _Run(tuple(updates), tuple(parameters), dict(optimizer.param_groups[0]))
    _assert_finite_nonzero(optimizer_class.__name__, run)
    final_loss = parameter.square().sum().item()
    assert final_loss < initial_loss, (
        f"{optimizer_class.__name__}: fallback training check did not decrease loss "
        f"({initial_loss:.6g} -> {final_loss:.6g})"
    )


def _assert_truegrad_training(optimizer_class: type[optim.HeavyBallOptimizer]) -> None:
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(_INITIAL_PARAMETER.clone())
    optimizer = _build_default(optimizer_class.__name__, parameter)
    initial_loss = parameter.square().sum().item()
    for _ in range(5):
        parameter.grad.copy_(parameter.detach())
        optimizer.produce(parameter, "sum_grad_squared", parameter.grad.square())
        optimizer.step()
    final_loss = parameter.square().sum().item()
    assert final_loss < initial_loss, (
        f"{optimizer_class.__name__}: truegrad training check did not decrease loss "
        f"({initial_loss:.6g} -> {final_loss:.6g})"
    )


_PROPERTY_RULES = {
    "ADOPT": _assert_adopt,
    "AdEMAMix": _assert_ademamix,
    "AdamW": _assert_adamw,
    "CautiousAdamW": _assert_cautious,
    "HyperBallAdamW": _assert_hyperball,
    "KLSOAP": _assert_cross_axis_preconditioning,
    "KLShampoo": _assert_cross_axis_preconditioning,
    "LATHER": _assert_cross_axis_preconditioning,
    "LaProp": _assert_laprop,
    "LaPropOrtho": _assert_laprop_ortho,
    "Lion": _assert_rms_or_sign,
    "MARSAdamW": _assert_mars,
    "MSAM": _assert_msam,
    "Muon": _assert_muon,
    "MuonLaProp": _assert_muon,
    "NAdam": _assert_nadam,
    "OrthoGradAdamW": _assert_orthograd_first,
    "OrthoLaProp": _assert_orthograd_first,
    "PSGD": _assert_cross_axis_preconditioning,
    "PSGDKron": _assert_cross_axis_preconditioning,
    "PSGDPro": _assert_cross_axis_preconditioning,
    "QSGD": _assert_cross_axis_preconditioning,
    "RMSprop": _assert_rms_or_sign,
    "SGD": _assert_sgd,
    "SOAP": _assert_cross_axis_preconditioning,
    "SUDSAdamW": _assert_suds,
    "SFAdamW": _assert_schedule_free,
    "Scion": _assert_scion,
    "Shampoo": _assert_cross_axis_preconditioning,
    "SignLaProp": _assert_sign_laprop,
    "SignSGD": _assert_rms_or_sign,
    "TrueGradAdam": _assert_truegrad_training,
    "TrueGradLaProp": _assert_truegrad_training,
    "TrueGradNAdam": _assert_truegrad_training,
    "TrueGradRMSprop": _assert_truegrad_training,
    "UnscaledAdamW": _assert_unscaled_adam,
    "Whitening": _assert_whitening,
}


@pytest.mark.parametrize("optimizer_class", _facade_classes(), ids=lambda cls: cls.__name__)
def test_default_optimizer_matches_its_advertised_structure(optimizer_class):
    """Every concrete ``heavyball.optim`` export is built at its defaults and behaviorally exercised."""

    rule = _PROPERTY_RULES.get(optimizer_class.__name__, _assert_generic_training)
    rule(optimizer_class)


def test_frozen_whitening_identity_basis_is_a_red_control():
    """With refresh probability zero, the main whitening assertion above must fail."""

    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(torch.zeros_like(_INITIAL_PARAMETER))
    torch._dynamo.reset()
    optimizer = optim.Whitening([parameter], preconditioner_update_probability=0.0)
    for _ in _CONSTANT_GRADIENTS:
        parameter.detach().zero_()
        parameter.grad.copy_(_ANISOTROPIC_GRADIENT)
        optimizer.step()
    applied, update_defect, input_defect = _is_clearly_whiter(-parameter.detach())
    assert not applied, (
        f"frozen identity unexpectedly passed whitening: update defect={update_defect:.6g}, "
        f"input defect={input_defect:.6g}"
    )
