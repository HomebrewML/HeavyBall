"""Fast CPU-only smoke tests for non-PSGD HeavyBall features."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

import heavyball


def _train_once(optimizer, model: nn.Module, data: torch.Tensor, target: torch.Tensor, steps: int = 3) -> float:
    """Run a few optimization steps and return the final loss."""

    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
    return loss.item()


def _parameter_drift(model: nn.Module, original: list[torch.Tensor]) -> float:
    current = [param.detach() for param in model.parameters()]
    diffs = [curr - init for curr, init in zip(current, original, strict=True)]
    stacked = torch.cat([diff.reshape(-1) for diff in diffs])
    return stacked.norm().item()


def _make_batch(
    in_features: int = 8, out_features: int = 4, batch: int = 16
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0x172893)
    model = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features))
    data = torch.randn(batch, in_features)
    target = torch.randn(batch, out_features)
    return model, data, target


@pytest.mark.parametrize(
    "opt_name",
    [
        "ForeachSOAP",
        "Muon",
        "ForeachAdamW",
    ],
)
def test_selected_optimizers_run_on_cpu(opt_name: str) -> None:
    model, data, target = _make_batch()
    init = [param.detach().clone() for param in model.parameters()]

    opt_cls = getattr(heavyball, opt_name)
    optimizer = opt_cls(model.parameters(), warmup_steps=0)
    final_loss = _train_once(optimizer, model, data, target, steps=3)

    assert torch.isfinite(torch.tensor(final_loss))
    assert _parameter_drift(model, init) > 0.0


def test_caution_reduces_update_magnitude() -> None:
    baseline_model, data, target = _make_batch()
    cautious_model = deepcopy(baseline_model)

    baseline_init = [param.detach().clone() for param in baseline_model.parameters()]
    cautious_init = [param.detach().clone() for param in cautious_model.parameters()]

    baseline_opt = heavyball.SGD(
        baseline_model.parameters(),
        lr=1e-3,
        caution=False,
    )
    cautious_opt = heavyball.SGD(
        cautious_model.parameters(),
        lr=1e-3,
        caution=True,
    )

    _train_once(baseline_opt, baseline_model, data, target)
    _train_once(cautious_opt, cautious_model, data, target)

    baseline_drift = _parameter_drift(baseline_model, baseline_init)
    cautious_drift = _parameter_drift(cautious_model, cautious_init)

    assert cautious_drift <= baseline_drift * 1.05  # caution should not overshoot compared to baseline


def test_mars_flag_changes_behavior() -> None:
    model_a, data, target = _make_batch()
    model_b = deepcopy(model_a)

    opt_a = heavyball.ForeachAdamW(model_a.parameters(), mars=False, warmup_steps=0)
    opt_b = heavyball.ForeachAdamW(model_b.parameters(), mars=True, warmup_steps=0)

    init = [param.detach().clone() for param in model_a.parameters()]

    _train_once(opt_a, model_a, data, target)
    _train_once(opt_b, model_b, data, target)

    baseline_drift = _parameter_drift(model_a, init)
    mars_drift = _parameter_drift(model_b, init)
    assert baseline_drift > 0.0
    assert mars_drift > 0.0

    deltas = [a.detach() - b.detach() for a, b in zip(model_a.parameters(), model_b.parameters(), strict=True)]
    combined = torch.cat([delta.reshape(-1) for delta in deltas])
    assert combined.norm().item() > 1e-6  # mars path should diverge from baseline


def test_sam_wrapper_requires_closure() -> None:
    model = nn.Linear(4, 2)
    base = heavyball.ForeachAdamW(model.parameters())
    wrapper = heavyball.SAMWrapper(model.parameters(), wrapped_optimizer=base)

    with pytest.raises(ValueError):
        wrapper.step()

    data = torch.randn(8, 4)
    target = torch.randn(8, 2)

    def closure():
        wrapper.zero_grad()
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        return loss

    before = [param.detach().clone() for param in model.parameters()]
    wrapper.step(closure)
    after = [param.detach() for param in model.parameters()]
    diff = torch.cat([(a - b).reshape(-1) for a, b in zip(after, before, strict=True)])
    assert diff.norm().item() > 0.0
