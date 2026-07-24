"""AdamC: AdamW with weight decay scaled by lr/max_lr (Defazio et al., arXiv 2506.02285).

The commit applies decoupled weight decay with an effective coefficient ``weight_decay * lr / max_lr``
(matching legacy ``update_by_adamc``: ``decay = lr*wd/max_lr``, applied as ``update + param*decay`` then
scaled by ``lr``). So AdamC reduces exactly to AdamW when ``max_lr == lr``, and halves the decay when
``max_lr`` is doubled.
"""

from unittest.mock import patch

import pytest
import torch

import heavyball


@pytest.fixture(autouse=True)
def _eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _trajectory(name, lr, weight_decay, grads, **extra):
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(8))
    optimizer = getattr(heavyball, name)([param], lr=lr, weight_decay=weight_decay, **extra)
    for grad in grads:
        param.grad.copy_(grad)
        optimizer.step()
    return param.detach().clone()


def test_adamc_equals_adamw_when_max_lr_equals_lr():
    torch.manual_seed(1)
    grads = [torch.randn(8) for _ in range(6)]
    adamc = _trajectory("AdamC", lr=0.01, weight_decay=0.1, grads=grads, max_lr=0.01)
    adamw = _trajectory("AdamW", lr=0.01, weight_decay=0.1, grads=grads)
    torch.testing.assert_close(adamc, adamw, rtol=0, atol=0)


def test_adamc_halves_weight_decay_when_max_lr_doubled():
    # Zero gradient isolates weight decay: param *= (1 - lr * wd_effective).
    zero = [torch.zeros(8)]
    base = torch.nn.Parameter(torch.ones(8))  # reference start
    adamw = _trajectory("AdamW", lr=0.01, weight_decay=0.5, grads=zero)
    adamc = _trajectory("AdamC", lr=0.01, weight_decay=0.5, grads=zero, max_lr=0.02)
    start = _trajectory("AdamW", lr=0.0, weight_decay=0.0, grads=zero)  # unchanged init
    adamw_decay = start - adamw
    adamc_decay = start - adamc
    assert adamw_decay.abs().max() > 1e-4  # weight decay actually moved the param
    # halved to fp32 precision (init - init*(1-lr*wd) is a catastrophic-cancellation subtraction)
    torch.testing.assert_close(adamc_decay, adamw_decay / 2, rtol=1e-3, atol=1e-6)


def test_adamc_omitted_max_lr_inherits_lr():
    # Omitting ``max_lr`` must resolve it to the construction ``lr`` (regression: the recipe default pinned
    # it to 0.0025, so a non-default ``lr`` silently scaled weight decay by ``lr / 0.0025``). With max_lr
    # inheriting lr, AdamC's zero-gradient decay must match AdamW's and an explicit max_lr=lr.
    zero = [torch.zeros(8)]
    adamw = _trajectory("AdamW", lr=0.01, weight_decay=0.5, grads=zero)
    adamc_omitted = _trajectory("AdamC", lr=0.01, weight_decay=0.5, grads=zero)
    adamc_explicit = _trajectory("AdamC", lr=0.01, weight_decay=0.5, grads=zero, max_lr=0.01)
    torch.testing.assert_close(adamc_omitted, adamw, rtol=0, atol=0)
    torch.testing.assert_close(adamc_omitted, adamc_explicit, rtol=0, atol=0)


def test_adamc_multigroup_max_lr_inherits_each_group_lr():
    # Omitted max_lr must resolve to EACH group's own lr, not a single top-level lr (regression: it was
    # pinned to the construction lr, silently mis-scaling decay for any group with a different lr).
    p1 = torch.nn.Parameter(torch.ones(4))
    p2 = torch.nn.Parameter(torch.ones(4))
    optimizer = heavyball.AdamC(
        [{"params": [p1], "lr": 0.01}, {"params": [p2], "lr": 0.02}], weight_decay=0.5
    )
    assert optimizer.param_groups[0]["max_lr"] == 0.01
    assert optimizer.param_groups[1]["max_lr"] == 0.02
    optimizer.zero_grad()
    p1.grad.copy_(torch.zeros_like(p1))
    p2.grad.copy_(torch.zeros_like(p2))
    optimizer.step()
    # decay = wd * lr / max_lr = wd * lr / lr = wd = 0.5  ->  param = 1 - lr * 0.5
    torch.testing.assert_close(p1, torch.full((4,), 1 - 0.01 * 0.5), rtol=0, atol=1e-6)
    torch.testing.assert_close(p2, torch.full((4,), 1 - 0.02 * 0.5), rtol=0, atol=1e-6)


def test_adamc_direct_build_omitted_max_lr_does_not_crash():
    # The low-level build() path must also resolve the max_lr sentinel (regression: only the facade
    # resolved it, so build(..., adamc, lr=...) hit _scalar(None) -> TypeError).
    from heavyball.core import build

    parameter = torch.nn.Parameter(torch.ones(2))
    engine = build([parameter], heavyball.adamc, lr=0.01, weight_decay=0.5, _leaf_indices=range(1))
    assert engine is not None


def test_adamc_zero_lr_does_not_nan():
    # lr=0 with inherited max_lr is 0/0; it must not NaN-poison the (no-op) update (regression).
    p = torch.nn.Parameter(torch.ones(4))
    heavyball.AdamC([p], lr=0.0, weight_decay=0.5).step()
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p, torch.ones(4))  # lr=0 -> no movement


def test_non_adamc_optimizers_have_no_max_lr():
    # The max_lr sentinel resolution must only touch AdamC; it used to add a spurious max_lr to every
    # optimizer, breaking pre-pass checkpoints via the strict hyperparameter-name check (regression).
    opt = heavyball.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    assert "max_lr" not in opt.param_groups[0]
    assert not hasattr(opt._engine.groups[0].hyper, "max_lr")


def test_adamc_is_a_facade_and_declares_max_lr():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        opt = heavyball.AdamC([torch.nn.Parameter(torch.randn(4))], lr=1e-3, max_lr=1e-2)
    assert isinstance(opt, heavyball.AdamC)
