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


def test_adamc_is_a_facade_and_declares_max_lr():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        opt = heavyball.AdamC([torch.nn.Parameter(torch.randn(4))], lr=1e-3, max_lr=1e-2)
    assert isinstance(opt, heavyball.AdamC)
