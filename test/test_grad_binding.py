"""Fail loud when a gradient is reassigned off its slab-bound view instead of written in place.

HeavyBall binds ``param.grad`` to a persistent slab view (core.py) and reads that slab in ``step()``.
``loss.backward()`` and ``p.grad.copy_(g)`` write in place and work. Reassigning ``p.grad = g`` (or
``p.grad = None``) breaks the binding, so the slab -- not the user's gradient -- is used. That
silent-ignore must fail loud, consistent with ``zero_grad(set_to_none=True)``.
"""

from unittest.mock import patch

import pytest
import torch

import heavyball


@pytest.fixture(autouse=True)
def _eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def test_reassigning_grad_raises():
    param = torch.nn.Parameter(torch.zeros(4))
    opt = heavyball.SGD([param], lr=0.1)
    param.grad = torch.ones(4)
    with pytest.raises(ValueError, match="in place"):
        opt.step()


def test_grad_set_to_none_raises():
    param = torch.nn.Parameter(torch.zeros(4))
    opt = heavyball.SGD([param], lr=0.1)
    param.grad = None
    with pytest.raises(ValueError, match="in place"):
        opt.step()


def test_reassign_in_second_group_raises():
    first = torch.nn.Parameter(torch.zeros(4))
    second = torch.nn.Parameter(torch.zeros(3))
    opt = heavyball.AdamW([{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.01}])
    first.grad.fill_(1.0)
    second.grad = torch.ones(3)
    with pytest.raises(ValueError, match="in place"):
        opt.step()


def test_reassigning_param_data_raises():
    param = torch.nn.Parameter(torch.zeros(4))
    opt = heavyball.SGD([param], lr=0.1)
    param.grad.fill_(1.0)
    param.data = torch.ones(4)  # reassigning weights detaches them from the slab the step updates
    with pytest.raises(ValueError, match="in place"):
        opt.step()


def test_inplace_weight_update_is_kept():
    param = torch.nn.Parameter(torch.zeros(4))
    opt = heavyball.SGD([param], lr=0.1, weight_decay=0.0)
    with torch.no_grad():
        param.copy_(torch.full((4,), 3.0))  # in-place weight surgery stays bound
    param.grad.fill_(1.0)
    opt.step()  # must not raise
    assert torch.allclose(param, torch.full((4,), 2.9))


@pytest.mark.parametrize(
    ("binding", "message"),
    (("weights", "weights.*no longer slab-bound"), ("gradient", "gradient.*no longer slab-bound")),
)
def test_same_pointer_transposed_view_raises(binding, message):
    param = torch.nn.Parameter(torch.tensor(((1.0, 2.0), (3.0, 4.0))))
    opt = heavyball.Engine([param], heavyball.sgd, lr=1.0, weight_decay=0.0)
    group = opt.groups[0]
    before_step = opt.step_count.detach().clone()
    before_age = group.age.detach().clone()
    before_slab = group.param_slab.detach().clone()

    if binding == "weights":
        param.data = param.data.t()
        param.grad.zero_()
    else:
        param.grad = param.grad.t()

    with pytest.raises(ValueError, match=message):
        opt.step()

    assert torch.equal(opt.step_count, before_step)
    assert torch.equal(group.age, before_age)
    assert torch.equal(group.param_slab, before_slab)
