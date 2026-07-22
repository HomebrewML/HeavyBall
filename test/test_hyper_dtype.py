"""Per-group hyperparameters carry the group's own dtype, not the first param's.

When one optimizer holds params of mixed dtype, HeavyBall buckets them into per-dtype groups. Each
group's hyperparameter scalars must match that group's dtype, independent of construction order, so an
fp64 group does not silently compute with fp32 betas/eps (precision loss + order-dependence). The
checkpoint hyper map must also keep the per-dtype hypers distinct.
"""

from unittest.mock import patch

import pytest
import torch

import heavyball


@pytest.fixture(autouse=True)
def _eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


_DT = {"f32": torch.float32, "f64": torch.float64}


@pytest.mark.parametrize("order", [("f32", "f64"), ("f64", "f32")])
def test_each_group_hyper_matches_its_own_dtype(order):
    params = [torch.nn.Parameter(torch.randn(3, dtype=_DT[name])) for name in order]
    opt = heavyball.AdamW(params, lr=1e-3)
    for engine in opt._engines:
        for group in engine.groups:
            group_dtype = group.params[0].dtype
            assert group.hyper.beta1.dtype == group_dtype
            assert group.hyper.eps.dtype == group_dtype


def test_uniform_fp32_hyper_stays_fp32():
    opt = heavyball.AdamW([torch.nn.Parameter(torch.randn(3))], lr=1e-3)
    assert opt._engines[0].groups[0].hyper.beta1.dtype == torch.float32


def test_set_hyper_updates_every_mixed_dtype_group():
    params = [torch.nn.Parameter(torch.randn(3, dtype=_DT[name])) for name in ("f32", "f64")]
    opt = heavyball.AdamW(params, lr=1e-3)
    opt.param_groups[0]["lr"] = 0.05  # scheduler-style mutation, synced at step()
    for param in params:
        param.grad.fill_(1.0)
    opt.step()
    for engine in opt._engines:
        for group in engine.groups:
            assert group.hyper.lr.item() == pytest.approx(0.05)
            assert group.hyper.lr.dtype == group.params[0].dtype


def test_mixed_dtype_optimizer_updates_both_params_finitely():
    for order in (("f32", "f64"), ("f64", "f32")):
        params = [torch.nn.Parameter(torch.zeros(3, dtype=_DT[name])) for name in order]
        opt = heavyball.AdamW(params, lr=1e-2)
        for param in params:
            param.grad.fill_(1.0)
        before = [p.detach().clone() for p in params]
        opt.step()
        for param, prior in zip(params, before):
            assert torch.isfinite(param).all()
            assert not torch.equal(param, prior)


def test_mixed_dtype_checkpoint_roundtrip_preserves_both_hypers():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(3, dtype=_DT[name])) for name in ("f32", "f64")]
    opt = heavyball.AdamW(params, lr=1e-2)
    for _ in range(2):
        for param in params:
            param.grad.copy_(torch.randn_like(param))
        opt.step()
    checkpoint = opt.state_dict()

    restored = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt2 = heavyball.AdamW(restored, lr=0.5)  # different lr, must be overwritten by load
    opt2.load_state_dict(checkpoint)

    grads = [torch.randn_like(p) for p in params]
    for param, restored_param, grad in zip(params, restored, grads):
        param.grad.copy_(grad)
        restored_param.grad.copy_(grad)
    opt.step()
    opt2.step()
    for param, restored_param in zip(params, restored):
        torch.testing.assert_close(param, restored_param, rtol=0, atol=0)
