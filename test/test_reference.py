"""Shipped first-order optimizers match pure fp64 textbook references.

These checks compare the core first-order optimizers with math implemented from scratch in
``reference.py``, so agreement between two optimizer implementations cannot mask a shared bug.
"""

from unittest.mock import patch

import pytest
import reference
import torch

import heavyball
from heavyball.core import build


@pytest.fixture(autouse=True)
def _eager():
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        yield


def _run_shipped(recipe, init, grads, **hyper):
    param = torch.nn.Parameter(init.clone())
    optimizer = build([param], recipe, **hyper)
    for grad in grads:
        param.grad.copy_(grad)
        optimizer.step()
    return param.detach()


_CASES = [
    ("adamw", heavyball.adamw, reference.adam, dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.05)),
    ("rmsprop", heavyball.rmsprop, reference.rmsprop, dict(lr=1e-2, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("laprop", heavyball.laprop, reference.laprop, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("sgd", heavyball.sgd, reference.sgd, dict(lr=1e-2, weight_decay=0.05)),
    ("adopt", heavyball.adopt, reference.adopt, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("lion", heavyball.lion, reference.lion, dict(lr=1e-2, beta1=0.9, beta2=0.99, weight_decay=0.05)),
    ("signsgd", heavyball.signsgd, reference.signsgd, dict(lr=1e-2, weight_decay=0.05)),
    ("unscaled_adamw", heavyball.unscaled_adamw, reference.unscaled_adam, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("cautious_adamw", heavyball.cautious_adamw, reference.cautious_adam, dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.05)),
    ("ademamix", heavyball.ademamix, reference.ademamix, dict(lr=1e-2, beta1=0.9, beta2=0.99, beta3=0.99, alpha=2.0, eps=1e-8, weight_decay=0.05)),
    ("nadam", heavyball.nadam, reference.nadam, dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.05, momentum_decay=4e-3)),
    ("mars_adamw", heavyball.mars_adamw, reference.mars_adam, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05, mars_gamma=0.05)),
    ("orthograd_adamw", heavyball.orthograd_adamw, reference.orthograd_adam, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("ortho_laprop", heavyball.ortho_laprop, reference.ortho_laprop, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("laprop_ortho", heavyball.laprop_ortho, reference.laprop_ortho, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
    ("sign_laprop", heavyball.sign_laprop, reference.sign_laprop, dict(lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.05)),
]


@pytest.mark.parametrize("name, recipe, ref_fn, hyper", _CASES, ids=[case[0] for case in _CASES])
def test_shipped_matches_pure_fp64_reference(name, recipe, ref_fn, hyper):
    torch.manual_seed(0)
    init = torch.randn(16, dtype=torch.float64)
    grads = [torch.randn(16, dtype=torch.float64) for _ in range(10)]
    shipped = _run_shipped(recipe, init, grads, **hyper)
    expected = ref_fn(init, grads, **hyper)
    torch.testing.assert_close(shipped, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name, recipe, ref_fn, hyper", _CASES, ids=[case[0] for case in _CASES])
def test_shipped_bf16_is_closer_to_fp64_than_naive_bf16(name, recipe, ref_fn, hyper):
    """bfloat16 is where the precision design pays off: shipped keeps state in fp32 and rounds the
    param unbiasedly (stochastic rounding), so its seed-averaged result is markedly closer to the fp64
    truth than a naive bf16 implementation of the same math. fp32 and fp16 add no signal over the
    exact fp64 parity above -- both just round identical fp64 values, and fp16's write floor washes out
    the fp32 state -- so bf16 is the honest place to prove the design."""
    torch.manual_seed(0)
    init = torch.randn(64, dtype=torch.float64)
    grads = [torch.randn(64, dtype=torch.float64) for _ in range(20)]
    truth = ref_fn(init, grads, **hyper)

    samples = []
    for seed in range(48):  # stochastic bf16 rounding is unbiased; the expectation needs averaging
        torch.manual_seed(seed)
        samples.append(_run_shipped(recipe, init.bfloat16(), [g.bfloat16() for g in grads], **hyper).double())
    shipped = torch.stack(samples).mean(0)
    naive = ref_fn(init.bfloat16(), [g.bfloat16() for g in grads], **hyper).double()

    shipped_error = (shipped - truth).abs().max()
    naive_error = (naive - truth).abs().max()
    assert shipped_error < naive_error


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def test_adamw_is_at_least_as_close_to_fp64_as_torch_optim(dtype):
    """HeavyBall keeps optimizer state in fp32 and rounds a low-precision param unbiasedly, so its
    expected result is at least as close to the fp64 truth as torch.optim's. bf16 is stochastically
    rounded, so the seed-averaged expectation -- not one sample -- is the honest metric."""
    torch.manual_seed(0)
    init = torch.randn(256, dtype=torch.float64)
    grads = [torch.randn(256, dtype=torch.float64) for _ in range(10)]
    hyper = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.05)
    truth = reference.adam(init, grads, **hyper)

    seeds = 48 if dtype is torch.bfloat16 else 1  # only bf16 stochastic-rounds; fp32/fp16 are deterministic
    samples = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        samples.append(_run_shipped(heavyball.adamw, init.to(dtype), [g.to(dtype) for g in grads], **hyper).double())
    shipped = torch.stack(samples).mean(0)

    param = torch.nn.Parameter(init.to(dtype).clone())
    baseline = torch.optim.AdamW(
        [param], lr=hyper["lr"], betas=(hyper["beta1"], hyper["beta2"]), eps=hyper["eps"], weight_decay=hyper["weight_decay"]
    )
    for gradient in grads:
        param.grad = gradient.to(dtype)
        baseline.step()

    shipped_error = (shipped - truth).abs().max()
    baseline_error = (param.detach().double() - truth).abs().max()
    assert shipped_error <= baseline_error
