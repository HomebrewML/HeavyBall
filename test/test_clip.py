import os

os.environ["TORCH_LOGS"] = "+recompiles"

import pytest
import math

import torch
from torch import nn
from torch import linalg
from torch._dynamo import config

import heavyball
from heavyball.utils import (
    _compilable_l2_clip_,
    _compilable_rmsnorm_clip_,
    _compilable_global_rmsnorm_clip_,
    _compilable_global_l2norm_clip_,
)
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch
from heavyball import utils

config.cache_size_limit = 128


def _make_tensors_with_grad(x):
    out = [torch.zeros_like(y, requires_grad=True) for y in x]
    for y, grad in zip(out, x):
        y.grad = grad.clone()
    return out


def _in_assertions(x):
    if utils.compile_mode == None:
        assert all([not torch.isnan(y).any() for y in x]), "nan before clipping"
        assert all([not torch.isinf(y).any() for y in x]), "inf before clipping"


def _out_assertions(torch_clipped, heavyball_clipped):
    if utils.compile_mode == None:
        assert all([torch.allclose(a, b) for a, b in zip(torch_clipped, heavyball_clipped)])
        assert all([not (torch.isnan(y).any() or torch.isinf(y).any()) for y in heavyball_clipped])


def _clip_non_global(x, norm, clip_at):
    scalar = [max(clip_at / n.item(), 1.0) for n in norm]
    return [s * y for s, y in zip(scalar, x)]


def _test_rmsnorm(x, clip_at):
    _in_assertions(x)
    norm = [torch.sqrt(torch.mean(y**2)) + 1e-6 for y in x]
    torch_clipped = _clip_non_global(x, norm, clip_at)
    heavyball_clipped = _compilable_rmsnorm_clip_(x, clip_at)
    _out_assertions(torch_clipped, heavyball_clipped)
    return heavyball_clipped


def _test_l2(x, clip_at):
    _in_assertions(x)
    norm = [linalg.vector_norm(y) + 1e-6 for y in x]
    torch_clipped = _clip_non_global(x, norm, clip_at)
    heavyball_clipped = _compilable_l2_clip_(x, clip_at)
    _out_assertions(torch_clipped, heavyball_clipped)
    return heavyball_clipped


def _test_global_l2norm(x, clip_at):
    _in_assertions(x)
    parameters = _make_tensors_with_grad(x)
    nn.utils.clip_grad_norm_(parameters, clip_at)
    torch_clipped = [y.grad for y in parameters]
    heavyball_clipped = _compilable_global_l2norm_clip_(x, clip_at)
    _out_assertions(torch_clipped, heavyball_clipped)
    return heavyball_clipped


def _test_global_rmsnorm(x, clip_at):
    _in_assertions(x)
    l2_norm = nn.utils.get_total_norm(x)
    rms_norm = l2_norm / math.sqrt(sum(y.numel() for y in x))
    parameters = _make_tensors_with_grad(x)
    nn.utils.clip_grads_with_norm_(parameters, clip_at, rms_norm)
    torch_clipped = [y.grad for y in parameters]
    heavyball_clipped = _compilable_global_rmsnorm_clip_(x, clip_at)
    _out_assertions(torch_clipped, heavyball_clipped)
    return heavyball_clipped


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 2)])
def test_clip(opt, size, depth: int, iterations: int = 16, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    for mode in ["max-autotune-no-cudagraphs", None]:
        utils.compile_mode = mode
        for clip_func in [_test_rmsnorm, _test_l2, _test_global_l2norm, _test_global_rmsnorm]:
            torch.manual_seed(0x2131290)

            for i in range(outer_iterations):
                model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
                o = get_optim(
                    opt,
                    model.parameters(),
                    lr=1e-5,
                    gradient_clipping=lambda x: clip_func(x, clip_at=5.0),
                    update_clipping=lambda x: clip_func(x, clip_at=0.05),
                )

                for _ in range(iterations):
                    loss = model(torch.randn((1024, size), device="cuda")).square().mean()
                    loss.backward()
                    o.step()
                    if utils.compile_mode != None:
                        assert all([not (torch.isnan(y).any() or torch.isinf(y).any()) for y in model.parameters()])
                    o.zero_grad()

                del model, o
                clean()
