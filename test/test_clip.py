import functools
import math

import pytest
import torch

import heavyball
from heavyball import utils
from heavyball.utils import (
    _compilable_global_l2norm_clip_,
    _compilable_global_rmsnorm_clip_,
    _compilable_l2_clip_,
    _compilable_rmsnorm_clip_,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _isolate_compile_mode():
    mode = utils.compile_mode
    yield
    utils.compile_mode = mode


def _reference_clip(values, kind, threshold):
    values = [value.double() for value in values]
    if kind == "l2":
        norms = [value.norm() for value in values]
    elif kind == "rms":
        norms = [value.norm() / math.sqrt(value.numel()) for value in values]
    else:
        norm = torch.stack([value.square().sum() for value in values]).sum().sqrt()
        if kind == "global_rms":
            norm /= math.sqrt(sum(value.numel() for value in values))
        norms = [norm] * len(values)
    return [
        value * (threshold / norm.clamp_min(torch.finfo(torch.float64).tiny)).clamp_max(1)
        for value, norm in zip(values, norms)
    ]


CLIP_FNS = [
    ("l2", _compilable_l2_clip_),
    ("rms", _compilable_rmsnorm_clip_),
    ("global_l2", _compilable_global_l2norm_clip_),
    ("global_rms", _compilable_global_rmsnorm_clip_),
]


@pytest.mark.parametrize("kind,clip_fn", CLIP_FNS)
def test_clip(kind, clip_fn):
    gradients = [
        torch.tensor([[-6.0, 8.0], [3.0, -4.0]], device=DEVICE),
        torch.tensor([1.0, -2.0, 4.0], device=DEVICE),
    ]
    expected = _reference_clip(_reference_clip(gradients, kind, 2.0), kind, 0.05)

    for mode in (None, "default"):
        utils.compile_mode = mode
        params = [torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in gradients]
        global_fn = {
            "global_l2": utils.global_l2norm_clip,
            "global_rms": utils.global_rmsnorm_clip,
        }.get(kind)
        optimizer = heavyball.SGD(
            params,
            lr=1,
            beta=0,
            warmup_steps=0,
            weight_decay=0,
            compile_step=True,
            gradient_clipping=(
                functools.partial(global_fn, clip_at=2.0) if global_fn else lambda values: clip_fn(values, 2.0)
            ),
            update_clipping=(
                functools.partial(global_fn, clip_at=0.05)
                if global_fn
                else lambda values: clip_fn(values, 0.05)
            ),
        )

        params[0].grad = torch.zeros_like(params[0])
        optimizer.step()
        for param, gradient in zip(params, gradients):
            param.grad = gradient.clone()
        optimizer.step()

        for param, reference in zip(params, expected):
            torch.testing.assert_close(param, -reference.to(param), atol=1e-6, rtol=1e-6)
