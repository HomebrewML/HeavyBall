import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from utils import REPRESENTATIVE_OPTS, set_grad
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128

PSGD_OPTS = [o for o in REPRESENTATIVE_OPTS if "PSGD" in o]


@pytest.mark.parametrize("opt", PSGD_OPTS)
def test_foreach(opt, size: int = 256, depth: int = 2, iterations: int = 32, outer_iterations: int = 2):
    set_torch()

    opt = getattr(heavyball, opt)

    all_params = []

    for q_dtype in ["float32", "bfloat16"]:
        torch.manual_seed(0x2131290)
        all_params.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, q_dtype=q_dtype)

            for _ in range(iterations):
                set_grad(model)
                o.step()
                o.zero_grad()

            all_params[-1].append([p.data.clone() for p in model.parameters()])

            del model, o
            clean()

    for params_f32, params_bf16 in zip(*all_params):
        for p0, p1 in zip(params_f32, params_bf16):
            assert torch.allclose(p0, p1, rtol=0.1, atol=1e-3)
