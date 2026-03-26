import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config
from utils import REPRESENTATIVE_OPTS

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128

PSGD_OPTS = [o for o in REPRESENTATIVE_OPTS if "PSGD" in o]


@pytest.mark.parametrize("opt", PSGD_OPTS)
def test_foreach(opt, size: int = 256, depth: int = 2, iterations: int = 32, outer_iterations: int = 2):
    set_torch()

    opt = getattr(heavyball, opt)

    # Pre-generate all gradients so bf16 stochastic rounding inside the optimizer
    # doesn't desync the CUDA RNG between q_dtype runs.
    torch.manual_seed(0x2131290)
    all_grads = []
    for i in range(outer_iterations):
        model_tmp = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
        grads_for_run = []
        for _ in range(iterations):
            grads_for_run.append([torch.randn_like(p) for p in model_tmp.parameters()])
        all_grads.append(grads_for_run)
        del model_tmp

    all_params = []

    for q_dtype in ["float32", "bfloat16"]:
        all_params.append([])

        for i in range(outer_iterations):
            torch.manual_seed(0x2131290)
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, q_dtype=q_dtype)

            for j, step_grads in enumerate(all_grads[i]):
                for p, g in zip(model.parameters(), step_grads):
                    p.grad = g.clone()
                # Re-seed CUDA RNG before each step so dampen_grad and other
                # internal RNG calls produce identical values despite bf16
                # stochastic rounding consuming extra RNG state.
                torch.cuda.manual_seed(0x9999 + i * iterations + j)
                o.step()
                o.zero_grad()

            all_params[-1].append([p.data.clone() for p in model.parameters()])

            del model, o
            clean()

    for params_f32, params_bf16 in zip(*all_params):
        for p0, p1 in zip(params_f32, params_bf16):
            assert torch.allclose(p0, p1, rtol=0.1, atol=5e-3)
