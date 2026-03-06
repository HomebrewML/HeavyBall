import copy
import os

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from utils import REPRESENTATIVE_OPTS, set_grad
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"

os.environ["TORCH_LOGS"] = "+recompiles"

config.cache_size_limit = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to run bf16 foreach parameter tests.",
)


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
def test_foreach(opt, size: int = 256, depth: int = 1, iterations: int = 64, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x123131)
    model = nn.Sequential(*[nn.Linear(size, size, bias=False) for _ in range(depth)]).to(torch.double).cuda()

    all_params = []

    for dtype in [torch.float32, torch.bfloat16]:
        all_params.append([])

        for i in range(outer_iterations):
            torch.manual_seed(0x2131290 + i)
            seeds = torch.randint(0, 2**30, (iterations,), device='cpu')
            mdl = copy.deepcopy(model).to(dtype)
            o = get_optim(opt, mdl.parameters(), lr=1e-4, update_clipping=None)

            for s in seeds.numpy():
                torch.manual_seed(s)
                set_grad(mdl, dtype=torch.float32)
                o.step()
                o.zero_grad()

            all_params[-1].append([p.data.clone() for p in mdl.parameters()])

            del mdl, o
            clean()

    for params_f32, params_bf16 in zip(*all_params):
        for p0, p1 in zip(params_f32, params_bf16):
            assert torch.allclose(p0.float(), p1.float(), rtol=0.1, atol=1e-3)
