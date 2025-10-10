import copy
import os

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from heavyball.utils import clean, set_torch

os.environ["TORCH_LOGS"] = "+recompiles"

config.cache_size_limit = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to run bf16 foreach parameter tests.",
)


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 1)])
def test_foreach(opt, size, depth: int, iterations: int = 512, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    torch.manual_seed(0x123131)
    model = nn.Sequential(*[nn.Linear(size, size, bias=False) for _ in range(depth)]).to(torch.double).cuda()

    for dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            mdl = copy.deepcopy(model).to(dtype)
            o = get_optim(opt, mdl.parameters(), lr=1e-4, update_clipping=None, warmup_steps=128)
            print(f"\n\n\n{dtype} {opt} {size} {depth}\n\n\n")
            for _ in range(iterations):
                loss = mdl(torch.randn((1024, size), device="cuda", dtype=dtype)).double().abs().mean()
                loss.backward()
                print(mdl[0].weight.double().norm().item())
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del mdl, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        # assert torch.allclose(l0.float(), l1.float(), rtol=0.1)
