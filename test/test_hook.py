import os

os.environ["TORCH_LOGS"] = "+recompiles"

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from heavyball.utils import clean, hook_optimizer_into_model, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 1)])
def test_foreach(opt, size, depth: int, iterations: int = 128, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for use_hook in [False, True]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()

            if use_hook:
                hook_optimizer_into_model(model, opt, lr=1e-3, weight_decay=1e-4, warmup_steps=16)
            else:
                o = get_optim(opt, model.parameters(), lr=1e-3, weight_decay=1e-4, warmup_steps=16)
            for _ in range(iterations):
                loss = model(torch.randn((1024, size), device="cuda")).square().mean()
                loss.backward()
                if not use_hook:
                    o.step()
                    o.zero_grad()
                losses[-1].append(loss.detach())

            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert torch.allclose(l0.float(), l1.float(), rtol=0.1)
