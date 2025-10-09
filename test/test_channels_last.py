import os

os.environ["TORCH_LOGS"] = "+recompiles"

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.zeroth_power_mode = "newtonschulz"
heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 1)])
def test_foreach(opt, size, depth: int, iterations: int = 1024, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for is_channels_last in [False, True]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Conv2d(size, size, 3) for _ in range(depth)]).cuda()
            if is_channels_last:
                model.to(memory_format=torch.channels_last)

            o = get_optim(opt, model.parameters(), lr=1e-3, weight_decay=1e-4, warmup_steps=16)

            for _ in range(iterations):
                loss = model(torch.randn((1024, size, 4, 4), device="cuda")).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert torch.allclose(l0.float(), l1.float(), rtol=0.1)
