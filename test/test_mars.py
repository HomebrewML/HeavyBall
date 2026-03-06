import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from utils import REPRESENTATIVE_OPTS
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128

MARS_OPTS = [o for o in REPRESENTATIVE_OPTS if "SF" not in o and "ScheduleFree" not in o]


@pytest.mark.parametrize("opt", MARS_OPTS)
def test_mars(opt, size: int = 128, depth: int = 2, iterations: int = 32, outer_iterations: int = 1):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for mars in [True, False]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-5, mars=mars)

            for _ in range(iterations):
                loss = model(torch.randn((1024, size), device="cuda")).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert l0.item() <= l1.item() * 1.1
