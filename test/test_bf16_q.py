import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from heavyball.utils import clean, set_torch

config.cache_size_limit = 128


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 2)])
def test_foreach(opt, size, depth: int, iterations: int = 128, outer_iterations: int = 3):
    set_torch()

    opt = getattr(heavyball, opt)
    if "PSGD" not in opt.__name__:
        raise pytest.skip("Only PSGD is supported")

    peaks = []
    losses = []

    for q_dtype in ["float32", "bfloat16"]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, q_dtype=q_dtype)

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
        assert torch.allclose(l0, l1, rtol=0.1)
