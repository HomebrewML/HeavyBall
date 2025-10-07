import pytest
import torch
from lightbench.utils import get_optim
from torch import nn

import heavyball
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 1)])
def test_foreach(opt, size, depth: int, iterations: int = 8192, outer_iterations: int = 3):
    set_torch()

    opt = getattr(heavyball, opt)
    if "PSGD" not in opt.__name__:
        raise pytest.skip("Only PSGD is supported")

    peaks = []
    losses = []

    for stochastic in [False, True]:
        print("stochastic", stochastic)
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size, bias=False) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, stochastic_schedule=stochastic)

            for _ in range(iterations):
                loss = model(torch.randn((128, size), device="cuda")).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    stochastic = sum([l.item() for l in losses[1]])
    deterministic = sum([l.item() for l in losses[0]])
    print(f"{deterministic=}, {stochastic=}")
    assert deterministic < stochastic
