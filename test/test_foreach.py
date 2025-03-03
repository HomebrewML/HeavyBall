import pytest
import torch
from torch import nn

import heavyball
import heavyball.utils
from benchmark.utils import get_optim
from heavyball.utils import PSGDBase, clean, set_torch


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 128)])
def test_foreach(opt, size, depth: int, iterations: int = 4096, outer_iterations: int = 2):
    set_torch()

    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for foreach in [True, False]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            clean()
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            clean()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_accumulated_memory_stats()

            clean()
            o = get_optim(opt, model.parameters(), lr=1e-3, foreach=foreach)
            clean()

            for _ in range(iterations):
                loss = model(torch.randn((1, size), device="cuda")).sum()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

            if i > 0:
                peaks[-1].append(peak)

    for p0, p1 in zip(*peaks):
        assert p0 > p1
    for l0, l1 in zip(
        *losses
    ):  # increase error tolerance for PSGD, as we have different RNGs -> expected differences
        assert torch.allclose(l0, l1, rtol=0.01 if isinstance(opt, PSGDBase) else 1e-5)
