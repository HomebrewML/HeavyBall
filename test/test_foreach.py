import os

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


def _read_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


DEFAULT_SIZE = _read_int("HB_FOREACH_TEST_SIZE", 128, minimum=1)
DEFAULT_DEPTH = _read_int("HB_FOREACH_TEST_DEPTH", 16, minimum=1)
DEFAULT_ITERATIONS = _read_int("HB_FOREACH_TEST_ITERATIONS", 64, minimum=1)
DEFAULT_OUTER = _read_int("HB_FOREACH_TEST_OUTER", 1, minimum=1)
DEFAULT_WARMUP = _read_int("HB_FOREACH_TEST_WARMUP", 1, minimum=0)


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(DEFAULT_SIZE, DEFAULT_DEPTH)])
def test_foreach(
    opt,
    size,
    depth: int,
    iterations: int = DEFAULT_ITERATIONS,
    outer_iterations: int = DEFAULT_OUTER,
    warmup_runs: int = DEFAULT_WARMUP,
):
    set_torch()

    opt = getattr(heavyball, opt)

    total_runs = warmup_runs + outer_iterations
    assert total_runs >= 1

    peaks = []
    losses = []

    for foreach in [True, False]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(total_runs):
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

            if i < warmup_runs:
                continue

            peaks[-1].append(peak)

    if warmup_runs:
        cutoff = warmup_runs * iterations
        losses = [loss_list[cutoff:] for loss_list in losses]

    for p0, p1 in zip(*peaks):
        assert p0 > p1
    for l0, l1 in zip(*losses):  # increase error tolerance for PSGD, as we have different RNGs -> expected differences
        assert torch.allclose(l0, l1, rtol=0.01 if "PSGD" in opt.__class__.__name__ else 1e-5)
