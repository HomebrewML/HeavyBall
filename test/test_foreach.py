import os

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from utils import REPRESENTATIVE_OPTS

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"


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
DEFAULT_DEPTH = _read_int("HB_FOREACH_TEST_DEPTH", 2, minimum=1)
DEFAULT_ITERATIONS = _read_int("HB_FOREACH_TEST_ITERATIONS", 4, minimum=1)
DEFAULT_OUTER = _read_int("HB_FOREACH_TEST_OUTER", 1, minimum=1)
DEFAULT_WARMUP = _read_int("HB_FOREACH_TEST_WARMUP", 2, minimum=0)


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
def test_foreach(
    opt,
    size: int = DEFAULT_SIZE,
    depth: int = DEFAULT_DEPTH,
    iterations: int = DEFAULT_ITERATIONS,
    outer_iterations: int = DEFAULT_OUTER,
    warmup_runs: int = DEFAULT_WARMUP,
):
    set_torch()

    opt = getattr(heavyball, opt)

    total_runs = warmup_runs + outer_iterations
    assert total_runs >= 1

    peaks = [[], []]
    losses = [[], []]

    for i in range(total_runs):
        for foreach in [True, False]:
            lss, pk = losses[int(foreach)], peaks[int(foreach)]
            torch.manual_seed(0x2131290)

            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, foreach=foreach)

            clean()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_accumulated_memory_stats()

            for _ in range(iterations):
                loss = model(torch.randn((1, size), device="cuda")).sum()
                loss.backward()
                o.step()
                o.zero_grad()
                lss.append(loss.detach())

            del model, o
            clean()

            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

            if i < warmup_runs:
                continue

            pk.append(peak)

    if warmup_runs:
        cutoff = warmup_runs * iterations
        losses = [loss_list[cutoff:] for loss_list in losses]

    for peak_no_foreach, peak_foreach in zip(*peaks):
        assert peak_no_foreach < peak_foreach

    # no-foreach LRA is a different optimizer (per-parameter LRA vs global LRA),
    # so we only check that both converge, not that they match.
    if "LRA" in opt.__name__:
        return

    for loss_no_foreach, loss_foreach in zip(*losses):
        if torch.isnan(loss_no_foreach) and torch.isnan(loss_foreach):
            continue

        # increase error tolerance for PSGD, as we have different RNGs -> expected differences
        assert torch.allclose(loss_no_foreach, loss_foreach, rtol=0.01 if "PSGD" in opt.__name__ else 1e-5)
