import pytest
import torch
from torch import nn
from utils import REPRESENTATIVE_OPTS

import heavyball
from heavyball.utils import clean, set_torch

_SAVED_COMPILE_MODE = heavyball.utils.compile_mode
heavyball.utils.compile_mode = "default"


@pytest.fixture(autouse=True)
def _isolate_compile_mode():
    heavyball.utils.compile_mode = "default"
    yield
    heavyball.utils.compile_mode = _SAVED_COMPILE_MODE


PSGD_OPTS = [o for o in REPRESENTATIVE_OPTS if "PSGD" in o]


@pytest.mark.parametrize("opt", PSGD_OPTS)
def test_foreach(opt, size: int = 128, depth: int = 1, iterations: int = 512, outer_iterations: int = 2):
    set_torch()

    opt = getattr(heavyball, opt)

    losses = []

    for stochastic in [False, True]:
        print("stochastic", stochastic)
        torch.manual_seed(0x2131290)
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size, bias=False) for _ in range(depth)]).cuda()
            o = opt(
                model.parameters(),
                lr=1e-3,
                stochastic_schedule=stochastic,
                preconditioner_update_probability=lambda step: 0.1,
            )

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
    assert not torch.isclose(torch.tensor(deterministic), torch.tensor(stochastic), rtol=0.01)
