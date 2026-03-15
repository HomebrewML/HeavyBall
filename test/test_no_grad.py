import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from utils import REPRESENTATIVE_OPTS

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = None


class Param(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size))

    def forward(self, inp):
        return self.weight.mean() * inp


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
def test_no_grad_step(opt, size: tuple = (4, 4, 4, 4), depth: int = 2, iterations: int = 5, outer_iterations: int = 3):
    clean()
    set_torch()

    opt = getattr(heavyball, opt)

    for _ in range(outer_iterations):
        clean()
        model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
        o = get_optim(opt, model.parameters(), lr=1e-3)

        for i in range(iterations):
            o.step()
            o.zero_grad()
            assert o.state_size() == 0

        del model, o
