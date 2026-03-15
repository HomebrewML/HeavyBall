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
def test_closure(opt, size: tuple = (4, 4, 4, 4), depth: int = 2, iterations: int = 5, outer_iterations: int = 3):
    clean()
    set_torch()

    opt = getattr(heavyball, opt)

    for _ in range(outer_iterations):
        clean()
        model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
        o = get_optim(opt, model.parameters(), lr=1e-3)

        def _closure():
            loss = model(torch.randn((1, size[0]), device="cuda")).sum()
            loss.backward()
            return loss

        for i in range(iterations):
            o.step(_closure)
            o.zero_grad()
            print(o.state_size())
