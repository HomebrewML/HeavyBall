from typing import List

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn

import heavyball
from heavyball.utils import clean, set_torch


class Param(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size))

    def forward(self, inp):
        return self.weight.mean() * inp


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize(
    "size",
    [
        (4, 4, 4, 4),
    ],
)
def test_closure(opt, size: List[int], depth: int = 2, iterations: int = 5, outer_iterations: int = 3):
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
