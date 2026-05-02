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
def test_closure(opt, size: tuple = (4, 4, 4, 4), depth: int = 2, iterations: int = 5):
    clean()
    set_torch()
    opt = getattr(heavyball, opt)

    model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    o = get_optim(opt, model.parameters(), lr=1e-3)

    losses = []

    def _closure():
        loss = model(torch.randn((1, size[0]), device="cuda")).sum()
        loss.backward()
        return loss

    for _ in range(iterations):
        loss = o.step(_closure)
        o.zero_grad()
        losses.append(loss.detach().float().item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    moved = [(p.detach() - q).abs().max().item() for p, q in zip(model.parameters(), initial)]
    assert max(moved) > 0, "no parameter changed across {iterations} closure steps"
    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after closure step"
