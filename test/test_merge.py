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


@pytest.mark.parametrize("opt", ["PSGDKron"])
@pytest.mark.parametrize("size", [(16, 16, 16, 16), (4, 4, 4, 4), (512, 1, 128), (32128, 768)])
@pytest.mark.parametrize("merge,split", [(False, False), (True, False), (True, True)])
def test_merge(opt, size: List[int], merge, split, depth: int = 2, iterations: int = 5):
    clean()
    set_torch()
    opt = getattr(heavyball, opt)
    heavyball.utils.zeroth_power_mode = "qr"

    model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    o = get_optim(
        opt,
        model.parameters(),
        lr=1e-3,
        merge_dims=merge,
        split=split,
        max_precond_dim=256,
        max_size_triangular=256,
    )

    for _ in range(iterations):
        for p in model.parameters():
            p.grad = torch.randn_like(p, requires_grad=False)
        o.step()
        o.zero_grad()

    moved = [(p.detach() - q).abs().max().item() for p, q in zip(model.parameters(), initial)]
    assert max(moved) > 0, f"no parameter changed (merge={merge}, split={split})"
    for p in model.parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter (merge={merge}, split={split})"
