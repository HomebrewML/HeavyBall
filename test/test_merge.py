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


@pytest.mark.parametrize("opt", ["ForeachPSGDKron"])
@pytest.mark.parametrize("method", ["qr", "newtonschulz2", "svd", "eigh"])
@pytest.mark.parametrize("size", [(16, 16, 16, 16), (4, 4, 4, 4), (512, 1, 128), (32128, 768)])
@pytest.mark.parametrize("merge,split", [(False, False), (True, False), (True, True)])
def test_merge(
    opt,
    method,
    size: List[int],
    merge,
    split,
    depth: int = 2,
    iterations: int = 5,
    outer_iterations: int = 3,
):
    if "soap" not in opt.lower() and method != "qr":
        raise pytest.skip("Only SOAP supports `method` argument")
    clean()
    set_torch()

    opt = getattr(heavyball, opt)
    heavyball.utils.zeroth_power_mode = method

    for _ in range(outer_iterations):
        clean()
        model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
        # We don't know if merging will use more or less memory, but we do know that it shouldn't crash. This test is to check if it crashes
        o = get_optim(
            opt,
            model.parameters(),
            lr=1e-3,
            merge_dims=merge,
            split=split,
            max_precond_dim=256,
            max_size_triangular=256,
        )

        for i in range(iterations):
            model(torch.randn((1, size[0]), device="cuda")).sum().backward()
            o.step()
            o.zero_grad()
            print(o.state_size())

        del model, o
