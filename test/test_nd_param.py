import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from heavyball.utils import set_torch

config.cache_size_limit = 2**20


class Model(nn.Module):
    def __init__(self, size, dims):
        super().__init__()
        self.params = nn.Parameter(torch.randn((size,) * dims))

    def forward(self, x):
        return self.params.square().mean() * x.square().mean()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size", [1, 4, 16])
@pytest.mark.parametrize("dims", list(range(1, 6)))
@pytest.mark.parametrize("params", list(range(1, 6)))
def test_ndim_tensor(opt, size, dims: int, params: int, iterations: int = 4):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x2131290)

    model = nn.Sequential(*[Model(size, dims) for _ in range(params)]).cuda()
    o = get_optim(opt, model.parameters(), lr=1e-5)

    for _ in range(iterations):
        loss = model(torch.randn((1, size), device="cuda")).square().mean()
        loss.backward()
        o.step()
        o.zero_grad()
