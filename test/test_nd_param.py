import pytest
import torch
from torch import nn
from torch._dynamo import config

import heavyball
from utils import REPRESENTATIVE_OPTS, set_grad
from heavyball.utils import set_torch

heavyball.utils.compile_mode = None
config.cache_size_limit = 2**20


class Model(nn.Module):
    def __init__(self, size, dims):
        super().__init__()
        self.params = nn.Parameter(torch.randn((size,) * dims))

    def forward(self, x):
        return self.params.square().mean() * x.square().mean()


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
@pytest.mark.parametrize("size,dims", [(1, 1), (1, 5), (4, 1), (4, 3), (4, 5)])
def test_ndim_tensor(opt, size, dims: int, iterations: int = 4):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x2131290)

    model = nn.Sequential(Model(size, dims)).cuda()
    o = opt(model.parameters(), lr=1e-5)

    for _ in range(iterations):
        set_grad(model)
        o.step()
        o.zero_grad()

    assert all(p.isfinite().all() for p in model.parameters())


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
@pytest.mark.parametrize("params", [2, 5])
def test_multi_param(opt, params: int, size: int = 4, iterations: int = 4):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x2131290)

    model = nn.Sequential(*[Model(size, 2) for _ in range(params)]).cuda()
    o = opt(model.parameters(), lr=1e-5)

    for _ in range(iterations):
        set_grad(model)
        o.step()
        o.zero_grad()

    assert all(p.isfinite().all() for p in model.parameters())
