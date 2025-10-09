import copy
import os

os.environ["TORCH_LOGS"] = "+recompiles"

import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config
from torch.utils._pytree import tree_map

import heavyball
from heavyball.utils import set_torch

config.cache_size_limit = 128


def _train_one(dataset, model, opt):
    torch.manual_seed(0x2131290)
    for d in dataset:
        opt.zero_grad()

        def _closure():
            loss = (model(d) - d.square()).square().mean()
            loss.backward()
            return loss

        opt.step(_closure)
    return model


def _allclose(x, y):
    if isinstance(x, torch.Tensor):
        assert torch.allclose(x, y)
    elif isinstance(x, (list, tuple)):
        assert all(_allclose(x, y) for x, y in zip(x, y))
    elif not isinstance(x, bytes):  # bytes -> it's a pickle
        assert x == y


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(32, 2)])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("merge_dims", [False, True])
def test_save_restore(
    opt, size, depth: int, split: bool, merge_dims: bool, iterations: int = 32, outer_iterations: int = 8
):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x2131290)
    data = torch.randn((iterations, size), device="cuda", dtype=torch.double)

    model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda().double()
    o: torch.optim.Optimizer = get_optim(
        opt, model.parameters(), lr=1e-3, merge_dims=merge_dims, split=split, storage_dtype="float64", q_dtype="float64"
    )

    for x in range(outer_iterations):
        new_m = copy.deepcopy(model)
        new_o = get_optim(opt, new_m.parameters(), lr=1e-3)
        state_dict = copy.deepcopy(o.state_dict())
        m = _train_one(data, model, o)

        new_o.load_state_dict(state_dict)
        new_m = _train_one(data, new_m, new_o)

        tree_map(_allclose, new_o.state_dict(), o.state_dict())

        for normal_param, state_param in zip(m.parameters(), new_m.parameters()):
            assert torch.allclose(normal_param, state_param)
