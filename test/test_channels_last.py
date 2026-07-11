import os

os.environ["TORCH_LOGS"] = "+recompiles"

import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import get_optim

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128


@pytest.mark.parametrize("opt", ["AdamW", "PSGDKron"])
def test_foreach(opt, size: int = 8, iterations: int = 3):
    set_torch()
    opt = getattr(heavyball, opt)

    deltas = []

    for is_channels_last in [False, True]:
        torch.manual_seed(0x2131290)
        model = nn.Conv2d(size, size, 3).cuda()
        if is_channels_last:
            model.to(memory_format=torch.channels_last)
        initial = [p.detach().clone() for p in model.parameters()]
        o = get_optim(opt, model.parameters(), lr=1e-3, weight_decay=1e-4, warmup_steps=0)
        for step in range(iterations):
            torch.manual_seed(0x2131290 + step)
            loss = model(torch.randn((4, size, 8, 8), device="cuda")).square().mean()
            loss.backward()
            o.step()
            o.zero_grad()
        deltas.append(torch.cat([(p - p0).flatten() for p, p0 in zip(model.parameters(), initial, strict=True)]))
        del model, o
        clean()

    torch.testing.assert_close(deltas[1], deltas[0], rtol=2e-3, atol=2e-6)
