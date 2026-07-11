import copy
import os

import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import get_optim, set_grad

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"

os.environ["TORCH_LOGS"] = "+recompiles"

config.cache_size_limit = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to run bf16 foreach parameter tests.",
)


@pytest.mark.parametrize("opt", ["SGD", "AdamW", "PSGDKron"])
def test_foreach(opt, size: int = 64, iterations: int = 8):
    set_torch()
    opt = getattr(heavyball, opt)

    torch.manual_seed(0x123131)
    model = nn.Linear(size, size, bias=False).double().cuda()
    seeds = torch.randint(0, 2**30, (iterations,), device="cpu")

    deltas = []

    for dtype in [torch.float32, torch.bfloat16]:
        mdl = copy.deepcopy(model).to(dtype)
        initial = [p.detach().float().clone() for p in mdl.parameters()]
        o = get_optim(opt, mdl.parameters(), lr=1e-3, warmup_steps=0, update_clipping=None)
        for seed in seeds:
            torch.manual_seed(seed)
            set_grad(mdl, dtype=torch.float32)
            o.step()
            o.zero_grad()
        deltas.append(torch.cat([(p.float() - p0).flatten() for p, p0 in zip(mdl.parameters(), initial)]))
        del mdl, o
        clean()

    cosine = torch.nn.functional.cosine_similarity(*deltas, dim=0)
    ratio = deltas[1].norm() / deltas[0].norm()
    assert cosine > 0.9
    assert 0.7 < ratio < 1.3
