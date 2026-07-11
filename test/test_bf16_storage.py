import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import get_optim, set_grad

import heavyball
from heavyball.utils import clean, set_torch

_SAVED_COMPILE_MODE = heavyball.utils.compile_mode
heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128


@pytest.fixture(autouse=True)
def _isolate_compile_mode():
    heavyball.utils.compile_mode = "default"
    yield
    heavyball.utils.compile_mode = _SAVED_COMPILE_MODE


STORAGE_OPTS = ["AdamW", "SOAP", "PSGDKron"]


@pytest.mark.parametrize("opt", STORAGE_OPTS)
def test_foreach(opt, size: int = 32, iterations: int = 6):
    set_torch()

    opt = getattr(heavyball, opt)

    deltas = []

    for dtype_name in ["float32", "bfloat16"]:
        torch.manual_seed(0x2131290)
        model = nn.Linear(size, size).cuda()
        initial = [p.detach().clone() for p in model.parameters()]
        o = get_optim(opt, model.parameters(), lr=1e-3, storage_dtype=dtype_name, warmup_steps=0)
        for step in range(iterations):
            torch.manual_seed(0x2131290 + step)
            set_grad(model)
            o.step()
            o.zero_grad()
        deltas.append(torch.cat([(p - p0).flatten() for p, p0 in zip(model.parameters(), initial, strict=True)]))
        del model, o
        clean()

    cosine = torch.nn.functional.cosine_similarity(*deltas, dim=0)
    ratio = deltas[1].norm() / deltas[0].norm()
    assert cosine > 0.95
    assert 0.8 < ratio < 1.2
