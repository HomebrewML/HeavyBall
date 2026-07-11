import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import get_optim

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128

PSGD_OPTS = ["PSGDKron", "PSGDPRO", "QSGD", "PSGDLRA"]


@pytest.mark.parametrize("opt", PSGD_OPTS)
def test_foreach(opt, size: int = 32, iterations: int = 6):
    set_torch()

    opt = getattr(heavyball, opt)

    torch.manual_seed(0x2131290)
    model_tmp = nn.Linear(size, size).cuda()
    all_grads = [[torch.randn_like(p) for p in model_tmp.parameters()] for _ in range(iterations)]
    del model_tmp
    deltas = []

    for q_dtype in ["float32", "bfloat16"]:
        torch.manual_seed(0x2131290)
        model = nn.Linear(size, size).cuda()
        initial = [p.detach().clone() for p in model.parameters()]
        o = get_optim(opt, model.parameters(), lr=1e-3, q_dtype=q_dtype, warmup_steps=0)
        for step, gradients in enumerate(all_grads):
            for p, grad in zip(model.parameters(), gradients, strict=True):
                p.grad = grad.clone()
            torch.cuda.manual_seed(0x9999 + step)
            o.step()
            o.zero_grad()
        deltas.append(torch.cat([(p - p0).flatten() for p, p0 in zip(model.parameters(), initial, strict=True)]))
        del model, o
        clean()

    cosine = torch.nn.functional.cosine_similarity(*deltas, dim=0)
    ratio = deltas[1].norm() / deltas[0].norm()
    assert cosine > 0.95
    assert 0.8 < ratio < 1.2
