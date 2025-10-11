import pytest
import torch
from lightbench.utils import get_optim
from torch import nn

import heavyball
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


expected_memory = {
    "adamw": {"after": 4, "peak": 5.1},
    "soap": {"after": 7, "peak": 14},
    "psgd": {"after": 4, "peak": 11.5},
}


@pytest.mark.parametrize("opt", ["NewtonHybrid2PSGDKron"])
@pytest.mark.parametrize("method", ["qr", "newtonschulz2", "svd", "eigh"])
@pytest.mark.parametrize("size,depth", [(8192, 2), (2048, 16)])
def test_memory(opt, method, size, depth: int, iterations: int = 5, outer_iterations: int = 3):
    if "soap" not in opt.lower() and method != "qr":
        raise pytest.skip("Only SOAP supports `method` argument")
    set_torch()

    for k, v in expected_memory.items():
        if k in opt.lower():
            break
    else:
        raise pytest.skip(f"Opt {opt} not supported")

    opt = getattr(heavyball, opt)
    heavyball.utils.zeroth_power_mode = method

    for i in range(outer_iterations):
        model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
        print(model)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_accumulated_memory_stats()

        model_allocated = get_memory()
        o = get_optim(opt, model.parameters(), lr=1e-3)
        for _ in range(iterations):
            data = torch.randn((1, size), device="cuda").requires_grad_(True)

            def _closure():
                nonlocal model
                loss = (model(data) - data).square().mean()
                loss.backward()
                return loss

            o.step(_closure)

            opt_allocated = get_memory()

        del model, o
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        print(f"Peak: {peak / model_allocated:.2f}x | Opt: {opt_allocated / model_allocated:.2f}x")
        if i > 0:
            assert peak / model_allocated < v["peak"]
            assert opt_allocated / model_allocated < v["after"]
