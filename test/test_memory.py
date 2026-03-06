import dataclasses

import pytest
import torch
from torch import nn

import heavyball
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@dataclasses.dataclass
class Memory:
    peak_compiled: float
    peak_uncompiled: float
    optimizer: float


expected_memory = {"adamw": Memory(peak_compiled=4, peak_uncompiled=8, optimizer=2)}


@pytest.mark.parametrize("size,depth", [(8192, 2), (2048, 16)])
@pytest.mark.parametrize("compiled", [True, False])
def test_memory(size, depth: int, compiled: bool, opt: str = "AdamW", iterations: int = 2, outer_iterations: int = 3):
    set_torch()

    heavyball.utils.compile_mode = "default" if compiled else None

    for k, v in expected_memory.items():
        if k in opt.lower():
            break
    else:
        raise pytest.skip(f"Opt {opt} not supported")

    opt = getattr(heavyball, opt)
    heavyball.utils.zeroth_power_mode = "qr"

    for i in range(outer_iterations):
        model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
        model_allocated = get_memory()
        o = opt(model.parameters(), lr=1e-3)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_accumulated_memory_stats()

        for _ in range(iterations):
            with torch.no_grad():
                for p in model.parameters():
                    p.grad = torch.randn_like(p, requires_grad=False)
                o.step()

        opt_allocated = get_memory()

        del model, o
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        print(f"Peak: {peak / model_allocated:.2f}x | Opt: {opt_allocated / model_allocated:.2f}x")
        if i > 0:
            peak -= model_allocated
            opt_allocated -= model_allocated

            assert peak / model_allocated < (v.peak_compiled if compiled else v.peak_uncompiled)
            assert opt_allocated / model_allocated < v.optimizer
