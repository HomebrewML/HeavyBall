import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import REPRESENTATIVE_OPTS, get_optim, set_grad

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


STORAGE_OPTS = [
    o for o in REPRESENTATIVE_OPTS if "PSGD" not in o and "soap" not in o.lower() and "solp" not in o.lower()
]


@pytest.mark.parametrize("opt", STORAGE_OPTS)
def test_foreach(opt, size: int = 256, depth: int = 2, iterations: int = 32, outer_iterations: int = 2):
    set_torch()

    opt = getattr(heavyball, opt)

    all_params = []

    for dtype_name in ["float32", "bfloat16"]:
        all_params.append([])

        for i in range(outer_iterations):
            torch.manual_seed(0x2131290 + i)
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, storage_dtype=dtype_name)

            for j in range(iterations):
                torch.manual_seed(0x2131290 + outer_iterations + i * iterations + j)
                set_grad(model)
                o.step()
                o.zero_grad()

            all_params[-1].append([p.data.clone() for p in model.parameters()])

            del model, o
            clean()

    cos_threshold = 0.5 if opt.__name__ == "SGD" else 0.9
    for params_f32, params_bf16 in zip(*all_params):
        flat_f32 = torch.cat([p.float().flatten() for p in params_f32])
        flat_bf16 = torch.cat([p.float().flatten() for p in params_bf16])
        cos = torch.nn.functional.cosine_similarity(flat_f32, flat_bf16, dim=0)
        assert cos > cos_threshold, f"cosine similarity {cos:.4f} too low"
        norm_ratio = flat_bf16.norm() / flat_f32.norm()
        assert 0.9 < norm_ratio < 1.1, f"norm ratio {norm_ratio:.4f} out of range"
