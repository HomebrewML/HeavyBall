import pytest
import torch
from lightbench.utils import get_optim
from torch import nn
from torch._dynamo import config

import heavyball
from utils import REPRESENTATIVE_OPTS, set_grad
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"
config.cache_size_limit = 128


@pytest.mark.parametrize("opt", REPRESENTATIVE_OPTS)
def test_foreach(opt, size: int = 256, depth: int = 2, iterations: int = 32, outer_iterations: int = 2):
    set_torch()
    opt = getattr(heavyball, opt)

    weights = []
    for do_ema in [True, False]:
        torch.manual_seed(0x2131290)
        weights.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3)
            init_params = [p.data.clone() for p in model.parameters()]

            for _ in range(iterations):
                set_grad(model)
                o.step()
                o.zero_grad()
                if do_ema:
                    o.ema_update()
                    o.copy_emas_to_params()
                    o.copy_params_to_emas()

            if do_ema:
                o.copy_emas_to_params()

            delta = sum((p.data - p0).float().square().sum().item() for p, p0 in zip(model.parameters(), init_params))
            weights[-1].append(delta)

            del model, o
            clean()

    for i, (w_ema, w_no_ema) in enumerate(zip(*weights)):
        print(i, w_ema, w_no_ema)
        assert w_ema > 0, "EMA weights should have changed"
        assert w_no_ema > 0, "Non-EMA weights should have changed"
        assert torch.isclose(torch.tensor(w_ema), torch.tensor(w_no_ema), rtol=0.1), \
            f"EMA and non-EMA weight changes differ too much: {w_ema} vs {w_no_ema}"
