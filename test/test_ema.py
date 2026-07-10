import pytest
import torch
from torch import nn
from torch._dynamo import config
from utils import REPRESENTATIVE_OPTS, get_optim, set_grad

import heavyball
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

            delta = sum((p.data - p0).float().square().sum().item() for p, p0 in zip(model.parameters(), init_params))
            if do_ema:
                live_params = [p.detach().clone() for p in model.parameters()]
                o.copy_emas_to_params()
                assert any(not torch.equal(p, live) for p, live in zip(model.parameters(), live_params))
                o.copy_params_to_emas()
                assert all(torch.equal(p, live) for p, live in zip(model.parameters(), live_params))

            weights[-1].append(delta)

            del model, o
            clean()

    for i, (w_ema, w_no_ema) in enumerate(zip(*weights)):
        print(i, w_ema, w_no_ema)
        assert w_ema > 0, "EMA weights should have changed"
        assert w_no_ema > 0, "Non-EMA weights should have changed"
        assert torch.isclose(torch.tensor(w_ema), torch.tensor(w_no_ema), rtol=1e-6)


def test_normalized_ema_recurrence_and_swap():
    p = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = heavyball.SGD([p], use_ema=True)
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    for value in values:
        p.data.copy_(value)
        opt.ema_update()

    beta = 1 - opt.ema_decay
    weights = beta ** torch.arange(len(values) - 1, -1, -1, dtype=torch.float64)
    expected = (values * weights).sum() / weights.sum()
    ema = opt.state[p]["_root"]["param_ema"]
    torch.testing.assert_close(ema, expected)

    live = p.detach().clone()
    opt.copy_emas_to_params()
    torch.testing.assert_close(p, expected)
    opt.copy_params_to_emas()
    assert torch.equal(p, live)
    torch.testing.assert_close(opt.state[p]["_root"]["param_ema"], expected)
