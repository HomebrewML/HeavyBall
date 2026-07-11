import torch

import heavyball


def test_ema_handles_noncontiguous_merged_parameters():
    p = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.float64).mT)
    assert not p.is_contiguous()
    opt = heavyball.SOAP([p], use_ema=True, ema_decay=0.25, merge_dims=True)
    values = torch.arange(1, 4, dtype=torch.float64).view(-1, 1, 1) * torch.ones_like(p)
    for value in values:
        p.data.copy_(value)
        opt.ema_update()

    beta = 1 - opt.ema_decay
    weights = beta ** torch.arange(len(values) - 1, -1, -1, dtype=torch.float64)
    expected = (values * weights.view(-1, 1, 1)).sum(0) / weights.sum()
    ema = opt.state[p]["_root"]["param_ema"]
    assert ema.is_contiguous()
    torch.testing.assert_close(ema, expected)

    live = p.detach().clone()
    opt.copy_emas_to_params()
    torch.testing.assert_close(p, expected)
    opt.copy_params_to_emas()
    assert torch.equal(p, live)


def test_normalized_ema_recurrence_and_swap():
    p = torch.nn.Parameter(torch.zeros(()))
    opt = heavyball.SGD([p], use_ema=True, param_ecc="bf16+16")
    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    for value in values:
        p.data.copy_(value)
        opt.ema_update()

    beta = 1 - opt.ema_decay
    weights = beta ** torch.arange(len(values) - 1, -1, -1)
    expected = (values * weights).sum() / weights.sum()
    ema = opt.state[p]["_root"]["param_ema"]
    torch.testing.assert_close(ema, expected)

    group = opt.param_groups[0]
    view = opt._set_views(p, group)[0]
    state = opt.state_(view)
    config = group["_param_ecc_config"]

    def decoded():
        with config.attached([view], [state["param::ecc"].view_as(view)]):
            return heavyball.utils.promote(view).clone()

    live = decoded()
    opt.copy_emas_to_params()
    torch.testing.assert_close(decoded(), expected, rtol=0, atol=1e-4)
    opt.copy_params_to_emas()
    torch.testing.assert_close(decoded(), live, rtol=0, atol=1e-4)
    torch.testing.assert_close(opt.state[p]["_root"]["param_ema"], expected)
