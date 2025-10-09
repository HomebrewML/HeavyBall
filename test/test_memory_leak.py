import pytest
import torch
import tqdm
from lightbench.utils import get_optim
from torch import nn
from torch.nn import functional as F

import heavyball
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    clean()
    torch.cuda.synchronize()
    out = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.reset_accumulated_memory_stats()
    return out


class LayerNorm2dParam(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2dParam, self).__init__()
        self.param = nn.Parameter(torch.ones(2, num_features))

    def forward(self, x):
        weight, bias = self.param.unbind(0)
        return F.layer_norm(x, [x.size(-1)], weight, bias)


@pytest.mark.parametrize(
    "opt", ["NewtonHybrid2PSGDKron"]
)  # leak with NewtonHybrid2PSGDKron, but not ForeachCachedNewtonPSGD or ForeachCachedPSGDKron
@pytest.mark.parametrize("size,depth", [(64, 2)])  # happens across all sizes
@pytest.mark.parametrize("mars", [False])  # happens with True and False
@pytest.mark.parametrize("cached", [False])  # happens with True and False
@pytest.mark.parametrize("delayed", [False])  # happens with True and False
@pytest.mark.parametrize("merge_dims", [False])  # happens with True and False
@pytest.mark.parametrize("split", [True])  # happens with True and False
@pytest.mark.parametrize("finite_differences", [False])  # only happens with False - does not happen with True
def test_memory(
    opt,
    size,
    depth: int,
    mars: bool,
    cached: bool,
    delayed: bool,
    merge_dims: bool,
    split: bool,
    finite_differences: bool,
    iterations: int = 10000,
    warmup: int = 100,
    check_every: int = 10,
    max_growth: float = 1.10,
):
    set_torch()

    opt = getattr(heavyball, opt)
    model = nn.Sequential(*[LayerNorm2dParam(size) for _ in range(depth)]).cuda()
    print(model)

    o = get_optim(
        opt,
        model.parameters(),
        lr=1e-3,
        mars=mars,
        merge_dims=merge_dims,
        split=split,
        cached=cached,
        delayed=delayed,
        preconditioner_update_probability=1.0,
    )
    if finite_differences:
        if not o.hessian_approx:
            pytest.skip("Finite Differences is an HVP calculation - can't do it on non-hvp optimizers")
        o.finite_differences = True

    peak = 0
    for i in tqdm.trange(iterations):
        data = torch.randn((1, size), device="cuda").requires_grad_(True)

        def _closure():
            nonlocal model
            loss = (model(data) - data).square().mean()
            loss.backward()
            return loss

        o.step(_closure)

        if i % check_every == 0:
            if i <= warmup:
                peak = max(peak, get_memory())
            if i > warmup:
                new = get_memory()
                print(i, peak, new)
                assert peak * max_growth >= new  # fudge factor
