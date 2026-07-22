"""Guard that repeated optimizer steps do not trigger per-step recompilation.

A silent recompile is a large performance cliff invisible to correctness tests;
existing coverage runs each normal and refresh step type only once.
"""

import pytest
import torch
import torch.nn as nn
import torch._dynamo

import heavyball


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compile-first GPU path"
)


@pytest.mark.parametrize("name", ("AdamW", "SOAP", "PSGDKron"))
def test_no_per_step_recompilation(name):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(0)
    model = nn.Linear(32, 32).cuda()
    opt = getattr(heavyball, name)(model.parameters(), lr=1e-2)

    def step():
        x = torch.randn(16, 32, device="cuda")
        y = torch.randn(16, 32, device="cuda")
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()

    for _ in range(15):
        step()
    warm = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    for _ in range(15):
        step()
    final = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    assert final == warm, f"{name} recompiled per-step: {warm}->{final}"
    assert warm <= 2
    assert not any(torch._dynamo.utils.counters.get("recompiles", {}).values())
