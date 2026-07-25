"""GPU compiled-vs-eager oracle across every optimizer facade.

The ordinary facades use a 1e-4 bound (versus their measured ~1e-7 level).
``_DIVERGENT`` is limited to iterative matmul/eigendecomposition recipes whose
compiled and eager reduction orders produce distinct, valid training
trajectories; those use 1e-1 only as a gross-breakage bound.
"""

import inspect
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import heavyball

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compile-first GPU path requires CUDA")

OPTIMIZERS = sorted(
    name
    for name in dir(heavyball)
    if isinstance(getattr(heavyball, name), type)
    and issubclass(getattr(heavyball, name), heavyball.HeavyBallOptimizer)
    and getattr(heavyball, name) is not heavyball.HeavyBallOptimizer
)
assert len(OPTIMIZERS) >= 43, f"optimizer feature matrix unexpectedly shrank: {len(OPTIMIZERS)} < 43"

# These orthogonalization/polar/spectral optimizers amplify compiled-vs-eager matmul reduction-order differences
# through Newton-Schulz/eigendecomposition iterations, so the trajectories separate like different seeds while
# both converge. Measured rel at 12 steps on RTX 5060 Ti: SpEL 3.4e-2, PolarGrad 9.2e-3, AdaMuon 4.8e-3,
# Aurora 3.7e-3, NorMuon 3.4e-3, Muon 2.8e-3, KLSOAP 2.6e-3, MuonLaProp 1.1e-3, Scion 1.6e-4. Thus a 1e-2
# threshold is already known to reject a valid fp32 SpEL run; 1e-1 leaves platform headroom without being used
# for ordinary facades.
_DIVERGENT = {"AdaMuon", "Aurora", "HeavyKLSOAP", "KLSOAP", "Muon", "MuonLaProp", "NorMuon", "PolarGrad", "Scion", "SpEL"}


def run(name, eager):
    def run_steps():
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 12), nn.LayerNorm(12), nn.Linear(12, 8)).cuda()
        facade = getattr(heavyball, name)
        lr = inspect.signature(facade).parameters["lr"].default
        opt = facade(model.parameters(), lr=lr)
        torch.manual_seed(100)
        for _ in range(12):
            x = torch.randn(16, 16, device="cuda")
            y = torch.randn(16, 8, device="cuda")
            ((model(x) - y) ** 2).mean().backward()
            opt.step()
            opt.zero_grad()
        return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

    if eager:
        with patch("heavyball.core.torch.compile", lambda f, **k: f):
            return run_steps()
    return run_steps()


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_compiled_matches_eager_gpu(name):
    if name.startswith("TrueGrad"):
        pytest.skip("TrueGrad facades require an external observation producer")
    compiled = run(name, False)
    eager = run(name, True)
    rel = (compiled - eager).abs().max().item() / max(eager.abs().max().item(), 1e-12)
    tol = 1e-1 if name in _DIVERGENT else 1e-4
    assert rel < tol, f"{name}: compiled-vs-eager rel={rel:.2e} exceeds {tol}"
