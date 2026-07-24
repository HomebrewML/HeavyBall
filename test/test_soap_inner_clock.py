"""SOAP's inner Adam must bias-correct by the per-parameter update count (heavyball's per-leaf `age`),
matching the paper (arXiv:2409.11321) and the official soap.py (state["step"] increments only when the
parameter has a gradient), NOT the global per-device step.

Contract: a leaf's update on its first observation (age=1) must be invariant to the global step counter,
because the age-1 bias correction is 1 - beta**1 regardless of how many global steps have elapsed. The
same must hold for every SOAP variant (they share _soap_factory). Under always-observed training age==step
so this is invisible, but it is reachable through the public Engine.step(observed=[...]) API for
conditionally-active / late-joining parameters.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

with patch("heavyball.core.torch.compile", lambda f, **k: f):
    import heavyball  # noqa: F401
from heavyball.matrix import soap, soap_init


def _hyper():
    return SimpleNamespace(
        beta1=torch.tensor(0.9, dtype=torch.float64),
        beta2=torch.tensor(0.999, dtype=torch.float64),
        eps=torch.tensor(1e-12, dtype=torch.float64),
        shampoo_beta=torch.tensor(0.95, dtype=torch.float64),
        beta3=torch.tensor(0.9999, dtype=torch.float64),
        alpha=torch.tensor(2.0, dtype=torch.float64),
        beta3_warmup=torch.tensor(0.0, dtype=torch.float64),
        alpha_warmup=torch.tensor(0.0, dtype=torch.float64),
    )


def _tempo(age, step, count=1):
    from heavyball.transforms import Tempo

    return Tempo(
        torch.tensor(step, dtype=torch.long),
        torch.full((count,), age, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        _hyper(),
        False,
    )


def test_soap_first_update_is_invariant_to_global_step():
    # All SOAP variants share _soap_factory, so the adam-inner `soap` gates the shared clock.
    torch.manual_seed(0)
    ref = torch.randn(6, 4, dtype=torch.float64)  # unbatched leaf; the engine adds the slab axis
    g = torch.randn(1, 6, 4, dtype=torch.float64)
    unbatched = soap_init(ref, max_precond_dim=torch.tensor(9999))
    state0 = {k: v.unsqueeze(0) for k, v in unbatched.items()}
    at_step_1 = soap(g.clone(), None, None, deepcopy(state0), _tempo(age=1, step=1))[0]
    at_step_10 = soap(g.clone(), None, None, deepcopy(state0), _tempo(age=1, step=10))[0]
    err = (at_step_1 - at_step_10).abs().max().item()
    assert err < 1e-12, (
        f"a leaf's first (age=1) update changed by {err:.3e} when only the global step differed "
        "-> the inner Adam bias-corrects by the global step, not the per-leaf update count"
    )
