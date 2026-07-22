"""HeavyBall must respect autograd's mutation contract: committing a parameter update must bump the
Parameter's ._version so that a retained-graph double-backward through a mutated parameter ERRORS (as
torch.optim does) instead of silently returning a gradient computed from the post-update value.

Reproduces the silent-wrong-differentiation P0: heavyball's slab write updates the parameter value via
the aliased slab storage without bumping Parameter._version, so autograd does not detect the mutation.
"""
from unittest.mock import patch

import torch
import torch.nn as nn


def _probe(make_opt):
    p = nn.Parameter(torch.tensor([2.0]))
    opt = make_opt([p])
    y = (p * p).sum()  # backward of y saves p; dy/dp = 2*p
    opt.zero_grad()
    y.backward(retain_graph=True)  # p.grad = 4 at p == 2
    v0 = p._version
    opt.step()  # mutate p (2.0 -> 1.6 for lr=0.1)
    v1 = p._version
    p.grad = None
    errored = False
    try:
        y.backward()  # second backward through the retained graph
    except RuntimeError:
        errored = True
    second_grad = None if errored else float(p.grad.item())
    return v0, v1, errored, second_grad


def test_torch_optim_reference_bumps_version_and_rejects_stale_graph():
    # Control: proves the probe measures the real autograd contract.
    v0, v1, errored, _ = _probe(lambda ps: torch.optim.SGD(ps, lr=0.1))
    assert v1 > v0
    assert errored


def test_heavyball_commit_bumps_version_like_torch():
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball

        v0, v1, errored, second_grad = _probe(lambda ps: heavyball.SGD(ps, lr=0.1))
    assert v1 > v0, (
        f"commit did not bump Parameter._version ({v0} -> {v1}); the update is invisible to autograd"
    )
    assert errored, (
        "double-backward through the mutated parameter did not error; it silently returned "
        f"{second_grad} (a gradient computed from the post-update value) instead of raising"
    )
