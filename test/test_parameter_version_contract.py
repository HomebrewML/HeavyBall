"""HeavyBall must respect autograd's mutation contract: committing a parameter update must bump the
Parameter's ._version so that a retained-graph double-backward through a mutated parameter ERRORS (as
torch.optim does) instead of silently returning a gradient computed from the post-update value.

Reproduces the silent-wrong-differentiation P0: heavyball's slab write updates the parameter value via
the aliased slab storage without bumping Parameter._version, so autograd does not detect the mutation.
"""
from unittest.mock import patch

import torch
import torch.nn as nn


def test_heavyball_commit_bumps_version_like_torch():
    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball

    p = nn.Parameter(torch.tensor([2.0]))
    opt = heavyball.SGD([p], lr=0.1)
    y = (p * p).sum()
    opt.zero_grad()
    y.backward(retain_graph=True)
    v0 = p._version
    opt.step()
    v1 = p._version
    p.grad = None
    errored = False
    try:
        y.backward()
    except RuntimeError:
        errored = True
    second_grad = None if errored else float(p.grad.item())
    assert v1 > v0, (
        f"commit did not bump Parameter._version ({v0} -> {v1}); the update is invisible to autograd"
    )
    assert errored, (
        "double-backward through the mutated parameter did not error; it silently returned "
        f"{second_grad} (a gradient computed from the post-update value) instead of raising"
    )
