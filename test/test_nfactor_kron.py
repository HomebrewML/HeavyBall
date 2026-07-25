from unittest.mock import patch

import torch
from torch import nn

from heavyball import ParamInfo, PSGDNfactor, PSGDPro as PSGD
from heavyball.matrix import nfactor_route


def test_nfactor_route_selects_convolutions_not_vectors_or_matrices():
    model = nn.Linear(4, 4)
    scalar = nn.Parameter(torch.randn(()))

    assert nfactor_route(ParamInfo(nn.Conv1d(2, 2, 3).weight))
    assert nfactor_route(ParamInfo(nn.Conv2d(2, 2, 3).weight))
    assert nfactor_route(ParamInfo(nn.Conv3d(2, 2, 3).weight))
    assert not nfactor_route(ParamInfo(model.bias))
    assert not nfactor_route(ParamInfo(scalar))
    assert not nfactor_route(ParamInfo(model.weight))


def test_nfactor_differs_from_2factor():
    torch.manual_seed(8)
    nfactor_model = nn.Conv2d(4, 8, 3, bias=False)
    two_factor_model = nn.Conv2d(4, 8, 3, bias=False)
    two_factor_model.load_state_dict(nfactor_model.state_dict())
    inputs = torch.randn(8, 4, 5, 5)
    initial_loss = nfactor_model(inputs).square().mean()

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        nfactor_optimizer = PSGDNfactor(nfactor_model.parameters(), lr=1e-3)
        two_factor_optimizer = PSGD(two_factor_model.parameters(), lr=1e-3)
        for step in range(10):
            torch.manual_seed(step)
            nfactor_optimizer.zero_grad()
            nfactor_model(inputs).square().mean().backward()
            nfactor_optimizer.step()

            torch.manual_seed(step)
            two_factor_optimizer.zero_grad()
            two_factor_model(inputs).square().mean().backward()
            two_factor_optimizer.step()

    nfactor_loss = nfactor_model(inputs).square().mean()
    two_factor_loss = two_factor_model(inputs).square().mean()
    assert all(torch.isfinite(param).all() for param in nfactor_model.parameters())
    assert all(torch.isfinite(param).all() for param in two_factor_model.parameters())
    assert not torch.allclose(nfactor_model.weight, two_factor_model.weight)
    assert nfactor_loss < initial_loss
    assert two_factor_loss < initial_loss
