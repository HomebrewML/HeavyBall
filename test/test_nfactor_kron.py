from unittest.mock import patch

import torch
from torch import nn

from heavyball import ParamInfo, PSGDNfactor, PSGDPro as PSGD
from heavyball.matrix import nfactor_route
from heavyball.psgd_pro import _precondition_nfactor


def test_conv256_steps_finite():
    torch.manual_seed(0)
    model = nn.Conv2d(256, 256, 3, bias=False)
    inputs = torch.randn(1, 256, 3, 3)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
        for _ in range(5):
            optimizer.zero_grad()
            model(inputs).square().mean().backward()
            optimizer.step()

    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_state_has_4_factors():
    model = nn.Conv2d(4, 4, 3, bias=False)
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        model(torch.randn(2, 4, 5, 5)).square().mean().backward()
        optimizer.step()

    state = optimizer._engine.groups[0].states[0]
    assert all(f"Q_{index}" in state for index in range(4))
    assert all(f"running_lower_bound_{index}" in state for index in range(4))


def test_linear_not_captured():
    model = nn.Linear(64, 64, bias=False)

    assert nfactor_route(ParamInfo(model.weight)) is False


def test_identity_precondition():
    update = torch.randn(2, 4, 5, 3, 3)
    factors = [torch.eye(size).expand(update.shape[0], -1, -1) for size in update.shape[1:]]

    torch.testing.assert_close(_precondition_nfactor(update, factors), update)


def test_nfactor_loss_decreases():
    torch.manual_seed(1)
    model = nn.Conv2d(3, 4, 3, bias=False)
    inputs = torch.randn(8, 3, 5, 5)
    target = torch.randn(8, 4, 3, 3)
    initial_loss = torch.nn.functional.mse_loss(model(inputs), target)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
        for _ in range(20):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(inputs), target).backward()
            optimizer.step()

    final_loss = torch.nn.functional.mse_loss(model(inputs), target)
    assert final_loss < initial_loss


def test_compiles_fullgraph():
    torch.manual_seed(2)
    model = nn.Conv2d(16, 16, 3, bias=False)
    optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
    compiled_model = torch.compile(model, fullgraph=True)
    inputs = torch.randn(2, 16, 5, 5)

    for _ in range(3):
        optimizer.zero_grad()
        compiled_model(inputs).square().mean().backward()
        optimizer.step()

    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_diagonal_factor_path():
    torch.manual_seed(3)
    model = nn.Conv2d(8, 8, 3, bias=False)
    inputs = torch.randn(1, 8, 5, 5)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3, max_size_triangular=4)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()

    state = optimizer._engine.groups[0].states[0]
    channel_factors = [state["Q_0"][0], state["Q_1"][0]]
    assert all(factor.ndim == 1 for factor in channel_factors)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_conv1d_routed():
    torch.manual_seed(4)
    model = nn.Conv1d(2, 2, 3, bias=False)
    inputs = torch.randn(2, 2, 8)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        assert nfactor_route(ParamInfo(model.weight)) is True
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()

    assert torch.isfinite(loss)


def test_conv3d_routed():
    torch.manual_seed(5)
    model = nn.Conv3d(2, 2, 3, bias=False)
    inputs = torch.randn(1, 2, 4, 4, 4)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        assert nfactor_route(ParamInfo(model.weight)) is True
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_mixed_diagonal_triangular():
    torch.manual_seed(6)
    model = nn.Conv2d(4, 16, 3, bias=False)
    inputs = torch.randn(1, 4, 5, 5)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = PSGDNfactor(model.parameters(), lr=1e-3, max_size_triangular=8)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()

    state = optimizer._engine.groups[0].states[0]
    factors = [state[f"Q_{index}"] for index in range(4)]
    assert any(factor[0].ndim == 1 for factor in factors)
    assert any(factor.ndim == 3 for factor in factors)
    assert torch.isfinite(loss)


def test_scalar_and_1d_not_routed():
    torch.manual_seed(7)
    model = nn.Linear(4, 4)
    scalar = nn.Parameter(torch.randn(()))

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        assert nfactor_route(ParamInfo(model.bias)) is False
        assert nfactor_route(ParamInfo(scalar)) is False
        assert nfactor_route(ParamInfo(model.weight)) is False


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
