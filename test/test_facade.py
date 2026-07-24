"""Proofs for the torch.optim facade over HeavyBall Engines."""

import pytest
import torch
from torch import nn

import heavyball


def test_truegrad_facade_and_recipe_both_importable():
    from heavyball import truegrad_adam

    assert hasattr(heavyball, "TrueGradAdam")
    assert heavyball.truegrad_adam is truegrad_adam


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.linear = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(inputs).flatten(1))


def test_facade_trains():
    torch.manual_seed(21)
    direct_model = _TinyNet()
    facade_model = _TinyNet()
    facade_model.load_state_dict(direct_model.state_dict())
    inputs = torch.randn(8, 1, 4, 4)
    targets = torch.zeros(8, 1)

    direct = heavyball.build(direct_model.parameters(), heavyball.adamw, lr=3e-3)
    facade = heavyball.AdamW(facade_model.parameters(), lr=3e-3)
    initial_loss = torch.nn.functional.mse_loss(facade_model(inputs), targets)

    for _ in range(5):
        direct.zero_grad()
        direct_loss = torch.nn.functional.mse_loss(direct_model(inputs), targets)
        direct_loss.backward()
        direct.step()

        facade.zero_grad()
        facade_loss = torch.nn.functional.mse_loss(facade_model(inputs), targets)
        facade_loss.backward()
        facade.step()

        assert torch.equal(direct_loss, facade_loss)
        for direct_param, facade_param in zip(direct_model.parameters(), facade_model.parameters(), strict=True):
            assert torch.equal(direct_param, facade_param)

    final_loss = torch.nn.functional.mse_loss(facade_model(inputs), targets)
    assert final_loss < initial_loss
    assert all(torch.isfinite(param).all() for param in facade_model.parameters())


def test_facade_lr_scheduler():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    param = nn.Parameter(torch.tensor(10.0))
    optimizer = heavyball.SGD([param], lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for _ in range(3):
        before = param.detach().clone()
        param.grad.fill_(1)
        optimizer.step()
        assert torch.equal(param, before - 0.25)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.125
    for _ in range(3):
        before = param.detach().clone()
        param.grad.fill_(1)
        optimizer.step()
        assert torch.equal(param, before - 0.125)
    assert torch.equal(optimizer._engine.hyper.lr, torch.tensor(0.125))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_set_hyper_updates_routed_namespaces_without_recompile():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    matrix = nn.Parameter(torch.eye(2))
    vector = nn.Parameter(torch.ones(2))
    optimizer = heavyball.build([matrix, vector], heavyball.muon, lr=0.25)

    for _ in range(3):
        matrix.grad.fill_(1)
        vector.grad.fill_(1)
        optimizer.step()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    compiled_step = optimizer.compiled_step

    optimizer.set_hyper("lr", 0.125)
    assert len(optimizer._hyper_locations) == 2
    assert all(torch.equal(namespace.lr, torch.tensor(0.125)) for namespace in optimizer._hyper_locations.values())
    for _ in range(3):
        matrix.grad.fill_(1)
        vector.grad.fill_(1)
        optimizer.step()

    assert optimizer.compiled_step is compiled_step
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs
    with pytest.raises(ValueError, match="unknown hyperparameter"):
        optimizer.set_hyper("not_a_hyper", 1)


def test_facade_state_dict_roundtrip():
    torch.manual_seed(22)
    param_a = nn.Parameter(torch.randn(3))
    optimizer_a = heavyball.AdamW([param_a], lr=0.03, weight_decay=0.01)
    gradients = [torch.randn_like(param_a) for _ in range(6)]

    for gradient in gradients[:3]:
        param_a.grad.copy_(gradient)
        optimizer_a.step()

    checkpoint = optimizer_a.state_dict()
    assert set(checkpoint) == {"state", "param_groups", "engines"}
    assert len(checkpoint["engines"]) == 1
    param_b = nn.Parameter(param_a.detach().clone())
    optimizer_b = heavyball.AdamW([param_b], lr=0.9, weight_decay=0.0)
    optimizer_b.load_state_dict(checkpoint)
    assert optimizer_b.param_groups[0]["lr"] == optimizer_a.param_groups[0]["lr"]
    assert optimizer_b.param_groups[0]["weight_decay"] == optimizer_a.param_groups[0]["weight_decay"]
    assert torch.equal(optimizer_b._engine.hyper.lr, optimizer_a._engine.hyper.lr)

    for gradient in gradients[3:]:
        param_a.grad.copy_(gradient)
        param_b.grad.copy_(gradient)
        optimizer_a.step()
        optimizer_b.step()
        assert torch.equal(param_a, param_b)


def test_facade_zero_grad():
    param = nn.Parameter(torch.ones(2))
    optimizer = heavyball.AdamW([param])
    grad = param.grad
    grad.fill_(1)

    optimizer.zero_grad()
    assert param.grad is grad
    assert torch.count_nonzero(param.grad) == 0
    with pytest.raises(ValueError, match="persistent gradient buffers"):
        optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize(
    "facade",
    (
        heavyball.SOAP,
        heavyball.Shampoo,
        heavyball.KLSOAP,
        heavyball.KLShampoo,
        heavyball.PSGD,
        heavyball.PSGDKron,
        heavyball.PSGDPro,
        heavyball.QSGD,
        heavyball.LATHER,
        heavyball.Whitening,
    ),
)
def test_matrix_facades_step_with_weight_and_bias_at_defaults(facade):
    model = nn.Linear(4, 3)
    before = [param.detach().clone() for param in model.parameters()]
    optimizer = facade(model.parameters())

    for _ in range(2):  # KL facades seed on step 1 (no parameter write); the update lands from step 2
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()

    assert all(not torch.equal(param, initial) for param, initial in zip(model.parameters(), before, strict=True))


def test_soap_facade_matches_raw_recipe_at_defaults_for_matrix_param():
    initial = torch.tensor(
        [[0.5, -1.0, 0.25, 2.0], [-0.75, 1.5, -2.0, 0.125], [1.25, -0.5, 0.75, -1.5]],
        dtype=torch.float64,
    )
    gradient = torch.tensor(
        [[0.2, -0.1, 0.4, -0.3], [-0.5, 0.7, -0.2, 0.6], [0.9, -0.8, 0.3, -0.4]],
        dtype=torch.float64,
    )
    routed_param = nn.Parameter(initial.clone())
    raw_param = nn.Parameter(initial.clone())
    routed = heavyball.SOAP([routed_param])
    raw = heavyball.build([raw_param], heavyball.soap)

    for name, value in heavyball.soap.defaults.items():
        assert routed.param_groups[0][name] == value

    routed_param.grad.copy_(gradient)
    raw_param.grad.copy_(gradient)
    routed.step()
    raw.step()

    assert torch.equal(initial - routed_param, initial - raw_param)


@pytest.mark.parametrize("facade", (heavyball.Muon, heavyball.Scion, heavyball.Whitening))
def test_existing_mixed_shape_facades_still_build(facade):
    model = nn.Linear(4, 3)

    optimizer = facade(model.parameters())

    assert isinstance(optimizer, torch.optim.Optimizer)


def test_psgd_routes_mixed_matrix_and_vector_params_and_steps():
    matrix = nn.Parameter(torch.eye(2))
    vector = nn.Parameter(torch.ones(3))
    optimizer = heavyball.PSGD([matrix, vector], max_size_triangular=2)
    groups = {tuple(group.params[0].shape): group for group in optimizer._engine.groups}

    assert groups[(2, 2)].recipe is heavyball.kron
    assert groups[(3,)].recipe is heavyball.psgd_nfactor
    assert groups[(3,)].states[0]["Q_0"].ndim == 2  # slab axis plus a diagonal vector factor

    matrix.grad.copy_(torch.tensor(((1.0, -0.5), (0.25, 2.0))))
    vector.grad.copy_(torch.tensor((0.5, -1.0, 2.0)))
    optimizer.step()

    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()


@pytest.mark.parametrize(
    ("alias", "replacement"),
    (
        (heavyball.ScheduleFree, heavyball.SFAdamW),
        (heavyball.WhitenAdamW, heavyball.Whitening),
    ),
)
def test_deprecated_facade_aliases_warn_on_construction(alias, replacement):
    parameter = nn.Parameter(torch.ones(2))

    with pytest.warns(DeprecationWarning, match=rf"use {replacement.__name__} instead"):
        optimizer = alias([parameter])

    assert isinstance(optimizer, replacement)


def test_route_defaults_prefer_primary_branch():
    primary = heavyball.Recipe(
        chain=(),
        commit=heavyball.sgd_commit,
        defaults={"lr": 0.3, "weight_decay": 0.2, "primary_only": 1.0},
    )
    fallback = heavyball.Recipe(
        chain=(),
        commit=heavyball.sgd_commit,
        defaults={"lr": 0.1, "weight_decay": 0.0, "fallback_only": 2.0},
    )
    route = heavyball.Route(lambda info: info.ndim == 2, primary, fallback)
    matrix = nn.Parameter(torch.ones(2, 2))
    vector = nn.Parameter(torch.ones(2))

    optimizer = heavyball.HeavyBallOptimizer([matrix, vector], route)

    assert optimizer.defaults == {
        "lr": 0.3,
        "weight_decay": 0.2,
        "primary_only": 1.0,
        "fallback_only": 2.0,
    }
    for group in optimizer._engine.groups:
        assert torch.equal(group.hyper.lr, torch.tensor(0.3))
        assert torch.equal(group.hyper.weight_decay, torch.tensor(0.2))
