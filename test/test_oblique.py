"""Independent oracle for Oblique: row-normalized Adam on the oblique manifold (2nd retraction commit).

Oblique's defining behavior, verified independently of the source: the update is projected onto the
tangent space of the unit-norm-rows manifold (orthogonal to each row), and after each step the
weight's rows are re-normalized to unit L2 norm. Property oracle; reuses the RetractionCommit from a
different angle than SpEL (a tangent projection before the base, a normalize retraction after).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, oblique
from heavyball.transforms import Tempo, oblique_tangent_projection


def _tempo(count):
    return Tempo(
        torch.ones((), dtype=torch.long),
        torch.ones(count, dtype=torch.long),
        torch.ones(count, dtype=torch.bool),
        SimpleNamespace(),
        False,
    )


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])  # tall AND wide: the constrained axis (rows) is fixed
def test_oblique_tangent_projection_is_orthogonal_to_the_rows(shape):
    torch.manual_seed(0)
    param = torch.nn.functional.normalize(torch.randn(1, *shape, dtype=torch.float64), dim=-1)  # unit-norm rows
    update = torch.randn(1, *shape, dtype=torch.float64)
    projected = oblique_tangent_projection(update.clone(), None, param, {}, _tempo(1))[0]
    radial = (param * projected).sum(dim=-1)  # per-row inner product with the row -> must be ~0
    assert radial.abs().max().item() < 1e-10


def test_oblique_keeps_the_rows_unit_norm():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], oblique, lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.0)
    param.grad.copy_(torch.randn_like(param))
    optimizer.step()
    row_norms = param.detach().norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


def test_oblique_compiles_fullgraph_and_keeps_rows_unit_norm():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], oblique, lr=0.05, beta1=0.9, beta2=0.99, weight_decay=0.0)
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        row_norms = param.detach().norm(dim=1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-2)
    finally:
        torch._dynamo.reset()


def test_oblique_facade_keeps_rows_unit_norm():
    from heavyball import Oblique

    assert Oblique.recipe is oblique
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)  # weight 6x4; each of the 6 rows normalized over its 4 columns
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = Oblique(model.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    row_norms = model.weight.detach().norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_oblique_descends_toward_a_unit_row_target():
    # The unit-norm-rows checks hold for any update; this pins that Oblique actually DESCENDS.
    torch.manual_seed(3)
    target = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
    param = torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(6, 4), dim=1).clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], oblique, lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(200):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial
