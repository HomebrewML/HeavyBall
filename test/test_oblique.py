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


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_oblique_tangent_projection_is_orthogonal_to_the_rows(shape):
    torch.manual_seed(0)
    param = torch.nn.functional.normalize(torch.randn(1, *shape, dtype=torch.float64), dim=-1)
    update = torch.randn(1, *shape, dtype=torch.float64)
    projected = oblique_tangent_projection(update.clone(), None, param, {}, _tempo(1))[0]
    radial = (param * projected).sum(dim=-1)
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

