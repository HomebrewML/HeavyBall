"""Independent oracle for SpEL: Stiefel-projected Muon, the first retraction-commit optimizer.

SpEL's defining behavior, verified independently of the source: after each step the weight is
re-projected onto the Stiefel manifold (orthonormal columns/rows) by a RetractionCommit that wraps
the base commit with the orthogonalize projection. Property oracle.
"""

from unittest.mock import patch

import torch

from heavyball import Engine, spel


def _orthonormal_spread(weight):
    tall = weight if weight.shape[-2] >= weight.shape[-1] else weight.mT
    return torch.linalg.svdvals(tall)


def test_spel_reprojects_the_weight_onto_the_stiefel_manifold():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(6, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], spel, lr=0.1, beta1=0.9, weight_decay=0.0)
    param.grad.copy_(torch.randn_like(param))
    optimizer.step()
    singular = _orthonormal_spread(param.detach())
    assert singular.min() > 0.9 and singular.max() < 1.1


def test_spel_compiles_fullgraph_and_stays_on_the_manifold():
    torch.manual_seed(4)
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Engine([param], spel, lr=0.05, beta1=0.9, weight_decay=0.0)
    try:
        param.grad.copy_(torch.randn_like(param))
        optimizer.step()
        singular = _orthonormal_spread(param.detach())
        assert singular.min() > 0.9 and singular.max() < 1.1
    finally:
        torch._dynamo.reset()


def test_spel_facade_keeps_the_weight_orthonormal():
    from heavyball import SpEL

    assert SpEL.recipe is spel
    torch.manual_seed(21)
    model = torch.nn.Linear(4, 6)
    inputs = torch.randn(8, 4)
    targets = torch.zeros(8, 6)
    optimizer = SpEL(model.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    singular = _orthonormal_spread(model.weight.detach())
    assert singular.min() > 0.9 and singular.max() < 1.1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_spel_is_muon_reorthogonalized_onto_the_stiefel_manifold():
    from heavyball.transforms import momentum, orthogonalize

    matrix = spel.then
    assert matrix.chain == (momentum, orthogonalize)  # a missing inner orthogonalize is caught here
    assert matrix.commit.config == {"base": "sgd_commit", "projection": "stiefel", "projection_fn": "stiefel_projection"}


def test_spel_descends_toward_an_orthonormal_target():
    # The manifold checks pass for any update (the retraction enforces orthonormality unconditionally),
    # so this pins that SpEL actually DESCENDS -- a sign-flipped or inert update fails here.
    torch.manual_seed(3)
    target = torch.linalg.qr(torch.randn(6, 4))[0]
    param = torch.nn.Parameter(torch.linalg.qr(torch.randn(6, 4))[0].clone())
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], spel, lr=0.1, beta1=0.9, weight_decay=0.0)

    def squared_error() -> float:
        return float(((param.detach() - target) ** 2).sum())

    initial = squared_error()
    best = initial
    for _ in range(200):
        param.grad.copy_(2.0 * (param.detach() - target))
        optimizer.step()
        best = min(best, squared_error())
    assert best < 0.1 * initial
