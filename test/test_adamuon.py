"""Independent oracle for AdaMuon's sign-stabilized polar and raw adaptive second moment.

The paper applies NS5 to ``sign(momentum)``, updates an uncorrected elementwise second moment of
that polar direction, divides by ``sqrt(v) + eps``, and RMS-aligns the result to 0.2 before a plain
SGD commit.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import heavyball
from heavyball.transforms import Tempo, adamuon_rmsprop


def test_adamuon_first_update_depends_only_on_gradient_sign():
    gradient = torch.tensor(
        ((1.0, -2.0, 3.0), (-4.0, 5.0, -6.0), (7.0, -8.0, 9.0)),
        dtype=torch.float64,
    )
    rescaled = gradient * torch.tensor(
        ((9.0, 0.5, 3.0), (0.25, 7.0, 2.0), (4.0, 0.75, 6.0)),
        dtype=torch.float64,
    )
    parameters = [
        torch.nn.Parameter(torch.zeros_like(gradient)),
        torch.nn.Parameter(torch.zeros_like(gradient)),
    ]
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizers = [
            heavyball.AdaMuon([parameter], lr=0.1, weight_decay=0.0)
            for parameter in parameters
        ]

    for parameter, optimizer, value in zip(
        parameters, optimizers, (gradient, rescaled), strict=True
    ):
        parameter.grad.copy_(value)
        optimizer.step()

    torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)


def test_adamuon_second_moment_uses_raw_beta_and_additive_epsilon():
    update = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]], dtype=torch.float64)
    previous = torch.tensor([[[0.5, 0.75], [1.0, 1.25]]], dtype=torch.float64)
    beta1 = torch.tensor(0.8, dtype=torch.float64)
    epsilon = torch.tensor(1e-4, dtype=torch.float64)
    tempo = Tempo(
        step=torch.ones((), dtype=torch.long),
        age=torch.full((1,), 7, dtype=torch.long),
        live=torch.ones(1, dtype=torch.bool),
        hyper=SimpleNamespace(
            beta1=beta1,
            beta2=torch.tensor(0.1, dtype=torch.float64),
            eps=epsilon,
        ),
        refresh=False,
    )

    output, state, _ = adamuon_rmsprop(
        update, None, None, {"exp_avg_sq": previous}, tempo
    )

    variance = beta1 * previous.square() + (1 - beta1) * update.square()
    torch.testing.assert_close(state["exp_avg_sq"].square(), variance)
    torch.testing.assert_close(output, update / (variance.sqrt() + epsilon))
