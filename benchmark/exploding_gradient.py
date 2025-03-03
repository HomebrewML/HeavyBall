"""
Tests optimizer's ability to handle exploding gradients.

This benchmark creates a scenario where gradients can grow exponentially,
testing the optimizer's:
1. Gradient clipping/scaling mechanisms
2. Numerical stability
3. Ability to make progress despite extreme gradient values
"""

import itertools
from typing import List

import torch
import torch.nn as nn
import typer

from benchmark.utils import param_norm_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class ExplodingGradient(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim))
        self.scale = 5.0  # Controls how quickly gradients grow

    def forward(self):
        # Creates exponentially growing gradients
        # Gradient will be scale * exp(|param|) * sign(param)
        return torch.exp(self.scale * torch.abs(self.param)).mean()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    dim: int = 512,
    steps: int = 500,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["adamw"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 3,
):
    """Run exploding gradient benchmark with specified parameters."""
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [dim], opt, [weight_decay]):
        m, d, dim, o, wd = args

        model = ExplodingGradient(dim)

        def data():
            return None, None

        # Win condition: loss should be close to 1.0 (exp(0) = 1)
        # Using 1.1 as threshold since perfect convergence is hard
        trial(
            model,
            data,
            None,
            param_norm_win_condition(0.01 * win_condition_multiplier, 0),
            steps,
            [o],
            [d],
            1,
            1,
            wd,
            m,
            1,
            1,
            base_lr=0.001,  # Lower learning rate due to large gradients
            trials=trials,
        )


if __name__ == "__main__":
    app()
