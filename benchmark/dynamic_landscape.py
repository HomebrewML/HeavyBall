"""
Tests optimizer's ability to track moving targets.

This benchmark simulates a dynamic loss landscape where the optimal parameters
continuously shift over time. This tests the optimizer's ability to:
1. Track moving targets
2. Adapt to non-stationary objectives
3. Handle continuous parameter updates
"""

import itertools
import math
from typing import List

import torch
import torch.nn as nn
import typer

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class ShiftingSphere(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim))
        self.phase = 0
        self.frequency = 0.1  # How fast target moves

    def forward(self):
        self.phase += self.frequency
        target = torch.linspace(
            0, 2 * math.pi, len(self.param), device=self.param.device, dtype=self.param.dtype
        )
        target = torch.sin(target + self.phase)
        return (self.param - target).square().mean()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    dim: int = 16384,
    steps: int = 500,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["adamw"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 3,
):
    """Run dynamic landscape benchmark with specified parameters."""
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [dim], opt, [weight_decay]):
        m, d, dim, o, wd = args

        model = ShiftingSphere(dim)

        def data():
            return None, None

        # Win condition: average squared error should be small (parameters close to target)
        trial(
            model,
            data,
            None,
            loss_win_condition(0.01 * win_condition_multiplier, 0),
            steps,
            [o],
            [d],
            1,
            1,
            wd,
            m,
            1,
            1,
            base_lr=0.1,
            trials=trials,
        )


if __name__ == "__main__":
    app()
