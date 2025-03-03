import pathlib
import random
from typing import List

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from torch import nn
from utils import Plotter

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def objective(x, y):
    """Classic saddle point objective - tests ability to escape saddle points."""
    return x**2 - y**2  # Saddle point at (0,0)


class Model(nn.Module):
    def __init__(self, x, offset):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(x).float())
        self.offset = offset

    def forward(self):
        return objective(*self.param) + self.offset


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
):
    dtype = [getattr(torch, d) for d in dtype]
    coords = (1e-6, 1e-6)  # Start near but not at saddle point

    # Clean up old plots
    for path in pathlib.Path(".").glob("saddle_point.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    offset = win_condition_multiplier * 10

    if show_image:
        model = Plotter(
            lambda *x: objective(*x).add(offset).log(),
            coords=coords,
            xlim=(-2, 2),
            ylim=(-2, 2),
            normalize=8,
            after_step=torch.exp,
        )
    else:
        model = Model(coords, offset)
    model.double()

    def data():
        return None, None

    trial(
        model,
        data,
        None,
        loss_win_condition(0.1),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        failure_threshold=3,
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
