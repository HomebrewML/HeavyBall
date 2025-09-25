import math
import pathlib
import random
from typing import List, Optional

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from torch import nn
from utils import Plotter

from benchmark.utils import SkipConfig, loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def _formula(x, A):
    return x**2 + A * (1 - torch.cos(2 * math.pi * x))


def objective(*args, A=10):
    if len(args) == 1:
        return _formula(args[0], A).mean()

    return sum(_formula(x, A) for x in args) / len(args)


class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(x).float())

    def forward(self):
        return objective(self.param)


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
    size: int = 2,
    config: Optional[str] = None,
    ema_beta: float = 0.9,
):
    if config is not None and config != "trivial":
        raise SkipConfig("'config' must be 'trivial'.")
    if show_image:
        assert size == 2, "Image can only be displayed for 2D functions"
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-2.2,) * size

    # Clean up old plots
    for path in pathlib.Path(".").glob("rastrigin.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        model = Model(coords)
        model = Plotter(
            model,
            x_limits=(-8, 2),
            y_limits=(-8, 2),
        )
    else:
        model = Model(coords)
    model.double()

    model = trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-2 * (not show_image)),
        steps,
        opt,
        weight_decay,
        trials=trials,
        return_best=show_image,
        ema_beta=ema_beta,
    )

    if not show_image:
        return

    model.plot(title=f"{method[0]} {opt[0]}", save_path="rastrigin.png")


if __name__ == "__main__":
    app()
