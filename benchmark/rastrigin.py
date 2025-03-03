import math
import pathlib
import random
from typing import List

import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import torch.backends.opt_einsum
import typer
from hyperopt import early_stop
from torch import nn
from utils import Plotter

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

early_stop.no_progress_loss()
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
):
    if show_image:
        assert size == 2, "Image can only be displayed for 2D functions"
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-2.2,) * size

    # Clean up old plots
    for path in pathlib.Path(".").glob("rastrigin.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    stride = max(1, steps // 20)
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        model = Plotter(
            lambda *x: objective(*x).log(),
            coords=coords,
            xlim=(-8, 2),
            ylim=(-8, 2),
            normalize=8,
            after_step=torch.exp,
        )
    else:
        model = Model(coords)
    model.double()

    def data():
        return None, None

    model = trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-2 * (not show_image)),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        base_lr=1e-4,
        trials=trials,
        return_best=show_image,
    )

    if not show_image:
        return

    fig, ax = model.plot_image(cmap="gray", levels=20, return_fig=True, xlim=(-8, 2), ylim=(-8, 2))
    ax.set_frame_on(False)

    c = colors[0]
    ax.plot(
        *list(zip(*model.coords_history)),
        linewidth=1,
        color=c,
        zorder=2,
        label=f"{method[0]} {opt[0]}",
    )
    ax.scatter(
        *list(zip(*model.coords_history[::stride])), s=8, zorder=1, alpha=0.75, marker="x", color=c
    )
    ax.scatter(*model.coords_history[-1], s=64, zorder=3, marker="x", color=c)

    fig.legend()
    fig.savefig("rastrigin.png", dpi=1000)
    plt.close(fig)


if __name__ == "__main__":
    app()
