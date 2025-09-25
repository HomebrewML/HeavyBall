import pathlib
import random
from typing import List, Optional

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import Plotter, SkipConfig, loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def objective(x, y):
    return (1 - x) ** 2 + 1 * (y - x**2) ** 2


class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(x).float())

    def forward(self):
        return objective(*self.param)


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
    config: Optional[str] = None,
    ema_beta: float = 0.9,
):
    if config is not None and config != "trivial":
        raise SkipConfig("'config' must be 'trivial'.")
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-7, -4)

    # Clean up old plots
    for path in pathlib.Path(".").glob("rosenbrock.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        model = Plotter(Model(coords), x_limits=(-8, 2), y_limits=(-8, 2), should_normalize=True)
    else:
        model = Model(coords)
    model.double()

    model = trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-9 * (not show_image)),
        steps,
        opt,
        weight_decay,
        trials=trials,
        return_best=show_image,
        ema_beta=ema_beta,
    )

    if not show_image:
        return

    model.plot(save_path="rosenbrock.png")


if __name__ == "__main__":
    app()
