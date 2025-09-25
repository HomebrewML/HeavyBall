import pathlib
import random
from typing import List, Optional

import matplotlib.colors
import torch
import typer
from torch import nn

from benchmark.utils import Plotter, SkipConfig, loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def objective(xy, a=0.5, b=0.30, along=1e-3, across=1.0):
    """
    Archimedean-spiral test function for optimizers.

    xy      : (..., 2) tensor or (x, y) tuple
    a, b    : spiral parameters for r = a + b*theta
    along   : curvature along the valley (make small for a long valley)
    across  : curvature across the valley (make large for narrow valley)

    Returns:
        f(x, y) with a unique minimum at (r=a, theta=0), and a single curved
        valley that repeats outward every 2π.
    """
    # Support either a (...,2) tensor or a tuple
    if isinstance(xy, (tuple, list)):
        x, y = xy
    else:
        x, y = xy[..., 0], xy[..., 1]

    # Polar coordinates
    r = torch.sqrt(x * x + y * y + 1e-12)  # eps avoids a gradient hiccup at the origin
    theta = torch.atan2(y, x)

    # Unwrap theta to the nearest spiral turn so the valley repeats infinitely
    k = torch.round((r - a - b * theta) / (2 * torch.pi * b))
    theta_u = theta + 2 * torch.pi * k  # unwrapped angle (piecewise-constant k)

    # Orthogonal and tangential errors relative to the spiral
    err_across = r - (a + b * theta_u)  # distance to the nearest spiral branch
    err_along = theta_u  # position along the valley (min at 0)

    # Rosenbrock-like anisotropy: steep across, flat along the valley
    return across * err_across.pow(2) + along * err_along.pow(2)


class Model(nn.Module):
    """
    A 2D spiral valley test function for optimization in PyTorch.
    The function has low values along logarithmic spiral paths leading to the minimum at (0,0).
    It is similar to the Rosenbrock function but with a spiral-shaped valley.
    The parameter 'scale' controls the number of turns in the spiral (larger scale means more turns, longer valley).
    The parameter 'lambda_' controls the narrowness of the valley (larger means narrower).

    f(x, y) = r^2 + lambda_ * sin^2(scale * phi - log(r))
    where r = sqrt(x^2 + y^2 + epsilon), phi = atan2(y, x)

    The minimum is approximately 0 at (0, 0), but due to the nature of the function, optimizers must follow the spiral valley to reach it effectively.
    """

    def __init__(self, x, scale=0.1, lambda_=100.0, epsilon=1e-8):
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
    config: Optional[str] = None,
):
    if config is not None and config != "trivial":
        raise SkipConfig("'config' must be 'trivial'.")
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-7, -4)

    # Clean up old plots
    for path in pathlib.Path(".").glob("beale.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        lim = max(map(abs, coords)) * 1.5  # assume it's perfectly on a diagonal
        model = Plotter(Model(coords), x_limits=(-lim, lim), y_limits=(-lim, lim), should_normalize=True)
    else:
        model = Model(coords)
    model.double()

    model = trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-8 * (not show_image)),
        steps,
        opt,
        weight_decay,
        trials=trials,
        return_best=show_image,
    )

    if not show_image:
        return

    title = ", ".join(opt)
    model.plot(title=title if len(opt) > 1 else None, save_path="beale.png")


if __name__ == "__main__":
    app()
