from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size, powers, target_mult):
        super().__init__()
        self.target = nn.Buffer(
            torch.arange(powers * size).view(size, powers).transpose(0, 1).float()
            * target_mult
            / powers
            / size
        )
        self.param = nn.Parameter(torch.rand(powers, size) * 2)
        self.register_buffer("scale", torch.arange(powers).float().add(1))

    def forward(self):
        x = self.param - self.target
        x = x ** self.scale.view(-1, 1)
        return x.square().mean()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    size: int = 64,
    powers: int = 8,
    steps: int = 10,
    target_mult: float = 1.0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(size, powers, target_mult).cuda().double()

    def data():
        return None, None

    trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-6),
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
