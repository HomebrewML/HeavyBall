from typing import List

import torch
import torch.backends.opt_einsum
import typer
from torch import nn
from torch.nn import functional as F

from benchmark.utils import param_norm_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn((size, size)))
        self.target = nn.Buffer(torch.triu(torch.ones_like(self.param)))

    def forward(self, inp):
        return inp @ self.param


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    size: int = 1024,
    depth: int = 4,
    batch: int = 16,
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(size).cuda()

    def data():
        inp = torch.randn((batch, size), device="cuda", dtype=dtype[0])
        return inp, inp.cumsum(1)

    trial(
        model,
        data,
        F.mse_loss,
        param_norm_win_condition(1e-7 * win_condition_multiplier, model.target),
        steps,
        opt[0],
        dtype[0],
        size,
        batch,
        weight_decay,
        method[0],
        1,
        depth,
        failure_threshold=depth * 2,
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
