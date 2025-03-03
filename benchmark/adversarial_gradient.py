from typing import List

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import param_norm_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("step", torch.zeros(1))

    def forward(self):
        """Test optimizer's robustness to adversarial gradient patterns."""
        self.step += 1
        # Create an oscillating adversarial component
        direction = torch.sin(self.step * torch.pi / 10)
        # Main objective plus adversarial component
        return self.param.square().mean() + direction * self.param.mean()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model().cuda().double()

    def data():
        return None, None

    # More lenient condition due to adversarial component
    trial(
        model,
        data,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-3, 0),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        failure_threshold=7,
        base_lr=1e-3,
        trials=trials,
    )  # More attempts for adversarial case


if __name__ == "__main__":
    app()
