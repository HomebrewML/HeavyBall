import math
import random
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
        self.register_buffer("batch_sizes", torch.tensor([1, 4, 16, 64]))
        self.rng = random.Random(0x1238192)

    def forward(self):
        """Test optimizer's ability to handle different batch sizes and noise scales."""
        batch_size = self.rng.choice(self.batch_sizes)
        generator = torch.Generator(device=self.param.device).manual_seed(
            self.rng.randint(0, 2**31)
        )
        noise = torch.randn(self.param.shape, generator=generator, device=self.param.device)
        scale = self.param.norm() / (noise.norm() + 1e-6)
        noise *= scale.detach() / math.sqrt(batch_size)
        return (self.param + noise).square().mean()


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

    # Use a more lenient win condition since we have inherent noise
    trial(
        model,
        data,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-8, 0),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        failure_threshold=5,
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
