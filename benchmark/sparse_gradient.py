from typing import List

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size=2**12, sparsity=2**-6):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.sparsity = sparsity
        self.register_buffer("prev_mask", torch.zeros_like(self.param))

    def forward(self):
        """Test optimizer's ability to handle sparse gradients."""
        # Generate new random mask each time, but keep some consistency
        new_mask = (torch.rand_like(self.param) < self.sparsity).float()
        mask = (new_mask + self.prev_mask) > 0  # Union of current and previous mask
        self.prev_mask.copy_(new_mask)

        return (self.param * mask.float()).square().mean()


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

    # Win condition accounts for sparsity - harder to reach very low loss
    trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
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
    )  # More failure attempts allowed


if __name__ == "__main__":
    app()
