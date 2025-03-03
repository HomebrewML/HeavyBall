from collections import deque
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
    def __init__(self, num_params=16, param_size=256):
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(param_size)) for _ in range(num_params)
        ])
        # Different update frequencies for each parameter
        self.delays = [i for i in range(num_params)]
        self.step = 0
        self.grad_queues = [deque(maxlen=i + 1) for i in self.delays]

    def forward(self):
        """Test optimizer's ability to handle delayed gradients."""
        total_loss = 0
        self.step += 1

        for param, delay, queue in zip(self.params, self.delays, self.grad_queues):
            # Current loss for this parameter
            loss = param.square().mean()

            # Store the gradient in the queue
            queue.append(loss)

            # Only add to total loss when we have enough history
            if len(queue) == queue.maxlen and self.step % (delay + 1) == 0:
                total_loss = total_loss + queue[0]  # Use oldest gradient

        return total_loss / len(self.params)


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

    # More lenient win condition and more steps due to delayed updates
    trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
        steps * 2,
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
    )  # Double steps, more attempts


if __name__ == "__main__":
    app()
