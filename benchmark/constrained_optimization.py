import pathlib
from typing import List

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

# Objective: Minimize (x-2)^2 subject to x <= 1
# Implemented using a penalty: (x-2)^2 + penalty * max(0, x - 1)
PENALTY = 1e6
TARGET_X = 1.0
TOLERANCE = 1e-3


def objective(x):
    """Objective function with a penalty for violating the constraint x <= 1."""
    return (x - 2.0) ** 2 + PENALTY * torch.relu(x - TARGET_X)


class Model(nn.Module):
    def __init__(self, initial_x):
        super().__init__()
        # Using a tensor with requires_grad=True directly as the parameter
        self.param = nn.Parameter(torch.tensor(initial_x).float())

    def forward(self):
        return objective(self.param)


def win_condition(model, loss):
    """Check if the parameter x is close to the constraint boundary."""
    with torch.no_grad():
        final_x = model.param.item()
        success = abs(final_x - TARGET_X) < TOLERANCE
        # print(f"Final x: {final_x}, Target: {TARGET_X}, Success: {success}") # Debug print
        return success, {"final_x": final_x}


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 200,  # Increased steps slightly
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    trials: int = 50,  # Reduced trials slightly for faster testing
    win_condition_multiplier: float = 1.0,  # Not used directly, but kept for consistency
):
    dtype = [getattr(torch, d) for d in dtype]
    initial_x = 0.0  # Start within the feasible region

    # Clean up old plots if any (though this benchmark doesn't plot)
    for path in pathlib.Path(".").glob("constrained_optimization*.png"):
        path.unlink()

    model = Model(initial_x)
    model.double()  # Use double for precision if needed

    # No external data needed for this simple objective
    def data():
        return None, None

    # The loss is the objective value itself
    loss_fn = None

    print(f"Running constrained optimization benchmark for {opt[0]}...")
    print(f"Objective: (x-2)^2 + {PENALTY}*relu(x-{TARGET_X})")
    print(f"Target x: {TARGET_X}, Tolerance: {TOLERANCE}")

    trial(
        model,
        data,
        loss_fn,
        win_condition,
        steps,
        opt[0],
        dtype[0],
        1,  # size (not relevant here)
        1,  # batch (not relevant here)
        weight_decay,
        method[0],
        1,  # length (not relevant here)
        1,  # depth (not relevant here)
        failure_threshold=3,
        base_lr=1e-3,  # Default base LR, hyperopt will search
        trials=trials,
        group=32,  # Smaller group size might be better for simple problems
    )


if __name__ == "__main__":
    app()
