from pathlib import Path
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torchvision import datasets, transforms

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

app = typer.Typer()


class Model(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        # self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def set_deterministic_weights(model, seed=42):
    """Initialize model with deterministic weights using a fixed seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Re-initialize all parameters
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot uniform initialization with fixed seed
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    return model


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 128,
    batch: int = 128,
    steps: int = 0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    test_loader: bool = None,
):
    dtype = [getattr(torch, d) for d in dtype]

    # Usage in your script:
    model = Model(hidden_size).cuda()
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Download data to a data directory relative to the script
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True
    )

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True
    )

    data_iter = iter(train_loader)

    def data():
        nonlocal data_iter
        try:
            batch_data, batch_targets = next(data_iter)
        except StopIteration:
            # Reset iterator when exhausted
            data_iter = iter(train_loader)
            batch_data, batch_targets = next(data_iter)

        return batch_data.cuda(), batch_targets.cuda()

    # Custom loss function that matches the expected signature
    def loss_fn(output, target):
        return F.nll_loss(output, target)

    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0),
        steps,
        opt[0],
        weight_decay,
        failure_threshold=10,
        trials=trials,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    app()
