from pathlib import Path
from typing import List

import torch
import torch._dynamo
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
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


# A very deep plain CNN for SVHN (no BatchNorm, no residuals)
class DeepCNN(nn.Module):
    def __init__(self, num_classes: int = 10, channels: int = 32, depth: int = 12):
        super().__init__()
        layers = []
        in_channels = 3  # SVHN has 3-channel RGB input
        for i in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def set_deterministic_weights(model, seed=42):
    """Initialize model with deterministic weights using a fixed seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return model


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    channels: int = 32,
    depth: int = 12,
    batch: int = 128,
    steps: int = 0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    test_loader: bool = None,
):
    dtype = [getattr(torch, d) for d in dtype]

    model = DeepCNN(num_classes=10, channels=channels, depth=depth).cuda()

    # SVHN normalization (roughly like CIFAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    train_dataset = datasets.SVHN(root=data_dir, split="train", download=True, transform=transform)
    test_dataset = datasets.SVHN(root=data_dir, split="test", download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True
    )

    data_iter = iter(train_loader)

    def data():
        nonlocal data_iter
        try:
            batch_data, batch_targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_data, batch_targets = next(data_iter)

        return batch_data.cuda(), batch_targets.cuda()

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
