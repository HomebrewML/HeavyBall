from pathlib import Path
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Take0(nn.Module):
    def forward(self, x):
        return x[0]


class Model(nn.Module):
    def __init__(self, features: int, sequence: int):
        super().__init__()
        self.sequence = sequence
        self.net = nn.Sequential(
            nn.Embedding(256, features),
            nn.LSTM(features, features, 1, batch_first=True),  # Removed dropout since num_layers=1
            Take0(),
            nn.Linear(features, 256),
        )

    def forward(self, inp):
        return self.net(inp)


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    features: int = 512,
    sequence: int = 256,
    batch: int = 16,
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(features, sequence).cuda()

    # Load text data
    benchmark_dir = Path(__file__).parent
    with open(benchmark_dir / "shakespeare.txt", "rb") as f:
        text = f.read()
    chars = torch.frombuffer(text, dtype=torch.uint8).cuda().long()

    # Create holdout set
    chars = chars[(sequence + 1) * batch :]
    offsets = torch.arange(0, sequence + 1, device="cuda").repeat(batch, 1)

    def data():
        batch_offsets = torch.randint(0, len(chars) - sequence - 1, (batch,), device="cuda")
        batch_offsets = batch_offsets[:, None] + offsets
        batch_chars = chars[batch_offsets]
        batch_chars = batch_chars.view(batch, sequence + 1)
        src = batch_chars[:, :-1]
        tgt = batch_chars[:, 1:]
        return src, tgt

    trial(
        model,
        data,
        F.cross_entropy,
        loss_win_condition(win_condition_multiplier * 2.0),
        steps,
        opt[0],
        dtype[0],
        features,
        batch,
        weight_decay,
        method[0],
        sequence,
        1,
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
