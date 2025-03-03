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


class Model(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.embed0 = nn.Embedding(2, size)
        self.embed1 = nn.Embedding(2, size)
        self.enc = nn.LSTM(size, size, depth, batch_first=False)
        self.dec = nn.LSTM(size, size, depth, batch_first=False)
        self.enc.flatten_parameters()
        self.dec.flatten_parameters()
        self.proj = nn.Sequential(
            nn.LayerNorm(size),  #
            nn.Linear(size, 1),
        )

    def forward(self, inp):
        i0, i1 = inp.chunk(2, 1)
        i0 = i0.transpose(0, 1)
        i1 = i1.transpose(0, 1)
        i0 = self.embed0(i0)
        i1 = self.embed1(i1)
        _, state = torch.compiler.disable()(self.enc)(i0)
        out, _ = torch.compiler.disable()(self.dec)(i1, state)
        return self.proj(out.transpose(0, 1))


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    length: int = 14,
    size: int = 16,
    depth: int = 1,
    batch: int = 256,
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    torch.manual_seed(0x1239121)
    model = Model(size, depth).cuda()

    def data():
        inp = torch.randn((batch, length, 1), device="cuda", dtype=dtype[0])
        inp = inp > 0
        i0, i1 = inp.chunk(2, 1)
        xored = torch.logical_xor(i0, i1)
        return inp.long().squeeze(-1), xored.to(dtype[0])

    trial(
        model,
        data,
        F.binary_cross_entropy_with_logits,
        loss_win_condition(win_condition_multiplier * 1e-2),
        steps,
        opt[0],
        dtype[0],
        size,
        batch,
        weight_decay,
        method[0],
        length,
        depth,
        failure_threshold=10,
        base_lr=0.001,
        trials=trials,
    )


if __name__ == "__main__":
    app()
