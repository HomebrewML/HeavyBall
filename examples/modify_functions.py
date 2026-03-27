"""Adds orthogonalize_update to a SOAP optimizer after construction.

Shows runtime modification of the transform chain via assignment to optimizer.fns.
Run: python examples/modify_functions.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

import heavyball

heavyball.utils.compile_mode = "default"
heavyball.utils.set_torch()


def main(epochs: int, batch: int, features: int = 16, steps: int = 1024):
    model = nn.Sequential(nn.Linear(features, features * 4), nn.ReLU(), nn.Linear(features * 4, 1))
    model.cuda()

    optimizer = heavyball.SOAP(
        model.parameters(), lr=1e-3, precondition_frequency=1
    )  # initial_d is required by scale_by_lr_adaptation but not used in standard SOAP - we'll get a warning about it
    optimizer.fns = optimizer.fns + [
        heavyball.chainable.orthogonalize_update
    ]  # important that we assign and don't just .append()!

    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps):
            data = torch.randn((batch, features), device="cuda")
            target = data.square().mean(1, keepdim=True)

            def _closure():
                output = model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                return loss

            loss = optimizer.step(_closure)
            optimizer.zero_grad()
            with torch.no_grad():
                total_loss = total_loss + loss.detach()

        avg_loss = (total_loss / steps).item()
        print(f"[{epoch:{len(str(epochs))}d}/{epochs}]  Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main(epochs=100, batch=1024)
