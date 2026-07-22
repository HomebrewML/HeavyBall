from dataclasses import replace

import torch
import torch.nn as nn
from torch.nn import functional as F

import heavyball

# HeavyBall 4.0 replaces mutable chainable fns with Recipe composition; precondition_frequency was removed.
ORTHOGONAL_SOAP = replace(heavyball.soap, chain=(*heavyball.soap.chain, heavyball.orthogonalize))
MODIFIED_SOAP = heavyball.Route(lambda info: info.ndim == 2, ORTHOGONAL_SOAP, heavyball.adamw)


def build_optimizer(params):
    return heavyball.HeavyBallOptimizer(params, MODIFIED_SOAP, lr=1e-3, beta2=0.95, weight_decay=0.01)


def main(epochs: int, batch: int, features: int = 16, steps: int = 1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(features, features * 4), nn.ReLU(), nn.Linear(features * 4, 1)).to(device)
    optimizer = build_optimizer(model.parameters())

    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps):
            data = torch.randn((batch, features), device=device)
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
