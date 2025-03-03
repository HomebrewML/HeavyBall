import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

import heavyball

device = "cuda"
heavyball.utils.compile_mode = None
heavyball.utils.dynamic = True
heavyball.utils.set_torch()


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Residual(nn.Module):
    def __init__(self, wrapped):
        super(Residual, self).__init__()
        self.wrapped = wrapped

    def forward(self, x):
        return x + self.wrapped(x)


class DatasetNorm(nn.Module):
    def __init__(self, features: int, momentum: float = 0.99):
        super().__init__()
        self.weight = nn.Parameter(torch.stack([torch.ones(features), torch.zeros(features)], 1))
        self.register_buffer("stats", torch.zeros(features * 2))
        self.register_buffer("step", torch.tensor(0))
        self.momentum = momentum

    def forward(self, x):
        if True:
            with torch.no_grad():
                mean, sq_mean = x.mean(dim=0), (x**2).mean(dim=0)
                stats = torch.cat([mean, sq_mean])
                self.step.add_(1)
                self.stats.lerp_(stats, 1 - heavyball.utils.beta_debias(self.momentum, self.step))
                # self.stats.lerp_(stats, self.step == 1)
                mean, sq_mean = self.stats.chunk(2)
                std = (sq_mean - mean**2).clamp_min_(1e-6).sqrt()
        else:
            std, mean = 1, 0
        weight, bias = self.weight.unbind(1)
        return (x - mean) / std * weight + bias


class MLP(nn.Module):
    def __init__(self, in_shape, out_shape, width, depth, act=Sine(), expanded: int = 256):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_shape, width))

        for _ in range(depth - 1):
            layers.append(
                Residual(
                    nn.Sequential(
                        nn.Linear(width, expanded),  #
                        act,  #
                        DatasetNorm(expanded),  #
                        nn.Linear(expanded, width),
                    )
                )
            )
        layers.append(DatasetNorm(width))
        layers.append(nn.Linear(width, out_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def generate_two_moons_torch(n_samples=1000, noise=0.1, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)

    half_samples = n_samples // 2

    theta1 = torch.linspace(0, np.pi, half_samples, device=device)
    theta2 = torch.linspace(0, np.pi, half_samples, device=device)

    X1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
    X2 = torch.stack([1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5], dim=1)

    X = torch.cat([X1, X2], dim=0)
    y = torch.cat(
        [torch.zeros(half_samples, device=device), torch.ones(half_samples, device=device)], dim=0
    )

    X += noise * torch.randn(n_samples, 2, device=device)

    indices = torch.randperm(n_samples, device=device)
    X = X[indices]
    y = y[indices]

    return X, y


def train_and_generate_frames(
    model,
    X_train,
    y_train,
    domain,
    epochs,
    lr,
    filename="training_video",
    resolution: int = 128,
    subsample: int = 1,
    train_samples: int = 1024,
):
    X_train = X_train.to(device).float()
    y_train = y_train.view(-1, 1).to(device).float()

    optimizers = {
        "ForeachSOAP": heavyball.ForeachSOAP(model.parameters(), lr=lr),
        "PaLMForeachSOAP": heavyball.PaLMForeachSOAP(model.parameters(), lr=lr),
        "PrecondScheduleForeachSOAP": heavyball.PrecondScheduleForeachSOAP(
            model.parameters(), lr=lr
        ),
    }
    criterion = nn.BCEWithLogitsLoss()

    xx, yy = torch.meshgrid(
        torch.linspace(domain[0][0], domain[1][0], resolution, device=device),
        torch.linspace(domain[0][1], domain[1][1], resolution, device=device),
        indexing="xy",
    )
    grid_points = torch.stack((xx.ravel(), yy.ravel()), dim=1).float()

    base_model = copy.deepcopy(model)

    for optimizer_name, optimizer in optimizers.items():
        model = copy.deepcopy(base_model)
        print(f"\nTraining with {optimizer_name}")
        model.train()

        os.makedirs("frames", exist_ok=True)

        for epoch in tqdm.tqdm(range(epochs)):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % subsample == 0:
                model.eval()
                with torch.no_grad():
                    Z = model(grid_points).reshape(resolution, resolution)
                    plt.figure(figsize=(10, 8))
                    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), levels=20)
                    plt.colorbar(label="Model Output")
                    plt.scatter(
                        X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train.cpu(), cmap="coolwarm"
                    )
                    plt.title(f"{optimizer_name} - Epoch {epoch}, Loss: {loss.item():.4f}")
                    plt.savefig(f"frames/{optimizer_name}_epoch_{epoch:05d}.png")
                    plt.close()
                model.train()


if __name__ == "__main__":
    X, y = generate_two_moons_torch(n_samples=1024, noise=0.05, random_state=42)

    domain = np.array([
        [X[:, 0].min().item() - 1, X[:, 1].min().item() - 1],
        [X[:, 0].max().item() + 1, X[:, 1].max().item() + 1],
    ])

    model = torch.compile(
        MLP(in_shape=2, out_shape=1, width=2, depth=32), mode="max-autotune-no-cudagraphs"
    ).to(device)

    epochs = 100
    lr = 1e-4
    train_and_generate_frames(model, X, y, domain, epochs, lr)
