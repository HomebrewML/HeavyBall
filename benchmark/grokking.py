import copy
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

import heavyball
from benchmark.utils import get_optim
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class ModularMLP(nn.Module):
    def __init__(self, numbers, p, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(p, hidden_dim),
            nn.Flatten(),
            nn.Linear(numbers * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, p),
        )

    def forward(self, x):
        return self.net(x)


class ModuloDataset(torch.utils.data.Dataset):
    def __init__(self, p, numbers, min_idx, length, batch_size):
        length = length // batch_size
        self.p = p
        self.numbers = numbers
        self.n_samples = length
        self.min_idx = min_idx
        self.max_idx = min_idx + length
        self.batch_size = batch_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(
            random.Random(min(idx + self.min_idx, self.max_idx)).randint(0, 2**32)
        )
        x = torch.randint(0, self.p, (self.batch_size, self.numbers), generator=generator)
        y = (x.sum(dim=-1) % self.p).long()
        return x, y


def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().detach()
            total += y.size(0)
    return correct / total


def plot_results(train_losses, test_accs, steps_to_grok=None, save_path=None):
    """Plot training curves"""
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot training loss
    ax1.plot(train_losses, label="Training Loss")
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True)

    # Plot test accuracy
    eval_steps = np.arange(0, len(train_losses), len(train_losses) // len(test_accs))
    ax2.plot(eval_steps, test_accs, label="Test Accuracy", color="orange")
    ax2.axhline(y=0.9, color="r", linestyle="--", label="Grokking Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Steps")
    ax2.set_title("Test Accuracy Over Time")
    ax2.grid(True)

    if steps_to_grok is not None:
        ax2.axvline(
            x=steps_to_grok, color="g", linestyle="--", label=f"Grokking Step ({steps_to_grok})"
        )

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    opt: List[str] = typer.Option(
        ["ForeachSOAP", "PaLMForeachSOAP", "PrecondScheduleForeachSOAP"], help="Optimizers to use"
    ),
    steps: int = 100,
    batch_size: int = 32,
    hidden_dim: int = 32,
    p: int = 257,
    numbers: int = 4,
    weight_decay: float = 0,
    lr: float = 1e-4,
    train_percent: float = 0.1,
    eval_samples: int = 1024,
    printervall: int = 1000,
):
    dtype = [getattr(torch, d) for d in dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clean up old plots
    plot_dir = Path(".")
    for path in plot_dir.glob("grokking_*.png"):
        path.unlink()

    # Pre-generate datasets
    unique_samples = p**numbers
    train_data = ModuloDataset(p, numbers, 0, int(unique_samples * train_percent), batch_size)
    test_data = ModuloDataset(p, numbers, train_data.max_idx, eval_samples, eval_samples)

    print(f"Training on {train_data.n_samples * batch_size:,} samples - {train_percent * 100}%")
    print(f"Testing on {eval_samples:,} samples")

    train_loader = DataLoader(
        train_data,
        collate_fn=lambda x: x[0],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        prefetch_factor=16,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_data,
        collate_fn=lambda x: x[0],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        prefetch_factor=32,
    )
    test_loader = list(test_loader)
    test_loader = [[x.pin_memory() for x in i] for i in test_loader]

    train_iter = iter(train_loader)
    history = defaultdict(list)

    def data():
        """Get next batch from the dataloader"""
        nonlocal train_iter
        try:
            x, y = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        return x.to(device), y.to(device)

    criterion = nn.CrossEntropyLoss()

    def win_condition(model, loss_hist):
        """Check if model has achieved grokking"""
        if not isinstance(loss_hist, float):
            loss = loss_hist
        else:
            loss = loss_hist

        history["loss"].append(loss)

        if loss > 0.1:  # Not converged yet
            return False, {}

        # If loss is low, check test accuracy
        acc = evaluate(model, test_loader, device)
        history["test_acc"].append(acc)
        return acc > 0.9, {"test_acc": acc}

    global_model = ModularMLP(numbers, p, hidden_dim).to(device)
    global_model = torch.compile(global_model, mode="max-autotune-no-cudagraphs")
    for d, o in itertools.product(dtype, opt):
        print(f"\nRunning {o} with {d}")
        model = copy.deepcopy(global_model)
        model.to(dtype=d)

        history.clear()

        # Get optimizer class
        optimizer_class = getattr(heavyball, o)
        optimizer = get_optim(
            optimizer_class, model.parameters(), lr=lr, weight_decay=weight_decay
        )

        loss_hist = torch.empty(steps)

        # Training loop
        for step in range(steps):
            model.train()
            x, y = data()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_hist[step] = loss.detach()

                if step % printervall == 0:
                    lh = loss_hist[:step][-printervall:].mean().item()
                    acc = evaluate(model, test_loader, device).item()
                    history["test_acc"].append(acc)
                    print(f"Step {step}: Loss = {lh:.4f}, Test Acc = {acc:.4f}")

        # Plot results
        plot_name = plot_dir / f"grokking_{o}_{d}_lr{lr}_h{hidden_dim}_p{p}.png"
        plot_results(
            loss_hist.cpu().numpy(),
            history["test_acc"],
            next((i for i, acc in enumerate(history["test_acc"]) if acc > 0.9), None),
            plot_name,
        )
        print(f"Training curves saved to {plot_name}")


if __name__ == "__main__":
    app()
