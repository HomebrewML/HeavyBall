"""A SYNTHETIC controlled-conditioning demonstration for HeavyBall optimizers.

Loss magnitudes are task- and seed-dependent; this is not a universal or production benchmark. The
point is the REGIME: whitening+momentum (LATHER) increasingly outperforms diagonal+momentum (AdamW)
as input conditioning worsens, SOAP is strong when well-conditioned, and PSGDKron (whitening with
NO update momentum) trails.
"""

import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import heavyball

D = 24
H = 24
STEPS = 500
BATCH_SIZE = 384
CONDITIONS = (1, 10, 100, 1000)
LEARNING_RATES = (3e-3, 1e-2, 3e-2)
SEEDS = range(5)


def _mlp() -> torch.nn.Sequential:
    return torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.Tanh(), torch.nn.Linear(H, D))


def _task(condition: int, seed: int) -> tuple[torch.Tensor, torch.nn.Sequential]:
    torch.manual_seed(2000 + seed)
    orthogonal, _ = torch.linalg.qr(torch.randn(D, D))
    eigenvalues = torch.logspace(0, math.log10(condition), D)
    root = orthogonal @ torch.diag(eigenvalues.sqrt()) @ orthogonal.T
    teacher = _mlp().requires_grad_(False)
    return root, teacher


def _train(
    optimizer_class: type[heavyball.HeavyBallOptimizer],
    optimizer_kwargs: dict[str, float],
    root: torch.Tensor,
    teacher: torch.nn.Sequential,
    seed: int,
    learning_rate: float,
) -> float:
    torch.manual_seed(seed)
    student = _mlp()
    optimizer = optimizer_class(student.parameters(), lr=learning_rate, **optimizer_kwargs)

    for _ in range(STEPS):
        x = torch.randn(BATCH_SIZE, D) @ root.T
        with torch.no_grad():
            y = teacher(x)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(student(x), y)
        loss.backward()
        optimizer.step()

    return loss.item()


def _print_table(medians: dict[int, dict[str, float]], names: list[str]) -> None:
    condition_width = len("condition")
    column_widths = {name: max(len(name), len("0.000e+00*")) for name in names}
    header = f"{'condition':>{condition_width}} | " + " | ".join(
        f"{name:>{column_widths[name]}}" for name in names
    )
    print(header)
    print("-" * len(header))
    for condition in CONDITIONS:
        best = min(medians[condition].values())
        cells = []
        for name in names:
            value = medians[condition][name]
            marker = "*" if value == best else " "
            cells.append(f"{value:.3e}{marker:1}")
        print(f"{condition:>{condition_width}} | " + " | ".join(cells))
    print("* best median final loss in row")


def main() -> None:
    optimizer_configs = [
        ("AdamW", heavyball.AdamW, {}),
        ("SOAP", heavyball.SOAP, {}),
        ("LATHER", heavyball.LATHER, {}),
        ("PSGDKron", heavyball.PSGDKron, {"precond_lr": 0.1}),
    ]
    final_losses = {
        condition: {name: [] for name, _, _ in optimizer_configs} for condition in CONDITIONS
    }

    for condition in CONDITIONS:
        for seed in SEEDS:
            root, teacher = _task(condition, seed)
            for name, optimizer_class, optimizer_kwargs in optimizer_configs:
                best_loss = min(
                    _train(
                        optimizer_class,
                        optimizer_kwargs,
                        root,
                        teacher,
                        seed,
                        learning_rate,
                    )
                    for learning_rate in LEARNING_RATES
                )
                final_losses[condition][name].append(best_loss)

    medians = {
        condition: {
            name: statistics.median(losses) for name, losses in optimizer_losses.items()
        }
        for condition, optimizer_losses in final_losses.items()
    }
    _print_table(medians, [name for name, _, _ in optimizer_configs])


if __name__ == "__main__":
    main()
