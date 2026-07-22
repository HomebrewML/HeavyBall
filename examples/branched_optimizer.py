import random
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import heavyball

# HeavyBall 4.0 has no chainable Parallel/merge grafting; Route branches by ParamInfo instead.
BRANCHED_RECIPE = heavyball.Route(lambda info: info.ndim == 2, heavyball.adamw, heavyball.sgd)


def build_optimizer(
    params: Iterable[torch.Tensor],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
):
    return heavyball.HeavyBallOptimizer(
        params,
        BRANCHED_RECIPE,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
    )


def main(epochs: int = 20, batch_size: int = 256, subset_size: int = 4096):
    from torchvision import datasets, transforms

    torch.manual_seed(2024)
    random.seed(2024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    if subset_size < len(train_data):
        train_data = Subset(train_data, range(subset_size))
    if subset_size // 4 < len(test_data):
        test_data = Subset(test_data, range(max(1, subset_size // 4)))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=3e-4, betas=(0.9, 0.995), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            def closure():
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            optimizer.zero_grad()

            running_loss += loss.item()
            total += labels.size(0)

            with torch.no_grad():
                preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total else 0.0

        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)

        eval_acc = eval_correct / eval_total if eval_total else 0.0
        print(
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} - train acc: {train_acc:.3f} - eval acc: {eval_acc:.3f}"
        )


if __name__ == "__main__":
    main()
