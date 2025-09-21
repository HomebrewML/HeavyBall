import random
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import heavyball.chainable as C
import heavyball.utils

heavyball.utils.set_torch()


def _graft(outputs: Iterable[list[torch.Tensor]], eps: float = 1e-8) -> list[torch.Tensor]:
    adam_update, sgd_update = outputs
    merged = []
    for adam, sgd in zip(adam_update, sgd_update):
        merged.append(sgd * (adam.norm() / sgd.norm().add(eps)))
    return merged


class GraftedAdam(C.BaseOpt):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
            weight_decay: float = 1e-4, warmup_steps: int = 0, foreach: bool = True, ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup_steps=warmup_steps, )
        branch = C.Branch(branches=[[C.scale_by_adam], [C.identity]], merge_fn=_graft)
        super().__init__(params, defaults, foreach, fns=(branch,))


def main(epochs: int = 20, batch_size: int = 256, subset_size: int = 4096):
    torch.manual_seed(2024)
    random.seed(2024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

    train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    if subset_size < len(train_data):
        train_data = Subset(train_data, range(subset_size))
    if subset_size // 4 < len(test_data):
        test_data = Subset(test_data, range(max(1, subset_size // 4)))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10), ).to(device)

    optimizer = GraftedAdam(model.parameters(), lr=3e-4, betas=(0.9, 0.995), weight_decay=1e-4)
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
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} - train acc: {train_acc:.3f} - eval acc: {eval_acc:.3f}")


if __name__ == "__main__":
    main()
