"""Compare ForeachMuon optimizer-state configs on CIFAR-100:
fp32 (gold standard) vs naive bf16 (lossy) vs ecc bf16+int8 (corrected)."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

import heavyball
import heavyball.utils

heavyball.utils.set_torch()

CONFIGS = {
    "fp32": {},
    "naive_bf16": {"storage_dtype": "bfloat16"},
    "ecc": {"ecc": "bf16+8"},
}

_orig_orthogonalize_update = heavyball.chainable.orthogonalize_update

def _orthogonalize_update(group, update, grad, param):
    return _orig_orthogonalize_update(group, update, grad, param, scale_mode="graft")

heavyball.chainable.orthogonalize_update = _orig_orthogonalize_update

class TinyCNN(nn.Sequential):
    def __init__(self, classes=100):
        super().__init__()
        in_features, out_features = 3, 32
        for i in range(3):
            self.add_module(f'conv{i}', nn.Conv2d(in_features, out_features, 4, padding=1, stride=2))
            self.add_module(f'norm{i}', nn.BatchNorm2d(out_features))
            self.add_module(f'relu{i}', nn.ReLU())
            in_features, out_features = out_features, out_features * 2
        self.add_module('pool', nn.AdaptiveAvgPool2d(1))
        self.add_module('flatten', nn.Flatten())
        self.add_module('head', nn.Linear(in_features, classes))


def train(config, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    assert torch.cuda.is_available(), "CUDA required"

    mean = torch.tensor((0.5071, 0.4867, 0.4408), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor((0.2675, 0.2565, 0.2761), device="cuda").view(1, 3, 1, 1)
    coord = torch.arange(32, device="cuda")

    train_set = datasets.CIFAR100("./data", train=True, download=True)
    test_set = datasets.CIFAR100("./data", train=False, download=True)
    train_images = F.pad(torch.from_numpy(train_set.data).to("cuda").permute(0, 3, 1, 2).contiguous(), (4, 4, 4, 4))
    train_labels = torch.tensor(train_set.targets, device="cuda")
    test_images = torch.from_numpy(test_set.data).to("cuda").permute(0, 3, 1, 2).contiguous()
    test_images = test_images.float().div_(255).sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)
    test_labels = torch.tensor(test_set.targets, device="cuda")

    model = TinyCNN().to("cuda", memory_format=torch.channels_last)
    optimizer = heavyball.MuonLaProp(model.parameters(), lr=lr, weight_decay=0., **CONFIGS[config])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = torch.zeros((), device="cuda")
        perm = torch.randperm(train_labels.numel(), device="cuda")
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start:start + batch_size]
            images = train_images[idx]
            rows = torch.randint(9, (idx.numel(), 1), device="cuda") + coord
            cols = torch.randint(9, (idx.numel(), 1), device="cuda") + coord
            images = images.gather(2, rows[:, None, :, None].expand(-1, 3, -1, 40))
            images = images.gather(3, cols[:, None, None, :].expand(-1, 3, 32, -1))
            flip = torch.rand(idx.numel(), device="cuda") < 0.5
            images = torch.where(flip[:, None, None, None], images.flip(-1), images)
            images = images.float().div_(255).sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)
            labels = train_labels[idx]

            def closure():
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            optimizer.zero_grad()
            total_loss += loss.detach()

        avg_loss = total_loss / ((perm.numel() + batch_size - 1) // batch_size)
        model.eval()
        correct = torch.zeros((), device="cuda", dtype=torch.long)
        with torch.inference_mode():
            for start in range(0, test_labels.numel(), 1024):
                images = test_images[start:start + 1024]
                labels = test_labels[start:start + 1024]
                correct += (model(images).argmax(1) == labels).sum()

        acc = 100.0 * correct.item() / test_labels.numel()
        print(f"[{config:>10}] epoch {epoch:3d}  loss {avg_loss.item():.4f}  test_acc {acc:.2f}%")
        scheduler.step()

    return acc


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 ECC benchmark")
    parser.add_argument("--config", choices=list(CONFIGS), help="single config to run")
    parser.add_argument("--all", action="store_true", help="run all configs and print summary")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("specify --config or --all")

    results = {cfg: train(cfg, args.epochs, args.lr, args.batch_size, args.seed) for cfg in (CONFIGS if args.all else [args.config])}
    if args.all:
        print("\n" + "=" * 40)
        print(f"{'Config':<12} {'Test Acc':>8}")
        print("-" * 40)
        for cfg, acc in results.items():
            print(f"{cfg:<12} {acc:>7.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    main()
