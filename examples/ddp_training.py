"""DDP training with HeavyBall optimizers.

Launch with torchrun:
    torchrun --nproc_per_node=2 examples/ddp_training.py
    torchrun --nproc_per_node=2 examples/ddp_training.py --opt ForeachSOAP
    torchrun --nproc_per_node=2 examples/ddp_training.py --opt ForeachMuon --lr 0.01

All HeavyBall optimizers work transparently with DDP
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import heavyball


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="ForeachAdamW")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    model = DDP(make_model().cuda())
    opt = getattr(heavyball, args.opt)(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if rank == 0:
        datasets.FashionMNIST("./data", train=True, download=True)
    dist.barrier()
    dataset = datasets.FashionMNIST("./data", train=True, transform=tf)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = correct = total = 0

        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

            running_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={running_loss / len(loader):.4f}  acc={correct / total:.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
