from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import typer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
app = typer.Typer()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Model, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    num_classes: int = 100,
    batch: int = 128,
    steps: int = 0,
    weight_decay: float = 5e-4,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    test_loader: bool = None,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(num_classes).cuda()

    # CIFAR-100 data loading with image augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    # Create data iterator that matches the expected format
    train_iter = iter(trainloader)

    def data():
        nonlocal train_iter
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            inputs, targets = next(train_iter)
        return inputs.cuda(), targets.cuda()

    trial(
        model,
        data,
        F.cross_entropy,
        loss_win_condition(win_condition_multiplier * 0.0),
        steps,
        opt[0],
        weight_decay,
        failure_threshold=10,
        trials=trials,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    app()
