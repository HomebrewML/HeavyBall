import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torchvision.utils import make_grid

import heavyball

heavyball.utils.set_torch()


class Residual(nn.Sequential):
    def forward(self, input):
        out = super().forward(input)
        return out + F.interpolate(input, out.shape[2:])


class Block(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        intermediate: int,
        out_features: int,
        kernel: int,
        stride: int,
        up: bool,
        depth: int,
    ):
        padding = kernel // 2
        layers = [nn.Conv2d(in_features, intermediate, kernel_size=kernel, padding=padding)]

        for _ in range(depth):
            layers.append(
                Residual(
                    nn.Upsample(scale_factor=stride) if up else nn.MaxPool2d(stride),
                    nn.BatchNorm2d(intermediate),
                    nn.ReLU(),
                    nn.Conv2d(intermediate, intermediate, kernel_size=kernel, padding=padding),
                )
            )

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(intermediate, out_features, kernel_size=kernel, padding=padding))

        super().__init__(*layers)


class Autoencoder(nn.Module):
    def __init__(self, kernel: int = 3, stride: int = 2, hidden: int = 1, intermediate: int = 128):
        super(Autoencoder, self).__init__()
        self.enc = Block(1, intermediate, hidden, kernel, stride, False, 3)
        self.dec = Block(hidden, intermediate, 1, kernel, stride, True, 3)

    def forward(self, x):
        x = self.enc(x).sigmoid()
        # label = x > torch.rand_like(x)
        # x = label.detach().float() + x - x.detach()
        out = self.dec(x)
        return out


class RandomPad(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount
        self.rng = np.random.default_rng(0x12312)

    def forward(self, inp):
        new = []
        xs, ys = np.split((np.random.randint(0, self.amount, size=2 * inp.size(0)) * self.amount).round(), 2)
        for val, x, y in zip(inp, xs, ys):
            padded = F.pad(val, (x, self.amount - x, y, self.amount - y))
            new.append(padded)
        return torch.stack(new)


def main(epochs: int, batch: int, log_interval: int = 16):
    # Setup tensorboard logging
    torch.manual_seed(0x12783)
    np.random.seed(0x12783)
    random.seed(0x12783)
    log_dir = os.path.join("runs", f"soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    model = torch.compile(Autoencoder().cuda(), mode="default")
    optimizer = heavyball.PSGDKron(
        model.parameters(),
        lr=1e-4,
        mars=True,
        lower_bound_beta=0.9,
        precond_update_power_iterations=6,
        store_triu_as_line=False,
    )

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    train = [img for img, _ in MNIST(root="./data", train=True, download=True, transform=transform)]
    test = [img for _, (img, _) in zip(range(8), MNIST(root="./data", train=False, download=True, transform=transform))]

    train = torch.stack(train).cuda() / 255.0
    eval_batch = torch.stack(test) / 255.0

    transform = RandomPad(4)
    eval_batch = transform(eval_batch)
    eval_batch_cuda = eval_batch.cuda()
    step = 0
    total_loss = 0

    for epoch in range(epochs):
        train = train[torch.randperm(train.size(0))].contiguous()
        batches = transform(train)
        batches = batches[: batches.size(0) // batch * batch]
        batches = batches.view(-1, batch, *batches.shape[1:])

        for i in tqdm.tqdm(range(batches.size(0))):
            img = batches[i]
            step += 1

            def _closure():
                output = model(img)
                loss = F.mse_loss(output, img)
                loss.backward()
                return loss

            loss = optimizer.step(_closure)
            optimizer.zero_grad()
            with torch.no_grad():
                total_loss = total_loss + loss.detach()

            if step % log_interval == 0:
                avg_loss = (total_loss / log_interval).item()
                writer.add_scalar("Loss/train", avg_loss, step)
                total_loss = 0
            if step % (log_interval * 10) == 0:
                writer.flush()

        with torch.no_grad():
            model.eval()
            samples = model(eval_batch_cuda)
            comparison = torch.cat([eval_batch, samples.cpu()], dim=0)
            grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
            writer.add_image("reconstructions", grid, epoch)
            model.train()
        writer.flush()


if __name__ == "__main__":
    main(epochs=100, batch=128)
