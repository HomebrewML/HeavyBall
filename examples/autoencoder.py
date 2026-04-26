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


class Autoencoder(nn.Module):
    def __init__(self, d=1024, widths=(768, 384, 160, 96), latent=24):
        super().__init__()
        enc, dec = [], []
        prev = d
        for w in widths:
            enc += [nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU()]
            prev = w
        enc.append(nn.Linear(prev, latent))

        prev = latent
        for w in reversed(widths):
            dec += [nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU()]
            prev = w
        dec.append(nn.Linear(prev, d))

        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(1)
        x = self.dec(self.enc(x))
        return x.view(shape)


class RandomPad(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, inp):
        new = []
        xs, ys = np.split(np.random.randint(0, self.amount + 1, size=2 * inp.size(0)), 2)
        for val, x, y in zip(inp, xs, ys):
            padded = F.pad(val, (x, self.amount - x, y, self.amount - y))
            new.append(padded)
        return torch.stack(new)


def main(epochs: int, batch: int, log_interval: int = 16):
    torch.manual_seed(0x12783)
    np.random.seed(0x12783)
    random.seed(0x12783)
    log_dir = os.path.join("runs", f"psgdpro_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    model = torch.compile(Autoencoder().cuda(), mode="default")
    optimizer = heavyball.PSGDPRO(
        model.parameters(), lr=1e-3, precond_update_power_iterations=6, store_triu_as_line=False, cached=True
    )

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    train = [img for img, _ in MNIST(root="./data", train=True, download=True, transform=transform)]
    test = [img for _, (img, _) in zip(range(8), MNIST(root="./data", train=False, download=True, transform=transform))]

    train = torch.stack(train).cuda() / 255.0
    eval_batch = torch.stack(test) / 255.0

    pad = RandomPad(4)
    eval_batch_raw = eval_batch
    eval_batch_cuda = F.pad(eval_batch, (2, 2, 2, 2)).cuda()
    step = 0
    total_loss = 0

    for epoch in range(epochs):
        train = train[torch.randperm(train.size(0))].contiguous()
        batches = pad(train)
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
            comparison = torch.cat([eval_batch_raw, samples.cpu()[:, :, 2:-2, 2:-2]], dim=0)
            grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
            writer.add_image("reconstructions", grid, epoch)
            model.train()
        writer.flush()


if __name__ == "__main__":
    main(epochs=100, batch=128)
