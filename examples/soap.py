import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torchvision.utils import make_grid

import heavyball

heavyball.utils.compile_mode = "default"
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
    def __init__(self, kernel: int = 5, stride: int = 2, hidden: int = 8, intermediate: int = 256):
        super(Autoencoder, self).__init__()
        self.enc = Block(1, intermediate, hidden, kernel, stride, False, 5)
        self.balancer = nn.BatchNorm2d(hidden, affine=False)
        self.dec = Block(hidden, intermediate, 1, kernel, stride, True, 5)

    def forward(self, x):
        x = self.enc(x)
        x = self.balancer(x).sigmoid()
        label = (x + torch.rand_like(x)) > 1
        x = label.detach().float() + x - x.detach()
        out = self.dec(x)
        return out


def plot_samples(model, data, epoch, save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        samples = model(data.cuda())
        # Create a grid of original and reconstructed images
        comparison = torch.cat([data, samples.cpu() * 255.0], dim=0)
        grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
        plt.close()
    model.train()


class RandomPad(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, inp):
        x = torch.randint(0, self.amount, (inp.size(0),))
        y = torch.randint(0, self.amount, (inp.size(0),))
        new = torch.zeros(
            [inp.shape[0], inp.shape[1] + self.amount, inp.shape[2] + self.amount],
            device=inp.device,
            dtype=inp.dtype,
        )
        new[:, x : x + inp.size(1), y : y + inp.size(2)] = inp
        return new


def main(epochs: int, batch: int):
    # Setup tensorboard logging
    log_dir = os.path.join("runs", f"soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    model = torch.compile(Autoencoder().cuda(), mode="default")
    optimizer = heavyball.SOAP(model.parameters(), lr=1e-3, precondition_frequency=1)
    # optimizer = heavyball.PSGDKron(optimizer, lr=1e-3, mars=True)
    # optimizer = heavyball.AdamW(model.parameters(), lr=1e-3, mars=True)

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32), RandomPad(4)])
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch * 8, shuffle=False, num_workers=1, pin_memory=True
    )

    for epoch in range(epochs):
        total_loss = 0

        for data in tqdm.tqdm(dataloader):
            img, _ = data
            img = img.cuda()

            def _closure():
                output = model(img)
                loss = F.mse_loss(output, img)
                loss.backward()
                return loss

            loss = optimizer.step(_closure)
            optimizer.zero_grad()
            with torch.no_grad():
                total_loss = total_loss + loss.detach()

        avg_loss = (total_loss / len(dataloader)).item()
        print(f"epoch [{epoch}/{epochs}], loss:{avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Plot samples every 2 epochs
        if epoch % 2 == 0:
            eval_batch = next(iter(testloader))[0][:8]
            plot_samples(model, eval_batch, epoch)

            # Log reconstructions to tensorboard
            with torch.no_grad():
                model.eval()
                samples = model(eval_batch.cuda())
                comparison = torch.cat([eval_batch, samples.cpu()], dim=0)
                grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
                writer.add_image("reconstructions", grid, epoch)
                model.train()
        writer.flush()


if __name__ == "__main__":
    main(epochs=10, batch=1024)
