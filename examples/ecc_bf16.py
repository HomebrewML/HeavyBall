"""Plot precision_toy results. Run precision_toy.py first, then this."""

import warnings

import matplotlib.pyplot as plt

import heavyball.utils

heavyball.utils.set_torch()
warnings.filterwarnings("ignore", message="Learning rate changed")

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import heavyball
import heavyball.utils

heavyball.utils.set_torch()
warnings.filterwarnings("ignore", message="Learning rate changed")

DEPTH, WIDTH, IN_DIM = 6, 64, 8
STEPS, LR, BETAS, EPS = 10000, 3e-3, (0.9, 0.999), 1e-8
N_TRAIN = 2048
LOG_EVERY = 500

CONFIGS = {"naive_fp32": lambda p: NaiveAdamW(p, lr=LR, betas=BETAS, eps=EPS, state_dtype=torch.float32),
    "naive_bf16": lambda p: NaiveAdamW(p, lr=LR, betas=BETAS, eps=EPS, state_dtype=torch.bfloat16),

    "heavyball_fp32": lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, storage_dtype="float32"),
    "heavyball_bf16": lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, weight_decay=0,
                                                       storage_dtype="bfloat16"),
    "ecc_bf16+8": lambda p: heavyball.ForeachAdamW(p, lr=LR, betas=BETAS, eps=EPS, weight_decay=0, ecc="bf16+8"), }

COLORS = {"naive_fp32": "#888888", "naive_bf16": "#d62728", "heavyball_fp32": "#2d2d2d", "heavyball_bf16": "#1f77b4",
          "ecc_bf16+8": "#2ca02c", }
LABELS = {"naive_fp32": "naive fp32", "naive_bf16": "naive bf16", "heavyball_fp32": "heavyball fp32",
          "heavyball_bf16": "heavyball bf16 (stochastic rounding)", "ecc_bf16+8": "ECC bf16+8 (correction tensor)", }
STYLES = {"naive_fp32": dict(linewidth=2, linestyle="--", alpha=0.7), "naive_bf16": dict(linewidth=2.5, linestyle="-"),
          "heavyball_fp32": dict(linewidth=2.5, linestyle="-"), "heavyball_bf16": dict(linewidth=1.8, linestyle="--"),
          "ecc_bf16+8": dict(linewidth=1.8, linestyle="--"), }


class MLP(nn.Sequential):
    def __init__(self):
        layers = [nn.Linear(IN_DIM, WIDTH), nn.GELU()]
        for _ in range(DEPTH - 2):
            layers += [nn.Linear(WIDTH, WIDTH), nn.GELU()]
        layers.append(nn.Linear(WIDTH, 1))
        super().__init__(*layers)


class NaiveAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, state_dtype=torch.float32):
        self.params = [p for p in params if p.requires_grad]
        self.param_groups = [{"lr": lr, "params": self.params}]
        self.eps, self.wd = eps, weight_decay
        self.beta1, self.beta2 = betas
        self.state_dtype = state_dtype
        self.state = {}
        self.t = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        self.t += 1
        lr = self.param_groups[0]["lr"]
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.float()
            if p not in self.state:
                self.state[p] = {"m": torch.zeros_like(g, dtype=self.state_dtype),
                                 "v": torch.zeros_like(g, dtype=self.state_dtype)}
            s = self.state[p]
            m, v = s["m"].float(), s["v"].float()
            if self.wd:
                p.data.mul_(1 - lr * self.wd)
            m.lerp_(g, 1 - self.beta1)
            v.lerp_(g * g, 1 - self.beta2)
            bc1 = 1 - self.beta1 ** self.t
            bc2 = 1 - self.beta2 ** self.t
            p.data.addcdiv_(m / bc1, (v / bc2).sqrt().add_(self.eps), value=-lr)
            s["m"].copy_(m)
            s["v"].copy_(v)
        return loss

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()


@torch.no_grad()
def make_data(teacher, n, seed):
    x = torch.randn(n, IN_DIM, generator=torch.Generator().manual_seed(seed)).cuda()
    return x, teacher(x)


def train(name, make_opt, init_state, train_x, train_y):
    model = MLP().cuda()
    model.load_state_dict(init_state)
    opt = make_opt(model.parameters())

    log = []
    for step in range(1, STEPS + 1):
        lr = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr

        def closure():
            loss = F.mse_loss(model(train_x), train_y)
            loss.backward()
            return loss

        opt.step(closure)
        opt.zero_grad()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse = F.mse_loss(model(train_x), train_y).item()
            log_mse = math.log10(mse) if mse > 0 else -float("inf")
            log.append((step, mse, log_mse))
            print(f"[{name:>20}] step {step:5d}  log10(mse) {log_mse:+.2f}  mse {mse:.2e}")

    return log, opt


def main():
    torch.manual_seed(0)
    teacher = MLP().cuda().eval()
    torch.manual_seed(42)
    init = MLP().cuda().state_dict()
    train_x, train_y = make_data(teacher, N_TRAIN, seed=0)

    results, opts = {}, {}
    for name, make_opt in CONFIGS.items():
        results[name], opts[name] = train(name, make_opt, init, train_x, train_y)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in CONFIGS:
        steps = [r[0] for r in results[name]]
        mses = [r[1] for r in results[name]]
        ax.plot(steps, mses, color=COLORS[name], label=LABELS[name], **STYLES[name])

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    fig.tight_layout()
    fig.savefig("precision_toy.png", dpi=180)
    print(f"\nSaved precision_toy.png")


if __name__ == "__main__":
    main()
