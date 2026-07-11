"""Convergence test for PSGD FP64 lower-bound fix.

Trains a model for N steps and logs per-step loss to JSON.
Run before and after the fix with the same seed to produce
comparable convergence curves.

    python benchmarks/convergence_test.py --tag before --steps 500
    python benchmarks/convergence_test.py --tag after --steps 500
"""

import argparse
import json
import os

import torch
from torch import nn

import heavyball


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"convergence_{args.tag}.json")

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(args.dim, args.dim, bias=True, device="cuda", dtype=torch.bfloat16),
        nn.ReLU(),
        nn.Linear(args.dim, args.dim, bias=True, device="cuda", dtype=torch.bfloat16),
    )
    opt = heavyball.PSGDPRO(model.parameters(), lr=args.lr)
    data = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)

    losses = []
    for step in range(args.steps):
        out = model(data)
        loss = nn.functional.mse_loss(out, target).float()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}/{args.steps}  loss={losses[-1]:.6e}")

    result = {
        "tag": args.tag,
        "dim": args.dim,
        "batch": args.batch,
        "steps": args.steps,
        "lr": args.lr,
        "losses": losses,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal loss: {losses[-1]:.6e}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
