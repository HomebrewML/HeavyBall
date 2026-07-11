"""Generate convergence plot from before/after JSON files.

    python benchmarks/plot_convergence.py --before convergence_before.json --after convergence_after.json --out convergence.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", default="convergence.png")
    args = parser.parse_args()

    with open(args.before) as f:
        before = json.load(f)
    with open(args.after) as f:
        after = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(before["losses"], label="before (fp64 lower bound)",
             color="#1f77b4", alpha=0.9, linewidth=2.5)
    ax1.plot(after["losses"], label="after (fp32 lower bound)",
             color="#d62728", alpha=0.7, linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Convergence: linear scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(before["losses"], label="before (fp64 lower bound)",
                 color="#1f77b4", alpha=0.9, linewidth=2.5)
    ax2.semilogy(after["losses"], label="after (fp32 lower bound)",
                 color="#d62728", alpha=0.7, linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Convergence: log scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"PSGD FP64 Fix - Convergence Test\n"
        f"MLP(Linear({before['dim']},{before['dim']}) -> ReLU -> Linear({before['dim']},{before['dim']})), "
        f"bf16, batch={before['batch']}, lr={before['lr']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
