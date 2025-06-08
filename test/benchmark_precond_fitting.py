import argparse
import itertools
import math
import os
import string
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import tqdm
from matplotlib.colors import LogNorm
from torch._dynamo import config as dyn_cfg

from heavyball.utils import _gg_inverse_via_newtonschulz, set_torch

set_torch()
dyn_cfg.cache_size_limit = 10**6
dyn_cfg.accumulated_cache_size_limit = 10**6

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

LETTERS = string.ascii_lowercase


def generate_spd_matrix_with_cond(n: int, cond: float, device="cpu") -> torch.Tensor:
    """Generate an SPD matrix with specified condition number."""
    eigvals = torch.logspace(0, math.log10(cond), n, device=device)
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return Q @ torch.diag(eigvals) @ Q.T


def generate_symmetric_matrix_with_eig_range(n: int, eig_min: float, eig_max: float, device="cpu") -> torch.Tensor:
    """Generate a symmetric matrix with specified eigenvalue range."""
    eigvals = torch.linspace(eig_min, eig_max, n, device=device)
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return Q @ torch.diag(eigvals) @ Q.T


def generate_matrix(n: int, cfg: Dict, device="cpu") -> torch.Tensor:
    """Generate a matrix based on configuration."""
    if cfg["matrix_type"] == "spd":
        return generate_spd_matrix_with_cond(n, cfg["cond_number"], device)
    elif cfg["matrix_type"] == "non_spd":
        return generate_symmetric_matrix_with_eig_range(n, cfg["eig_min"], cfg["eig_max"], device)
    else:
        raise ValueError("Invalid matrix_type")


def kron_prod(mats: List[torch.Tensor]) -> torch.Tensor:
    """Kronecker-product of a list of square matrices."""
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def precondition_gradient(G_raw: torch.Tensor, hs: List[torch.Tensor]) -> torch.Tensor:
    """Apply Kronecker-structured Hessian list *hs* to *G_raw*."""
    n = G_raw.ndim
    assert n == len(hs) <= 13, "Too many dims (einsum limited to 26 letters)."
    in_idx = LETTERS[n : 2 * n]
    out_idx = LETTERS[:n]
    h_terms = [f"{out_idx[i]}{in_idx[i]}" for i in range(n)]
    expr = ",".join(h_terms + ["".join(in_idx)]) + f"->{''.join(out_idx)}"
    return torch.einsum(expr, *hs, G_raw)


def relative_fro_error(est: List[torch.Tensor], true: List[torch.Tensor]) -> float:
    """Compute relative Frobenius error; returns NaN if not applicable."""
    try:
        return math.exp(sum(torch.linalg.cond(e @ e.T @ t).log().item() for t, e in zip(true, est)))
    except Exception:
        return float("nan")  # Return NaN if inverse is unstable (e.g., non-SPD)


# --------------------------------------------------------------------------------------
# Benchmark runner
# --------------------------------------------------------------------------------------


class BenchmarkRunner:
    def __init__(self, cfg_grid: Dict[str, List], n_steps: int = 40, seed: int = 0, device="cpu"):
        self.cfg_grid = cfg_grid
        self.n_steps = n_steps
        self.seed = seed
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def run(self) -> pd.DataFrame:
        records = []
        keys, values = zip(*self.cfg_grid.items())
        for combo in tqdm.tqdm(list(itertools.product(*values))):
            cfg = dict(zip(keys, combo))
            shape: Tuple[int, ...] = cfg["matrix_shape"]
            shape_str = "x".join(map(str, shape))

            # Build true Hessian list
            hs = [generate_matrix(d, cfg, device=self.device) for d in shape]
            Q = [torch.eye(d, device=self.device) for d in shape]  # Preconditioners
            oq = Q

            for step in range(self.n_steps // 10):
                torch.manual_seed(self.seed + step)
                G_raw = torch.randn(*shape, device=self.device)
                G = precondition_gradient(G_raw, hs).contiguous().clone()
                _gg_inverse_via_newtonschulz(
                    G=G,
                    oq=oq,
                    inverse_order=cfg["inverse_order"],
                    precond_lr=torch.tensor(cfg["precond_lr"], device=self.device),
                )

            error = relative_fro_error(Q, hs)
            records.append({**cfg, "shape_str": shape_str, "rel_error": error})
        return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------


def plot_heatmap(df: pd.DataFrame, shape_str: str, matrix_type: str, out_dir: str):
    sub = df[(df["shape_str"] == shape_str) & (df["matrix_type"] == matrix_type)]
    pivot = sub.pivot_table(index="inverse_order", columns="precond_lr", values="rel_error", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2e", linewidths=0.5, norm=LogNorm())
    plt.title(f"Rel. error – shape {shape_str}, type {matrix_type}")
    plt.ylabel("inverse_order")
    plt.xlabel("precond_lr")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heat_{shape_str}_{matrix_type}.png"), dpi=150)
    plt.close()


def create_unified_visualization(
    df: pd.DataFrame, best_config: pd.Series, convergence_errors: List[float], out_dir: str
):
    """Create a unified visualization with all results in a single figure."""

    # Set up the figure with a sophisticated layout
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.2, 1, 1, 1], width_ratios=[1, 1, 1, 1, 0.05], hspace=0.35, wspace=0.25)

    # Use a professional color scheme
    plt.style.use("seaborn-v0_8-darkgrid")
    cmap = "viridis"

    # Title
    fig.suptitle("Newton-Schulz Preconditioner Convergence Analysis", fontsize=24, fontweight="bold", y=0.98)

    # 1. Convergence curve (top row, spanning 4 columns)
    ax_conv = fig.add_subplot(gs[0, :4])
    ax_conv.semilogy(
        range(1, len(convergence_errors) + 1), convergence_errors, linewidth=3, color="#2E86AB", label="Best SPD config"
    )
    ax_conv.fill_between(range(1, len(convergence_errors) + 1), convergence_errors, alpha=0.3, color="#2E86AB")
    ax_conv.set_xlabel("Update Step", fontsize=14)
    ax_conv.set_ylabel("Relative Frobenius Error", fontsize=14)
    ax_conv.set_title(
        f"Convergence of Best Configuration: shape={best_config.shape_str}, "
        f"order={int(best_config.inverse_order)}, lr={best_config.precond_lr}",
        fontsize=16,
        pad=10,
    )
    ax_conv.grid(True, alpha=0.3, linestyle="--")
    ax_conv.legend(fontsize=12)

    # 2. Heatmaps grid
    shapes = sorted(df.shape_str.unique(), key=lambda x: (len(x.split("x")), x))
    matrix_types = ["spd", "non_spd"]

    # Create a shared colorbar axis
    cbar_ax = fig.add_subplot(gs[1:, -1])

    # Determine global color scale for better comparison
    valid_errors = df["rel_error"].replace([float("inf"), -float("inf")], float("nan")).dropna()
    vmin, vmax = valid_errors.min(), valid_errors.max()

    # Create heatmaps
    heatmap_axes = []
    for i, shape in enumerate(shapes):
        for j, mtype in enumerate(matrix_types):
            row = 1 + i // 2
            col = (i % 2) * 2 + j

            ax = fig.add_subplot(gs[row, col])
            heatmap_axes.append(ax)

            # Filter data
            sub = df[(df["shape_str"] == shape) & (df["matrix_type"] == mtype)]

            if not sub.empty:
                # Create pivot table
                pivot = sub.pivot_table(index="inverse_order", columns="precond_lr", values="rel_error", aggfunc="mean")

                sns.heatmap(
                    pivot,
                    ax=ax,
                    cmap=cmap,
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    annot=True,
                    fmt=".1e",
                    linewidths=1,
                    linecolor="white",
                    cbar=False,
                    square=True,
                    annot_kws={"size": 10},
                )

                # Styling
                ax.set_title(f"{shape} - {mtype.upper()}", fontsize=14, fontweight="bold", pad=10)
                ax.set_xlabel("Learning Rate" if row == 3 else "", fontsize=12)
                ax.set_ylabel("Inverse Order" if col == 0 else "", fontsize=12)

                # Rotate labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                # Highlight best configuration if it matches
                if shape == best_config.shape_str and mtype == "spd":
                    best_order_idx = list(pivot.index).index(int(best_config.inverse_order))
                    best_lr_idx = list(pivot.columns).index(float(best_config.precond_lr))
                    rect = plt.Rectangle((best_lr_idx, best_order_idx), 1, 1, fill=False, edgecolor="red", linewidth=3)
                    ax.add_patch(rect)
            else:
                ax.text(
                    0.5, 0.5, "No Data", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="gray"
                )
                ax.set_xticks([])
                ax.set_yticks([])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Relative Frobenius Error", fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    # Add summary statistics box
    stats_text = f"""Summary Statistics:
    Total configurations tested: {len(df)}
    Best error (SPD): {df[df["matrix_type"] == "spd"]["rel_error"].min():.2e}
    Worst error (SPD): {df[df["matrix_type"] == "spd"]["rel_error"].max():.2e}
    Best configuration: {best_config.shape_str}, order={int(best_config.inverse_order)}, lr={best_config.precond_lr}
    """

    fig.text(
        0.98,
        0.02,
        stats_text,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        fontsize=11,
        ha="right",
        va="bottom",
        family="monospace",
    )

    # Save the unified figure
    plt.savefig(os.path.join(out_dir, "unified_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_comparison_plot(df: pd.DataFrame, out_dir: str):
    """Create a comparison plot showing the effect of different parameters."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Parameter Effects on Convergence", fontsize=18, fontweight="bold")

    # 1. Effect of inverse order
    ax = axes[0, 0]
    for shape in df.shape_str.unique():
        spd_data = df[(df["shape_str"] == shape) & (df["matrix_type"] == "spd")]
        if not spd_data.empty:
            order_effect = spd_data.groupby("inverse_order")["rel_error"].mean()
            ax.semilogy(order_effect.index, order_effect.values, marker="o", label=shape, linewidth=2, markersize=8)
    ax.set_xlabel("Inverse Order")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Effect of Inverse Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Effect of learning rate
    ax = axes[0, 1]
    for shape in df.shape_str.unique():
        spd_data = df[(df["shape_str"] == shape) & (df["matrix_type"] == "spd")]
        if not spd_data.empty:
            lr_effect = spd_data.groupby("precond_lr")["rel_error"].mean()
            ax.loglog(lr_effect.index, lr_effect.values, marker="s", label=shape, linewidth=2, markersize=8)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Effect of Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. SPD vs Non-SPD comparison
    ax = axes[1, 0]
    shape_groups = df.groupby(["shape_str", "matrix_type"])["rel_error"].mean().unstack()
    shape_groups.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("SPD vs Non-SPD Performance")
    ax.set_yscale("log")
    ax.legend(["Non-SPD", "SPD"])
    ax.set_xlabel("Shape")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 4. Condition number effect (for SPD matrices)
    ax = axes[1, 1]
    spd_data = df[df["matrix_type"] == "spd"]
    for shape in spd_data.shape_str.unique():
        shape_data = spd_data[spd_data["shape_str"] == shape]
        cond_effect = shape_data.groupby("cond_number")["rel_error"].mean()
        ax.loglog(cond_effect.index, cond_effect.values, marker="^", label=shape, linewidth=2, markersize=8)
    ax.set_xlabel("Condition Number")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Effect of Condition Number (SPD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parameter_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_grid = {
        "matrix_shape": [(32,), (128,), (32, 32), (16, 16, 16)],
        "inverse_order": [1, 4],
        "precond_lr": [1, 1e-1],
        "matrix_type": ["spd", "non_spd"],
        "cond_number": [1e2, 1e4, 1e6],  # Used when matrix_type is "spd"
        "eig_min": [-1e-1, -1],  # Used when matrix_type is "non_spd"
        "eig_max": [10],  # Used when matrix_type is "non_spd"
    }

    runner = BenchmarkRunner(cfg_grid, n_steps=args.steps, device=args.device)
    df = runner.run()
    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    # Get convergence data for best configuration
    best = df[df["matrix_type"] == "spd"].nsmallest(1, "rel_error").iloc[0]
    shape = best.matrix_shape
    hs = [generate_matrix(d, best.to_dict(), device=args.device) for d in shape]
    Q = [torch.eye(d, device=args.device) for d in shape]
    oq = Q
    errors = []

    for step in range(min(args.steps, 1000)):  # Limit convergence plot to 1000 steps for clarity
        torch.manual_seed(runner.seed + step)
        G_raw = torch.randn(*shape, device=args.device)
        G = precondition_gradient(G_raw, hs)
        _gg_inverse_via_newtonschulz(
            G=G.contiguous().clone(),
            oq=oq,
            inverse_order=int(best.inverse_order),
            precond_lr=torch.tensor(float(best.precond_lr), device=args.device),
        )
        errors.append(relative_fro_error(Q, hs))

    # Create unified visualization
    create_unified_visualization(df, best, errors, args.out_dir)

    # Create additional comparison plots
    create_comparison_plot(df, args.out_dir)

    print(f"Done. Unified visualizations saved to {args.out_dir}")
    print("  - unified_analysis.png: Complete overview with heatmaps and convergence")
    print("  - parameter_comparison.png: Parameter effect analysis")


if __name__ == "__main__":
    main()
