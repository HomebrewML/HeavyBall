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
# Paste or import your actual implementation of `_gg_inverse_via_newtonschulz` here.
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

LETTERS = string.ascii_lowercase


def generate_spd_matrix(n: int, scale: float = 1.0, device="cpu") -> torch.Tensor:
    """Return an SPD matrix of size n × n (single precision)."""
    a = torch.randn(n, n, device=device)
    spd = a @ a.T
    eigvals = torch.linalg.eigvals(spd).real
    min_eig = eigvals.min().clamp(min=1e-3)
    spd += (scale - min_eig) * torch.eye(n, device=device)
    return spd.float()


def kron_prod(mats: List[torch.Tensor]) -> torch.Tensor:
    """Kronecker‑product of a list of square matrices."""
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def precondition_gradient(G_raw: torch.Tensor, hs: List[torch.Tensor]) -> torch.Tensor:
    """Apply Kronecker‑structured Hessian list *hs* to *G_raw*.

    G_raw shape = dims (d₀, d₁, …, dₙ₋₁)
    hs        = [H₀ (d₀×d₀), …, Hₙ₋₁ (dₙ₋₁×dₙ₋₁)]
    Returns   = G_pre of same shape as G_raw.
    """
    n = G_raw.ndim
    assert n == len(hs) <= 13, "Too many dims (einsum limited to 26 letters)."
    in_idx = LETTERS[n : 2 * n]  # d, e, f, …
    out_idx = LETTERS[:n]  # a, b, c, …
    h_terms = [f"{out_idx[i]}{in_idx[i]}" for i in range(n)]
    expr = ",".join(h_terms + ["".join(in_idx)]) + f"->{''.join(out_idx)}"
    return torch.einsum(expr, *hs, G_raw)


def relative_fro_error(est: List[torch.Tensor], true: List[torch.Tensor]) -> float:
    return math.exp(sum(torch.linalg.cond(e @ t).log().item() for t, e in zip(true, est)))


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

            # Build true Hessian list & its inverse
            hs = [generate_spd_matrix(d, device=self.device) for d in shape]

            # Preconditioners initialise to identity
            Q = [torch.eye(d, device=self.device) for d in shape]  # will be updated in‑place by algo
            oq = Q  # TriuOrLine alias

            for step in range(self.n_steps // 10):
                torch.manual_seed(self.seed + step)  # deterministic re‑sampling per step
                G_raw = torch.randn(*shape, device=self.device)
                G = precondition_gradient(G_raw, hs).contiguous().clone()  # apply true Hessian

                _gg_inverse_via_newtonschulz(
                    G=G,
                    oq=oq,
                    inverse_order=cfg["inverse_order"],
                    precond_lr=torch.tensor(cfg["precond_lr"], device=self.device),
                )

            # After optimisation, reconstruct full estimated inverse
            error = relative_fro_error(Q, hs)

            records.append({**cfg, "shape_str": shape_str, "rel_error": error})
        return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------


def plot_heatmap(df: pd.DataFrame, shape_str: str, out_dir: str):
    sub = df[df["shape_str"] == shape_str]
    pivot = sub.pivot_table(index="inverse_order", columns="precond_lr", values="rel_error", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2e", linewidths=0.5, norm=LogNorm())
    plt.title(f"Rel. error – shape {shape_str}")
    plt.ylabel("inverse_order")
    plt.xlabel("precond_lr")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heat_{shape_str}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_grid = {
        "matrix_shape": [(128, 128)],
        "inverse_order": [16],
        "precond_lr": [1],
    }

    runner = BenchmarkRunner(cfg_grid, n_steps=args.steps, device=args.device)
    df = runner.run()
    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    # Heat‑maps per shape
    for shape_str in df.shape_str.unique():
        plot_heatmap(df, shape_str, args.out_dir)

    # Convergence curve for best config
    best = df.nsmallest(1, "rel_error").iloc[0]
    shape = best.matrix_shape
    hs = [generate_spd_matrix(d, device=args.device) for d in shape]
    Q = [torch.eye(d, device=args.device) for d in shape]
    oq = Q
    errors = []
    for step in range(args.steps):
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

    plt.figure(figsize=(6, 4))
    plt.semilogy(range(1, args.steps + 1), errors, linewidth=0.8)
    plt.grid(True, which="both", ls="--", linewidth=0.4)
    plt.xlabel("Update step")
    plt.ylabel("Relative Frobenius error")
    plt.title(f"Convergence – shape {best.shape_str}, order {best.inverse_order}, lr {best.precond_lr}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"convergence_{best.shape_str}.png"), dpi=150)
    plt.close()

    print("Done. Results & plots saved to", args.out_dir)


if __name__ == "__main__":
    main()
