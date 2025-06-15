# benchmark_fast.py
# minimal, Typer‑based rewrite focused on runtime

import functools
import itertools
import math
import os
import string
import typing as tp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import tqdm
import typer
from matplotlib.colors import LogNorm
from torch._dynamo import config as dyn_cfg

from heavyball.utils import _gg_inverse_via_newtonschulz, set_torch

set_torch()
dyn_cfg.cache_size_limit = dyn_cfg.accumulated_cache_size_limit = 1_000_000
LETTERS = string.ascii_lowercase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- matrices


@functools.lru_cache(maxsize=None)
def _spd(n: int, cond: float, device: str):
    eig = torch.logspace(0, math.log10(cond), n, device=device)
    q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return q @ torch.diag(eig) @ q.T


@functools.lru_cache(maxsize=None)
def _sym_range(n: int, lo: float, hi: float, device: str):
    eig = torch.linspace(lo, hi, n, device=device)
    q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return q @ torch.diag(eig) @ q.T


def gen_mat(n: int, cfg: dict, device: str):
    if cfg["matrix_type"] == "spd":
        return _spd(n, cfg["cond_number"], device)
    return _sym_range(n, cfg["eig_min"], cfg["eig_max"], device)


# ----------------------------------------------------------------------------- gradients

_DIST = {
    "normal": lambda s, d: torch.randn(*s, device=d),
    "laplace": lambda s, d: torch.distributions.Laplace(0.0, 1.0).sample(s).to(d),
    "cauchy": lambda s, d: torch.distributions.Cauchy(0.0, 1.0).sample(s).to(d),
    "uniform": lambda s, d: torch.rand(*s, device=d) * 2 - 1,
    "rademacher": lambda s, d: (torch.randint(0, 2, s, device=d, dtype=torch.float32) * 2 - 1),
    "poisson": lambda s, d: torch.poisson(torch.ones(*s, device=d)),
}


def _parse(spec: str):
    if "_sparse" in spec:
        base, p = spec.split("_sparse")
        return base, ("static", float(p))
    if "_anneal" in spec:
        base, rng = spec.split("_anneal")
        p0, p1 = map(float, rng.split("-"))
        return base, ("anneal", (p0, p1))
    return spec, None


def gen_grad(shape: tp.Tuple[int, ...], cfg: dict, step: int, n_steps: int, device: str):
    base, sparse = _parse(cfg["grad_dist"])
    g = _DIST[base](shape, device)
    if sparse is None:
        return g
    kind, param = sparse
    p = param if kind == "static" else param[0] + (param[1] - param[0]) * step / (n_steps - 1)
    return g * (torch.rand(*shape, device=device) > p)


# ----------------------------------------------------------------------------- hessian dyn


def blend(a, b, t):
    return a * (1 - t) + b * t


def hess_init(shape, cfg, device):
    return [gen_mat(d, cfg, device) for d in shape]


def hess_update(hs0, hstgt, cfg, step, n):
    k = cfg["hess_dynamic"]
    if k == "static":
        return hs0
    if k == "lerp":
        t = step / (n - 1)
        return [blend(a, b, t) for a, b in zip(hs0, hstgt)]
    if k == "random_walk":
        std = cfg.get("perturb_std", 1e-3)
        return [h + std * torch.randn_like(h) for h in hs0]
    raise ValueError(k)


# ----------------------------------------------------------------------------- util


def kron_list(ms):
    out = ms[0]
    for m in ms[1:]:
        out = torch.kron(out, m)
    return out


def precond(G, hs):
    n = G.ndim
    ins = LETTERS[n : 2 * n]
    outs = LETTERS[:n]
    expr = ",".join(f"{outs[i]}{ins[i]}" for i in range(n)) + "," + "".join(ins) + "->" + "".join(outs)
    return torch.einsum(expr, *hs, G)


@torch.no_grad()
def rel_err(est, true):
    vals = torch.stack([torch.linalg.cond(e @ t).log() for t, e in zip(true, est)])
    return torch.exp(vals.sum()).item()


# ----------------------------------------------------------------------------- runner


class Runner:
    def __init__(self, grid: dict, steps: int, seed: int, device: str):
        self.grid, self.steps, self.seed, self.device = grid, steps, seed, device
        torch.manual_seed(seed)

    def run(self):
        out = []
        keys, vals = zip(*self.grid.items())
        for combo in tqdm.tqdm(list(itertools.product(*vals))):
            cfg = dict(zip(keys, combo))
            shape = cfg["matrix_shape"]
            sstr = "x".join(map(str, shape))
            hs0 = hess_init(shape, cfg, self.device)
            hstgt = hess_init(shape, cfg, self.device) if cfg["hess_dynamic"] == "lerp" else hs0
            Q = [torch.eye(d, device=self.device) for d in shape]
            oq = Q
            for step in range(self.steps):
                torch.manual_seed(self.seed + step)
                Graw = gen_grad(shape, cfg, step, self.steps, self.device)
                hs = hess_update(hs0, hstgt, cfg, step, self.steps)
                G = precond(Graw, hs).contiguous()
                _gg_inverse_via_newtonschulz(
                    G=G,
                    oq=oq,
                    inverse_order=cfg["inverse_order"],
                    precond_lr=torch.tensor(cfg["precond_lr"], device=self.device),
                )
            err = rel_err(Q, hs)
            out.append({**cfg, "shape_str": sstr, "rel_error": err})
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------- plots


def heatmaps(df: pd.DataFrame, out_dir: str):
    g = sns.FacetGrid(df, row="grad_dist", col="hess_dynamic", height=3.4, despine=False, margin_titles=True)

    def _hm(data, **kw):
        pivot = data.pivot_table(index="inverse_order", columns="precond_lr", values="rel_error", aggfunc="mean")
        sns.heatmap(pivot, norm=LogNorm(), cmap="viridis", cbar=False, **kw)

    g.map_dataframe(_hm)
    b_ax = g.fig.add_axes([0.92, 0.2, 0.015, 0.6])
    vmin, vmax = df.rel_error.min(), df.rel_error.max()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=LogNorm(vmin, vmax))
    sm.set_array([])
    g.fig.colorbar(sm, cax=b_ax, label="rel error")
    g.fig.savefig(os.path.join(out_dir, "faceted_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)


# ----------------------------------------------------------------------------- CLI

app = typer.Typer()


@app.command()
def main(
    out_dir: str = "plots",
    steps: int = 100,
    device: str = DEVICE.type,
    grad_dists: tp.List[str] = typer.Option(
        [
            "normal",
            "laplace",
            "cauchy",
            "rademacher",
            "uniform",
            "poisson",
            "normal_sparse0.8",
            "laplace_anneal0.9-0.3",
        ],
        help="gradient specs",
    ),
    hess_dynamics: tp.List[str] = typer.Option(["static", "lerp", "random_walk"]),
):
    os.makedirs(out_dir, exist_ok=True)
    grid = {
        "matrix_shape": [(4, 4), (32, 32), (256, 256)],
        "inverse_order": [1, 4],
        "precond_lr": [1.0, 1e-1, 1e-2],
        "matrix_type": ["spd", "non_spd"],
        "cond_number": [1e2, 1e4, 1e12, 1e30],
        "eig_min": [1, -10],
        "eig_max": [1e2, 1e6],
        "grad_dist": grad_dists,
        "hess_dynamic": hess_dynamics,
    }
    df = Runner(grid, steps, seed=0, device=device).run()
    csv = os.path.join(out_dir, "benchmark.csv")
    df.to_csv(csv, index=False)
    heatmaps(df, out_dir)
    typer.echo(f"done -> {csv}")


if __name__ == "__main__":
    app()
