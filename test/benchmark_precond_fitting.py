import hashlib
import itertools
import math
import os
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import tqdm
import typer
from matplotlib.colors import LogNorm
from torch._dynamo import config as dyn_cfg

from heavyball.utils import init_Q_exprs, psgd_update_precond, set_torch

set_torch()
dyn_cfg.cache_size_limit = dyn_cfg.accumulated_cache_size_limit = 1_000_000
LETTERS = string.ascii_lowercase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _spd(n: int, cond: float, device: str):
    eig = torch.logspace(0, math.log10(cond), n, device=device)
    q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return q @ torch.diag(eig) @ q.T


def _sym_range(n: int, lo: float, hi: float, device: str):
    eig = torch.linspace(lo, hi, n, device=device)
    q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
    return q @ torch.diag(eig) @ q.T


def gen_mat(n: int, cfg: dict, device: str):
    if cfg["matrix_type"] == "spd":
        return _spd(n, cfg["cond_number"], device)
    return _sym_range(n, cfg["eig_min"], cfg["eig_max"], device)


_DIST = {
    "normal": lambda s, d: torch.randn(*s, device=d),
    "laplace": lambda s, d: torch.distributions.Laplace(0.0, 1.0).sample(s).to(d),
    "cauchy": lambda s, d: torch.distributions.Cauchy(0.0, 1.0).sample(s).to(d),
    "uniform": lambda s, d: torch.rand(*s, device=d) * 2 - 1,
    "rademacher": lambda s, d: torch.randint(0, 2, s, device=d, dtype=torch.float32) * 2 - 1,
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


def gen_grad(shape: tuple[int, ...], cfg: dict, step: int, n_steps: int, device: str):
    base, sparse = _parse(cfg["grad_dist"])
    g = _DIST[base](shape, device)
    if sparse is None:
        return g
    kind, param = sparse
    p = param if kind == "static" else param[0] + (param[1] - param[0]) * step / (n_steps - 1)
    return g * (torch.rand(*shape, device=device) > p)


def hess_init(shape, cfg, device):
    return [gen_mat(d, cfg, device) for d in shape]


def matrix_seed(cfg, seed):
    identity = cfg["matrix_shape"], cfg["matrix_type"], cfg.get("cond_number"), cfg.get("eig_min"), cfg.get("eig_max")
    digest = hashlib.blake2b(repr(identity).encode(), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def precond(G, hs):
    n = G.ndim
    ins = LETTERS[n : 2 * n]
    outs = LETTERS[:n]
    expr = ",".join(f"{outs[i]}{ins[i]}" for i in range(n)) + "," + "".join(ins) + "->" + "".join(outs)
    return torch.einsum(expr, *hs, G)


@torch.no_grad()
def fit_condition(preconditioners, hessians):
    vals = torch.stack([torch.linalg.cond(p @ h).log() for p, h in zip(preconditioners, hessians)])
    return torch.exp(vals.sum()).item()


def drift_hessian(hessian, std, spd):
    noise = torch.randn_like(hessian)
    drifted = hessian + std * (noise + noise.mT) / 2
    if not spd:
        return drifted
    values, vectors = torch.linalg.eigh(drifted)
    floor = torch.finfo(values.dtype).eps * values.abs().amax().clamp_min(1)
    return (vectors * values.clamp_min(floor)) @ vectors.mT


class Runner:
    def __init__(self, grid: dict, steps: int, seed: int, device: str):
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.grid, self.steps, self.seed, self.device = grid, steps, seed, device
        torch.manual_seed(seed)

    def run(self):
        out = []
        keys, vals = zip(*self.grid.items())
        for combo in tqdm.tqdm(list(itertools.product(*vals))):
            cfg = dict(zip(keys, combo))
            cfg.update(cfg.pop("matrix"))
            shape = cfg["matrix_shape"]
            sstr = "x".join(map(str, shape))
            seed = matrix_seed(cfg, self.seed)
            torch.manual_seed(seed)
            hs0 = hess_init(shape, cfg, self.device)
            if cfg["hess_dynamic"] == "lerp":
                torch.manual_seed(seed + 1)
                hstgt = hess_init(shape, cfg, self.device)
            else:
                hstgt = hs0
            hs = hs0
            stacked_shape = (1, *shape)
            seed_grad = torch.zeros(stacked_shape, device=self.device)
            Q = init_Q_exprs(seed_grad, 1.0, 1.0, 0.0, max(shape), 1, None, None, None)
            running_lb = [torch.zeros((1,), device=self.device, dtype=torch.float64) for _ in shape]
            for step in range(self.steps):
                torch.manual_seed(self.seed + step)
                vector = gen_grad(shape, cfg, step, self.steps, self.device).unsqueeze(0)
                if cfg["hess_dynamic"] == "static":
                    hs = hs0
                elif cfg["hess_dynamic"] == "lerp":
                    t = step / (self.steps - 1)
                    hs = [a * (1 - t) + b * t for a, b in zip(hs0, hstgt)]
                elif cfg["hess_dynamic"] == "random_walk":
                    torch.manual_seed(seed + step + 1)
                    hs = [drift_hessian(h, cfg.get("perturb_std", 1e-3), cfg["matrix_type"] == "spd") for h in hs]
                else:
                    raise ValueError(cfg["hess_dynamic"])
                hessian_vector = precond(vector.squeeze(0), hs).unsqueeze(0).contiguous()
                psgd_update_precond(
                    hessian_vector,
                    cfg["precond_lr"],
                    Q,
                    False,
                    vector,
                    running_lb,
                    cfg["lower_bound_beta"],
                    cfg["power_iter"],
                )
            factors = [q[0] if q.ndim == 3 else torch.diag(q[0]) for q in Q]
            condition = fit_condition([q.mT @ q for q in factors], hs)
            out.append({**cfg, "shape_str": sstr, "condition": condition})
        return pd.DataFrame(out)


def heatmaps(df: pd.DataFrame, out_dir: str):
    g = sns.FacetGrid(df, row="grad_dist", col="hess_dynamic", height=3.4, despine=False, margin_titles=True)
    norm = LogNorm(df.condition.min(), df.condition.max())

    def _hm(data, **kw):
        pivot = data.pivot_table(index="shape_str", columns="precond_lr", values="condition", aggfunc="mean")
        sns.heatmap(pivot, norm=norm, cmap="viridis", cbar=False, **kw)

    g.map_dataframe(_hm)
    b_ax = g.fig.add_axes([0.92, 0.2, 0.015, 0.6])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    g.fig.colorbar(sm, cax=b_ax, label="condition number")
    g.fig.savefig(os.path.join(out_dir, "faceted_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)


app = typer.Typer()


@app.command()
def main(
    out_dir: str = "plots",
    steps: int = 32,
    device: str = DEVICE.type,
    grad_dists: list[str] = typer.Option(
        [
            "normal",
            "cauchy",
            "normal_sparse0.8",
        ],
        help="gradient specs",
    ),
    hess_dynamics: list[str] = typer.Option(["static", "lerp", "random_walk"]),
):
    os.makedirs(out_dir, exist_ok=True)
    grid = {
        "matrix_shape": [(4, 4), (32, 32)],
        "power_iter": [4],
        "precond_lr": [1.0, 1e-1, 1e-2],
        "lower_bound_beta": [0.95],
        "matrix": [
            *({"matrix_type": "spd", "cond_number": cond} for cond in (1e2, 1e8)),
            *({"matrix_type": "non_spd", "eig_min": lo, "eig_max": hi} for lo, hi in ((-10, 1e2), (-10, 1e6))),
        ],
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
