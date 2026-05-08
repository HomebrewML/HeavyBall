import argparse
import csv
from collections import defaultdict
from itertools import product

import torch

SHAPES = [(64, 64), (128, 32), (32, 128), (256, 16), (16, 256)]
WARMUP_K = [0, 1, 2, 4, 8, 16]
EPS_VALS = [1e-12, 1e-8, 1e-4]
DTYPES = {torch.float32: "fp32", torch.float64: "fp64"}
METHODS = ["clamp", "pinv"]


def make_L(shape, k, seed, dtype):
    d_a, d_b = shape
    g = torch.Generator().manual_seed(seed)
    Gs = [torch.randn(d_a, d_b, generator=g, dtype=dtype) for _ in range(k + 1)]
    L = sum(G @ G.T for G in Gs) / (d_b * (k + 1))
    return L, Gs[-1]


def reciprocal(method, eig, eps):
    if method == "clamp":
        return eig.clamp_min(eps).reciprocal()
    keep = eig > eps * eig.amax(dim=-1, keepdim=True)
    return torch.where(keep, eig.reciprocal(), 0.0)


def apply_inv(Q, inv_eig, X):
    return Q @ (inv_eig.unsqueeze(-1) * (Q.T @ X))


def run_case(shape, k, eps, dtype, seed):
    L, G = make_L(shape, k, seed, dtype)
    eig64, Q64 = torch.linalg.eigh(L.double())
    eig64 = eig64.clamp_min(0)
    eig, Q = eig64.to(dtype), Q64.to(dtype)
    truth_eps = eig64.shape[-1] * torch.finfo(eig64.dtype).eps
    truth = apply_inv(Q64, reciprocal("pinv", eig64, truth_eps), G.double()).to(dtype)
    ref_norm = truth.norm().item()
    rank = (eig64 > eig64.max() * 1e-12).sum().item()

    row = {
        "shape": f"{shape[0]}x{shape[1]}",
        "dtype": DTYPES[dtype],
        "k": k,
        "eps": eps,
        "seed": seed,
        "rank": rank,
        "d": shape[0],
    }
    for m in METHODS:
        inv = reciprocal(m, eig, eps)
        out = apply_inv(Q, inv, G)
        row[f"maxinv_{m}"] = inv.max().item()
        row[f"err_{m}"] = (out - truth).norm().item() / ref_norm
    return row


def summarize(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["shape"], r["dtype"], r["k"], r["eps"])].append(r)

    cols = [(stat, m) for stat in ("maxinv", "err") for m in METHODS]
    header = f"{'shape':<10} {'dtype':<5} {'k':>3} {'eps':>9}  {'rank/d':>8}  " + "  ".join(f"{stat + '_' + m:<13}" for stat, m in cols)
    print(f"\n{header}\n{'-' * len(header)}")
    for (shape, dt, k, eps), items in sorted(buckets.items()):
        rank, d = items[0]["rank"], items[0]["d"]
        vals = "  ".join(f"{max(r[f'{stat}_{m}'] for r in items):>13.3e}" for stat, m in cols)
        print(f"{shape:<10} {dt:<5} {k:>3} {eps:>9.0e}  {rank}/{d:<5}  {vals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Write all rows to CSV file")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    rows = [
        run_case(shape, k, eps, dtype, seed)
        for shape, k, eps, dtype, seed in product(
            SHAPES, WARMUP_K, EPS_VALS, DTYPES, range(args.seeds)
        )
    ]
    summarize(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
