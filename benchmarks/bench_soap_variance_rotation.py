"""Numerics of SOAP/KLSOAP second-moment transport under a Q rotation.

Pre-fix (none):     _apply_soap_preconditioner did not pass exp_avg_sq into
                    get_orthogonal_matrix_QR. v stayed in the OLD eigenframe
                    while Q (and m) moved -- drifts with each rotation.
Strawman (linear):  apply m's einsum to v. Not what was shipped; included to
                    show why a naive rotation does not work.
Post-fix (hadamard): v <- (R*R)^T v per side, R = Q_old^T Q_new. Equals
                    diag(R^T diag(v) R) -- diagonal of the rotated covariance.

Reports per method vs analytical truth (= hadamard by construction):
  err   ||method - truth||_inf       (hadamard: 0; none/linear grow with theta)
  min   min(method)                  (linear can go negative)
  dvar  (sum(method) - sum(v))/sum(v) (hadamard preserves; linear does not)
"""

import argparse
import csv
import math
from collections import defaultdict
from itertools import product

import torch

ANGLES = [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, math.pi / 2]
SHAPES = [(16,), (256,), (16, 16), (32, 8), (8, 32), (64, 64)]
V_KINDS = ["uniform", "exponential", "spike"]
DTYPES = {torch.float32: "fp32", torch.float64: "fp64"}
METHODS = ["none", "linear", "hadamard"]


def haar(d, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g, dtype=dtype))[0]


def rotate(Q, theta, seed):
    d = Q.shape[-1]
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=Q.dtype)
    S = (A - A.T) / 2
    S = S / S.norm() * math.sqrt(d)
    return Q @ torch.linalg.matrix_exp(theta * S)


def make_v(shape, kind, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.rand(shape, generator=g, dtype=dtype) + 0.1
    if kind == "exponential":
        return -(torch.rand(shape, generator=g, dtype=dtype) + 1e-6).log()
    v = torch.full(shape, 1e-3, dtype=dtype)
    v.view(-1)[0] = 1.0
    return v


def transport(method, v, Q_old, Q_new):
    if method == "none":
        return v
    n = len(Q_old)
    in_, out, mid = "abcd"[:n], "efgh"[:n], "ABCD"[:n]
    if method == "linear":
        from_ = ",".join(m + i for m, i in zip(mid, in_))
        to_ = ",".join(m + o for m, o in zip(mid, out))
        return torch.einsum(f"{in_},{from_},{to_}->{out}", v, *Q_old, *Q_new)
    Rs_sq = [(Qo.T @ Qn).pow(2) for Qo, Qn in zip(Q_old, Q_new)]
    sides = ",".join(i + o for i, o in zip(in_, out))
    return torch.einsum(f"{in_},{sides}->{out}", v, *Rs_sq)


def measure(out, truth, total):
    return {
        "err": (out - truth).abs().max().item(),
        "min": out.min().item(),
        "dvar": (out.sum().item() - total) / total,
    }


def run_case(shape, theta, kind, dtype, seed):
    Q_old = [haar(d, seed + 100 * i, dtype) for i, d in enumerate(shape)]
    Q_new = [rotate(Q, theta, seed + 1000 + 100 * i) for i, Q in enumerate(Q_old)]
    v = make_v(shape, kind, seed + 2000, dtype)
    results = {m: transport(m, v, Q_old, Q_new) for m in METHODS}
    total = v.sum().item()
    return {
        "shape": "x".join(map(str, shape)),
        "dtype": DTYPES[dtype],
        "theta": theta,
        "kind": kind,
        "seed": seed,
        **{f"{k}_{m}": val for m in METHODS for k, val in measure(results[m], results["hadamard"], total).items()},
    }


def summarize(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(len(r["shape"].split("x")), r["dtype"], r["theta"])].append(r)

    cols = [(stat, m) for stat in ("err", "min", "dvar") for m in METHODS]
    header = f"{'mode':<4} {'dtype':<5} {'theta':>8}  " + "  ".join(f"{stat + '_' + m:<13}" for stat, m in cols)
    print(f"\n{header}\n{'-' * len(header)}")
    agg = {"err": max, "min": min, "dvar": lambda xs: max(map(abs, xs))}
    for (n, dt, theta), items in sorted(buckets.items()):
        vals = "  ".join(f"{agg[stat]([r[f'{stat}_{m}'] for r in items]):>13.3e}" for stat, m in cols)
        print(f"{n}d   {dt:<5} {theta:>8.4f}  {vals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Write all rows to CSV file")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    rows = [
        run_case(shape, theta, kind, dtype, seed)
        for shape, theta, kind, dtype, seed in product(SHAPES, ANGLES, V_KINDS, DTYPES, range(args.seeds))
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
