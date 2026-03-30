import argparse

import torch
from torch._dynamo import config as dynamo_config

from heavyball.utils import _max_singular_value_ndim, max_singular_value, min_singular_value

dynamo_config.cache_size_limit = 2**20
dynamo_config.accumulated_cache_size_limit = 2**20


def hilbert_matrix(n):
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(0)
    return 1.0 / (i + j - 1).cuda()


def make_matrix(shape, cond=10, dtype=torch.float32, symmetric=False, seed=0):
    torch.manual_seed(seed)
    m, n = shape
    r = min(m, n)
    q_left, _ = torch.linalg.qr(torch.randn(m, r, dtype=torch.float32))
    q_right, _ = torch.linalg.qr(torch.randn(n, r, dtype=torch.float32))
    exponents = torch.linspace(0, -1, r, dtype=torch.float32)
    spectrum = cond**exponents
    diag = torch.diag(spectrum)
    if symmetric:
        return (q_left @ diag @ q_left.T).contiguous().to(dtype).cuda()
    return (q_left @ diag @ q_right.T).contiguous().to(dtype).cuda()


SHAPES_2D = [(4, 4), (32, 32), (128, 128), (10, 5), (5, 10)]
SHAPES_SYM = [(4, 4), (32, 32), (128, 128)]
SHAPES_NDIM = [(3, 4, 5), (16, 32, 64), (16, 16, 512)]
CONDS = [1, 10, 1e4, 1e10, 1e18, 1e30, 1e300]
DTYPES = [torch.bfloat16, torch.float32, torch.float64]
POWER_ITERS = [0, 5, 20]


def _dtype_name(dt):
    return {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float64: "fp64"}[dt]


def bench_max_sv(rows):
    for dtype in DTYPES:
        for pi in POWER_ITERS:
            for shape in SHAPES_2D:
                for cond in CONDS:
                    A = make_matrix(shape, cond=cond, dtype=dtype)
                    exact = torch.linalg.svdvals(A.double()).max()
                    try:
                        approx = max_singular_value(A, power_iter=pi)
                        rerr = abs((approx.double() - exact) / exact).item()
                        status = "ok"
                    except Exception as e:
                        rerr = float("nan")
                        status = type(e).__name__
                    rows.append(("max_sv", _dtype_name(dtype), pi, shape, cond, rerr, status))


def bench_min_sv(rows):
    for dtype in DTYPES:
        for pi in POWER_ITERS:
            for shape in SHAPES_SYM:
                for cond in CONDS:
                    A = make_matrix(shape, cond=cond, dtype=dtype, symmetric=True)
                    exact = torch.linalg.svdvals(A.double()).min()
                    try:
                        approx = min_singular_value(A, power_iter=pi)
                        if exact.abs() < 1e-8:
                            rerr = abs(approx.double() - exact).item()
                        else:
                            rerr = abs((approx.double() - exact) / exact).item()
                        status = "ok"
                    except Exception as e:
                        rerr = float("nan")
                        status = type(e).__name__
                    rows.append(("min_sv", _dtype_name(dtype), pi, shape, cond, rerr, status))


def bench_ndim(rows):
    for shape in SHAPES_NDIM:
        torch.manual_seed(0x172893)
        A = torch.randn(shape).cuda()
        exact = torch.linalg.svdvals(A.double()).max()
        try:
            approx = _max_singular_value_ndim(A, power_iter=2)
            rerr = abs((approx.double() - exact) / exact).item()
            is_upper = (approx.double() >= exact.double()).item()
            status = "ok" if is_upper else "not_upper_bound"
        except Exception as e:
            rerr = float("nan")
            status = type(e).__name__
        rows.append(("ndim", "fp32", 2, shape, 0, rerr, status))


def print_pareto(rows):
    from itertools import groupby

    print(f"\n{'func':<8} {'dtype':<5} {'pi':>3}  {'best_rerr':>10}  {'worst_rerr':>10}  {'errors':>6}  {'total':>5}")
    print("-" * 60)

    def key_fn(r):
        return r[0], r[1], r[2]

    for key, group in groupby(sorted(rows, key=key_fn), key=key_fn):
        items = list(group)
        ok = [r for r in items if r[6] == "ok"]
        errs = len(items) - len(ok)
        if ok:
            rerrs = [r[5] for r in ok]
            print(
                f"{key[0]:<8} {key[1]:<5} {key[2]:>3}  {min(rerrs):>10.6f}  {max(rerrs):>10.6f}  {errs:>6}  {len(items):>5}"
            )
        else:
            print(f"{key[0]:<8} {key[1]:<5} {key[2]:>3}  {'-':>10}  {'-':>10}  {errs:>6}  {len(items):>5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Write results to CSV file")
    args = parser.parse_args()

    rows = []
    bench_max_sv(rows)
    bench_min_sv(rows)
    bench_ndim(rows)

    print_pareto(rows)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["func", "dtype", "power_iter", "shape", "cond", "rel_error", "status"])
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
