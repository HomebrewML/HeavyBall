"""Reproducer for FP64 promotion in PSGD preconditioner lower bound.

Run BEFORE and AFTER the fix to compare:
    python benchmarks/bench_psgd_fp64.py --tag before
    python benchmarks/bench_psgd_fp64.py --tag after

Outputs:
    bench_psgd_{tag}.json   Perfetto trace (open in chrome://tracing or ui.perfetto.dev)
    bench_psgd_{tag}.txt    Profiler key_averages table
"""

import argparse
import os
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

import heavyball


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="run", help="Tag for output files (e.g. 'before' or 'after')")
    parser.add_argument("--dim", type=int, default=2048, help="Linear layer dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps (not profiled)")
    parser.add_argument("--steps", type=int, default=20, help="Profiled steps")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, f"bench_psgd_{args.tag}.json")
    txt_path = os.path.join(out_dir, f"bench_psgd_{args.tag}.txt")

    torch.manual_seed(42)
    model = nn.Linear(args.dim, args.dim, bias=False, device="cuda", dtype=torch.bfloat16)
    opt = heavyball.PSGDPRO(model.parameters(), lr=1e-3, compile_step=False)
    data = torch.randn(64, args.dim, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(64, args.dim, device="cuda", dtype=torch.bfloat16)

    for _ in range(args.warmup):
        loss = ((model(data) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.cuda.synchronize()

    wall_times = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            loss = ((model(data) - target) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - t0)

    prof.export_chrome_trace(json_path)

    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
    with open(txt_path, "w") as f:
        f.write(table)

    kernel_events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    fp64_kernels = [e for e in kernel_events if "d884" in e.key]
    total_cuda = sum(e.self_device_time_total for e in kernel_events)

    print(f"\n{'=' * 60}")
    print(f"PSGD FP64 Benchmark [{args.tag}]")
    print(f"{'=' * 60}")
    print(f"Model:          Linear({args.dim}, {args.dim}) bfloat16")
    print(f"Steps:          {args.steps} (after {args.warmup} warmup)")
    print(f"Wall time/step: {sum(wall_times) / len(wall_times) * 1000:.2f}ms "
          f"(+/- {torch.tensor(wall_times).std().item() * 1000:.2f}ms)")
    print(f"Total GPU:      {total_cuda / 1000:.2f}ms")
    print()

    if fp64_kernels:
        fp64_time = sum(e.self_device_time_total for e in fp64_kernels)
        print(f"FP64 GEMM (d884) FOUND: {len(fp64_kernels)} kernel type(s), "
              f"{fp64_time / 1000:.2f}ms ({fp64_time / total_cuda * 100:.1f}% of GPU time)")
    else:
        print("FP64 GEMM (d884): NONE -- fix is working")

    print(f"\nTop 5 GPU kernels by self device time:")
    sorted_kernels = sorted(kernel_events, key=lambda e: e.self_device_time_total, reverse=True)
    for e in sorted_kernels[:5]:
        pct = e.self_device_time_total / total_cuda * 100 if total_cuda else 0
        print(f"  {pct:5.1f}% | {e.self_device_time_total / 1000:8.2f}ms | {e.count:4}x | {e.key[:90]}")

    print(f"\nTraces saved to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
