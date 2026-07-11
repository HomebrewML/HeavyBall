"""Benchmark PSGD FP64 lower-bound fix on a production-like model.

An alternative to the toy benchmark bench_psgd_fp64.py
  - 2-layer MLP: Linear(512,512) -> ReLU -> Linear(512,512)
  - bfloat16, batch=64, bias=True
  - PSGDPRO with default settings (compile_step default, max-autotune)
  - MSE loss

Run BEFORE and AFTER the fix:
    python benchmarks/bench_psgd_prod.py --tag before
    python benchmarks/bench_psgd_prod.py --tag after
"""

import argparse
import json
import os
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

import heavyball

def make_model(dim, device):
    return nn.Sequential(
        nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16),
        nn.ReLU(),
        nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, f"prod_{args.tag}.json")
    txt_path = os.path.join(out_dir, f"prod_{args.tag}.txt")
    loss_path = os.path.join(out_dir, f"prod_{args.tag}_losses.json")

    torch.manual_seed(42)
    model = make_model(args.dim, "cuda")
    opt = heavyball.PSGDPRO(model.parameters(), lr=1e-3)
    data = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)

    for _ in range(args.warmup):
        with record_function("fwdbwd"):
            with record_function("forward"):
                out = model(data)
                loss = nn.functional.mse_loss(out, target)
            with record_function("backward"):
                loss.backward()
        with record_function("optimizer_step"):
            with record_function("psgd_pro_step"):
                opt.step()
                opt.zero_grad()

    torch.cuda.synchronize()

    wall_times = []
    losses = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=args.steps),
        record_shapes=True,
        with_flops=True,
        on_trace_ready=lambda p: p.export_chrome_trace(json_path),
    ) as prof:
        for _ in range(1 + 1 + args.steps):
            t0 = time.perf_counter()
            with record_function("fwdbwd"):
                with record_function("forward"):
                    out = model(data)
                    loss = nn.functional.mse_loss(out, target)
                with record_function("backward"):
                    loss.backward()
            with record_function("optimizer_step"):
                with record_function("psgd_pro_step"):
                    opt.step()
                    opt.zero_grad()
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - t0)
            losses.append(loss.item())
            prof.step()

    active_times = wall_times[2:]
    active_losses = losses[2:]

    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
    with open(txt_path, "w") as f:
        f.write(table)

    with open(loss_path, "w") as f:
        json.dump({"wall_times": active_times, "losses": active_losses, "tag": args.tag}, f, indent=2)

    kernel_events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    fp64_kernels = [e for e in kernel_events if "d884" in e.key]
    total_cuda = sum(e.self_device_time_total for e in kernel_events)

    print(f"\n{'=' * 60}")
    print(f"PSGD Prod Benchmark [{args.tag}]")
    print(f"{'=' * 60}")
    print(f"Model:          MLP(Linear({args.dim},{args.dim}) -> ReLU -> Linear({args.dim},{args.dim})) bf16")
    print(f"Batch:          {args.batch}")
    print(f"Steps:          {args.steps} (after {args.warmup} warmup)")
    print(f"Wall time/step: {sum(active_times) / len(active_times) * 1000:.2f}ms")
    print(f"Total GPU:      {total_cuda / 1000:.2f}ms")

    if fp64_kernels:
        fp64_time = sum(e.self_device_time_total for e in fp64_kernels)
        print(f"\nFP64 GEMM (d884): {len(fp64_kernels)} type(s), "
              f"{fp64_time / 1000:.2f}ms ({fp64_time / total_cuda * 100:.1f}% of GPU time)")
    else:
        print("\nFP64 GEMM (d884): NONE")

    print(f"\nTop 10 GPU kernels:")
    sorted_kernels = sorted(kernel_events, key=lambda e: e.self_device_time_total, reverse=True)
    for e in sorted_kernels[:10]:
        pct = e.self_device_time_total / total_cuda * 100 if total_cuda else 0
        print(f"  {pct:5.1f}% | {e.self_device_time_total / 1000:8.2f}ms | {e.count:4}x | {e.key[:80]}")

    print(f"\nOutputs: {json_path}")
    print(f"         {txt_path}")
    print(f"         {loss_path}")


if __name__ == "__main__":
    main()
