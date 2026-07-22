"""Speed and state-memory of HeavyBall low-precision state vs torch.optim.AdamW(fused).

Run: python benchmarks/precision_speed.py   (uses CUDA if available)

Reports ms/step (median of a timed window after warmup) and optimizer-state bytes for
fp32 / bfloat16 / ecc8 state, against torch.optim.AdamW(fused) as the baseline to beat.
"""

import sys
import time
from pathlib import Path

import torch
import torch.optim as torch_optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import heavyball


def _bench_step(optimizer, model, steps: int, warmup: int) -> float:
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)

    def one() -> None:
        for parameter in model.parameters():
            parameter.grad.normal_()
        optimizer.step()

    for _ in range(warmup):
        one()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        one()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps * 1e3


def _state_megabytes(optimizer) -> float:
    total = 0
    for engine in optimizer._engines:
        for group in engine.groups:
            for state in group.states:
                total += sum(slab.numel() * slab.element_size() for slab in state.values())
            for corrections in getattr(group, "state_corrections", ()):
                total += sum(slab.numel() * slab.element_size() for slab in corrections.values())
    return total / 1e6


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused = device == "cuda"
    print(f"device={device} torch={torch.__version__}")
    configs = [
        ("torch AdamW(fused)", lambda m: torch_optim.AdamW(m.parameters(), fused=fused)),
        ("HeavyBall fp32", lambda m: heavyball.AdamW(m.parameters())),
        ("HeavyBall bf16", lambda m: heavyball.AdamW(m.parameters(), storage_dtype=torch.bfloat16)),
        ("HeavyBall ecc8", lambda m: heavyball.AdamW(m.parameters(), ecc=8)),
    ]
    for layers in (4, 16, 48):
        params = layers * 1024 * 1024 / 1e6
        print(f"\n{layers} x Linear(1024, 1024)  ({params:.0f}M params)")
        for name, make in configs:
            model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(layers)]).to(device)
            optimizer = make(model)
            milliseconds = _bench_step(optimizer, model, steps=50, warmup=15)
            memory = "" if name.startswith("torch") else f"   state {_state_megabytes(optimizer):.1f} MB"
            print(f"  {name:22} {milliseconds:6.3f} ms/step{memory}")


if __name__ == "__main__":
    main()
