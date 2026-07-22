"""Real compiled max-autotune optimizer steps remain correct under FSDP2.

This GPU-only oracle covers both shard-separable and owner-whole optimizer state while using
HeavyBall's actual compiled step. Each rank trains on different data, so bit-identical full
parameters also verify that FSDP2's reduced gradients reach the optimizer slabs.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or shutil.which("torchrun") is None,
    reason="requires torch.distributed and torchrun",
)

_WORKER = '''
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball


def cross_rank_max(t):
    ref = t.clone()
    dist.broadcast(ref, src=0)
    d = (t - ref).abs().max()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return d.item()


def main():
    cuda = torch.cuda.is_available()
    if not cuda:
        print("CUDA unavailable; compiled FSDP2 worker skipped", flush=True)
        print("FSDP2_COMPILED_STEP", flush=True)
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())

    for name in ("AdamW", "SOAP", "PSGDKron"):
        torch.manual_seed(0)
        model = nn.Linear(8, 6, bias=False).to(device)
        fully_shard(model)
        opt = getattr(heavyball, name).fsdp2(model, lr=1e-2)
        init = model.weight.full_tensor().flatten().clone()
        torch.manual_seed(100 + rank)  # different data per rank
        # >= the preconditioner-refresh cadence, so this exercises BOTH compiled graphs under FSDP2
        # (the normal step and the refresh step, where the owner-whole gather + eigendecomp live).
        for _ in range(25):
            X = torch.randn(16, 8, device=device)
            Y = torch.randn(16, 6, device=device)
            ((model(X) - Y) ** 2).mean().backward()
            opt.step()
            opt.zero_grad()
        flat = model.weight.full_tensor().flatten()
        diff = cross_rank_max(flat)
        trained = (flat - init).abs().max().item()
        if rank == 0:
            assert diff == 0.0, f"{name} diverged across ranks: {diff}"
            assert trained > 1e-4, f"{name} did not train (frozen): {trained}"
            assert torch.isfinite(flat).all(), f"{name} produced non-finite parameters"
            print(f"OK {name} diff={diff} trained={trained:.2e}", flush=True)

    if rank == 0:
        print("FSDP2_COMPILED_STEP", flush=True)
    dist.destroy_process_group()


main()
'''


def test_real_compiled_step_under_fsdp2(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs for real compiled FSDP2")
    script = tmp_path / "fsdp2_compiled_step_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 2000
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK AdamW" in output, output
    assert "OK SOAP" in output, output
    assert "OK PSGDKron" in output, output
    assert "FSDP2_COMPILED_STEP" in output, output
