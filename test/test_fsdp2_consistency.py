"""FSDP2 consistency: shard-separable optimizers train correctly under fully_shard.

The .fsdp2(model) adapter binds a HeavyBall slab into FSDP2's sharded storage: param.grad is a
prebound DTensor.from_local over the grad slab (FSDP2's reduce-scatter accumulates in place), and the
param's sharded storage is repointed at the param slab (the compiled step writes it; the next
all-gather reads it). This gates the guarantee: under fully_shard, with DIFFERENT data per rank,
HeavyBall's AdamW must (a) train and (b) produce a cross-rank BIT-IDENTICAL full parameter -- only
possible if the reduce-scattered gradients reach the slab. Two ranks, eager (the grad/param flow is
compile-independent, and it avoids CPU multi-process autotune flakiness), mirroring
test_ddp_consistency. Mechanism independently verified at world_size=2 before this oracle was written.
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
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


def cross_rank_max(t):
    ref = t.clone()
    dist.broadcast(ref, src=0)
    d = (t - ref).abs().max()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return d.item()


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    torch.manual_seed(0)
    model = nn.Linear(8, 6, bias=False).to(device)
    fully_shard(model)
    opt = heavyball.AdamW.fsdp2(model, lr=1e-2)
    init = model.weight.full_tensor().flatten().clone()
    torch.manual_seed(100 + rank)  # different data per rank
    for _ in range(8):
        X, Y = torch.randn(16, 8, device=device), torch.randn(16, 6, device=device)
        ((model(X) - Y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    flat = model.weight.full_tensor().flatten()
    diff = cross_rank_max(flat)
    trained = (flat - init).abs().max().item()
    if rank == 0:
        assert diff == 0.0, f"AdamW diverged across ranks: {diff}"
        assert trained > 1e-4, f"AdamW did not train (frozen): {trained}"
        print(f"OK AdamW diff={diff} trained={trained:.3e}", flush=True)
        print("FSDP2_CONSISTENT", flush=True)
    dist.destroy_process_group()


main()
'''


def test_shard_separable_optimizer_is_consistent_across_fsdp2_ranks(tmp_path):
    script = tmp_path / "fsdp2_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 2000
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK AdamW" in output, output
    assert "FSDP2_CONSISTENT" in output, output
