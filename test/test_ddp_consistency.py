"""DDP consistency: RNG-free optimizers produce bit-identical models across ranks.

HeavyBall binds ``param.grad`` into a slab view (core.py); DDP's reducer writes the all-reduced
gradient back into that view, so ``step()`` reads synchronized gradients. This gates that guarantee:
with DIFFERENT data per rank, AdamW and SOAP must produce bit-identical parameters across two ranks
(only possible if gradients are synchronized), and must actually train (non-vacuous). Runs eager --
the grad-flow guarantee is compile-independent, and it avoids CPU multi-process autotune flakiness.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or shutil.which("torchrun") is None,
    reason="requires torch.distributed and torchrun",
)

_WORKER = '''
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


def cross_rank_max(t):
    ref = t.clone()
    dist.broadcast(ref, src=0)
    d = (t - ref).abs().max()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return d.item()


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    for name in ("AdamW", "SOAP"):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        init = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        opt = getattr(heavyball, name)(model.parameters(), lr=1e-2)
        ddp = DDP(model)
        torch.manual_seed(100 + rank)  # different data per rank
        for _ in range(8):
            X, Y = torch.randn(16, 8), torch.randn(16, 4)
            (((ddp(X) - Y) ** 2).mean()).backward()
            opt.step()
            opt.zero_grad()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        diff = cross_rank_max(flat)
        trained = (flat - init).abs().max().item()
        if rank == 0:
            assert diff == 0.0, f"{name} diverged across ranks: {diff}"
            assert trained > 1e-4, f"{name} did not train (frozen): {trained}"
            print(f"OK {name} diff={diff} trained={trained:.3e}", flush=True)
    if rank == 0:
        print("DDP_CONSISTENT", flush=True)
    dist.destroy_process_group()


main()
'''


def test_rng_free_optimizers_are_bit_identical_across_ddp_ranks(tmp_path):
    script = tmp_path / "ddp_worker.py"
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
    assert "OK SOAP" in output, output
    assert "DDP_CONSISTENT" in output, output
