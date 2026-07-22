"""DDP stochastic-op consistency. Muon's orthogonalize stochastically rounds to bf16 (transforms.py
stochastic_round_bfloat16) drawing from the AMBIENT RNG; under DDP each rank's RNG diverges (different
data advances it differently), so the params silently DRIFT across ranks even though DDP keeps grads
bit-synced. This pins the fix: DDP + Muon (fp32, DIFFERENT data per rank) must keep the full parameter
BIT-IDENTICAL across ranks (only possible with a rank-consistent optimizer RNG), plus train and stay
finite. RED until the RNG service -- the drift is verified at 4.7e-4 over 8 steps on 2-GPU NCCL.
Run: torchrun --nproc_per_node=2 <worker>
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
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()  # eager; drift is in the RNG, not compile


def cross_rank_max(t):
    ref = t.clone()
    dist.broadcast(ref, src=0)
    d = (t - ref).abs().max()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return d.item()


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 15, bias=False), nn.ReLU(), nn.Linear(15, 10, bias=False))
    ddp = DDP(model)
    opt = heavyball.Muon(model.parameters(), lr=1e-2)  # fp32 -> stochastic bf16 rounding path
    init = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    torch.manual_seed(100 + rank)  # DIFFERENT data per rank -> ambient RNG diverges across ranks
    for _ in range(8):
        X = torch.randn(16, 8)
        Y = torch.randn(16, 10)
        ((ddp(X) - Y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    diff = cross_rank_max(flat)
    trained = (flat - init).abs().max().item()
    if rank == 0:
        assert diff == 0.0, f"DDP Muon params drifted across ranks (needs rank-consistent RNG): {diff}"
        assert trained > 1e-4, f"did not train: {trained}"
        assert bool(torch.isfinite(flat).all()), "non-finite params"
        print(f"OK DDP Muon consistency diff={diff} trained={trained:.3e}", flush=True)
        print("DDP_STOCHASTIC_CONSISTENCY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_ddp_muon_params_stay_bit_synced(tmp_path):
    script = tmp_path / "ddp_consistency_worker.py"
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
    assert "OK DDP Muon consistency" in output, output
    assert "DDP_STOCHASTIC_CONSISTENCY" in output, output
