"""HSDP replicas must share one optimizer seed, so stochastic steps stay bit-identical across replicas.

The Engine seed is broadcast over the shard subgroup; under a 2D replicate x shard HSDP mesh that agrees
only within one replicate row. Without a replicate-axis broadcast, replicas that started from different
ambient seeds draw different stochastic-rounding noise (Muon rounds its orthogonalized direction to bf16)
and drift apart, even though HSDP keeps their gradients synced. This pins the replicate-axis seed
broadcast: DIFFERENT per-rank ambient seeds + stochastic Muon must still leave corresponding shards
bit-identical across replicas. (test_fsdp2_hsdp_parity uses identical seeds + fp64 and cannot catch this.)
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or shutil.which("torchrun") is None
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 4,
    reason="requires torchrun and a 4-GPU HSDP mesh",
)

_WORKER = '''
import torch, torch.distributed as dist, torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
import heavyball


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    torch.manual_seed(0)                       # identical initial weights on every rank
    model = nn.Linear(32, 32, bias=False).cuda()
    fully_shard(model, mesh=mesh)
    torch.manual_seed(1000 + rank)             # DIFFERENT ambient seed per rank before construction
    opt = heavyball.Muon.fsdp2(model, lr=0.1)  # Muon stochastically rounds its direction to bf16
    torch.manual_seed(42)                      # identical data on every rank
    x = torch.randn(16, 32, device="cuda"); y = torch.randn(16, 32, device="cuda")
    for _ in range(3):
        opt.zero_grad(); ((model(x) - y) ** 2).mean().backward(); opt.step()
    local = model.weight.to_local().detach().contiguous()
    gathered = [torch.zeros_like(local) for _ in range(4)]
    dist.all_gather(gathered, local)
    if rank == 0:
        # ranks: 0=(rep0,shard0) 1=(rep0,shard1) 2=(rep1,shard0) 3=(rep1,shard1)
        d0 = (gathered[0] - gathered[2]).abs().max().item()
        d1 = (gathered[1] - gathered[3]).abs().max().item()
        assert d0 == 0.0 and d1 == 0.0, f"HSDP replicas diverged: shard0={d0:.3e} shard1={d1:.3e}"
        print("HSDP_SEED_CONSISTENT", flush=True)
    dist.destroy_process_group()


main()
'''


def test_hsdp_replicas_share_one_seed(tmp_path):
    script = tmp_path / "hsdp_seed_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 2000
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=4", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "HSDP_SEED_CONSISTENT" in output, output
