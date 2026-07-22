"""Matrix-family gradient sync under FSDP2. The parity tests use SAME data on both ranks (trivial grad
sync, exact float64 parity), so the reduce-scatter gradient synchronization COMPOSED WITH the matrix
regroup is only exercised for the componentwise family (test_fsdp2_consistency uses AdamW). This pins it
for the MATRIX family: Muon.fsdp2 with DIFFERENT data per rank must produce a cross-rank BIT-IDENTICAL
full parameter (only possible if the reduce-scattered gradient reaches the whole-matrix regroup), plus
train and stay finite. fp32 (the default deployment dtype, not the float64 the parity tests use).
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
    model = nn.Sequential(nn.Linear(8, 15, bias=False), nn.ReLU(), nn.Linear(15, 10, bias=False)).to(device)
    for m in model:
        if isinstance(m, nn.Linear):
            fully_shard(m)
    fully_shard(model)
    opt = heavyball.Muon.fsdp2(model, lr=1e-2)
    init = torch.cat([p.full_tensor().flatten() for p in model.parameters()]).clone()
    torch.manual_seed(100 + rank)  # DIFFERENT data per rank -> exercises the reduce-scatter grad sync
    for _ in range(8):
        X = torch.randn(16, 8, device=device)
        Y = torch.randn(16, 10, device=device)
        ((model(X) - Y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    flat = torch.cat([p.full_tensor().flatten() for p in model.parameters()])
    diff = cross_rank_max(flat)
    trained = (flat - init).abs().max().item()
    if rank == 0:
        assert diff == 0.0, f"matrix full param diverged across ranks: {diff}"
        assert trained > 1e-4, f"did not train: {trained}"
        assert bool(torch.isfinite(flat).all()), "non-finite params"
        print(f"OK Muon gradsync diff={diff} trained={trained:.3e}", flush=True)
        print("FSDP2_MATRIX_GRADSYNC", flush=True)
    dist.destroy_process_group()


main()
'''


def test_matrix_family_gradient_sync_is_cross_rank_consistent(tmp_path):
    script = tmp_path / "gradsync_worker.py"
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
    assert "OK Muon gradsync" in output, output
    assert "FSDP2_MATRIX_GRADSYNC" in output, output
