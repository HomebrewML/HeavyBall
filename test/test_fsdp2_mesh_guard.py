"""FSDP2 mesh guard: accept 2D replicate x shard HSDP, while unsupported layouts still fail closed.

The accepted HSDP layout shards parameters on dim 0 along mesh["shard"]. A named 2D mesh without the
required shard dimension must be rejected cleanly, never silently run optimizer collectives globally.
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
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
import heavyball


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    hsdp_mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("replicate", "shard"))
    supported = nn.Linear(8, 8, bias=False)
    fully_shard(supported, mesh=hsdp_mesh)
    heavyball.AdamW.fsdp2(supported, lr=1e-2)

    unsupported_mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("replicate", "other"))
    unsupported = nn.Linear(8, 8, bias=False)
    fully_shard(unsupported, mesh=unsupported_mesh)
    rejected = False
    try:
        heavyball.AdamW.fsdp2(unsupported, lr=1e-2)
    except ValueError as error:
        rejected = "replicate x shard HSDP mesh" in str(error)
    if rank == 0:
        assert rejected, "a 2D mesh without a shard dimension was not rejected by the mesh guard"
        print("HSDP_2D_MESH_ACCEPTED", flush=True)
        print("UNSUPPORTED_2D_MESH_REJECTED", flush=True)
    dist.destroy_process_group()


main()
'''


def test_hsdp_2d_mesh_is_accepted_and_unsupported_mesh_is_rejected(tmp_path):
    script = tmp_path / "mesh_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 2000
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "HSDP_2D_MESH_ACCEPTED" in output, output
    assert "UNSUPPORTED_2D_MESH_REJECTED" in output, output
