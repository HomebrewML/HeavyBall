"""HSDP (2D mesh: replicate x shard) under FSDP2 -- the common multi-node config. The regroup must run
on the SHARD sub-mesh only (gather the sharded rows within the shard group); the replicate dim is
handled by FSDP2's gradient sync (reduce-scatter within shard + all-reduce across replicate), so the
optimizer sees the fully-synced grad and a whole-matrix op on the shard sub-mesh matches single-process.
4 processes = 2 replicate x 2 shard. Muon.fsdp2 (whole-matrix) + SOAP (owner-whole state) on the 2D
mesh must match single-process. float64, same data on every rank. RED until the mesh guard accepts a 2D
shard+replicate mesh and the regroup + owner-state use mesh["shard"].
Run: torchrun --nproc_per_node=4 <worker>
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
from torch.distributed.device_mesh import init_device_mesh
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


class Par(nn.Module):
    def __init__(self, cin, cout, n=3):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def parity(Opt, cin, cout, device):
    mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("replicate", "shard"))
    torch.manual_seed(0)
    ref = Par(cin, cout).double().to(device)
    ref_opt = Opt(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par(cin, cout).double().to(device)
    for layer in fs.ls:
        fully_shard(layer, mesh=mesh)
    fully_shard(fs, mesh=mesh)
    fs_opt = Opt.fsdp2(fs, lr=2e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(8):
        X = torch.randn(8, cin * 3, dtype=torch.float64, device=device)
        tgt = torch.randn(8, cout, dtype=torch.float64, device=device)
        ((ref(X) - tgt) ** 2).mean().backward()
        ref_opt.step(); ref_opt.zero_grad()
        ((fs(X) - tgt) ** 2).mean().backward()
        fs_opt.step(); fs_opt.zero_grad()
        err = max(err, max(
            (r.detach() - f.full_tensor()).abs().max().item()
            for r, f in zip(ref.parameters(), fs.parameters())
        ))
    return err


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    muon = parity(heavyball.Muon, 4, 5, device)
    soap = parity(heavyball.SOAP, 4, 5, device)
    if rank == 0:
        assert muon < 1e-8, f"HSDP Muon: {muon}"
        assert soap < 1e-8, f"HSDP SOAP (owner-whole on shard sub-mesh): {soap}"
        print(f"OK HSDP parity muon={muon:.2e} soap={soap:.2e}", flush=True)
        print("FSDP2_HSDP_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_hsdp_2d_mesh_matches_single_process(tmp_path):
    script = tmp_path / "hsdp_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 1500
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=4", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK HSDP parity" in output, output
    assert "FSDP2_HSDP_PARITY" in output, output
