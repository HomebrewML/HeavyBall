"""P1: the Newton-Schulz / matrix-direction family under FSDP2 must MATCH single-process on the whole
matrix, same as Muon. Each reuses the P0 regroup (momentum/state stays row-sharded; the direction
transform runs on the whole matrix). Oblique is the exception: it is shard-separable on Shard(0)
(tangent projection + retraction reduce only over the last dim), so it needs NO regroup and must still
match single-process.

Reuses the debugged parallel model (3x Linear(4,5) on disjoint slices) -- valid forward, distinct grad
per weight, N=3 param padding, R=5 uneven shard. float64, same init+data => single-process parity.
RED until the P1 build authorizes these facades; that build moves them out of _REJECTED.
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

_P1 = ["PolarGrad", "Aurora", "AdaMuon", "MuonLaProp", "Oblique"]  # NorMuon is P2 (owner state)


class Par(nn.Module):
    def __init__(self):
        super().__init__()
        self.ls = nn.ModuleList([nn.Linear(4, 5, bias=False) for _ in range(3)])

    def forward(self, x):
        return sum(layer(x[:, i * 4:(i + 1) * 4]) for i, layer in enumerate(self.ls))


def parity(facade, device):
    torch.manual_seed(0)
    ref = Par().double().to(device)
    ref_opt = facade(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par().double().to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = facade.fsdp2(fs, lr=2e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(5):
        X = torch.randn(8, 12, dtype=torch.float64, device=device)
        tgt = torch.randn(8, 5, dtype=torch.float64, device=device)
        ((ref(X) - tgt) ** 2).mean().backward()
        ref_opt.step()
        ref_opt.zero_grad()
        ((fs(X) - tgt) ** 2).mean().backward()
        fs_opt.step()
        fs_opt.zero_grad()
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
    tested = 0
    for name in _P1:
        facade = getattr(heavyball, name, None)
        if facade is None:
            continue
        err = parity(facade, device)
        tested += 1
        if rank == 0:
            assert err < 1e-8, f"{name}.fsdp2 != single-process {name}: {err}"
            print(f"OK {name} parity err={err:.2e}", flush=True)
    if rank == 0:
        assert tested > 0, "no P1 facades were exercised"
        print("FSDP2_DIRECTION_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_matrix_direction_family_matches_single_process(tmp_path):
    script = tmp_path / "direction_parity_worker.py"
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
    assert "FSDP2_DIRECTION_PARITY" in output, output
