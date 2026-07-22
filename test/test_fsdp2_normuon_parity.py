"""NorMuon under FSDP2 -- the FIRST owner-whole state (moment2). NorMuon's reduced second moment plus
its Frobenius-preserving scale need the WHOLE matrix, so moment2 lives owner-whole (a [R,1]/[1,C]
vector), not row-sharded; momentum stays row-sharded.

Tests BOTH orientations, because they stress different things:
  - TALL: Linear(4,5) weight [5,4], R>=C -> moment2 [5,1] per-row. moment2 could in principle be
    row-sharded here, but the design owner-places it uniformly.
  - WIDE: Linear(5,3) weight [3,5], R<C -> moment2 [1,5] per-col, whose second moment is a mean across
    the FSDP-SHARDED row dimension. This is the case that is WRONG if moment2 stays sharded -- it
    forces the whole matrix. A build that kept moment2 sharded would pass tall and fail wide.

float64, same init + same data on every rank (FSDP2 averages grads -> the sharded grad equals the
single-process grad). Single-process parity < 1e-8. RED until the NorMuon owner-state build; that build
moves NorMuon out of test_fsdp2_scope_guard._REJECTED.
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


class Par(nn.Module):
    def __init__(self, cin, cout, n=3):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def parity(cin, cout, label, device):
    torch.manual_seed(0)
    ref = Par(cin, cout).double().to(device)
    ref_opt = heavyball.NorMuon(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par(cin, cout).double().to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = heavyball.NorMuon.fsdp2(fs, lr=2e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(6):
        X = torch.randn(8, cin * 3, dtype=torch.float64, device=device)
        tgt = torch.randn(8, cout, dtype=torch.float64, device=device)
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
    tall = parity(4, 5, "tall", device)   # weight [5,4], R>=C, moment2 [5,1]
    wide = parity(5, 3, "wide", device)   # weight [3,5], R<C,  moment2 [1,5] (reduces across sharded rows)
    if rank == 0:
        assert tall < 1e-8, f"NorMuon tall: {tall}"
        assert wide < 1e-8, f"NorMuon wide (moment2 must be owner-whole): {wide}"
        print(f"OK NorMuon parity tall={tall:.2e} wide={wide:.2e}", flush=True)
        print("FSDP2_NORMUON_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_normuon_fsdp2_matches_single_process(tmp_path):
    script = tmp_path / "normuon_parity_worker.py"
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
    assert "OK NorMuon parity" in output, output
    assert "FSDP2_NORMUON_PARITY" in output, output
