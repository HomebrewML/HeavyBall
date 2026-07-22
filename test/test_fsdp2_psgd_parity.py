"""PSGD-Kron under FSDP2 -- the validation of the RNG service's LAYOUT-CORRECTNESS. PSGD's whitening
probe is a random Gaussian (torch.randn_like, kron.py). For FSDP2 parity the owner's probe for leaf i
must equal single-process's probe for leaf i, which is only true if the probe draws from the
layout-correct Tempo RNG (seeded per-leaf + step), not the ambient RNG. The Kron factors Q0/Q1 are
owner-whole (like SOAP's Grams). This test is the whole reason the RNG service is layout-correct rather
than merely rank-consistent.

Both orientations, float64, same init + same data on every rank, 12 steps (crosses a preconditioner
refresh where the probe is drawn), single-process parity < 1e-8. RED until PSGD declares WHOLE and its
probe is wired to Tempo.randn_like.
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
    ref_opt = heavyball.PSGDKron(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par(cin, cout).double().to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = heavyball.PSGDKron.fsdp2(fs, lr=2e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(12):
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
    tall = parity(4, 5, "tall", device)
    wide = parity(5, 3, "wide", device)
    if rank == 0:
        assert tall < 1e-8, f"PSGD tall: {tall}"
        assert wide < 1e-8, f"PSGD wide (probe must be layout-correct): {wide}"
        print(f"OK PSGD parity tall={tall:.2e} wide={wide:.2e}", flush=True)
        print("FSDP2_PSGD_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_psgd_fsdp2_matches_single_process(tmp_path):
    script = tmp_path / "psgd_parity_worker.py"
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
    assert "OK PSGD parity" in output, output
    assert "FSDP2_PSGD_PARITY" in output, output
