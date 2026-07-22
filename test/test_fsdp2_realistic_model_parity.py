"""Realistic-model FSDP2 parity -- the coverage that toy Par/Sequential oracles miss. A block-model with
LayerNorm (1D params), biases (1D), and unique-shape matrix weights (1-leaf buckets = the empty-owner
case) exercises the routing (2D -> matrix op, 1D -> AdamW fallback), the empty-owner 0-leaf bypass, and
multiple parameter shapes all at once -- exactly the structure a real model has and the toy oracles do
not. AdamW (componentwise), Muon (whole-matrix), SOAP + PSGDKron (owner-whole) must all match single-
process. float64, same data. This is the oracle that would have caught the empty-owner bug directly.
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
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1 = nn.Linear(d, h); self.ln1 = nn.LayerNorm(h)
        self.l2 = nn.Linear(h, d); self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln2(self.l2(F.gelu(self.ln1(self.l1(x)))))


def model(device):
    torch.manual_seed(0)
    return nn.Sequential(Block(8, 20), Block(8, 12)).double().to(device)


def shard(m):
    for block in m:
        fully_shard(block)
    fully_shard(m)
    return m


def parity(Opt, device):
    torch.manual_seed(0)
    ref = model(device); ref_opt = Opt(ref.parameters(), lr=1e-2)
    torch.manual_seed(0)
    fs = shard(model(device)); fs_opt = Opt.fsdp2(fs, lr=1e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(6):
        X = torch.randn(16, 8, dtype=torch.float64, device=device); tgt = torch.randn(16, 8, dtype=torch.float64, device=device)
        ((ref(X) - tgt) ** 2).mean().backward(); ref_opt.step(); ref_opt.zero_grad()
        ((fs(X) - tgt) ** 2).mean().backward(); fs_opt.step(); fs_opt.zero_grad()
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
    results = {name: parity(getattr(heavyball, name), device)
               for name in ("AdamW", "Muon", "SOAP", "PSGDKron")}
    if rank == 0:
        for name, err in results.items():
            assert err < 1e-8, f"{name} realistic-model: {err}"
        print(f"OK realistic-model parity {{{', '.join(f'{k}={v:.0e}' for k, v in results.items())}}}", flush=True)
        print("FSDP2_REALISTIC_MODEL_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_realistic_block_model_matches_single_process(tmp_path):
    script = tmp_path / "realistic_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 1500
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK realistic-model parity" in output, output
    assert "FSDP2_REALISTIC_MODEL_PARITY" in output, output
