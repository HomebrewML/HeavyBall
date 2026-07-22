"""Empty-owner / unique-shape-param coverage. A model with a UNIQUE-shape matrix param is a 1-leaf
bucket, so under FSDP2 W>=2 a non-owner rank owns 0 leaves of that bucket. Owner-whole optimizers must
handle that 0-leaf owner (the merged reshape + the whole segment run on an empty batch). Every prior
owner-whole oracle uses the Par model (a single 3-leaf bucket -> every rank owns >=1 leaf), so this path
was never exercised -- and SOAP crashes on it (matrix.py merged: reshape(0, R, -1), -1 ambiguous).

Model: a Sequential of TWO unique-shape Linears (each its own 1-leaf bucket). Muon is the control (its
whole segment is already 0-leaf-safe); SOAP, NorMuon, PSGDKron are the owner-whole optimizers that must
also work. float64, same data on every rank, match single-process < 1e-8.
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


def model(device):
    torch.manual_seed(0)
    # two UNIQUE matrix shapes -> two 1-leaf buckets -> a non-owner rank gets 0 leaves on W=2
    return nn.Sequential(nn.Linear(8, 10, bias=False), nn.GELU(), nn.Linear(10, 6, bias=False)).double().to(device)


def parity(Opt, name, device):
    torch.manual_seed(0)
    ref = model(device)
    ref_opt = Opt(ref.parameters(), lr=1e-2)
    torch.manual_seed(0)
    fs = model(device)
    for layer in fs:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(fs)
    fs_opt = Opt.fsdp2(fs, lr=1e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(8):
        X = torch.randn(16, 8, dtype=torch.float64, device=device)
        tgt = torch.randn(16, 6, dtype=torch.float64, device=device)
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
    results = {}
    for name, Opt in (("Muon", heavyball.Muon), ("SOAP", heavyball.SOAP),
                      ("NorMuon", heavyball.NorMuon), ("PSGDKron", heavyball.PSGDKron)):
        results[name] = parity(Opt, name, device)
    if rank == 0:
        for name, err in results.items():
            assert err < 1e-8, f"{name} empty-owner (unique-shape param): {err}"
        print(f"OK uneven-bucket parity {{{', '.join(f'{k}={v:.1e}' for k, v in results.items())}}}", flush=True)
        print("FSDP2_UNEVEN_BUCKET_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_uneven_bucket_owner_whole_matches_single_process(tmp_path):
    script = tmp_path / "uneven_bucket_worker.py"
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
    assert "OK uneven-bucket parity" in output, output
    assert "FSDP2_UNEVEN_BUCKET_PARITY" in output, output
