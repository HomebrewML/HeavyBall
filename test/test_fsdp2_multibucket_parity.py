"""Multi-bucket regroup: a model with SEVERAL parameter shapes exercises multiple mega-batch buckets
(the regroup groups params by shape, so each distinct shape is its own all_to_all). The single-shape
parity tests (3x Linear(4,5)) never hit this. Muon.fsdp2 on a 3-shape MLP with uneven dim-0 shards
(15x8, 33x15, 11x33 -- all odd dim0) must still match single-process Muon on every parameter.
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


def mk(device):
    return nn.Sequential(
        nn.Linear(8, 15, bias=False), nn.ReLU(),
        nn.Linear(15, 33, bias=False), nn.ReLU(),
        nn.Linear(33, 11, bias=False),
    ).double().to(device)


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    torch.manual_seed(0)
    ref = mk(device)
    ref_opt = heavyball.Muon(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = mk(device)
    for m in fs:
        if isinstance(m, nn.Linear):
            fully_shard(m)
    fully_shard(fs)
    fs_opt = heavyball.Muon.fsdp2(fs, lr=2e-2)
    torch.manual_seed(42)
    err = 0.0
    for _ in range(5):
        X = torch.randn(16, 8, dtype=torch.float64, device=device)
        tgt = torch.randn(16, 11, dtype=torch.float64, device=device)
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
    if rank == 0:
        assert err < 1e-8, f"multi-bucket Muon.fsdp2 != single-process: {err}"
        print(f"OK multibucket parity err={err:.2e}", flush=True)
        print("FSDP2_MULTIBUCKET_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_multibucket_muon_matches_single_process(tmp_path):
    script = tmp_path / "multibucket_worker.py"
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
    assert "OK multibucket parity" in output, output
    assert "FSDP2_MULTIBUCKET_PARITY" in output, output
