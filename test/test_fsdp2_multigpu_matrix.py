"""Multi-GPU nccl parity matrix -- the real-hardware guard for owner-whole optimizers across world sizes.
Runs at ws = torch.cuda.device_count() (skips below 2 GPUs). SOAP/NorMuon/PSGDKron x {plain, ecc=8} x
{even, uneven leaf counts} must match single-process. The gloo/CPU oracles guard the topology-dependent
bug class in ordinary CI; this guards the actual nccl/CUDA path (real collectives, CUDA numerics, real
sharding) and needs a multi-GPU CI instance. Non-ecc is fp64 bit-exact; ecc step-parity is fp32 (the
multi-way reduce-scatter reorders vs a single process), tol 1e-4.
Run: torchrun --nproc_per_node=<num_gpus> <worker>
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
_HAVE = NGPU >= 2 and shutil.which("torchrun") is not None
# HB_REQUIRE_MULTIGPU=1 (set by ci/gpu_tests.py for the multi-GPU lane) turns a missing-hardware skip
# into a hard failure so a misprovisioned instance cannot report green.
_REQUIRE = os.environ.get("HB_REQUIRE_MULTIGPU") == "1"
pytestmark = pytest.mark.skipif(
    not _HAVE and not _REQUIRE,
    reason="requires >=2 CUDA GPUs and torchrun",
)

_WORKER = '''
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


class Par(nn.Module):
    def __init__(self, cin, cout, n):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def parity(name, ecc, n, dev):
    dt = torch.float32 if ecc else torch.float64
    kw = {"ecc": 8} if ecc else {}
    Opt = getattr(heavyball, name)
    torch.manual_seed(0)
    ref = Par(4, 5, n).to(dt).to(dev)
    ref_opt = Opt(ref.parameters(), lr=2e-2, **kw)
    torch.manual_seed(0)
    fs = Par(4, 5, n).to(dt).to(dev)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = Opt.fsdp2(fs, lr=2e-2, **kw)
    err = torch.zeros((), device=dev, dtype=dt)
    for step in range(6):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, 4 * n, dtype=dt, generator=g).to(dev)
        Y = torch.randn(8, 5, dtype=dt, generator=g).to(dev)
        ((ref(X) - Y) ** 2).mean().backward(); ref_opt.step(); ref_opt.zero_grad()
        ((fs(X) - Y) ** 2).mean().backward(); fs_opt.step(); fs_opt.zero_grad()
        for r, f in zip(ref.parameters(), fs.parameters()):
            err = torch.maximum(err, (r.detach() - f.full_tensor()).abs().max())
    return err.item()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    results = {}
    for name in ("SOAP", "NorMuon", "PSGDKron"):
        for ecc in (None, 8):
            for n in (ws, ws + 1):  # even, uneven on this world size
                results[f"{name}/ecc{ecc}/n{n}"] = parity(name, ecc, n, dev)
    if rank == 0:
        for key, err in results.items():
            tol = 1e-4 if "ecc8" in key else 1e-8
            assert err < tol, f"{key} ws={ws} nccl != single-process: {err}"
        print(f"OK multigpu matrix ws={ws} ({len(results)} cells, max={max(results.values()):.0e})", flush=True)
        print("FSDP2_MULTIGPU_MATRIX", flush=True)
    dist.destroy_process_group()


main()
'''


def test_multigpu_nccl_matrix(tmp_path):
    if not _HAVE:
        raise AssertionError(
            f"HB_REQUIRE_MULTIGPU=1 but the multi-GPU lane is misprovisioned "
            f"(CUDA GPUs={NGPU}, torchrun={'present' if shutil.which('torchrun') else 'absent'}); "
            "the required lane must run the real nccl matrix, not skip."
        )
    script = tmp_path / "multigpu_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 1500
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={NGPU}", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK multigpu matrix" in output, output
    assert "FSDP2_MULTIGPU_MATRIX" in output, output
