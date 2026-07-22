"""Scion under FSDP2 must match single-process Scion on the whole leaf. Scion is not transform-only:
its LMO acts on the whole leaf, AND scion_param_init orthogonally-initializes the whole parameter at
construction via a seed-indexed Generator (deterministic given a per-parameter seed, NOT the global
RNG). The FSDP2 build needs a group-init hook that regroups the initial parameter, calls
scion_param_init once per whole OWNED leaf with the same per-param seed, and scatters it back -- else
the init runs on the dim-0 shard, and orth-init(shard) != slice(orth-init(whole)), so the trajectory
diverges from step 0. This parity oracle catches exactly that.

Reuses the debugged parallel model (3x Linear(4,5) disjoint slices) -- N=3 padding, R=5 uneven shard.
float64, same init+data. RED until the Scion build; that build moves Scion out of _REJECTED.
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
    def __init__(self):
        super().__init__()
        self.ls = nn.ModuleList([nn.Linear(4, 5, bias=False) for _ in range(3)])

    def forward(self, x):
        return sum(layer(x[:, i * 4:(i + 1) * 4]) for i, layer in enumerate(self.ls))


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    torch.manual_seed(0)
    ref = Par().double().to(device)
    ref_opt = heavyball.Scion(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par().double().to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = heavyball.Scion.fsdp2(fs, lr=2e-2)
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
    if rank == 0:
        assert err < 1e-8, f"Scion.fsdp2 != single-process Scion: {err}"
        print(f"OK Scion parity err={err:.2e}", flush=True)
        print("FSDP2_SCION_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_scion_fsdp2_matches_single_process(tmp_path):
    script = tmp_path / "scion_parity_worker.py"
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
    assert "OK Scion parity" in output, output
    assert "FSDP2_SCION_PARITY" in output, output
