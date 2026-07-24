"""Cross-world DCP resharding -- the core of "build it out properly" for checkpointing. A checkpoint
saved on W ranks must load onto W' ranks (a DIFFERENT world size) with the logical model + optimizer
state preserved. Save on 2 ranks, load on 1, RESUME, and the params must match a single-process run of
the same total steps -- the FSDP2 parity oracles already prove W=2 training == single-process, so a
correct reshard-load-and-resume must land there too. The resume (not just a static param compare) is
what actually exercises optimizer-state resharding: a wrongly-resharded momentum/Gram diverges on
resume. Uses torch.distributed.checkpoint. float64. Two torchrun invocations sharing a checkpoint dir.
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

_COMMON = '''
import os
from unittest.mock import patch
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()
CKPT = sys.argv[1]
STEPS_BEFORE, STEPS_AFTER = 6, 4


def model(device):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 10, bias=False), nn.GELU(), nn.Linear(10, 6, bias=False)).double().to(device)


def data(step, device):
    g = torch.Generator().manual_seed(1000 + step)
    return (torch.randn(16, 8, dtype=torch.float64, generator=g).to(device),
            torch.randn(16, 6, dtype=torch.float64, generator=g).to(device))


def train(m, opt, lo, hi, device):
    for step in range(lo, hi):
        X, Y = data(step, device)
        ((m(X) - Y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()


def sharded(device):
    m = model(device)
    for layer in m:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(m)
    return m, heavyball.Muon.fsdp2(m, lr=1e-2)
'''

_SAVE = _COMMON + '''
def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    m, opt = sharded(device)
    train(m, opt, 0, STEPS_BEFORE, device)
    opt.dcp_save(CKPT)                         # NEW API -- codex builds this (DTensor state -> DCP)
    if rank == 0:
        print("SAVE_DONE", flush=True)
    dist.destroy_process_group()

main()
'''

_LOAD = _COMMON + '''
def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    m, opt = sharded(device)                    # world size 1 here
    opt.dcp_load(CKPT)                          # reshard W=2 -> W=1
    train(m, opt, STEPS_BEFORE, STEPS_BEFORE + STEPS_AFTER, device)
    loaded = {n: p.full_tensor().cpu() for n, p in m.named_parameters()}

    # single-process reference: train STEPS_BEFORE + STEPS_AFTER continuously
    ref = model(device)
    ref_opt = heavyball.Muon(ref.parameters(), lr=1e-2)
    for step in range(STEPS_BEFORE + STEPS_AFTER):
        X, Y = data(step, device)
        ((ref(X) - Y) ** 2).mean().backward()
        ref_opt.step()
        ref_opt.zero_grad()
    reference = {n: p.detach().cpu() for n, p in ref.named_parameters()}

    err = max((loaded[n] - reference[n]).abs().max().item() for n in reference)
    assert err < 1e-8, f"cross-world reshard+resume != single-process: {err}"
    print(f"OK DCP reshard+resume err={err:.2e}", flush=True)
    print("FSDP2_DCP_RESHARD", flush=True)
    dist.destroy_process_group()

main()
'''


def _run(script_text, nproc, ckpt, tmp_path, name, port):
    script = tmp_path / name
    script.write_text(script_text)
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(script), ckpt],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=300,
    )


def test_dcp_cross_world_resharding(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    base = 29500 + os.getpid() % 1500
    save = _run(_SAVE, 2, ckpt, tmp_path, "dcp_save.py", base)
    assert save.returncode == 0 and "SAVE_DONE" in save.stdout + save.stderr, save.stdout + save.stderr
    load = _run(_LOAD, 1, ckpt, tmp_path, "dcp_load.py", base + 1)
    out = load.stdout + load.stderr
    assert load.returncode == 0, out
    assert "OK DCP reshard+resume" in out, out
    assert "FSDP2_DCP_RESHARD" in out, out
