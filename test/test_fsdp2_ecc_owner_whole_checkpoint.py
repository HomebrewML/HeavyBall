"""Same-world state_dict/load_state_dict round-trip for OWNER-WHOLE optimizers WITH ecc=8. ECC stores
state as a bfloat16 slab plus a separate int8 correction DTensor; this pins that the owner-whole
correction survives the common opt.state_dict() path (distinct from both the non-ecc owner-whole
checkpoint oracle, which has no correction, and the DCP+ECC oracle, which only covers shard-separable
Muon). fp32 (ecc is a float32 codec; fp64+ecc is nonsensical), NaN-propagating so non-finite fails.
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


def build(Opt, device):
    torch.manual_seed(0)
    m = Par(4, 5).to(device)
    for layer in m.ls:
        fully_shard(layer)
    fully_shard(m)
    return m, Opt.fsdp2(m, lr=2e-2, ecc=8)


def train(m, opt, lo, hi, device):
    for step in range(lo, hi):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, 12, generator=g).to(device)
        Y = torch.randn(8, 5, generator=g).to(device)
        ((m(X) - Y) ** 2).mean().backward()
        opt.step(); opt.zero_grad()


def roundtrip(Opt, device):
    base, base_opt = build(Opt, device); train(base, base_opt, 0, 12, device)
    inter, inter_opt = build(Opt, device); train(inter, inter_opt, 0, 8, device)
    model_ck = {k: v.clone() for k, v in inter.state_dict().items()}
    opt_ck = inter_opt.state_dict()
    res, res_opt = build(Opt, device)
    res.load_state_dict(model_ck); res_opt.load_state_dict(opt_ck)
    train(res, res_opt, 8, 12, device)
    err = torch.zeros((), device=device)
    for b, r in zip(base.parameters(), res.parameters()):
        err = torch.maximum(err, (b.full_tensor() - r.full_tensor()).abs().max())
    return err.item()


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    results = {name: roundtrip(getattr(heavyball, name), device)
               for name in ("SOAP", "NorMuon", "PSGDKron")}
    # SOAP GG^T reduces layout-dependently on CUDA/cuBLAS ~fp32 eps, int8-ECC-amplified to ~1e-6;
    # NorMuon/PSGDKron bit-exact. CPU matmul deterministic -> bit-exact. See test_fsdp2_ecc_owner_whole.
    tol = 1e-4 if cuda else 1e-8
    if rank == 0:
        for name, err in results.items():
            assert err < tol, f"{name} owner-whole ecc=8 checkpoint: {err} (tol {tol})"
        print(f"OK owner-whole ecc checkpoint {{{', '.join(f'{k}={v:.0e}' for k, v in results.items())}}}", flush=True)
        print("FSDP2_ECC_OWNER_WHOLE_CHECKPOINT", flush=True)
    dist.destroy_process_group()


main()
'''


def test_ecc_owner_whole_same_world_checkpoint(tmp_path):
    script = tmp_path / "ecc_ow_ckpt_worker.py"
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
    assert "OK owner-whole ecc checkpoint" in output, output
    assert "FSDP2_ECC_OWNER_WHOLE_CHECKPOINT" in output, output


_DCP_WORKER = '''
import os
from unittest.mock import patch
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()
MODE, CKPT = sys.argv[1], sys.argv[2]
BEFORE, AFTER = 6, 4


class Par(nn.Module):
    def __init__(self, cin, cout, n=3):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def data(step, device):
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(8, 12, generator=g).to(device), torch.randn(8, 5, generator=g).to(device)


def train(m, opt, lo, hi, device):
    for step in range(lo, hi):
        X, Y = data(step, device)
        ((m(X) - Y) ** 2).mean().backward()
        opt.step(); opt.zero_grad()


def sharded(device):
    torch.manual_seed(0)
    m = Par(4, 5).to(device)
    for layer in m.ls:
        fully_shard(layer)
    fully_shard(m)
    return m, heavyball.SOAP.fsdp2(m, lr=2e-2, ecc=8)


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    if MODE == "save":
        m, opt = sharded(device); train(m, opt, 0, BEFORE, device); opt.dcp_save(CKPT)
        if rank == 0:
            print("SAVE_DONE", flush=True)
    else:
        m, opt = sharded(device); opt.dcp_load(CKPT); train(m, opt, BEFORE, BEFORE + AFTER, device)
        loaded = {n: p.full_tensor().cpu() for n, p in m.named_parameters()}
        torch.manual_seed(0)
        ref = Par(4, 5).to(device); ref_opt = heavyball.SOAP(ref.parameters(), lr=2e-2, ecc=8)
        for step in range(BEFORE + AFTER):
            X, Y = data(step, device)
            ((ref(X) - Y) ** 2).mean().backward()
            ref_opt.step(); ref_opt.zero_grad()
        err = torch.zeros(())
        for n, p in ref.named_parameters():
            err = torch.maximum(err, (loaded[n] - p.detach().cpu()).abs().max())
        # SOAP GG^T reduces layout-dependently on CUDA/cuBLAS ~fp32 eps, int8-ECC-amplified to ~1e-6
        # (observed 1.2e-6); CPU matmul deterministic -> bit-exact. See test_fsdp2_ecc_owner_whole.
        tol = 1e-4 if cuda else 1e-8
        assert err.item() < tol, f"owner-whole ecc DCP reshard != single-process: {err.item()} (tol {tol})"
        print(f"OK owner-whole ecc DCP reshard err={err.item():.0e}", flush=True)
        print("FSDP2_ECC_OWNER_WHOLE_DCP", flush=True)
    dist.destroy_process_group()


main()
'''


def test_ecc_owner_whole_dcp_cross_world(tmp_path):
    """W=2 save -> W=1 load -> resume, matching single-process: the owner-whole ecc correction reshards
    through the distinct dcp_save/dcp_load path (the DCP+ECC oracle only covers shard-separable Muon)."""
    script = tmp_path / "ecc_ow_dcp_worker.py"
    script.write_text(_DCP_WORKER)
    ckpt = str(tmp_path / "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    base = 29500 + os.getpid() % 1500

    def run(nproc, mode, port):
        return subprocess.run(
            ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(script), mode, ckpt],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=300,
        )

    save = run(2, "save", base)
    assert save.returncode == 0 and "SAVE_DONE" in save.stdout + save.stderr, save.stdout + save.stderr
    load = run(1, "load", base + 1)
    out = load.stdout + load.stderr
    assert load.returncode == 0, out
    assert "OK owner-whole ecc DCP reshard" in out, out
    assert "FSDP2_ECC_OWNER_WHOLE_DCP" in out, out
