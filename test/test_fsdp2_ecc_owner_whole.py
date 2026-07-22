"""ECC (low-precision state) under FSDP2 for OWNER-WHOLE optimizers, at fp32 -- the real ECC use case.
Single-process SOAP/NorMuon/PSGDKron with ecc=8 are finite and stable; the FSDP2 owner-whole path with
ecc=8 went non-finite. Since (a) the fp32 owner-whole mechanism is bit-exact vs single-process and (b)
shard-separable Muon ecc=8 is bit-exact vs single-process (the seeded ECC rounding aligns by leaf index),
a fixed FSDP2 owner-whole ecc=8 must match single-process ecc=8 bit-exactly. float32, same data on every
rank.
(fp64 + ecc is a separate, nonsensical combination -- bf16 storage of an fp64 model -- and is out of
scope here; decode() is a float32 codec by construction.)
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


def parity(Opt, cin, cout, device):
    torch.manual_seed(0)
    ref = Par(cin, cout).to(device)
    ref_opt = Opt(ref.parameters(), lr=2e-2, ecc=8)
    torch.manual_seed(0)
    fs = Par(cin, cout).to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = Opt.fsdp2(fs, lr=2e-2, ecc=8)
    err = torch.zeros((), device=device)
    for step in range(12):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, cin * 3, generator=g).to(device)
        tgt = torch.randn(8, cout, generator=g).to(device)
        ((ref(X) - tgt) ** 2).mean().backward()
        ref_opt.step(); ref_opt.zero_grad()
        ((fs(X) - tgt) ** 2).mean().backward()
        fs_opt.step(); fs_opt.zero_grad()
        for r, f in zip(ref.parameters(), fs.parameters()):
            err = torch.maximum(err, (r.detach() - f.full_tensor()).abs().max())
    return err.item()  # torch.maximum propagates NaN (Python max() masks it), so non-finite fails the assert


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    results = {name: parity(getattr(heavyball, name), 4, 5, device)
               for name in ("SOAP", "NorMuon", "PSGDKron")}
    # SOAP's GG^T matmul reduces layout-dependently on CUDA/cuBLAS (gathered-whole vs standalone)
    # ~fp32 eps, which the int8 ECC residual amplifies near a rounding boundary to ~1e-6 (observed
    # 1.6e-6); NorMuon/PSGDKron have no layout-sensitive matmul and stay bit-exact. CPU matmul is
    # deterministic -> bit-exact. SR noise is identical either way (leaf-keyed Philox), so 1e-4 still
    # verifies parity 10x tighter than ecc's own ~1e-3 precision (a dropped int8 correction is ~1e-3).
    tol = 1e-4 if cuda else 1e-8
    if rank == 0:
        for name, err in results.items():
            assert err < tol, f"{name} owner-whole ecc=8 != single-process: {err} (tol {tol})"
        print(f"OK owner-whole ecc parity {{{', '.join(f'{k}={v:.0e}' for k, v in results.items())}}}", flush=True)
        print("FSDP2_ECC_OWNER_WHOLE", flush=True)
    dist.destroy_process_group()


main()
'''


def test_ecc_owner_whole_matches_single_process(tmp_path):
    script = tmp_path / "ecc_owner_whole_worker.py"
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
    assert "OK owner-whole ecc parity" in output, output
    assert "FSDP2_ECC_OWNER_WHOLE" in output, output
