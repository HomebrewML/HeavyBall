"""SOAP under FSDP2 -- the first MULTI-SLOT owner-whole preconditioner. SOAP's left Gram GG_l = G@G.T is
[R,R] (needs every row pair), and its whole state is six owner-whole slots (GG_l, GG_r, Q_l, Q_r,
exp_avg, exp_avg_sq). This is the [R,R]-Gram case: the Gram carries the parameter's row dimension but is
NOT shard-separable (cross-row sums), so SOAP must declare distributed_scope=WHOLE and hold its state
owner-whole, not row-sharded. NorMuon proved a single owner-whole slot; SOAP proves the mechanism
generalizes to many, plus the periodic eigenbasis refresh running owner-local.

Both orientations:
  - TALL: Linear(4,5) weight [5,4], R>=C -> GG_l [5,5], GG_r [4,4].
  - WIDE: Linear(5,3) weight [3,5], R<C  -> GG_l [3,3] (reduces across the FSDP-sharded 3 rows), GG_r [5,5].

float64, same init + same data on every rank (FSDP2 averages grads -> the sharded grad equals the
single-process grad). 12 steps to cross a refresh boundary. Single-process parity < 1e-8. RED until SOAP
declares WHOLE (currently rejected by the scope guard).
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
    ref_opt = heavyball.SOAP(ref.parameters(), lr=2e-2)
    torch.manual_seed(0)
    fs = Par(cin, cout).double().to(device)
    for layer in fs.ls:
        fully_shard(layer)
    fully_shard(fs)
    fs_opt = heavyball.SOAP.fsdp2(fs, lr=2e-2)
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
    tall = parity(4, 5, "tall", device)   # weight [5,4], GG_l [5,5]
    wide = parity(5, 3, "wide", device)   # weight [3,5], GG_l [3,3] reduces across sharded rows
    if rank == 0:
        assert tall < 1e-8, f"SOAP tall: {tall}"
        assert wide < 1e-8, f"SOAP wide (Gram must be owner-whole): {wide}"
        print(f"OK SOAP parity tall={tall:.2e} wide={wide:.2e}", flush=True)
        print("FSDP2_SOAP_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_soap_fsdp2_matches_single_process(tmp_path):
    script = tmp_path / "soap_parity_worker.py"
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
    assert "OK SOAP parity" in output, output
    assert "FSDP2_SOAP_PARITY" in output, output
