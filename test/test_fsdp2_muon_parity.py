"""Muon under FSDP2 must MATCH single-process Muon on the whole matrix -- the all_to_all regroup is
correct only if orthogonalization sees the whole parameter, not a dim-0 shard.

Cross-rank consistency is NOT a sufficient oracle here: an impl that orthogonalizes each rank's dim-0
shard separately is still cross-rank consistent (both ranks gather the same shards) yet diverges from
single-process Muon. So this pins PARITY, the only eval that forces a real whole-matrix regroup.

Three Linear(4,5) applied to disjoint input slices and summed (valid forward, distinct gradient per
weight) with 2 ranks is chosen deliberately: N=3 forces parameter-count padding (not a multiple of
W=2); R=5 forces uneven dim-0 sharding (local padded P=3, valid 3 and 2); and the Muon commit scale
sqrt(max(R,C)/min) = sqrt(5/4)~1.118 differs from the local-shard scale max(1,3/4)^0.5 = 1.0, so a
commit that wrongly used the local padded shape fails here. float64 avoids bfloat16 stochastic-rounding
noise in orthogonalize. Identical init + identical data on every rank and the reference: FSDP2 averages
grads (verified), so the FSDP2 sharded gradient equals the single-process gradient and the trajectories
must coincide bitwise-close.

Eager (patched compile) tests the regroup MATH for CI; the compiled fullgraph + NCCL path is verified
separately on the 2-GPU node.
Run: torchrun --nproc_per_node=2 test_fsdp2_muon_parity.py
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

    def forward(self, x):  # x:[B,12] -> 3 disjoint [B,4] slices -> distinct grad per [5,4] weight
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
    ref_opt = heavyball.Muon(ref.parameters(), lr=2e-2)

    torch.manual_seed(0)
    fsdp = Par().double().to(device)
    for layer in fsdp.ls:
        fully_shard(layer)
    fully_shard(fsdp)
    fsdp_opt = heavyball.Muon.fsdp2(fsdp, lr=2e-2)

    torch.manual_seed(42)  # SAME data stream on every rank and the reference
    err = 0.0
    for step in range(6):
        X = torch.randn(8, 12, dtype=torch.float64, device=device)
        tgt = torch.randn(8, 5, dtype=torch.float64, device=device)
        ((ref(X) - tgt) ** 2).mean().backward()
        ref_opt.step()
        ref_opt.zero_grad()
        ((fsdp(X) - tgt) ** 2).mean().backward()
        fsdp_opt.step()
        fsdp_opt.zero_grad()
        step_err = max(
            (r.detach() - f.full_tensor()).abs().max().item()
            for r, f in zip(ref.parameters(), fsdp.parameters())
        )
        err = max(err, step_err)
        if rank == 0:
            assert step_err < 1e-8, f"step {step}: Muon.fsdp2 != single-process Muon: {step_err}"
    moved = max(r.detach().abs().max().item() for r in ref.parameters())
    if rank == 0:
        assert moved > 1e-3, f"reference did not train: {moved}"
        print(f"OK Muon parity max_err={err:.2e}", flush=True)
        print("FSDP2_MUON_PARITY", flush=True)
    dist.destroy_process_group()


main()
'''


def test_muon_fsdp2_matches_single_process_muon(tmp_path):
    script = tmp_path / "muon_parity_worker.py"
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
    assert "OK Muon parity" in output, output
    assert "FSDP2_MUON_PARITY" in output, output
