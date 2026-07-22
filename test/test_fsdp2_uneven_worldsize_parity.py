"""Owner-whole optimizers with UNEVEN leaf counts on world_size >= 3 -- the topology class that a W=2
suite structurally cannot exercise (balanced == canonical for ws<=2). The owner-leaf assignment must
match DTensor's canonical Shard(0) chunking or the owner-whole state DTensor is internally inconsistent
and the step's broadcast_leaf / random_like redistribute crash or mis-narrow. This matrix pins the three
bug variants at ws=3: (1a) plain step, (1b) ecc step (random_like + codec on the owner-whole DTensor),
and the same-world checkpoint round-trip, for SOAP/NorMuon/PSGDKron over uneven leaf counts. gloo runs
it on CPU in CI (the divergence is topology-dependent, not device-dependent); hardware ws=2..8 coverage
lives in the scratch matrix workers. NaN-propagating (torch.maximum), so a crash or non-finite fails.
Run: torchrun --nproc_per_node=3 <worker>
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
    def __init__(self, cin, cout, n):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def build(Opt, n, dt, kw, device):
    torch.manual_seed(0)
    m = Par(4, 5, n).to(dt).to(device)
    for layer in m.ls:
        fully_shard(layer)
    fully_shard(m)
    return m, Opt.fsdp2(m, lr=2e-2, **kw)


def train(m, opt, n, lo, hi, dt, device):
    for step in range(lo, hi):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, 4 * n, dtype=dt, generator=g).to(device)
        Y = torch.randn(8, 5, dtype=dt, generator=g).to(device)
        ((m(X) - Y) ** 2).mean().backward()
        opt.step(); opt.zero_grad()


def step_parity(Opt, n, ecc, device):
    dt = torch.float32 if ecc else torch.float64
    kw = {"ecc": 8} if ecc else {}
    torch.manual_seed(0)
    ref = Par(4, 5, n).to(dt).to(device)
    ref_opt = Opt(ref.parameters(), lr=2e-2, **kw)
    fs, fs_opt = build(Opt, n, dt, kw, device)
    err = torch.zeros((), device=device)
    for step in range(8):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, 4 * n, dtype=dt, generator=g).to(device)
        Y = torch.randn(8, 5, dtype=dt, generator=g).to(device)
        ((ref(X) - Y) ** 2).mean().backward(); ref_opt.step(); ref_opt.zero_grad()
        ((fs(X) - Y) ** 2).mean().backward(); fs_opt.step(); fs_opt.zero_grad()
        for r, f in zip(ref.parameters(), fs.parameters()):
            err = torch.maximum(err, (r.detach() - f.full_tensor()).abs().max())
    return err.item()


def ckpt_roundtrip(Opt, n, ecc, device):
    dt = torch.float32 if ecc else torch.float64
    kw = {"ecc": 8} if ecc else {}
    base, base_opt = build(Opt, n, dt, kw, device); train(base, base_opt, n, 0, 12, dt, device)
    inter, inter_opt = build(Opt, n, dt, kw, device); train(inter, inter_opt, n, 0, 8, dt, device)
    model_ck = {k: v.clone() for k, v in inter.state_dict().items()}
    opt_ck = inter_opt.state_dict()
    res, res_opt = build(Opt, n, dt, kw, device)
    res.load_state_dict(model_ck); res_opt.load_state_dict(opt_ck)
    train(res, res_opt, n, 8, 12, dt, device)
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
    results = {}
    for name in ("SOAP", "NorMuon", "PSGDKron"):
        Opt = getattr(heavyball, name)
        for ecc in (None, 8):
            for n in (4, 5):  # 4 % 3 == 1, 5 % 3 == 2: both uneven on ws=3
                results[f"step/{name}/ecc{ecc}/n{n}"] = step_parity(Opt, n, ecc, device)
                results[f"ckpt/{name}/ecc{ecc}/n{n}"] = ckpt_roundtrip(Opt, n, ecc, device)
    if rank == 0:
        for key, err in results.items():
            # ecc step-parity vs single-process is fp32 (the multi-way reduce-scatter reorders vs a
            # single process); everything else is bit-exact (fp64 step, or FSDP2-vs-FSDP2 checkpoint).
            tol = 1e-4 if (key.startswith("step") and "ecc8" in key) else 1e-8
            assert err < tol, f"{key} uneven ws=3 != single-process: {err}"
        print(f"OK uneven ws=3 matrix ({len(results)} cells, max={max(results.values()):.0e})", flush=True)
        print("FSDP2_UNEVEN_WORLDSIZE", flush=True)
    dist.destroy_process_group()


main()
'''


def test_uneven_worldsize_owner_whole_matches_single_process(tmp_path):
    script = tmp_path / "uneven_ws_worker.py"
    script.write_text(_WORKER)
    port = 29500 + os.getpid() % 1500
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=3", f"--master_port={port}", str(script)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "OK uneven ws=3 matrix" in output, output
    assert "FSDP2_UNEVEN_WORLDSIZE" in output, output
