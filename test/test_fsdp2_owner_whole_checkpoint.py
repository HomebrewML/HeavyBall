"""Same-world FSDP2 state_dict/load_state_dict round-trip for OWNER-WHOLE optimizers. The plain
state_dict/load_state_dict path (the common opt.state_dict() API, distinct from dcp_save/dcp_load) must
restore owner-whole DTensor state (SOAP's Grams, NorMuon's moment2, PSGD's Kron factors) so a resumed run
matches an uninterrupted one. Shard-separable optimizers (AdamW/Muon) already round-trip bit-exactly;
this pins the owner-whole state, which was previously untested here (only cross-world DCP owner-whole and
same-world shard-separable were covered) and was broken.
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
from torch.distributed.tensor import DTensor
import heavyball

patch("heavyball.core.torch.compile", lambda f, **k: f).start()


class Par(nn.Module):
    def __init__(self, cin, cout, n=3):
        super().__init__()
        self.cin = cin
        self.ls = nn.ModuleList([nn.Linear(cin, cout, bias=False) for _ in range(n)])

    def forward(self, x):
        return sum(layer(x[:, i * self.cin:(i + 1) * self.cin]) for i, layer in enumerate(self.ls))


def build(Opt, device, seed):
    torch.manual_seed(seed)
    m = Par(4, 5).double().to(device)
    for layer in m.ls:
        fully_shard(layer)
    fully_shard(m)
    return m, Opt.fsdp2(m, lr=2e-2)


def train(m, opt, lo, hi, device):
    for step in range(lo, hi):
        g = torch.Generator().manual_seed(1000 + step)
        X = torch.randn(8, 12, dtype=torch.float64, generator=g).to(device)
        Y = torch.randn(8, 5, dtype=torch.float64, generator=g).to(device)
        ((m(X) - Y) ** 2).mean().backward()
        opt.step(); opt.zero_grad()


def assert_nested_equal(actual, expected):
    if isinstance(actual, DTensor):
        assert isinstance(expected, DTensor)
        assert torch.equal(actual.full_tensor(), expected.full_tensor())
    elif isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def roundtrip(Opt, device):
    base, base_opt = build(Opt, device, 0); train(base, base_opt, 0, 12, device)
    inter, inter_opt = build(Opt, device, 0); train(inter, inter_opt, 0, 8, device)
    model_ck = {k: v.clone() for k, v in inter.state_dict().items()}
    opt_ck = inter_opt.state_dict()
    res, res_opt = build(Opt, device, 987654321)
    res.load_state_dict(model_ck); res_opt.load_state_dict(opt_ck)
    torch.manual_seed(123456789 + dist.get_rank())
    train(res, res_opt, 8, 12, device)
    assert_nested_equal(base_opt.state_dict(), res_opt.state_dict())
    return max((b.full_tensor() - r.full_tensor()).abs().max().item()
               for b, r in zip(base.parameters(), res.parameters()))


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    results = {name: roundtrip(getattr(heavyball, name), device)
               for name in ("SOAP", "NorMuon", "PSGDKron")}
    if rank == 0:
        for name, err in results.items():
            assert err < 1e-8, f"{name} same-world owner-whole checkpoint: {err}"
        print(f"OK owner-whole checkpoint {{{', '.join(f'{k}={v:.0e}' for k, v in results.items())}}}", flush=True)
        print("FSDP2_OWNER_WHOLE_CHECKPOINT", flush=True)
    dist.destroy_process_group()


main()
'''


def test_owner_whole_same_world_checkpoint_roundtrip(tmp_path):
    script = tmp_path / "owner_whole_ckpt_worker.py"
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
    assert "OK owner-whole checkpoint" in output, output
    assert "FSDP2_OWNER_WHOLE_CHECKPOINT" in output, output
