"""Same-world FSDP2 state_dict/load_state_dict stays bit-identical for sharded and whole state."""

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

_WORKER = r'''
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

import heavyball

patch("heavyball.core.torch.compile", lambda function, **kwargs: function).start()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 5, bias=False) for _ in range(3)])

    def forward(self, inputs):
        return sum(layer(inputs[:, index * 4:(index + 1) * 4]) for index, layer in enumerate(self.layers))


def data(step, device):
    generator = torch.Generator().manual_seed(1000 + step)
    return (
        torch.randn(8, 12, dtype=torch.float64, generator=generator).to(device),
        torch.randn(8, 5, dtype=torch.float64, generator=generator).to(device),
    )


def sharded(device, facade=None):
    torch.manual_seed(0)
    model = Model().double().to(device)
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    if facade is None:
        return model
    return model, facade.fsdp2(model, lr=2e-2)


def train(model, optimizer, start, stop, device):
    for step in range(start, stop):
        inputs, targets = data(step, device)
        ((model(inputs) - targets) ** 2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def roundtrip(facade, device):
    reference, reference_optimizer = sharded(device, facade)
    train(reference, reference_optimizer, 0, 8, device)

    interrupted, interrupted_optimizer = sharded(device, facade)
    train(interrupted, interrupted_optimizer, 0, 4, device)
    model_checkpoint = {
        name: value.detach().clone() for name, value in interrupted.state_dict().items()
    }
    optimizer_checkpoint = interrupted_optimizer.state_dict()

    resumed = sharded(device)
    resumed.load_state_dict(model_checkpoint)
    resumed_optimizer = facade.fsdp2(resumed, lr=2e-2)
    resumed_optimizer.load_state_dict(optimizer_checkpoint)
    train(resumed, resumed_optimizer, 4, 8, device)

    return max(
        (expected.full_tensor() - actual.full_tensor()).abs().max().item()
        for expected, actual in zip(reference.parameters(), resumed.parameters(), strict=True)
    )


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    errors = {facade.__name__: roundtrip(facade, device) for facade in (heavyball.AdamW, heavyball.Muon)}
    if rank == 0:
        assert all(error == 0 for error in errors.values()), errors
        print(f"OK FSDP2 checkpoint roundtrip {errors}", flush=True)
        print("FSDP2_CHECKPOINT_ROUNDTRIP", flush=True)
    dist.destroy_process_group()


main()
'''


def test_fsdp2_checkpoint_resume_is_bit_identical(tmp_path):
    script = tmp_path / "fsdp2_checkpoint_roundtrip_worker.py"
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
    assert "OK FSDP2 checkpoint roundtrip" in output, output
    assert "FSDP2_CHECKPOINT_ROUNDTRIP" in output, output
