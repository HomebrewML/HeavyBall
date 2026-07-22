"""FSDP2 plain-constructor guard reports the required adapter clearly."""

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
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
import heavyball


def main():
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank = dist.get_rank()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    model = nn.Linear(8, 8).to(device)
    fully_shard(model)

    raised_naming_fsdp2 = False
    try:
        heavyball.AdamW(model.parameters(), lr=1e-2)
    except ValueError as e:
        raised_naming_fsdp2 = "fsdp2" in str(e).lower()

    if rank == 0:
        assert raised_naming_fsdp2, (
            "plain constructor on an FSDP2 DTensor parameter must raise a ValueError naming .fsdp2()"
        )
        print("FSDP2_CONSTRUCT_GUARD", flush=True)
    dist.destroy_process_group()


main()
'''


def test_plain_constructor_rejects_fsdp2_parameters_clearly(tmp_path):
    script = tmp_path / "fsdp2_construct_guard_worker.py"
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
    assert "FSDP2_CONSTRUCT_GUARD" in output, output
