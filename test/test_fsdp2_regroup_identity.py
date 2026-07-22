"""Bit-exact reference checks for FSDP2 parameter-owner regroup and its inverse."""

import math

import pytest
import torch
from torch.distributed.tensor import Shard

from heavyball.core import FSDP2RegroupPlan


def _reference_roundtrip(param_count: int, rows: int, columns: int, world_size: int):
    padded_rows = math.ceil(rows / world_size)
    plan = FSDP2RegroupPlan.create(
        tuple(f"weight_{index}" for index in range(param_count)),
        (rows, columns),
        (padded_rows, columns),
        world_size,
    )
    whole = torch.arange(param_count * rows * columns, dtype=torch.int64).reshape(
        param_count, rows, columns
    )
    shards = []
    for rank in range(world_size):
        shard = torch.zeros((param_count, padded_rows, columns), dtype=whole.dtype)
        start = rank * padded_rows
        valid = min(max(rows - start, 0), padded_rows)
        if valid:
            shard[:, :valid].copy_(whole[:, start:start + valid])
        shards.append(shard)

    pack = torch.tensor(plan.pack_indices)
    unpack = torch.tensor(plan.unpack_indices)
    forward_sends = []
    for shard in shards:
        padded = torch.cat((shard, torch.zeros_like(shard[:1])), dim=0)
        forward_sends.append(padded.index_select(0, pack))

    inverse_sends = []
    for owner, (owner_start, owner_stop) in enumerate(plan.owner_ranges):
        received = torch.cat(
            [send.reshape(world_size, plan.max_owned, padded_rows, columns)[owner] for send in forward_sends],
            dim=0,
        )
        owned = (
            received.reshape(world_size, plan.max_owned, padded_rows, columns)
            .permute(1, 0, 2, 3)
            .reshape(plan.max_owned, world_size * padded_rows, columns)
            [:owner_stop - owner_start, :rows]
        )
        padded_owned = torch.zeros(
            (plan.max_owned, world_size * padded_rows, columns), dtype=whole.dtype
        )
        padded_owned[:owner_stop - owner_start, :rows].copy_(owned)
        inverse_sends.append(
            padded_owned.reshape(plan.max_owned, world_size, padded_rows, columns)
            .permute(1, 0, 2, 3)
            .reshape(world_size * plan.max_owned, padded_rows, columns)
        )

    reconstructed = []
    for destination in range(world_size):
        received = torch.cat(
            [
                send.reshape(world_size, plan.max_owned, padded_rows, columns)[destination]
                for send in inverse_sends
            ],
            dim=0,
        )
        reconstructed.append(received.index_select(0, unpack))
    return plan, shards, reconstructed


@pytest.mark.parametrize(
    ("param_count", "rows", "columns", "world_size"),
    [
        (4, 5, 3, 2),  # R % W != 0
        (4, 2, 3, 4),  # R < W
        (5, 6, 3, 3),  # N % W != 0
        (2, 8, 3, 4),  # N < W, including zero-matrix owners
        (3, 5, 4, 2),  # same padded local rows as the next logical shape
        (3, 6, 4, 2),
        (4, 5, 3, 3),  # N=4 on W=3: canonical owner counts [2,2,0] (max-min=2), the uneven-ws bug case
    ],
    ids=("uneven-rows", "rows-below-world", "uneven-params", "params-below-world", "r5-p3", "r6-p3", "uneven-ws3"),
)
def test_regroup_inverse_regroup_is_bit_exact(param_count, rows, columns, world_size):
    plan, shards, reconstructed = _reference_roundtrip(param_count, rows, columns, world_size)
    assert all(torch.equal(actual, expected) for actual, expected in zip(reconstructed, shards, strict=True))
    owner_counts = [stop - start for start, stop in plan.owner_ranges]
    expected = [Shard.local_shard_size_and_offset(param_count, world_size, r)[0] for r in range(world_size)]
    assert owner_counts == expected, f"owner assignment must match DTensor canonical Shard(0): {owner_counts} != {expected}"


def test_logical_shapes_sharing_padded_rows_have_distinct_plan_identities():
    r5, _, _ = _reference_roundtrip(3, 5, 4, 2)
    r6, _, _ = _reference_roundtrip(3, 6, 4, 2)
    assert r5.storage_shape == r6.storage_shape == (3, 4)
    assert r5.identity != r6.identity
