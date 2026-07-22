"""Tests for shared parameter storage at optimizer construction.

Heavyball binds each parameter into its own slab row, so two distinct parameters
sharing storage would silently break the tie; the guard rejects this at
construction. Same-object tying is deduped and accepted.
"""

import pytest
import torch
import torch.nn as nn

import heavyball


def test_shared_storage_parameters_rejected():
    storage = torch.randn(8, 8)
    p1 = nn.Parameter(storage)
    p2 = nn.Parameter(storage)

    with pytest.raises(ValueError, match="share"):
        heavyball.AdamW([p1, p2], lr=1e-2)


def test_distinct_parameters_accepted():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    heavyball.AdamW(model.parameters(), lr=1e-2)


def test_same_object_weight_tying_accepted():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    model[1].weight = model[0].weight
    heavyball.AdamW(model.parameters(), lr=1e-2)
