import pytest
import torch
from torch import nn

import heavyball


def test_channels_last_not_silently_flattened():
    param = nn.Parameter(torch.randn(8, 3, 3, 3).to(memory_format=torch.channels_last))
    original_storage = param.untyped_storage()
    original_data_ptr = param.data_ptr()
    original_offset = param.storage_offset()
    original_stride = param.stride()

    with pytest.raises(ValueError) as raised:
        heavyball.AdamW([param])

    assert str(tuple(param.shape)) in str(raised.value)
    assert "contiguous" in str(raised.value)
    assert param.untyped_storage() is original_storage
    assert param.data_ptr() == original_data_ptr
    assert param.storage_offset() == original_offset
    assert param.stride() == original_stride


def test_strided_overlap_is_detected():
    base = torch.randn(10)
    first = nn.Parameter(base[::2])
    second = nn.Parameter(base[5:])

    with pytest.raises(ValueError, match="overlapping storage"):
        heavyball.AdamW([first, second])


def test_contiguous_singleton_layout_is_accepted():
    param = nn.Parameter(torch.randn(1, 8).t())
    assert param.is_contiguous()
    heavyball.AdamW([param])
