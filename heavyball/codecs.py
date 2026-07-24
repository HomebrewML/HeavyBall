"""Bit-extraction codec for bfloat16-plus-integer tensor storage.

Each float32 is split into a bfloat16 holding its upper 16 bits and an integer
correction holding the leading bits of its lower 16 bits. Int8 corrections keep
eight lower bits and stochastically round the discarded eight; int16 corrections
keep all 16 lower bits and therefore reconstruct the float32 bit-exactly.
"""

import torch
from torch.distributed.tensor import DTensor

from .numerics import _stochastic_keep_finite

_CORRECTION_BITS = {torch.int8: 8, torch.int16: 16}


def _from_local(local: torch.Tensor, reference: DTensor) -> DTensor:
    return DTensor.from_local(
        local,
        reference.device_mesh,
        reference.placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )


def _stochastic_keep(bits: torch.Tensor, keep_bits: int, random: torch.Tensor) -> torch.Tensor:
    discarded = 32 - keep_bits
    if discarded == 0:
        return bits
    noise = (random * (1 << discarded)).to(torch.int32)
    return _stochastic_keep_finite(bits, noise, keep_bits)


def encode(
    value: torch.Tensor,
    narrow_dtype: torch.dtype = torch.bfloat16,
    correction_dtype: torch.dtype = torch.int8,
    *,
    random: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a tensor as bfloat16 upper bits and an integer bit correction."""
    if isinstance(value, DTensor):
        local_random = random.to_local() if isinstance(random, DTensor) else random
        narrow, correction = encode(value.to_local(), narrow_dtype, correction_dtype, random=local_random)
        return _from_local(narrow, value), _from_local(correction, value)

    w = _CORRECTION_BITS[correction_dtype]
    bits = value.float().contiguous().view(torch.int32)
    if random is None:
        random = torch.rand_like(value, dtype=torch.float32)
    rounded = _stochastic_keep(bits, 16 + w, random)
    narrow = (rounded >> 16).to(torch.int16).view(torch.bfloat16)
    correction = ((rounded >> (16 - w)) & ((1 << w) - 1)).to(correction_dtype)
    return narrow, correction


def decode(
    narrow: torch.Tensor,
    correction: torch.Tensor,
    correction_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Decode bfloat16 upper bits and an integer correction into float32."""
    if isinstance(narrow, DTensor):
        local_correction = correction.to_local() if isinstance(correction, DTensor) else correction
        decoded = decode(narrow.to_local(), local_correction, correction_dtype)
        return _from_local(decoded, narrow)

    w = _CORRECTION_BITS[correction.dtype if correction_dtype is None else correction_dtype]
    narrow_bits = narrow.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).bitwise_left_shift(16)
    correction_bits = (correction.to(torch.int32).bitwise_and((1 << w) - 1)).bitwise_left_shift(16 - w)
    return narrow_bits.bitwise_or(correction_bits).view(torch.float32)


__all__ = ["decode", "encode"]
