import pytest
import torch

import heavyball
from heavyball.kl import kl_soap_init
from heavyball.matrix import shampoo_init, soap_init
from heavyball.transforms import whiten_init


def _state_slabs(optimizer):
    """Mirror the benchmark's accounting without importing its process-wide setup."""

    return [
        slab
        for engine in optimizer._engines
        for group in engine.groups
        for slots in (
            *group.states,
            group.commit_state,
            *group.state_corrections,
            group.commit_corrections,
        )
        for slab in slots.values()
    ]


def _state_bytes(state) -> int:
    return sum(value.nelement() * value.element_size() for value in state.values())


def _adamw_state_bytes(numel: int, **kwargs) -> int:
    parameter = torch.nn.Parameter(torch.zeros(numel))
    optimizer = heavyball.AdamW([parameter], lr=1e-3, **kwargs)
    return sum(slab.nelement() * slab.element_size() for slab in _state_slabs(optimizer))


def test_adamw_state_bytes_include_ecc_corrections():
    numel = 10_000
    fp32 = _adamw_state_bytes(numel)
    bf16 = _adamw_state_bytes(numel, storage_dtype=torch.bfloat16)
    ecc8 = _adamw_state_bytes(numel, ecc=8)
    ecc16 = _adamw_state_bytes(numel, ecc=16)

    assert fp32 == 8 * numel
    assert bf16 == 4 * numel
    assert ecc8 == 6 * numel
    assert ecc16 == 8 * numel


@pytest.mark.parametrize(
    ("initializer", "legacy_matrix_slabs", "vector_elements", "scale_count"),
    (
        (lambda ref, n: shampoo_init(ref, max_precond_dim=torch.tensor(n)), 4, 0, 2),
        (lambda ref, n: soap_init(ref, max_precond_dim=torch.tensor(n)), 6, 0, 2),
        (lambda ref, n: kl_soap_init(ref, max_precond_dim=torch.tensor(n)), 6, 2, 2),
        (lambda ref, n: whiten_init(ref), 2, 0, 1),
    ),
    ids=("Shampoo", "SOAP", "KLSOAP", "Whiten"),
)
def test_default_matrix_state_restores_pre_hardening_slabs(
    initializer, legacy_matrix_slabs, vector_elements, scale_count
):
    dimension = 2048
    state = initializer(torch.zeros(dimension, dimension), dimension)
    legacy_bytes = (
        legacy_matrix_slabs * dimension * dimension * torch.tensor([], dtype=torch.float32).element_size()
        + vector_elements * dimension * torch.tensor([], dtype=torch.float32).element_size()
    )
    # Scale normalization needs one scalar per active Gram, but adds no O(n) or O(n**2)
    # storage. Every pre-hardening matrix/vector slab is restored to fp32.
    assert _state_bytes(state) == legacy_bytes + scale_count * torch.tensor([], dtype=torch.float32).element_size()
    assert all(value.dtype is torch.float32 for value in state.values())
