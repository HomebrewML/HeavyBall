import os
from typing import Iterable, List, Sequence

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from utils import _global_l2_norm, _global_rms_norm, _local_l2_norm, _local_rms_norm

import heavyball
from heavyball.utils import (
    caution,
    global_l2norm_clip,
    global_rmsnorm_clip,
    l2_clip_,
    merge_group,
    rmsnorm_clip_,
    stochastic_add_,
    stochastic_add_divide_,
    stochastic_divide_with_eps_,
    stochastic_multiply_,
)

# Ensure torch.compile stays disabled on CPU-only CI runners.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


heavyball.utils.compile_mode = None

FLOAT_DTYPES: Sequence[torch.dtype] = (torch.float32, torch.bfloat16)
DTYPE_TOLERANCE = {torch.float32: 5e-5, torch.bfloat16: 2e-2}


def _to_float(value: float | torch.Tensor) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


@st.composite
def _tensor_shape(draw, min_rank: int = 1, max_rank: int = 4, max_side: int = 6) -> Sequence[int]:
    rank = draw(st.integers(min_value=min_rank, max_value=max_rank))
    return tuple(draw(st.integers(min_value=1, max_value=max_side)) for _ in range(rank))


@st.composite
def _tensor_list(
    draw,
    min_len: int = 1,
    max_len: int = 3,
    min_rank: int = 1,
    max_rank: int = 4,
    max_side: int = 6,
) -> List[torch.Tensor]:
    dtype = draw(st.sampled_from(FLOAT_DTYPES))
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    tensors = []
    for _ in range(length):
        shape = draw(_tensor_shape(min_rank=min_rank, max_rank=max_rank, max_side=max_side))
        tensor = torch.randn(shape, dtype=torch.float32)
        tensors.append(tensor.to(dtype))
    return tensors


@st.composite
def _stochastic_inputs(draw) -> tuple[torch.dtype, List[torch.Tensor], List[torch.Tensor]]:
    dtype = draw(st.sampled_from(FLOAT_DTYPES))
    length = draw(st.integers(min_value=1, max_value=3))
    shared_shape = draw(st.booleans())
    tensors: List[torch.Tensor] = []
    if shared_shape:
        shape = draw(_tensor_shape(max_side=5))
        tensors = [torch.randn(shape, dtype=torch.float32).to(dtype) for _ in range(length)]
    else:
        for _ in range(length):
            shape = draw(_tensor_shape(max_side=5))
            tensors.append(torch.randn(shape, dtype=torch.float32).to(dtype))
    partner = [torch.randn_like(tensor, dtype=torch.float32).to(dtype) for tensor in tensors]
    if shared_shape and len(partner) > 1 and draw(st.booleans()):
        partner = [partner[0]]
    return dtype, [tensor.clone() for tensor in tensors], partner


@st.composite
def _caution_inputs(draw) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = draw(_tensor_list(min_len=1, max_len=1, min_rank=1, max_rank=3, max_side=6))[0]
    update = torch.randn_like(tensor, dtype=torch.float32).to(tensor.dtype)
    return tensor, update


def _flatten_tensors(items) -> List[torch.Tensor]:
    if isinstance(items, torch.Tensor):
        return [items]
    flat: List[torch.Tensor] = []
    for item in items:
        flat.extend(_flatten_tensors(item))
    return flat


def _expand_like(reference: Iterable[torch.Tensor], candidate: List[torch.Tensor]) -> List[torch.Tensor]:
    reference = list(reference)
    if len(candidate) == len(reference):
        return [tensor.clone().float() for tensor in candidate]
    assert len(candidate) == 1
    source = candidate[0]
    return [source.clone().float() for _ in reference]


@settings(deadline=None, max_examples=75)
@given(
    tensors=_tensor_list(min_len=1, max_len=3),
    clip_at=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_local_clipping_respects_thresholds(tensors: List[torch.Tensor], clip_at: float):
    for clip_fn, metric in (
        (l2_clip_, _local_l2_norm),
        (rmsnorm_clip_, _local_rms_norm),
    ):
        clipped = [tensor.clone() for tensor in tensors]
        clip_fn(clipped, clip_at)
        for tensor in clipped:
            tolerance = clip_at * 1e-3 + DTYPE_TOLERANCE.get(tensor.dtype, 1e-5)
            assert metric(tensor) <= clip_at + tolerance


@settings(deadline=None, max_examples=75)
@given(
    tensors=_tensor_list(min_len=1, max_len=3),
    clip_at=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_global_clipping_bounds_group_norms(tensors: List[torch.Tensor], clip_at: float):
    for clip_fn, metric in (
        (
            global_l2norm_clip,
            _global_l2_norm,
        ),
        (
            global_rmsnorm_clip,
            _global_rms_norm,
        ),
    ):
        clipped = [tensor.clone() for tensor in tensors]
        clip_fn(clipped, clip_at)
        tolerance = clip_at * 1e-3 + DTYPE_TOLERANCE.get(clipped[0].dtype, 1e-5)
        assert metric(clipped) <= clip_at + tolerance


@settings(deadline=None, max_examples=75)
@given(
    data=_stochastic_inputs(),
    alpha=st.one_of(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        st.builds(torch.tensor, st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
    ),
)
def test_stochastic_add_matches_expected(data, alpha):
    torch.manual_seed(0x172893)
    dtype, tensors, partner = data
    x = [tensor.clone() for tensor in tensors]
    y = [tensor.clone() for tensor in partner]
    stochastic_add_(x, y, alpha)
    alpha_scalar = _to_float(alpha)
    expected_inputs = [tensor.clone().float() for tensor in tensors]
    expected_partner = _expand_like(expected_inputs, partner)
    expected = [xi + yi * alpha_scalar for xi, yi in zip(expected_inputs, expected_partner, strict=True)]
    max_error = max((result.float() - exp).abs().max().item() for result, exp in zip(x, expected, strict=True))
    assert max_error <= DTYPE_TOLERANCE[dtype] + 1e-6


@settings(deadline=None, max_examples=75)
@given(
    data=_stochastic_inputs(),
    alpha=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    divisor=st.one_of(
        st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.builds(torch.tensor, st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)),
    ),
)
def test_stochastic_add_divide_matches_expected(data, alpha, divisor):
    torch.manual_seed(0x172893)
    dtype, tensors, partner = data
    x = [tensor.clone() for tensor in tensors]
    y = [tensor.clone() for tensor in partner]
    stochastic_add_divide_(x, y, alpha=alpha, divisor=divisor)
    alpha_scalar = _to_float(alpha)
    divisor_scalar = _to_float(divisor)
    expected_inputs = [tensor.clone().float() for tensor in tensors]
    expected_partner = _expand_like(expected_inputs, partner)
    expected = [
        (xi + yi * alpha_scalar) / divisor_scalar for xi, yi in zip(expected_inputs, expected_partner, strict=True)
    ]
    max_error = max((result.float() - exp).abs().max().item() for result, exp in zip(x, expected, strict=True))
    assert max_error <= DTYPE_TOLERANCE[dtype] + 1e-6


@settings(deadline=None, max_examples=75)
@given(data=_stochastic_inputs())
def test_stochastic_multiply_matches_expected(data):
    torch.manual_seed(0x172893)
    dtype, tensors, partner = data
    x = [tensor.clone() for tensor in tensors]
    y = [tensor.clone() for tensor in partner]
    stochastic_multiply_(x, y)
    expected_inputs = [tensor.clone().float() for tensor in tensors]
    expected_partner = _expand_like(expected_inputs, partner)
    expected = [xi * yi for xi, yi in zip(expected_inputs, expected_partner, strict=True)]
    max_error = max((result.float() - exp).abs().max().item() for result, exp in zip(x, expected, strict=True))
    assert max_error <= DTYPE_TOLERANCE[dtype] + 1e-6


@settings(deadline=None, max_examples=75)
@given(
    data=_stochastic_inputs(),
    eps=st.one_of(
        st.floats(min_value=1e-5, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.builds(torch.tensor, st.floats(min_value=1e-5, max_value=1.0, allow_nan=False, allow_infinity=False)),
    ),
)
def test_stochastic_divide_with_eps_matches_expected(data, eps):
    torch.manual_seed(0x172893)
    dtype, tensors, partner = data
    # Avoid very small denominators to keep the numeric range reasonable.
    partner = [tensor.clamp(min=0.5) for tensor in partner]
    x = [tensor.clone() for tensor in tensors]
    y = [tensor.clone() for tensor in partner]
    stochastic_divide_with_eps_(x, y, eps)
    eps_scalar = _to_float(eps)
    expected_inputs = [tensor.clone().float() for tensor in tensors]
    expected_partner = _expand_like(expected_inputs, partner)
    expected = [xi / (yi + eps_scalar) for xi, yi in zip(expected_inputs, expected_partner, strict=True)]
    max_error = max((result.float() - exp).abs().max().item() for result, exp in zip(x, expected, strict=True))
    assert max_error <= DTYPE_TOLERANCE[dtype] + 1e-6


@settings(deadline=None, max_examples=75)
@given(tensor=_tensor_list(min_len=1, max_len=1, min_rank=2, max_rank=4, max_side=8))
def test_merge_group_passthrough_when_disabled(tensor: List[torch.Tensor]):
    original = tensor[0]
    group = {"merge_dims": False, "max_precond_dim": 8}
    result = merge_group(group, original)
    assert isinstance(result, tuple)
    assert result[0] is original


@settings(deadline=None, max_examples=75)
@given(
    tensor=st.builds(
        lambda tensor: tensor,
        _tensor_list(min_len=1, max_len=1, min_rank=2, max_rank=4, max_side=8),
    ),
    max_dim=st.integers(min_value=2, max_value=12),
    split=st.booleans(),
    use_triangular=st.booleans(),
)
def test_merge_group_preserves_structure(tensor: List[torch.Tensor], max_dim: int, split: bool, use_triangular: bool):
    base = tensor[0]
    group = {"merge_dims": True, "split": split}
    if use_triangular:
        group["max_size_triangular"] = max_dim
    else:
        group["max_precond_dim"] = max_dim
    merged = merge_group(group, base)
    flat = _flatten_tensors(merged)
    assert sum(chunk.numel() for chunk in flat) == base.numel()
    for chunk in flat:
        assert chunk.dtype == base.dtype
        assert chunk.is_contiguous()


@settings(deadline=None, max_examples=75)
@given(_caution_inputs())
def test_caution_masks_disagreeing_directions(data):
    grad_tensor, update_tensor = data
    result = caution(grad_tensor, update_tensor)
    assert result.shape == grad_tensor.shape
    mask = grad_tensor.signbit() ^ update_tensor.signbit()
    scale = mask.numel() / max(mask.numel() - mask.sum().item(), 1)
    expected = update_tensor.clone()
    expected[mask] = 0
    expected = expected * scale
    diff = (result.float() - expected.float()).abs().max().item()
    assert diff <= DTYPE_TOLERANCE[result.dtype] + 1e-6
    positive_alignment = (result * grad_tensor).float().sum().item()
    assert positive_alignment >= -DTYPE_TOLERANCE[result.dtype]
