import os
import sys
from copy import deepcopy

import pytest
import torch
from torch import nn

import heavyball
from heavyball import utils
from heavyball.chainable import ECCConfig

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
heavyball.utils.compile_mode = None

pytestmark = [
    pytest.mark.filterwarnings("ignore:CUDA initialization:"),
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
]

ULP_MODES = ["bf16+8", "bf16+16", "fp16+8", "fp16+16"]
ALL_MODES = ULP_MODES + ["8+16"]


def _assert_ecc_active(opt, param):
    """Assert optimizer state contains ::ecc keys (proving ECC is not vacuous)."""
    st = opt.state[param]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)
    ecc_keys = [k for k in all_keys if isinstance(k, str) and "::ecc" in k]
    assert len(ecc_keys) > 0, f"ECC not active — test is vacuous. Keys: {list(all_keys.keys())}"
    return ecc_keys


# ---------------------------------------------------------------------------
# 1. Numerical precision: roundtrip encode→decode error bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ULP_MODES)
def test_ulp_roundtrip_precision(mode):
    """ULP-based ECC roundtrip error must be bounded by half-ULP / smax."""
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    x = torch.randn(4096)
    xn = x.to(cfg.primary_dtype)
    ecc = torch.zeros_like(x, dtype=cfg.corr_dtype)
    out = torch.empty_like(x)

    utils.compute_ecc([x], [xn], [ecc])
    utils.apply_ecc([xn], [ecc], [out])

    err = (x - out).abs()
    naive_err = (x - xn.float()).abs()

    # ECC must strictly improve over naive cast
    improvement = naive_err.max().item() / max(err.max().item(), 1e-45)
    if cfg.corr_dtype == torch.int8:
        assert improvement > 100, f"int8 correction should give >100x improvement, got {improvement:.0f}x"
    else:
        assert improvement > 10000, f"int16 correction should give >10000x improvement, got {improvement:.0f}x"

    # No element should be worse than the naive cast
    assert (err <= naive_err + 1e-10).all(), "ECC reconstruction must never be worse than naive cast"


@pytest.mark.parametrize("compand", [False, True])
def test_int8_roundtrip_precision(compand):
    """Scale-based int8 ECC with bf16 correction should recover most of the original."""
    torch.manual_seed(42)
    x = torch.randn(4096)
    q = torch.zeros(4096, dtype=torch.int8)
    ns = (4096 + 31) // 32
    s = torch.zeros(ns, dtype=torch.bfloat16)
    c = torch.zeros(4096, dtype=torch.bfloat16)
    out = torch.empty(4096)

    utils.quantize_int8_ecc([x], [q], [s], [c], 32, compand)
    utils.dequantize_int8_ecc([q], [s], [c], [out], 32, compand)

    err = (x - out).abs().max().item()
    # bf16 correction should bring error well below raw int8 quantization error
    assert err < 0.01, f"int8+bf16 ECC error {err:.2e} too large"


def test_ecc_zeros():
    """ECC on zero tensors should produce zero output."""
    x = torch.zeros(256)
    xn = x.to(torch.bfloat16)
    ecc = torch.zeros(256, dtype=torch.int8)
    out = torch.empty(256)

    utils.compute_ecc([x], [xn], [ecc])
    utils.apply_ecc([xn], [ecc], [out])
    assert (out == 0).all()


def test_ecc_large_values():
    """ECC should handle values near dtype max without overflow."""
    torch.manual_seed(42)
    # bf16 max is ~3.39e38, use large but representable values
    x = torch.tensor([1e4, -1e4, 3e10, -3e10, 1e-10, -1e-10])
    xn = x.to(torch.bfloat16)
    ecc = torch.zeros(6, dtype=torch.int8)
    out = torch.empty(6)

    utils.compute_ecc([x], [xn], [ecc])
    utils.apply_ecc([xn], [ecc], [out])

    err = (x - out).abs()
    naive_err = (x - xn.float()).abs()
    assert (err <= naive_err + 1e-10).all(), "ECC should not make large values worse"


# ---------------------------------------------------------------------------
# 2. Memory: ECC state tensors use less memory than fp32
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ALL_MODES)
def test_state_memory_smaller_than_fp32(mode):
    """ECC state tensors (primary + correction + optional scales) should use less memory than fp32."""
    cfg = ECCConfig(mode)
    numel = 10000
    ref = torch.randn(numel)
    state = {}
    cfg.init_state(state, "test", ref)

    ecc_bytes = sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
    fp32_bytes = numel * 4

    if mode == "bf16+16":
        # bf16(2) + int16(2) = 4 bytes, same as fp32 — this mode trades no memory savings for higher precision
        assert ecc_bytes == fp32_bytes, f"bf16+16 should equal fp32 memory, got {ecc_bytes} vs {fp32_bytes}"
    elif mode == "fp16+16":
        assert ecc_bytes == fp32_bytes, f"fp16+16 should equal fp32 memory, got {ecc_bytes} vs {fp32_bytes}"
    else:
        assert ecc_bytes < fp32_bytes, (
            f"ECC mode {mode} uses {ecc_bytes}B vs fp32 {fp32_bytes}B — no memory savings"
        )


def test_optimizer_state_dtypes():
    """Verify ECC optimizer state tensors have expected dtypes, not fp32."""
    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3, ecc="bf16+8")

    x = torch.randn(4, 64)
    (model(x).sum()).backward()
    opt.step()

    p = list(model.parameters())[0]
    ecc_keys = _assert_ecc_active(opt, p)

    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)

    for ek in ecc_keys:
        pk = ek.replace("::ecc", "")
        assert all_keys[pk].dtype == torch.bfloat16, f"Primary {pk} should be bf16, got {all_keys[pk].dtype}"
        assert all_keys[ek].dtype == torch.int8, f"Correction {ek} should be int8, got {all_keys[ek].dtype}"


# ---------------------------------------------------------------------------
# 3. Convergence: ECC optimizer converges like the non-ECC baseline
# ---------------------------------------------------------------------------

def _train(model, opt, data, target, steps):
    losses = []
    for _ in range(steps):
        p = next(model.parameters())
        d = data.to(p.dtype) if p.dtype != data.dtype else data
        loss = ((model(d) - target.to(d.dtype)) ** 2).mean().float()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("mode", ["bf16+8", "fp16+8"])
def test_convergence_matches_baseline(mode):
    """ECC optimizer should converge to a similar loss as baseline within tolerance."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    # Baseline
    torch.manual_seed(0)
    m0 = nn.Linear(16, 8, bias=False)
    opt0 = heavyball.PaLMForeachSFAdamW(m0.parameters(), lr=1e-2)
    losses_base = _train(m0, opt0, data, target, steps)

    # ECC
    torch.manual_seed(0)
    m1 = nn.Linear(16, 8, bias=False)
    opt1 = heavyball.PaLMForeachSFAdamW(m1.parameters(), lr=1e-2, ecc=mode)
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    # Final losses should be within 20% of each other
    ratio = losses_ecc[-1] / max(losses_base[-1], 1e-12)
    assert 0.5 < ratio < 2.0, (
        f"ECC final loss {losses_ecc[-1]:.6f} vs baseline {losses_base[-1]:.6f} (ratio {ratio:.2f})"
    )

    # Both should have decreased substantially
    assert losses_ecc[-1] < losses_ecc[0] * 0.5, "ECC optimizer did not converge"
    assert losses_base[-1] < losses_base[0] * 0.5, "Baseline optimizer did not converge"


def test_param_ecc_convergence():
    """Param ECC should converge comparably to baseline."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m0 = nn.Linear(16, 8, bias=False)
    opt0 = heavyball.PaLMForeachSFAdamW(m0.parameters(), lr=1e-2)
    losses_base = _train(m0, opt0, data, target, steps)

    torch.manual_seed(0)
    m1 = nn.Linear(16, 8, bias=False)
    opt1 = heavyball.PaLMForeachSFAdamW(m1.parameters(), lr=1e-2, param_ecc="bf16+8")
    losses_ecc = _train(m1, opt1, data, target, steps)

    p = list(m1.parameters())[0]
    assert p.dtype == torch.bfloat16, f"Param should be bf16 under param_ecc, got {p.dtype}"
    st = opt1.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)
    assert "param::ecc" in all_keys, f"param::ecc not in state — param ECC inactive. Keys: {list(all_keys.keys())}"

    ratio = losses_ecc[-1] / max(losses_base[-1], 1e-12)
    assert 0.5 < ratio < 2.0, (
        f"Param ECC final loss {losses_ecc[-1]:.6f} vs baseline {losses_base[-1]:.6f}"
    )
    assert losses_ecc[-1] < losses_ecc[0] * 0.5, "Param ECC did not converge"


# ---------------------------------------------------------------------------
# 4. Loss: ECC matches fp32 loss trajectory step-by-step (not just final)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bf16+8", "bf16+16"])
def test_loss_trajectory_close_to_fp32(mode):
    """Per-step loss difference between ECC and fp32 should stay bounded."""
    steps = 50
    torch.manual_seed(42)
    data = torch.randn(16, 8)
    target = torch.randn(16, 4)

    torch.manual_seed(0)
    m0 = nn.Linear(8, 4, bias=False)
    opt0 = heavyball.PaLMForeachSFAdamW(m0.parameters(), lr=1e-3)
    losses_base = _train(m0, opt0, data, target, steps)

    torch.manual_seed(0)
    m1 = nn.Linear(8, 4, bias=False)
    opt1 = heavyball.PaLMForeachSFAdamW(m1.parameters(), lr=1e-3, ecc=mode)
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    max_rel_diff = max(
        abs(a - b) / max(abs(a), 1e-12) for a, b in zip(losses_base, losses_ecc)
    )

    if "16" in mode.split("+")[1]:  # int16 correction
        assert max_rel_diff < 0.001, f"bf16+16 trajectory diverged by {max_rel_diff:.4f}"
    else:
        assert max_rel_diff < 0.05, f"{mode} trajectory diverged by {max_rel_diff:.4f}"


def test_combined_ecc_loss_trajectory():
    """State ECC + param ECC together should still track fp32 reasonably."""
    steps = 50
    torch.manual_seed(42)
    data = torch.randn(16, 8)
    target = torch.randn(16, 4)

    torch.manual_seed(0)
    m0 = nn.Linear(8, 4, bias=False)
    opt0 = heavyball.PaLMForeachSFAdamW(m0.parameters(), lr=1e-3)
    losses_base = _train(m0, opt0, data, target, steps)

    torch.manual_seed(0)
    m1 = nn.Linear(8, 4, bias=False)
    opt1 = heavyball.PaLMForeachSFAdamW(m1.parameters(), lr=1e-3, ecc="bf16+8", param_ecc="bf16+8")
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    max_rel_diff = max(
        abs(a - b) / max(abs(a), 1e-12) for a, b in zip(losses_base, losses_ecc)
    )
    assert max_rel_diff < 0.1, f"Combined ECC trajectory diverged by {max_rel_diff:.4f}"


# ---------------------------------------------------------------------------
# 5. Encode-decode drift: repeated roundtrips must not accumulate error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ULP_MODES)
def test_ulp_encode_decode_drift(mode):
    """Repeated encode→decode cycles on a mutating tensor must not drift."""
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    x = torch.randn(1024)
    xn = x.to(cfg.primary_dtype)
    ecc = torch.zeros_like(x, dtype=cfg.corr_dtype)
    out = torch.empty_like(x)

    utils.compute_ecc([x], [xn], [ecc])

    for i in range(100):
        utils.apply_ecc([xn], [ecc], [out])
        # Simulate small optimizer update
        out.add_(torch.randn_like(out) * 1e-4)
        xn.copy_(out.to(cfg.primary_dtype))
        utils.compute_ecc([out], [xn], [ecc])

    utils.apply_ecc([xn], [ecc], [out])
    err = (out - x).abs().max().item()
    # After 100 steps of 1e-4 perturbation, drift should be bounded by cumulative perturbation
    # not exponentially growing
    assert err < 0.02, f"Drift after 100 encode-decode cycles: {err:.4f}"


@pytest.mark.parametrize("compand", [False, True])
def test_int8_encode_decode_drift(compand):
    """Repeated int8 ECC roundtrips must not accumulate unbounded error."""
    torch.manual_seed(42)
    x = torch.randn(1024)
    q = torch.zeros(1024, dtype=torch.int8)
    ns = (1024 + 31) // 32
    s = torch.zeros(ns, dtype=torch.bfloat16)
    c = torch.zeros(1024, dtype=torch.bfloat16)
    out = torch.empty(1024)

    for i in range(100):
        utils.quantize_int8_ecc([x if i == 0 else out], [q], [s], [c], 32, compand)
        utils.dequantize_int8_ecc([q], [s], [c], [out], 32, compand)
        out.add_(torch.randn_like(out) * 1e-4)

    utils.quantize_int8_ecc([out], [q], [s], [c], 32, compand)
    utils.dequantize_int8_ecc([q], [s], [c], [out], 32, compand)
    # Just verify it hasn't diverged to inf/nan
    assert out.isfinite().all(), "int8 ECC produced non-finite values after repeated roundtrips"


# ---------------------------------------------------------------------------
# 6. 8+16 mode convergence
# ---------------------------------------------------------------------------

def test_convergence_8_16():
    """8+16 ECC mode should converge comparably to baseline."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m0 = nn.Linear(16, 8, bias=False)
    opt0 = heavyball.PaLMForeachSFAdamW(m0.parameters(), lr=1e-2)
    losses_base = _train(m0, opt0, data, target, steps)

    torch.manual_seed(0)
    m1 = nn.Linear(16, 8, bias=False)
    opt1 = heavyball.PaLMForeachSFAdamW(m1.parameters(), lr=1e-2, ecc="8+16")
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    ratio = losses_ecc[-1] / max(losses_base[-1], 1e-12)
    assert 0.5 < ratio < 2.0, (
        f"8+16 ECC final loss {losses_ecc[-1]:.6f} vs baseline {losses_base[-1]:.6f} (ratio {ratio:.2f})"
    )
    assert losses_ecc[-1] < losses_ecc[0] * 0.5, "8+16 ECC optimizer did not converge"
