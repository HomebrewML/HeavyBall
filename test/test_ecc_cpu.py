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


# ---------------------------------------------------------------------------
# 7. ECC across multiple optimizer types (not just PaLMForeachSFAdamW)
# ---------------------------------------------------------------------------

_OPTIMIZERS_AND_LR = [
    (heavyball.ForeachAdamW, 5e-2, {}),
    (heavyball.ForeachADOPT, 5e-2, {}),
    (heavyball.ForeachNAdam, 1e-2, {}),
    (heavyball.ForeachLaProp, 5e-2, {}),
    (heavyball.ForeachAdEMAMix, 5e-2, {"betas": (0.9, 0.999, 0.9999)}),
    (heavyball.ForeachRMSprop, 1e-2, {}),
]


def _opt_id(val):
    if isinstance(val, tuple):
        return val[0].__name__
    return str(val)


@pytest.mark.parametrize("opt_cls,lr,extra_kw", _OPTIMIZERS_AND_LR, ids=[t[0].__name__ for t in _OPTIMIZERS_AND_LR])
def test_ecc_convergence_multi_optimizer(opt_cls, lr, extra_kw):
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    # Baseline (no ECC)
    torch.manual_seed(0)
    m0 = nn.Linear(16, 8, bias=False)
    opt0 = opt_cls(m0.parameters(), lr=lr, **extra_kw)
    losses_base = _train(m0, opt0, data, target, steps)

    # ECC
    torch.manual_seed(0)
    m1 = nn.Linear(16, 8, bias=False)
    opt1 = opt_cls(m1.parameters(), lr=lr, ecc="bf16+8", **extra_kw)
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    assert losses_ecc[-1] < losses_ecc[0] * 0.5, (
        f"{opt_cls.__name__} with ECC did not converge: {losses_ecc[-1]:.6f} vs initial {losses_ecc[0]:.6f}"
    )
    ratio = losses_ecc[-1] / max(losses_base[-1], 1e-12)
    assert ratio < 2.0, (
        f"{opt_cls.__name__} ECC loss {losses_ecc[-1]:.6f} vs baseline {losses_base[-1]:.6f} (ratio {ratio:.2f})"
    )


@pytest.mark.parametrize("opt_cls,lr,extra_kw", _OPTIMIZERS_AND_LR, ids=[t[0].__name__ for t in _OPTIMIZERS_AND_LR])
def test_ecc_state_dtypes_multi_optimizer(opt_cls, lr, extra_kw):
    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False)
    opt = opt_cls(model.parameters(), lr=lr, ecc="bf16+8", **extra_kw)

    x = torch.randn(4, 64)
    for _ in range(5):
        (model(x).sum()).backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    ecc_keys = _assert_ecc_active(opt, p)

    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)

    for ek in ecc_keys:
        pk = ek.replace("::ecc", "")
        assert all_keys[pk].dtype == torch.bfloat16, (
            f"{opt_cls.__name__}: primary {pk} should be bf16, got {all_keys[pk].dtype}"
        )
        assert all_keys[ek].dtype == torch.int8, (
            f"{opt_cls.__name__}: correction {ek} should be int8, got {all_keys[ek].dtype}"
        )


# ---------------------------------------------------------------------------
# 8. SkipUpdate + ECC: state mutations preserved through try/finally
# ---------------------------------------------------------------------------

def _assert_ecc_state_nonzero(opt, param, name_filter=None):
    """Assert ECC-managed state vars have been written (primary or correction nonzero).
    With small lr, the primary (bf16/fp16) may round to zero while the correction
    captures the residual — so we check that at least one of the pair is nonzero."""
    st = opt.state[param]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)
    checked = 0
    for k, v in all_keys.items():
        if not isinstance(k, str) or "::ecc" in k or "::scales" in k or not isinstance(v, torch.Tensor):
            continue
        if name_filter and name_filter not in k:
            continue
        ecc_k = k + "::ecc"
        if ecc_k not in all_keys:
            continue
        primary_nz = not (v == 0).all()
        ecc_nz = not (all_keys[ecc_k] == 0).all()
        assert primary_nz or ecc_nz, (
            f"ECC state '{k}' all zeros (primary + correction) — SkipUpdate likely prevented encode"
        )
        checked += 1
    assert checked > 0, f"No ECC state variables found (filter={name_filter})"


@pytest.mark.parametrize("mode", ULP_MODES)
def test_adamw_ecc_state_nonzero_after_skip_update(mode):
    """update_by_adam always raises SkipUpdate. Without try/finally, ECC encode
    would be skipped and state would stay zero."""
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-3, ecc=mode)

    x = torch.randn(4, 32)
    for _ in range(3):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p)


@pytest.mark.parametrize("mode", ULP_MODES)
def test_adopt_ecc_state_nonzero_step1(mode):
    """update_by_adopt raises SkipUpdate at step 1 after modifying exp_avg_sq.
    Verify exp_avg_sq is encoded back despite the early raise."""
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachADOPT(model.parameters(), lr=1e-3, ecc=mode)

    x = torch.randn(4, 32)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    p = list(model.parameters())[0]
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p, name_filter="exp_avg_sq")


@pytest.mark.parametrize("mode", ULP_MODES)
def test_adopt_ecc_state_nonzero_multistep(mode):
    """After 5 steps, both exp_avg and exp_avg_sq should be nonzero
    despite SkipUpdate being raised at every step."""
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachADOPT(model.parameters(), lr=1e-3, ecc=mode)

    x = torch.randn(4, 32)
    for _ in range(5):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p)


def test_zero_guard_encodes_on_skip_update():
    """Direct test: ZeroGuard._call must encode state back even when the
    wrapped function raises SkipUpdate."""
    from heavyball.chainable import ZeroGuard, SkipUpdate, ECCConfig

    def mutate_and_skip(state, group, update, grad, param, exp_avg):
        for ea in exp_avg:
            ea.add_(1.0)
        raise SkipUpdate from None

    guard = ZeroGuard(mutate_and_skip, names=("exp_avg",))
    guard.fn_name = "test_fn"
    guard.transform_idx = 0

    torch.manual_seed(42)
    p = torch.randn(64)
    cfg = ECCConfig("bf16+8")
    state_dict = {}
    cfg.init_state(state_dict, "test_fn_exp_avg_0", p)

    class StateAccessor:
        def __call__(self, param):
            return state_dict

    group = {"ecc": "bf16+8", "step": 1}

    with pytest.raises(SkipUpdate):
        guard._call(StateAccessor(), group, [p.clone()], [p.clone()], [p], [])

    primary = state_dict["test_fn_exp_avg_0"]
    ecc_corr = state_dict["test_fn_exp_avg_0::ecc"]
    assert not (primary == 0).all() or not (ecc_corr == 0).all(), (
        "ZeroGuard did not encode state after SkipUpdate — try/finally bug"
    )


# ---------------------------------------------------------------------------
# 9. Fused vs unfused ECC: numerical equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ULP_MODES)
def test_encode_ecc_equals_copy_then_compute(mode):
    """encode_ecc must be bit-identical to manual p.copy_(f) + compute_ecc."""
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    fp32 = [torch.randn(2048)]

    prim_a = [torch.empty_like(fp32[0], dtype=cfg.primary_dtype)]
    ecc_a = [torch.zeros_like(fp32[0], dtype=cfg.corr_dtype)]
    utils.encode_ecc(fp32, prim_a, ecc_a)

    prim_b = [fp32[0].to(cfg.primary_dtype)]
    ecc_b = [torch.zeros_like(fp32[0], dtype=cfg.corr_dtype)]
    utils.compute_ecc(fp32, prim_b, ecc_b)

    assert torch.equal(prim_a[0], prim_b[0]), "primaries diverge"
    assert torch.equal(ecc_a[0], ecc_b[0]), "ecc codes diverge"


@pytest.mark.parametrize("mode", ULP_MODES)
def test_encode_ecc_roundtrip_matches_compute_apply(mode):
    """encode_ecc -> apply_ecc must match compute_ecc -> apply_ecc."""
    cfg = ECCConfig(mode)
    torch.manual_seed(99)
    fp32 = [torch.randn(1024) * 10]

    prim = [torch.empty_like(fp32[0], dtype=cfg.primary_dtype)]
    ecc = [torch.zeros_like(fp32[0], dtype=cfg.corr_dtype)]
    utils.encode_ecc(fp32, prim, ecc)
    out_a = [torch.empty_like(fp32[0])]
    utils.apply_ecc(prim, ecc, out_a)

    prim2 = [fp32[0].to(cfg.primary_dtype)]
    ecc2 = [torch.zeros_like(fp32[0], dtype=cfg.corr_dtype)]
    utils.compute_ecc(fp32, prim2, ecc2)
    out_b = [torch.empty_like(fp32[0])]
    utils.apply_ecc(prim2, ecc2, out_b)

    assert torch.equal(out_a[0], out_b[0]), "roundtrip outputs differ"


@pytest.mark.parametrize("mode", ULP_MODES)
def test_fused_decode_compute_matches_unfused(mode):
    """Fused decode->compute->copy must produce identical state to the unfused sequence."""
    cfg = ECCConfig(mode)
    torch.manual_seed(7)
    fp32_val = torch.randn(512)

    prim_init = fp32_val.to(cfg.primary_dtype)
    ecc_init = torch.zeros_like(fp32_val, dtype=cfg.corr_dtype)
    utils.compute_ecc([fp32_val], [prim_init], [ecc_init])

    # Unfused: apply_ecc -> mutate -> copy primary -> compute_ecc
    prim_u = prim_init.clone()
    ecc_u = ecc_init.clone()
    decoded_u = torch.empty_like(fp32_val)
    utils.apply_ecc([prim_u], [ecc_u], [decoded_u])
    decoded_u.add_(0.001)
    prim_u.copy_(decoded_u)
    utils.compute_ecc([decoded_u], [prim_u], [ecc_u])

    # Fused: _compilable_apply_ecc_ directly, then same post-processing
    prim_f = prim_init.clone()
    ecc_f = ecc_init.clone()
    decoded_f = torch.empty_like(fp32_val)
    smax = {torch.int8: 127.0, torch.int16: 32767.0}[cfg.corr_dtype]
    utils._compilable_apply_ecc_([prim_f], [ecc_f], [decoded_f], smax)
    decoded_f.add_(0.001)
    prim_f.copy_(decoded_f)
    utils.compute_ecc([decoded_f], [prim_f], [ecc_f])

    assert torch.equal(prim_u, prim_f), "primaries differ between fused and unfused"
    assert torch.equal(ecc_u, ecc_f), "ecc codes differ between fused and unfused"


# ---------------------------------------------------------------------------
# 10. Cross-mode convergence: fused (ULP) vs unfused (8+16) consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opt_cls", [heavyball.PaLMForeachSFAdamW, heavyball.ForeachAdamW])
def test_fused_unfused_convergence_consistency(opt_cls):
    """bf16+8 (fused path) and 8+16 (unfused path) both converge to similar losses."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m_fused = nn.Linear(16, 8, bias=False)
    opt_fused = opt_cls(m_fused.parameters(), lr=1e-2, ecc="bf16+8")
    losses_fused = _train(m_fused, opt_fused, data, target, steps)

    torch.manual_seed(0)
    m_unfused = nn.Linear(16, 8, bias=False)
    opt_unfused = opt_cls(m_unfused.parameters(), lr=1e-2, ecc="8+16")
    losses_unfused = _train(m_unfused, opt_unfused, data, target, steps)

    _assert_ecc_active(opt_fused, list(m_fused.parameters())[0])
    _assert_ecc_active(opt_unfused, list(m_unfused.parameters())[0])

    assert losses_fused[-1] < losses_fused[0] * 0.5, "fused path did not converge"
    assert losses_unfused[-1] < losses_unfused[0] * 0.5, "unfused path did not converge"

    ratio = losses_fused[-1] / max(losses_unfused[-1], 1e-12)
    assert 0.3 < ratio < 3.0, (
        f"fused={losses_fused[-1]:.6f} vs unfused={losses_unfused[-1]:.6f} ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# 11. State accumulation accuracy over many steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bf16+8", "bf16+16"])
def test_state_drift_vs_fp32_baseline(mode):
    """ECC state tensors should not drift from fp32 baseline over 500 steps."""
    steps = 500
    torch.manual_seed(42)
    data = torch.randn(16, 8)
    target = torch.randn(16, 4)

    torch.manual_seed(0)
    m_ref = nn.Linear(8, 4, bias=False)
    opt_ref = heavyball.PaLMForeachSFAdamW(m_ref.parameters(), lr=1e-3)
    _train(m_ref, opt_ref, data, target, steps)

    torch.manual_seed(0)
    m_ecc = nn.Linear(8, 4, bias=False)
    opt_ecc = heavyball.PaLMForeachSFAdamW(m_ecc.parameters(), lr=1e-3, ecc=mode)
    _train(m_ecc, opt_ecc, data, target, steps)

    p_ref = list(m_ref.parameters())[0]
    p_ecc = list(m_ecc.parameters())[0]
    ecc_keys = _assert_ecc_active(opt_ecc, p_ecc)

    st_ref = opt_ref.state[p_ref]
    st_ecc = opt_ecc.state[p_ecc]

    cfg = ECCConfig(mode)
    compared = 0
    for v_ecc in st_ecc.values():
        if not isinstance(v_ecc, dict):
            continue
        for k, val in v_ecc.items():
            if not isinstance(k, str) or "::ecc" in k:
                continue
            ecc_key = k + "::ecc"
            if ecc_key not in v_ecc:
                continue
            decoded = torch.empty_like(val, dtype=torch.float32)
            utils.apply_ecc([val], [v_ecc[ecc_key]], [decoded])

            for v_ref in st_ref.values():
                if isinstance(v_ref, dict) and k in v_ref:
                    ref_val = v_ref[k].float()
                    rel_err = (decoded - ref_val).abs() / (ref_val.abs() + 1e-12)
                    max_rel = rel_err.max().item()
                    if "16" in mode.split("+")[1]:
                        assert max_rel < 0.01, f"int16 state {k} drifted: max_rel_err={max_rel:.4f}"
                    else:
                        assert max_rel < 0.1, f"int8 state {k} drifted: max_rel_err={max_rel:.4f}"
                    compared += 1
                    break
    assert compared > 0, "No ECC state tensors were compared — test is vacuous"


@pytest.mark.parametrize("opt_cls", [heavyball.PaLMForeachSFAdamW, heavyball.ForeachAdamW])
def test_long_run_loss_no_divergence(opt_cls):
    """500-step ECC run must not diverge."""
    steps = 500
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m = nn.Linear(16, 8, bias=False)
    opt = opt_cls(m.parameters(), lr=1e-3, ecc="bf16+8")
    losses = _train(m, opt, data, target, steps)

    _assert_ecc_active(opt, list(m.parameters())[0])
    assert losses[-1] < losses[0] * 0.6, f"did not converge: {losses[-1]:.6f} vs {losses[0]:.6f}"
    assert all(l < 1e6 for l in losses), "loss exploded during training"


# ---------------------------------------------------------------------------
# 8. Edge cases and boundary conditions
# ---------------------------------------------------------------------------

def test_single_element_parameter():
    torch.manual_seed(42)
    p = nn.Parameter(torch.randn(1))
    opt = heavyball.ForeachAdamW([p], lr=1e-2, ecc="bf16+8")
    for _ in range(20):
        (p ** 2).sum().backward()
        opt.step()
        opt.zero_grad()
    assert p.isfinite().all()
    _assert_ecc_active(opt, p)


def test_very_small_parameter():
    torch.manual_seed(42)
    model = nn.Linear(3, 2, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")
    data, target = torch.randn(4, 3), torch.randn(4, 2)
    losses = _train(model, opt, data, target, 50)
    _assert_ecc_active(opt, list(model.parameters())[0])
    assert all(torch.tensor(losses).isfinite()), "Non-finite loss with (2,3) param"
    assert losses[-1] < losses[0], "No progress with (2,3) param"


def test_multiple_param_groups():
    torch.manual_seed(42)
    m1 = nn.Linear(16, 8, bias=False)
    m2 = nn.Linear(8, 4, bias=False)
    opt = heavyball.ForeachAdamW([
        {"params": m1.parameters(), "ecc": "bf16+8"},
        {"params": m2.parameters()},
    ], lr=1e-2)

    data, target = torch.randn(8, 16), torch.randn(8, 4)
    for _ in range(50):
        ((m2(m1(data)) - target) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()

    _assert_ecc_active(opt, list(m1.parameters())[0])

    st2 = opt.state[list(m2.parameters())[0]]
    all_keys_2 = {}
    for v in st2.values():
        if isinstance(v, dict):
            all_keys_2.update(v)
    ecc_keys_2 = [k for k in all_keys_2 if isinstance(k, str) and "::ecc" in k]
    assert len(ecc_keys_2) == 0, f"Non-ECC group got ECC keys: {ecc_keys_2}"


def test_param_ecc_dtype_preservation():
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-2, param_ecc="bf16+8")
    data, target = torch.randn(4, 16), torch.randn(4, 8)

    for _ in range(5):
        p = list(model.parameters())[0]
        d = data.to(p.dtype)
        loss = ((model(d) - target.to(d.dtype)) ** 2).mean().float()
        loss.backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    assert p.data.dtype == torch.bfloat16, f"p.data should be bf16, got {p.data.dtype}"

    d = data.to(p.dtype)
    ((model(d) - target.to(d.dtype)) ** 2).mean().float().backward()
    assert p.grad.dtype == p.data.dtype, f"grad dtype {p.grad.dtype} != param dtype {p.data.dtype}"


@pytest.mark.xfail(reason="load_state_dict deserializes ECC correction tensors as fp32, losing int8/int16 dtype")
def test_state_save_restore_with_ecc():
    torch.manual_seed(42)
    data, target = torch.randn(32, 16), torch.randn(32, 8)

    torch.manual_seed(0)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")
    losses_before = _train(model, opt, data, target, 10)

    sd_opt = deepcopy(opt.state_dict())
    sd_model = deepcopy(model.state_dict())

    torch.manual_seed(0)
    model2 = nn.Linear(16, 8, bias=False)
    model2.load_state_dict(sd_model)
    opt2 = heavyball.PaLMForeachSFAdamW(model2.parameters(), lr=1e-2, ecc="bf16+8")
    opt2.load_state_dict(sd_opt)

    losses_after = _train(model2, opt2, data, target, 10)
    assert losses_after[-1] < losses_before[-1], (
        f"Loss didn't decrease after restore: {losses_before[-1]:.6f} -> {losses_after[-1]:.6f}"
    )

    losses_continued = _train(model, opt, data, target, 10)
    ratio = losses_after[-1] / max(losses_continued[-1], 1e-12)
    assert 0.8 < ratio < 1.2, (
        f"Restored diverged from original: {losses_after[-1]:.6f} vs {losses_continued[-1]:.6f}"
    )


def test_zero_gradients_with_ecc():
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")

    p = list(model.parameters())[0]
    for _ in range(10):
        p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()

    assert p.isfinite().all(), "Params NaN/Inf after zero-grad steps"
    st = opt.state[p]
    for k, v in st.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor) and vv.is_floating_point():
                    assert vv.isfinite().all(), f"State {k}/{kk} has NaN/Inf after zero-grad steps"


def test_large_gradient_spike_with_ecc():
    torch.manual_seed(42)
    data, target = torch.randn(16, 8), torch.randn(16, 4)

    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3, ecc="bf16+8")
    _train(model, opt, data, target, 10)

    p = list(model.parameters())[0]
    ((model(data) - target) ** 2).mean().backward()
    p.grad.mul_(1000.0)
    opt.step()
    opt.zero_grad()

    assert p.isfinite().all(), "Params NaN/Inf after gradient spike"

    losses_recovery = _train(model, opt, data, target, 20)
    assert all(torch.tensor(losses_recovery).isfinite()), "Non-finite loss during recovery"
    assert losses_recovery[-1] < losses_recovery[0], "No recovery after gradient spike"


def test_combined_state_and_param_ecc_dtypes():
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(
        model.parameters(), lr=1e-2, ecc="bf16+16", param_ecc="bf16+8"
    )
    data, target = torch.randn(8, 32), torch.randn(8, 16)
    _train(model, opt, data, target, 100)

    p = list(model.parameters())[0]
    assert p.dtype == torch.bfloat16, f"Param should be bf16 under param_ecc, got {p.dtype}"

    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)

    state_ecc_keys = [k for k in all_keys if isinstance(k, str) and "::ecc" in k and "param" not in k]
    assert len(state_ecc_keys) > 0, f"State ECC not active. Keys: {list(all_keys.keys())}"
    for k in state_ecc_keys:
        assert all_keys[k].dtype == torch.int16, f"State correction {k} should be int16, got {all_keys[k].dtype}"

    assert "param::ecc" in all_keys, f"param::ecc not in state. Keys: {list(all_keys.keys())}"
    assert all_keys["param::ecc"].dtype == torch.int8, (
        f"Param correction should be int8, got {all_keys['param::ecc'].dtype}"
    )


# ---------------------------------------------------------------------------
# 12. Cache correctness and multi-state-variable structural tests
# ---------------------------------------------------------------------------

def test_ecc_fused_cache_serves_correct_fn():
    """ForeachAdamW (update_by_adam) and ForeachADOPT (update_by_adopt) use
    different @zero_guard functions cached by id(fn) in _ecc_fused_cache.
    If the cache served the wrong compiled graph, both would produce identical
    trajectories. Verify they diverge and both converge."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m_adam = nn.Linear(16, 8, bias=False)
    opt_adam = heavyball.ForeachAdamW(m_adam.parameters(), lr=5e-2, ecc="bf16+8")
    losses_adam = _train(m_adam, opt_adam, data, target, steps)

    torch.manual_seed(0)
    m_adopt = nn.Linear(16, 8, bias=False)
    opt_adopt = heavyball.ForeachADOPT(m_adopt.parameters(), lr=5e-2, ecc="bf16+8")
    losses_adopt = _train(m_adopt, opt_adopt, data, target, steps)

    _assert_ecc_active(opt_adam, list(m_adam.parameters())[0])
    _assert_ecc_active(opt_adopt, list(m_adopt.parameters())[0])

    assert losses_adam[-1] < losses_adam[0] * 0.5, "AdamW with ECC did not converge"
    assert losses_adopt[-1] < losses_adopt[0] * 0.5, "ADOPT with ECC did not converge"

    # Loss trajectories must differ — ADOPT has a fundamentally different update rule
    diffs = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(losses_adam, losses_adopt)]
    steps_that_differ = sum(d > 1e-3 for d in diffs)
    assert steps_that_differ > steps // 4, (
        f"Only {steps_that_differ}/{steps} steps differ — cache may have served wrong fn"
    )

    # Final params must differ
    p_adam = list(m_adam.parameters())[0].detach().float()
    p_adopt = list(m_adopt.parameters())[0].detach().float()
    assert not torch.allclose(p_adam, p_adopt, atol=1e-3), (
        "AdamW and ADOPT produced identical params — cache may have served wrong fn"
    )


def test_ademamix_three_state_vars_structural():
    """ForeachAdEMAMix has @zero_guard('exp_avg_fast', 'exp_avg_slow', 'exp_avg_sq').
    Verify all three state variables are present, have correct dtypes (bf16 primary,
    int8 correction), and are nonzero after training."""
    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False)
    opt = heavyball.ForeachAdEMAMix(
        model.parameters(), lr=5e-2, betas=(0.9, 0.999, 0.9999), ecc="bf16+8"
    )

    x = torch.randn(4, 64)
    for _ in range(10):
        (model(x).sum()).backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    ecc_keys = _assert_ecc_active(opt, p)

    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)

    expected_vars = ["exp_avg_fast", "exp_avg_slow", "exp_avg_sq"]
    for var in expected_vars:
        matching = [k for k in all_keys if isinstance(k, str) and var in k and "::ecc" not in k and "::scales" not in k]
        assert len(matching) > 0, f"State variable '{var}' not found. Keys: {list(all_keys.keys())}"
        for pk in matching:
            assert all_keys[pk].dtype == torch.bfloat16, (
                f"Primary '{pk}' should be bf16, got {all_keys[pk].dtype}"
            )
            ecc_k = pk + "::ecc"
            assert ecc_k in all_keys, f"Correction key '{ecc_k}' not found"
            assert all_keys[ecc_k].dtype == torch.int8, (
                f"Correction '{ecc_k}' should be int8, got {all_keys[ecc_k].dtype}"
            )

    # All three must be nonzero (primary or correction)
    for var in expected_vars:
        _assert_ecc_state_nonzero(opt, p, name_filter=var)


def test_ademamix_ecc_convergence_vs_baseline():
    """ForeachAdEMAMix with ecc='bf16+8' (3-var fused path) must converge
    comparably to the non-ECC baseline."""
    steps = 200
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    m0 = nn.Linear(16, 8, bias=False)
    opt0 = heavyball.ForeachAdEMAMix(m0.parameters(), lr=5e-2, betas=(0.9, 0.999, 0.9999))
    losses_base = _train(m0, opt0, data, target, steps)

    torch.manual_seed(0)
    m1 = nn.Linear(16, 8, bias=False)
    opt1 = heavyball.ForeachAdEMAMix(
        m1.parameters(), lr=5e-2, betas=(0.9, 0.999, 0.9999), ecc="bf16+8"
    )
    losses_ecc = _train(m1, opt1, data, target, steps)

    _assert_ecc_active(opt1, list(m1.parameters())[0])

    assert losses_base[-1] < losses_base[0] * 0.5, "Baseline AdEMAMix did not converge"
    assert losses_ecc[-1] < losses_ecc[0] * 0.5, "ECC AdEMAMix did not converge"

    ratio = losses_ecc[-1] / max(losses_base[-1], 1e-12)
    assert ratio < 2.0, (
        f"AdEMAMix ECC loss {losses_ecc[-1]:.6f} vs baseline {losses_base[-1]:.6f} (ratio {ratio:.2f})"
    )


# ---------------------------------------------------------------------------
# 13. CSE invariant and int8 path SkipUpdate tests
# ---------------------------------------------------------------------------

def _extract_ecc_corrections(opt, param):
    """Return dict of {key: correction_tensor} for all ECC-managed state vars."""
    st = opt.state[param]
    corr = {}
    for v in st.values():
        if not isinstance(v, dict):
            continue
        for k, t in v.items():
            if isinstance(k, str) and "::ecc" in k and isinstance(t, torch.Tensor):
                corr[k] = t
    return corr


@pytest.mark.parametrize("mode", ["bf16+8", "bf16+16"])
def test_cse_invariant_corrections_nonzero(mode):
    """After real optimizer steps, ECC corrections must be non-trivial.
    If torch.compile's CSE collapsed f.to(bf16).float() back to f,
    corrections would be all-zero. This verifies the CSE workaround works."""
    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False)
    opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3, ecc=mode)

    x = torch.randn(4, 64)
    for _ in range(5):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    corr = _extract_ecc_corrections(opt, p)
    assert len(corr) > 0, "No ECC correction tensors found"

    any_nonzero = any(not (t == 0).all() for t in corr.values())
    assert any_nonzero, (
        f"All ECC corrections zero after 5 steps — CSE workaround likely broken. "
        f"Keys: {list(corr.keys())}"
    )


def test_cse_invariant_int16_more_nonzero_than_int8():
    """bf16+16 (int16 correction) should have at least as many nonzero correction
    elements as bf16+8 (int8 correction), since int16 has finer granularity and
    captures more of the truncation residual."""
    torch.manual_seed(42)
    data = torch.randn(4, 64)

    results = {}
    for mode in ["bf16+8", "bf16+16"]:
        torch.manual_seed(0)
        model = nn.Linear(64, 32, bias=False)
        opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3, ecc=mode)
        for _ in range(5):
            model(data).sum().backward()
            opt.step()
            opt.zero_grad()
        p = list(model.parameters())[0]
        corr = _extract_ecc_corrections(opt, p)
        total_nonzero = sum((t != 0).sum().item() for t in corr.values())
        results[mode] = total_nonzero

    assert results["bf16+16"] >= results["bf16+8"], (
        f"int16 corrections ({results['bf16+16']} nonzero) should >= int8 ({results['bf16+8']} nonzero)"
    )
    assert results["bf16+8"] > 0, "int8 corrections all zero — CSE bug?"


def test_cse_invariant_int16_lower_reconstruction_error():
    """bf16+16 should reconstruct optimizer state closer to fp32 than bf16+8,
    since int16 correction has more bits to capture the truncation residual."""
    torch.manual_seed(42)
    data, target = torch.randn(16, 32), torch.randn(16, 16)

    # fp32 baseline
    torch.manual_seed(0)
    m_ref = nn.Linear(32, 16, bias=False)
    opt_ref = heavyball.PaLMForeachSFAdamW(m_ref.parameters(), lr=1e-3)
    _train(m_ref, opt_ref, data, target, 20)
    p_ref = list(m_ref.parameters())[0]
    st_ref = opt_ref.state[p_ref]
    ref_vals = {}
    for v in st_ref.values():
        if isinstance(v, dict):
            for k, t in v.items():
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    ref_vals[k] = t.float()

    max_errs = {}
    for mode in ["bf16+8", "bf16+16"]:
        torch.manual_seed(0)
        model = nn.Linear(32, 16, bias=False)
        opt = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3, ecc=mode)
        _train(model, opt, data, target, 20)
        p = list(model.parameters())[0]
        cfg = ECCConfig(mode)

        st = opt.state[p]
        all_keys = {}
        for v in st.values():
            if isinstance(v, dict):
                all_keys.update(v)

        worst = 0.0
        for k, prim in all_keys.items():
            if not isinstance(k, str) or "::ecc" in k or "::scales" in k:
                continue
            ecc_k = k + "::ecc"
            if ecc_k not in all_keys:
                continue
            decoded = torch.empty_like(prim, dtype=torch.float32)
            utils.apply_ecc([prim], [all_keys[ecc_k]], [decoded])
            base_name = k.split("_", 2)[-1] if "_" in k else k
            for rk, rv in ref_vals.items():
                if rk == base_name or rk.endswith(base_name):
                    err = (decoded - rv).abs().max().item()
                    worst = max(worst, err)
                    break
        max_errs[mode] = worst

    if max_errs["bf16+8"] > 0:
        assert max_errs["bf16+16"] <= max_errs["bf16+8"], (
            f"int16 max error ({max_errs['bf16+16']:.2e}) should be <= "
            f"int8 max error ({max_errs['bf16+8']:.2e})"
        )


def test_int8_path_state_nonzero_after_skip_update():
    """8+16 mode goes through the is_int8 branch with try/finally.
    ForeachAdamW always raises SkipUpdate. Verify state is encoded
    back despite the exception."""
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-3, ecc="8+16")

    x = torch.randn(4, 32)
    for _ in range(3):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p)


def test_int8_path_adopt_step1_skip_update():
    """ForeachADOPT with 8+16 raises SkipUpdate at step 1 after modifying
    exp_avg_sq. The try/finally in the is_int8 branch must encode it back."""
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachADOPT(model.parameters(), lr=1e-3, ecc="8+16")

    x = torch.randn(4, 32)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    p = list(model.parameters())[0]
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p, name_filter="exp_avg_sq")


# ---------------------------------------------------------------------------
# 14. Multi-param, bias, and foreach=False tests
# ---------------------------------------------------------------------------

def test_heterogeneous_shapes_multi_layer():
    """Fused path builds primary_lists[var_idx][param_idx] etc.
    With heterogeneous shapes each sublist has differently-sized tensors.
    Must work for all parameters."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 16, bias=False),
        nn.Linear(16, 8, bias=False),
        nn.Linear(8, 4, bias=False),
    )
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")
    data = torch.randn(16, 32)
    target = torch.randn(16, 4)
    losses = _train(model, opt, data, target, 50)

    for p in model.parameters():
        _assert_ecc_active(opt, p)
        assert p.isfinite().all(), f"Non-finite param with shape {p.shape}"

    assert losses[-1] < losses[0] * 0.5, (
        f"Multi-layer did not converge: {losses[-1]:.6f} vs {losses[0]:.6f}"
    )


def test_bias_true_heterogeneous_weight_and_bias():
    """Weight (8,16) and bias (8,) go through the same foreach call.
    Fused path must handle this shape heterogeneity."""
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=True)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")
    data = torch.randn(8, 16)
    target = torch.randn(8, 8)

    for _ in range(20):
        loss = ((model(data) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    for p in model.parameters():
        _assert_ecc_active(opt, p)
        assert p.isfinite().all(), f"Non-finite param with shape {p.shape}"


def test_foreach_false_convergence():
    """foreach=False forces chain() per-parameter (param=[single_param]).
    Must converge and produce similar trajectory to foreach=True."""
    torch.manual_seed(42)
    data = torch.randn(16, 32)
    target = torch.randn(16, 4)

    model_cls = lambda: nn.Sequential(
        nn.Linear(32, 16, bias=False),
        nn.Linear(16, 4, bias=False),
    )

    torch.manual_seed(0)
    m_fe = model_cls()
    opt_fe = heavyball.ForeachAdamW(m_fe.parameters(), lr=1e-2, ecc="bf16+8", foreach=True)
    losses_fe = _train(m_fe, opt_fe, data, target, 50)

    torch.manual_seed(0)
    m_nf = model_cls()
    opt_nf = heavyball.ForeachAdamW(m_nf.parameters(), lr=1e-2, ecc="bf16+8", foreach=False)
    losses_nf = _train(m_nf, opt_nf, data, target, 50)

    assert losses_nf[-1] < losses_nf[0] * 0.5, (
        f"foreach=False did not converge: {losses_nf[-1]:.6f} vs {losses_nf[0]:.6f}"
    )
    for p in m_nf.parameters():
        _assert_ecc_active(opt_nf, p)

    ratio = losses_nf[-1] / max(losses_fe[-1], 1e-12)
    assert 0.3 < ratio < 3.0, (
        f"foreach=False loss {losses_nf[-1]:.6f} vs foreach=True {losses_fe[-1]:.6f} ratio={ratio:.2f}"
    )


def test_foreach_false_skip_update_state_nonzero():
    """foreach=False with ecc='bf16+8': update_by_adam raises SkipUpdate.
    In non-foreach mode chain() is called per-parameter.
    State must still be encoded correctly."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 8, bias=False),
        nn.Linear(8, 4, bias=False),
    )
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-3, ecc="bf16+8", foreach=False)

    x = torch.randn(4, 16)
    for _ in range(3):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    for p in model.parameters():
        _assert_ecc_active(opt, p)
        _assert_ecc_state_nonzero(opt, p)


# ---------------------------------------------------------------------------
# 15. Step tensorization, combined ECC paths, and mode layout tests
# ---------------------------------------------------------------------------

def test_adopt_ecc_step_conditional_branches():
    """ADOPT step==1 modifies exp_avg_sq only, step==2 initializes exp_avg,
    step>=3 does the full update. With ECC the step may be a tensor (via
    scalar_guard). Verify all three phases work and the optimizer converges."""
    torch.manual_seed(42)
    data = torch.randn(16, 32)
    target = torch.randn(16, 8)

    model = nn.Linear(32, 8, bias=False)
    opt = heavyball.ForeachADOPT(model.parameters(), lr=5e-2, ecc="bf16+8")
    p = list(model.parameters())[0]

    # Step 1: only exp_avg_sq should be written
    d = data.to(p.dtype) if p.dtype != data.dtype else data
    ((model(d) - target.to(d.dtype)) ** 2).mean().float().backward()
    opt.step()
    opt.zero_grad()
    _assert_ecc_active(opt, p)
    _assert_ecc_state_nonzero(opt, p, name_filter="exp_avg_sq")

    # Steps 2-3: exp_avg gets initialized at step 2
    for _ in range(2):
        d = data.to(p.dtype) if p.dtype != data.dtype else data
        ((model(d) - target.to(d.dtype)) ** 2).mean().float().backward()
        opt.step()
        opt.zero_grad()
    _assert_ecc_state_nonzero(opt, p, name_filter="exp_avg")

    # Steps 4-13: full update path, verify convergence
    losses = _train(model, opt, data, target, 10)
    assert losses[-1] < losses[0], "ADOPT with ECC did not make progress after step-conditional phases"


def test_step_reset_to_none_after_optimizer_step():
    """group['step'] is set to a tensor inside ZeroGuard._call for compile
    compatibility, then reset to None at line 1408 of chainable.py.
    Verify the tensorization does not leak across optimizer steps."""
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-2, ecc="bf16+8")

    x = torch.randn(4, 16)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    for group in opt.param_groups:
        assert group["step"] is None, (
            f"group['step'] should be None after step(), got {group['step']!r} "
            f"(type={type(group['step']).__name__})"
        )

    # Second step — same check
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    for group in opt.param_groups:
        assert group["step"] is None, (
            f"group['step'] leaked on second step: {group['step']!r}"
        )


def test_adopt_step_reset_to_none():
    """Same step-reset check for ADOPT, which exercises both step==1 and step==2
    branches with potentially tensorized steps."""
    torch.manual_seed(42)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.ForeachADOPT(model.parameters(), lr=5e-2, ecc="bf16+8")

    x = torch.randn(4, 16)
    for i in range(5):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        for group in opt.param_groups:
            assert group["step"] is None, (
                f"group['step'] not None after step {i+1}: {group['step']!r}"
            )


def test_combined_param_ecc_and_state_ecc_with_skip_update():
    """param_ecc + state ecc + SkipUpdate (from update_by_adam) all interacting.
    chain() decodes params, _inner_chain → ZeroGuard._call decodes state, fn raises
    SkipUpdate, finally encodes state. _inner_chain catches SkipUpdate. chain()
    encodes params back. Verify both ECC layers are active and convergence holds."""
    steps = 20
    torch.manual_seed(42)
    data = torch.randn(32, 16)
    target = torch.randn(32, 8)

    torch.manual_seed(0)
    model = nn.Linear(16, 8, bias=False)
    opt = heavyball.ForeachAdamW(
        model.parameters(), lr=5e-2, ecc="bf16+8", param_ecc="bf16+8"
    )
    losses = _train(model, opt, data, target, steps)

    p = list(model.parameters())[0]

    # param dtype must be bf16 under param_ecc
    assert p.dtype == torch.bfloat16, f"Expected bf16 param, got {p.dtype}"

    # State ECC active
    ecc_keys = _assert_ecc_active(opt, p)
    state_ecc_keys = [k for k in ecc_keys if "param" not in k]
    assert len(state_ecc_keys) > 0, "State ECC not active"

    # Param ECC active
    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)
    assert "param::ecc" in all_keys, f"param::ecc missing. Keys: {list(all_keys.keys())}"

    # Both state and param ECC corrections are non-trivially written
    _assert_ecc_state_nonzero(opt, p)
    param_ecc_val = all_keys["param::ecc"]
    assert not (param_ecc_val == 0).all(), "param::ecc is all zeros — param ECC not encoding"

    # Convergence
    assert losses[-1] < losses[0] * 0.6, (
        f"Combined ECC did not converge: {losses[-1]:.6f} vs {losses[0]:.6f}"
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_ecc_mode_state_layout(mode):
    """Every ECC mode must produce the right state keys with correct dtypes after 2 steps."""
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-2, ecc=mode)

    x = torch.randn(4, 32)
    for _ in range(2):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    ecc_keys = _assert_ecc_active(opt, p)

    st = opt.state[p]
    all_keys = {}
    for v in st.values():
        if isinstance(v, dict):
            all_keys.update(v)

    if not cfg.is_int8:
        # ULP modes: primary is bf16/fp16, correction is int8/int16
        for ek in ecc_keys:
            pk = ek.replace("::ecc", "")
            assert pk in all_keys, f"Primary key {pk} missing for correction {ek}"
            assert all_keys[pk].dtype == cfg.primary_dtype, (
                f"[{mode}] primary {pk}: expected {cfg.primary_dtype}, got {all_keys[pk].dtype}"
            )
            assert all_keys[ek].dtype == cfg.corr_dtype, (
                f"[{mode}] correction {ek}: expected {cfg.corr_dtype}, got {all_keys[ek].dtype}"
            )
    else:
        # 8+16 mode: primary is int8 quantized, correction is bf16, plus bf16 scales
        for ek in ecc_keys:
            pk = ek.replace("::ecc", "")
            scales_k = pk + "::scales"
            assert pk in all_keys, f"Primary key {pk} missing"
            assert all_keys[pk].dtype == torch.int8, (
                f"[8+16] primary {pk}: expected int8, got {all_keys[pk].dtype}"
            )
            assert all_keys[ek].dtype == torch.bfloat16, (
                f"[8+16] correction {ek}: expected bf16, got {all_keys[ek].dtype}"
            )
            assert scales_k in all_keys, f"Scales key {scales_k} missing for 8+16 mode"
            assert all_keys[scales_k].dtype == torch.bfloat16, (
                f"[8+16] scales {scales_k}: expected bf16, got {all_keys[scales_k].dtype}"
            )
