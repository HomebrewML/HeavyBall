"""Tests that param_ecc works correctly under torch.compile.

Inductor can fold f32->bf16->f32 roundtrips into identity, silently
zeroing out ECC corrections.  Stochastic rounding masks this because
randint_like prevents the fold.  These tests cover both rounding modes.
"""

import contextlib

import pytest
import torch

import heavyball
from heavyball import utils
from heavyball.utils import (
    _compilable_update_,
    _ULPState,
    clean,
    scalar_guard,
    set_torch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ------------------------------------------------------------------ helpers


def _reset_dynamo():
    torch._dynamo.reset()


def _make_param_with_ecc(value=1.0, dtype=torch.bfloat16, corr_dtype=torch.int8):
    p = torch.full((64,), value, device="cuda", dtype=dtype)
    corr = torch.zeros(64, device="cuda", dtype=corr_dtype)
    smax = _ULPState._SMAX[corr_dtype]
    return p, corr, smax


def _rne_round(ref, source=None):
    if source is None:
        source = ref
    return source.to(ref.dtype)


@contextlib.contextmanager
def _rne_mode():
    old = utils.stochastic_round_
    utils.stochastic_round_ = _rne_round
    try:
        yield
    finally:
        utils.stochastic_round_ = old


def _get_param_ecc(opt, p):
    for v in opt.state[p].values():
        if isinstance(v, dict) and "param::ecc" in v:
            return v["param::ecc"]
    return None


def _train_linear(compile_mode, rne=False, steps=50):
    torch.manual_seed(0)
    import heavyball.utils as _u

    old_mode = _u.compile_mode
    _u.compile_mode = compile_mode
    ctx = _rne_mode() if rne else contextlib.nullcontext()
    try:
        with ctx:
            model = torch.nn.Linear(64, 32, bias=False, device="cuda")
            opt = heavyball.ForeachAdamW(model.parameters(), lr=1e-2, param_ecc="bf16+8")
            data = torch.randn(32, 64, device="cuda")
            target = torch.randn(32, 32, device="cuda")
            for _ in range(steps):
                p = next(model.parameters())
                loss = ((model(data.to(p.dtype)) - target.to(p.dtype)) ** 2).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
            p = list(model.parameters())[0]
            ecc = _get_param_ecc(opt, p)
            return p, ecc, loss.item()
    finally:
        _u.compile_mode = old_mode


# --------------------------------- corrections populated (stochastic rounding)


def test_ecc_populated_stochastic():
    set_torch()
    _reset_dynamo()
    p, ecc, _ = _train_linear("max-autotune-no-cudagraphs", rne=False)
    assert ecc is not None, "param::ecc not found in optimizer state"
    assert ecc.any(), "param::ecc all zeros with stochastic rounding under compile"
    clean()


# --------------------------------- corrections populated (RNE rounding)
# This is the case Inductor could fold. It was broken before the _bf16_to_f32 fix.


def test_ecc_populated_rne():
    set_torch()
    _reset_dynamo()
    p, ecc, _ = _train_linear("max-autotune-no-cudagraphs", rne=True)
    assert ecc is not None, "param::ecc not found in optimizer state"
    assert ecc.any(), (
        "param::ecc all zeros with RNE rounding under compile — Inductor likely folded the f32->bf16->f32 roundtrip"
    )
    clean()


# --------------------------------- compiled ECC matches eager quality


@pytest.mark.parametrize("rne", [False, True], ids=["stochastic", "rne"])
def test_ecc_compiled_vs_eager(rne):
    set_torch()
    _reset_dynamo()
    _, ecc_c, loss_c = _train_linear("max-autotune-no-cudagraphs", rne=rne, steps=100)
    _reset_dynamo()
    _, ecc_e, loss_e = _train_linear(None, rne=rne, steps=100)

    assert ecc_c.any(), f"compiled corrections all zeros (rne={rne})"
    assert ecc_e.any(), f"eager corrections all zeros (rne={rne})"

    ratio = loss_c / max(loss_e, 1e-30)
    assert ratio < 5.0, f"compiled loss {loss_c:.2e} is {ratio:.1f}x worse than eager loss {loss_e:.2e} (rne={rne})"
    clean()


# --------------------------------- unit: _compilable_update_ with RNE


def test_compilable_update_rne():
    set_torch()
    _reset_dynamo()

    with _rne_mode():
        p, corr, smax = _make_param_with_ecc(1.0)
        p._ecc = _ULPState(corr, smax)
        update = torch.ones(64, device="cuda", dtype=torch.float32)
        lr = scalar_guard(0.001, p)
        decay = scalar_guard(0.0, p)

        for _ in range(10):
            _compilable_update_([p], [update], decay, lr, False, [None])

        del p._ecc

    assert corr.any(), "corrections all zeros through _compilable_update_ with RNE"
    assert p.float().mean().item() < 1.0
    clean()
