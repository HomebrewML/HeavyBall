import pytest
import torch
from torch import nn

import heavyball
from heavyball import utils
from heavyball.chainable import ECCConfig
from heavyball.utils import clean, set_torch

heavyball.utils.compile_mode = "default"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ULP_MODES = ["bf16+8", "bf16+16", "fp16+8", "fp16+16"]

_OPTIMIZERS = [
    (heavyball.AdamW, 5e-2, {}),
    (heavyball.ADOPT, 5e-2, {}),
    (heavyball.NAdam, 1e-2, {}),
    (heavyball.LaProp, 5e-2, {}),
    (heavyball.AdEMAMix, 5e-2, {"betas": (0.9, 0.999, 0.9999)}),
    (heavyball.RMSprop, 1e-2, {}),
    (heavyball.SFAdamW, 1e-2, {}),
]


def _flat_state(opt, param):
    out = {}
    for v in opt.state[param].values():
        if isinstance(v, dict):
            out.update(v)
    return out


def _ecc_keys(opt, param):
    st = _flat_state(opt, param)
    keys = [k for k in st if isinstance(k, str) and "::ecc" in k]
    assert keys, f"ECC not active. Keys: {list(st.keys())}"
    return st, keys


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


def _problem(in_dim=16, out_dim=8, n=32):
    torch.manual_seed(42)
    return torch.randn(n, in_dim, device="cuda"), torch.randn(n, out_dim, device="cuda")


def _model_opt(cls, in_dim, out_dim, lr, **kw):
    torch.manual_seed(0)
    m = nn.Linear(in_dim, out_dim, bias=False, device="cuda")
    return m, cls(m.parameters(), lr=lr, **kw)


@pytest.mark.parametrize("mode", ULP_MODES)
def test_ulp_roundtrip(mode):
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    x = torch.randn(4096, device="cuda")
    xn = x.to(cfg.primary_dtype)
    ecc = torch.zeros_like(x, dtype=cfg.corr_dtype)

    st = utils._ULPState(ecc, cfg.smax)
    st.compute_correction(x, xn)
    out = st.decode(xn)

    err, naive_err = (x - out).abs(), (x - xn.float()).abs()
    improvement = naive_err.max().item() / max(err.max().item(), 1e-45)
    assert improvement > (100 if cfg.corr_dtype == torch.int8 else 10000)
    assert (err <= naive_err + 1e-10).all()


def test_ecc_boundary_values():
    smax = utils._ULPState._SMAX[torch.int8]
    for vals in [torch.zeros(256, device="cuda"), torch.tensor([1e4, -1e4, 3e10, -3e10, 1e-10, -1e-10], device="cuda")]:
        xn = vals.to(torch.bfloat16)
        ecc = torch.zeros(vals.shape[0], dtype=torch.int8, device="cuda")
        st = utils._ULPState(ecc, smax)
        st.compute_correction(vals, xn)
        out = st.decode(xn)
        assert ((vals - out).abs() <= (vals - xn.float()).abs() + 1e-10).all()


@pytest.mark.parametrize("mode", ULP_MODES)
def test_state_memory_vs_fp32(mode):
    cfg = ECCConfig(mode)
    n = 10000
    state = {}
    cfg.init_state(state, "test", torch.randn(n))
    ecc_bytes = sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
    if mode in ("bf16+16", "fp16+16"):
        assert ecc_bytes == n * 4
    else:
        assert ecc_bytes < n * 4


@pytest.mark.parametrize("opt_cls,lr,extra_kw", _OPTIMIZERS, ids=[t[0].__name__ for t in _OPTIMIZERS])
@pytest.mark.parametrize("mode", ["bf16+8"])
def test_ecc_convergence(opt_cls, lr, extra_kw, mode):
    set_torch()
    data, target = _problem()
    m0, o0 = _model_opt(opt_cls, 16, 8, lr, **extra_kw)
    losses_base = _train(m0, o0, data, target, 200)
    m1, o1 = _model_opt(opt_cls, 16, 8, lr, ecc=mode, **extra_kw)
    losses_ecc = _train(m1, o1, data, target, 200)
    _ecc_keys(o1, list(m1.parameters())[0])
    assert losses_ecc[-1] < losses_ecc[0] * 0.5
    assert losses_ecc[-1] / max(losses_base[-1], 1e-12) < 2.0
    del m0, o0, m1, o1
    clean()


@pytest.mark.parametrize("combined", [False, True], ids=["param_ecc_only", "state+param_ecc"])
def test_param_ecc_convergence(combined):
    set_torch()
    data, target = _problem()
    m0, o0 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2)
    losses_base = _train(m0, o0, data, target, 200)
    kw = {"param_ecc": "bf16+8"}
    if combined:
        kw["ecc"] = "bf16+8"
    m1, o1 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, **kw)
    losses_ecc = _train(m1, o1, data, target, 200)
    p = list(m1.parameters())[0]
    assert p.dtype == torch.bfloat16
    st, _ = _ecc_keys(o1, p)
    assert "param::ecc" in st
    assert losses_ecc[-1] < losses_ecc[0] * 0.5
    assert 0.5 < losses_ecc[-1] / max(losses_base[-1], 1e-12) < 2.0
    del m0, o0, m1, o1
    clean()


@pytest.mark.parametrize("mode", ULP_MODES)
def test_state_layout_and_invariants(mode):
    set_torch()
    cfg = ECCConfig(mode)
    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False, device="cuda")
    opt = heavyball.AdamW(model.parameters(), lr=1e-2, ecc=mode)
    x = torch.randn(4, 32, device="cuda")
    for _ in range(3):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

    p = list(model.parameters())[0]
    st, ecc_keys = _ecc_keys(opt, p)

    for ek in ecc_keys:
        pk = ek.replace("::ecc", "")
        assert st[pk].dtype == cfg.primary_dtype
        assert st[ek].dtype == cfg.corr_dtype
        assert not (st[pk] == 0).all() or not (st[ek] == 0).all(), f"'{pk}' all zeros"

    for group in opt.param_groups:
        assert group["step"] is None
    del model, opt
    clean()


def test_ademamix_three_vars():
    set_torch()
    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False, device="cuda")
    opt = heavyball.AdEMAMix(model.parameters(), lr=5e-2, betas=(0.9, 0.999, 0.9999), ecc="bf16+8")
    x = torch.randn(4, 64, device="cuda")
    for _ in range(10):
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
    st, _ = _ecc_keys(opt, list(model.parameters())[0])
    for var in ["exp_avg_fast", "exp_avg_slow", "exp_avg_sq"]:
        pks = [k for k in st if isinstance(k, str) and var in k and "::" not in k]
        assert pks
        for pk in pks:
            assert st[pk].dtype == torch.bfloat16
            assert st[pk + "::ecc"].dtype == torch.int8
    del model, opt
    clean()


def test_combined_ecc_dtypes():
    set_torch()
    m, o = _model_opt(heavyball.SFAdamW, 32, 16, 1e-2, ecc="bf16+16", param_ecc="bf16+8")
    data, target = _problem(in_dim=32, out_dim=16, n=8)
    _train(m, o, data, target, 100)
    p = list(m.parameters())[0]
    assert p.dtype == torch.bfloat16
    st, ecc_keys = _ecc_keys(o, p)
    for k in ecc_keys:
        if "param" not in k:
            assert st[k].dtype == torch.int16
    assert st["param::ecc"].dtype == torch.int8
    del m, o
    clean()


def test_shapes_and_bias():
    set_torch()
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 16, bias=True), nn.Linear(16, 4, bias=False)).cuda()
    opt = heavyball.AdamW(model.parameters(), lr=1e-2, ecc="bf16+8")
    data, target = torch.randn(16, 32, device="cuda"), torch.randn(16, 4, device="cuda")
    losses = _train(model, opt, data, target, 50)
    for p in model.parameters():
        _ecc_keys(opt, p)
        assert p.isfinite().all()
    assert losses[-1] < losses[0] * 0.5
    del model, opt
    clean()


def test_foreach_false():
    set_torch()
    data, target = _problem(in_dim=32, out_dim=4, n=16)
    m_fe, o_fe = _model_opt(heavyball.AdamW, 32, 4, 1e-2, ecc="bf16+8", multi_tensor=True)
    losses_fe = _train(m_fe, o_fe, data, target, 50)
    m_nf, o_nf = _model_opt(heavyball.AdamW, 32, 4, 1e-2, ecc="bf16+8", multi_tensor=False)
    losses_nf = _train(m_nf, o_nf, data, target, 50)
    assert losses_nf[-1] < losses_nf[0] * 0.5
    assert 0.3 < losses_nf[-1] / max(losses_fe[-1], 1e-12) < 3.0
    del m_fe, o_fe, m_nf, o_nf
    clean()


def test_param_groups():
    set_torch()
    torch.manual_seed(42)
    m1, m2 = nn.Linear(16, 8, bias=False, device="cuda"), nn.Linear(8, 4, bias=False, device="cuda")
    opt = heavyball.AdamW(
        [
            {"params": m1.parameters(), "ecc": "bf16+8"},
            {"params": m2.parameters()},
        ],
        lr=1e-2,
    )
    data, target = torch.randn(8, 16, device="cuda"), torch.randn(8, 4, device="cuda")
    for _ in range(50):
        ((m2(m1(data)) - target) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    _ecc_keys(opt, list(m1.parameters())[0])
    assert not any(isinstance(k, str) and "::ecc" in k for k in _flat_state(opt, list(m2.parameters())[0]))
    del m1, m2, opt
    clean()


def test_zero_gradients():
    set_torch()
    torch.manual_seed(42)
    p = nn.Parameter(torch.randn(16, 8, device="cuda"))
    opt = heavyball.AdamW([p], lr=1e-2, ecc="bf16+8")
    for _ in range(10):
        p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()
    assert p.isfinite().all()
    del opt
    clean()


def test_state_save_restore():
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt, sd_model = deepcopy(o.state_dict()), deepcopy(m.state_dict())
    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, ecc="bf16+8")
    m2.load_state_dict(sd_model)
    o2.load_state_dict(sd_opt)
    losses_after = _train(m2, o2, data, target, 10)
    losses_continued = _train(m, o, data, target, 10)
    assert 0.8 < losses_after[-1] / max(losses_continued[-1], 1e-12) < 1.2
    del m, o, m2, o2
    clean()


def _measure_peak(cls, n, lr, ecc=None, param_ecc=None, steps=3):
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    p = torch.nn.Parameter(torch.randn(n, device="cuda"))
    kw = {}
    if ecc:
        kw["ecc"] = ecc
    if param_ecc:
        kw["param_ecc"] = param_ecc
    opt = cls([p], lr=lr, **kw)

    for _ in range(steps):
        p.grad = torch.randn_like(p)
        opt.step()
        opt.zero_grad()

    p.grad = torch.randn_like(p)
    pre = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    peak = torch.cuda.max_memory_allocated()
    opt.zero_grad()

    del opt, p
    gc.collect()
    torch.cuda.empty_cache()
    return pre / n, peak / n


@pytest.mark.parametrize("mode", ["bf16+8"])
@pytest.mark.parametrize(
    "cls,lr",
    [
        (heavyball.AdamW, 1e-3),
        (heavyball.SFAdamW, 1e-2),
        (heavyball.RMSprop, 1e-2),
    ],
    ids=["AdamW", "SFAdamW", "RMSprop"],
)
def test_ecc_peak_memory(cls, lr, mode):
    n = 2**24
    pre_base, peak_base = _measure_peak(cls, n, lr)
    pre_ecc, peak_ecc = _measure_peak(cls, n, lr, ecc=mode)

    # steady state must use less memory than baseline
    assert pre_ecc < pre_base, f"ECC pre-step {pre_ecc:.1f} B/p >= baseline {pre_base:.1f} B/p"

    assert peak_ecc < peak_base, f"ECC peak {peak_ecc:.1f} B/p >= baseline peak {peak_base:.1f} B/p"


@pytest.mark.parametrize("combined", [False, True], ids=["param_only", "state+param"])
def test_param_ecc_peak_memory(combined):
    n = 2**24
    cls, lr = heavyball.SFAdamW, 1e-2
    pre_base, peak_base = _measure_peak(cls, n, lr)
    ecc = "bf16+8" if combined else None
    pre_ecc, peak_ecc = _measure_peak(cls, n, lr, ecc=ecc, param_ecc="bf16+8")

    assert peak_ecc < peak_base, f"param_ecc peak {peak_ecc:.1f} B/p >= baseline peak {peak_base:.1f} B/p"


def test_ecc_live_path_nonzero_correction():
    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.AdamW, 16, 8, 5e-2, ecc="bf16+8")
    _train(m, o, data, target, 20)
    p = list(m.parameters())[0]
    st, ecc_keys = _ecc_keys(o, p)
    for ek in ecc_keys:
        assert st[ek].any(), f"ECC correction '{ek}' is all zeros - stochastic_round_ likely mutating source"
    del m, o
    clean()


def test_lerp_returns_valid_fp32():
    set_torch()
    m, o = _model_opt(heavyball.AdamW, 16, 8, 5e-2, ecc="bf16+8")
    data, target = _problem()
    _train(m, o, data, target, 10)
    p = list(m.parameters())[0]
    st = _flat_state(o, p)
    state_keys = [k for k in st if isinstance(k, str) and "::ecc" not in k and (k + "::ecc") in st]
    for k in state_keys:
        narrow = st[k]
        assert narrow.dtype == torch.bfloat16
        smax = utils._ULPState._SMAX[st[k + "::ecc"].dtype]
        narrow._ecc = utils._ULPState(st[k + "::ecc"], smax)
        fp32 = utils.promote(narrow)
        del narrow._ecc
        unique_vals = fp32.unique().numel()
        bf16_unique = narrow.float().unique().numel()
        assert unique_vals > bf16_unique, "promote() returned bf16-precision values in fp32 container"
    del m, o
    clean()


def test_param_ecc_merge_dims():
    set_torch()
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 8, bias=False),
    ).cuda()
    opt = heavyball.SOAP(model.parameters(), lr=1e-3, param_ecc="bf16+8")
    conv_p = list(model.parameters())[0]
    assert conv_p.dtype == torch.bfloat16
    st = _flat_state(opt, conv_p)
    assert "param::ecc" in st
    assert st["param::ecc"].shape == torch.Size([64, 27])
    data, target = torch.randn(4, 3, 8, 8, device="cuda"), torch.randn(4, 8, device="cuda")
    for _ in range(5):
        d = data.to(conv_p.dtype)
        ((model(d) - target.to(d.dtype)) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    assert conv_p.isfinite().all()
    del model, opt
    clean()


def test_param_ecc_dtype_at_construction():
    set_torch()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    p = list(m.parameters())[0]
    assert p.dtype == torch.bfloat16, "param should be bf16 immediately after construction"
    st = _flat_state(o, p)
    assert "param::ecc" in st, "param::ecc should be initialized at construction"
    del m, o
    clean()


def test_param_ecc_save_restore():
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt, sd_model = deepcopy(o.state_dict()), deepcopy(m.state_dict())

    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    m2.load_state_dict(sd_model)
    o2.load_state_dict(sd_opt)
    p2 = list(m2.parameters())[0]
    assert p2.dtype == torch.bfloat16, "param dtype not restored after load_state_dict"
    losses_after = _train(m2, o2, data, target, 10)
    losses_continued = _train(m, o, data, target, 10)
    assert 0.5 < losses_after[-1] / max(losses_continued[-1], 1e-12) < 2.0
    del m, o, m2, o2
    clean()


def test_param_ecc_partial_state_restore():
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt = deepcopy(o.state_dict())
    sd_model = deepcopy(m.state_dict())
    for param_state in sd_opt["state"].values():
        for idx_state in param_state.values():
            if isinstance(idx_state, dict):
                idx_state.pop("param::ecc", None)
    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    m2.load_state_dict(sd_model)
    o2.load_state_dict(sd_opt)
    st2 = _flat_state(o2, list(m2.parameters())[0])
    assert "param::ecc" in st2, "param::ecc not rehydrated after partial restore"
    _train(m2, o2, data, target, 5)
    p2 = list(m2.parameters())[0]
    assert p2.dtype == torch.bfloat16
    assert p2.isfinite().all()
    del m, o, m2, o2
    clean()


def test_param_ecc_empty_state_restore():
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt = deepcopy(o.state_dict())
    sd_model = deepcopy(m.state_dict())
    sd_opt["state"] = {}
    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    m2.load_state_dict(sd_model)
    o2.load_state_dict(sd_opt)
    p2 = list(m2.parameters())[0]
    st2 = _flat_state(o2, p2)
    assert "param::ecc" in st2, "param::ecc not created after empty state restore"
    assert st2["param::ecc"].dtype == torch.int8
    _train(m2, o2, data, target, 5)
    assert p2.dtype == torch.bfloat16
    assert p2.isfinite().all()
    del m, o, m2, o2
    clean()


def test_param_ecc_merged_view_partial_restore():
    from copy import deepcopy

    set_torch()
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 8, bias=False),
    ).cuda()
    opt = heavyball.SOAP(model.parameters(), lr=1e-3, param_ecc="bf16+8")
    data = torch.randn(4, 3, 8, 8, device="cuda")
    target = torch.randn(4, 8, device="cuda")
    conv_p = list(model.parameters())[0]
    for _ in range(5):
        d = data.to(conv_p.dtype)
        ((model(d) - target.to(d.dtype)) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
    sd_opt = deepcopy(opt.state_dict())
    sd_model = deepcopy(model.state_dict())
    for param_state in sd_opt["state"].values():
        for idx_state in param_state.values():
            if isinstance(idx_state, dict):
                idx_state.pop("param::ecc", None)
    model2 = nn.Sequential(
        nn.Conv2d(3, 64, 3, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 8, bias=False),
    ).cuda()
    opt2 = heavyball.SOAP(model2.parameters(), lr=1e-3, param_ecc="bf16+8")
    model2.load_state_dict(sd_model)
    opt2.load_state_dict(sd_opt)
    conv_p2 = list(model2.parameters())[0]
    st2 = _flat_state(opt2, conv_p2)
    assert "param::ecc" in st2, "param::ecc not rehydrated for merged-view param"
    assert st2["param::ecc"].shape == torch.Size([64, 27])
    for _ in range(5):
        d = data.to(conv_p2.dtype)
        ((model2(d) - target.to(d.dtype)) ** 2).mean().backward()
        opt2.step()
        opt2.zero_grad()
    assert conv_p2.isfinite().all()
    del model, opt, model2, opt2
    clean()


def test_optimizer_kwargs_not_in_param_groups():
    set_torch()
    p = torch.nn.Parameter(torch.randn(4, 4, device="cuda"))
    o = heavyball.AdamW([p], lr=1e-3, compile_step=True, promote=True)
    assert o.compile_step is True
    assert o.promote is True
    assert "compile_step" not in o.param_groups[0]
    assert "promote" not in o.param_groups[0]
    del o
    clean()


def test_param_ecc_load_order_model_before_optimizer():
    """Model-first load: param::ecc rehydrated as zeros (fp32 ref lost), training self-corrects."""
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt = deepcopy(o.state_dict())
    sd_model = deepcopy(m.state_dict())
    for param_state in sd_opt["state"].values():
        for idx_state in param_state.values():
            if isinstance(idx_state, dict):
                idx_state.pop("param::ecc", None)
    # model-first: load model, then optimizer
    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    m2.load_state_dict(sd_model)
    o2.load_state_dict(sd_opt)
    p2 = list(m2.parameters())[0]
    st2 = _flat_state(o2, p2)
    assert "param::ecc" in st2
    assert p2.dtype == torch.bfloat16
    _train(m2, o2, data, target, 10)
    assert p2.isfinite().all()
    st2_after = _flat_state(o2, list(m2.parameters())[0])
    assert st2_after["param::ecc"].any(), "correction should be non-zero after training"
    del m, o, m2, o2
    clean()


def test_param_ecc_load_order_optimizer_before_model():
    """Optimizer-first with stripped param::ecc: correction computed from stale params."""
    from copy import deepcopy

    set_torch()
    data, target = _problem()
    m, o = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    _train(m, o, data, target, 10)
    sd_opt = deepcopy(o.state_dict())
    sd_model = deepcopy(m.state_dict())
    for param_state in sd_opt["state"].values():
        for idx_state in param_state.values():
            if isinstance(idx_state, dict):
                idx_state.pop("param::ecc", None)
    # optimizer-first: load optimizer, then model
    m2, o2 = _model_opt(heavyball.SFAdamW, 16, 8, 1e-2, param_ecc="bf16+8")
    o2.load_state_dict(sd_opt)
    m2.load_state_dict(sd_model)
    p2 = list(m2.parameters())[0]
    assert p2.dtype == torch.bfloat16
    _train(m2, o2, data, target, 10)
    assert p2.isfinite().all()
    del m, o, m2, o2
    clean()
