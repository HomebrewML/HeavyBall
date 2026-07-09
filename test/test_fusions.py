import torch
from torch.fx.experimental.proxy_tensor import make_fx

from heavyball import fusions


def _targets(gm):
    return [node.target for node in gm.graph.nodes if node.op == "call_function"]


def _exec(fn, pass_fn, args, *, exact=False):
    """Trace fn, run pass_fn on its graph, recompile, execute, compare to eager.

    This catches value bugs (the structural-only tests missed the rsub-alpha one)."""
    gm = make_fx(fn)(*args)
    ref = fn(*args)
    pass_fn(gm.graph)
    gm.recompile()
    out = gm(*args)
    if isinstance(out, tuple):
        out = out[0]
    ok = torch.equal(ref, out) if exact else torch.allclose(ref, out, rtol=1e-5, atol=1e-6)
    assert ok, f"\n  eager {ref}\n  pass  {out}"


# ── stable 1 - x**y rewrite ──────────────────────────────────────────────────

def test_stable_rewrites_and_runs():
    # 1 - beta**step -> -expm1(step*log1p(beta-1)); executes and stays close to eager.
    _exec(lambda b, s: 1 - b ** s, fusions.stable_one_minus_pow,
          (torch.tensor(0.9999), torch.tensor(10)))
    gm = make_fx(lambda b, s: 1 - b ** s)(torch.tensor(0.9999), torch.tensor(10))
    fusions.stable_one_minus_pow(gm.graph)
    assert torch.ops.aten.expm1.default in _targets(gm)


def test_stable_skips_tensor_base():
    gm = make_fx(lambda x, s: 1 - x ** s)(torch.randn(8), torch.tensor(10))
    assert fusions.stable_one_minus_pow(gm.graph) == 0


def test_stable_skips_rsub_tensor_alpha():
    # rsub.Tensor(base**s, other, alpha!=1) is NOT 1 - base**s; must not rewrite (correct).
    a = torch.tensor([1.0, 2.0]); o = torch.tensor(2.0)
    gm = make_fx(lambda a, o: torch.ops.aten.rsub.Tensor(a ** 2, o, alpha=3.0))(a, o)
    assert fusions.stable_one_minus_pow(gm.graph) == 0


def test_stable_matches_sub_tensor_form():
    # dynamo emits 1 - x**y as sub.Tensor(1, x**y) (not rsub.Scalar); stable must catch it.
    try:
        gm = make_fx(lambda b, s: torch.ops.aten.sub.Tensor(1, b ** s))(torch.tensor(0.9999), torch.tensor(10))
    except Exception:
        import pytest
        pytest.skip("sub.Tensor(1, ...) not constructible via make_fx on this torch")
    assert fusions.stable_one_minus_pow(gm.graph) >= 1
    assert torch.ops.aten.expm1.default in _targets(gm)


def test_stable_preserves_nonpositive_scalar_base():
    # The log identity is only valid for x > 0; the rewrite keeps the original pow branch.
    _exec(lambda b, s: 1 - b ** s, fusions.stable_one_minus_pow,
          (torch.tensor(-2.0), torch.tensor(2)), exact=True)
    _exec(lambda b, s: 1 - b ** s, fusions.stable_one_minus_pow,
          (torch.tensor(0.0), torch.tensor(0)), exact=True)


# ── algebraic cancellation ───────────────────────────────────────────────────

def test_cancel_double_one_minus_rewrites_and_runs():
    _exec(lambda x: 1 - (1 - x), fusions.cancel_double_one_minus, (torch.rand(8),))
    gm = make_fx(lambda x: 1 - (1 - x))(torch.rand(8))
    assert fusions.cancel_double_one_minus(gm.graph) == 1
    assert all(getattr(t, "overloadpacket", t) not in (torch.ops.aten.sub, torch.ops.aten.rsub) for t in _targets(gm))


def test_cancel_double_one_minus_skips_integers():
    gm = make_fx(lambda x: 1 - (1 - x))(torch.arange(8))
    assert fusions.cancel_double_one_minus(gm.graph) == 0


def test_beta_debias_complement_cancels():
    from heavyball.utils import beta_debias
    gm = make_fx(lambda b, s: 1 - beta_debias(b, s))(torch.tensor(0.9999), torch.tensor(10))
    assert fusions.cancel_double_one_minus(gm.graph) == 1


def test_fold_affine_deep_one_minus():
    _exec(lambda x: 1 - (1 - (1 - (1 - x))), fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: 1 - (1 - (1 - (1 - x))))(torch.rand(8))
    assert fusions.fold_affine(gm.graph) >= 2
    assert all(getattr(t, "overloadpacket", t) != torch.ops.aten.rsub for t in _targets(gm))


def test_fold_affine_beta_debias_ping_pong():
    b = torch.tensor(0.9)
    s = torch.tensor(10)

    def f(b, s):
        denom = 1 - b ** s
        ratio = (1 - b) / denom
        a = 1 - ratio
        return 1 - a

    _exec(f, fusions.fold_affine, (b, s))
    gm = make_fx(f)(b, s)
    fusions.fold_affine(gm.graph)
    assert sum(1 for t in _targets(gm) if getattr(t, "overloadpacket", t) == torch.ops.aten.rsub) <= 2


def test_stable_positive_scalar_skips_pow():
    gm = make_fx(lambda b, s: 1 - b ** s)(torch.tensor(0.9999), torch.tensor(10))
    for n in gm.graph.nodes:
        if n.op == "placeholder" and n.name == "b_1":
            n.meta["val"] = torch.tensor(0.9999)
    fusions.stable_one_minus_pow(gm.graph)
    assert torch.ops.aten.pow.Tensor_Tensor not in _targets(gm)
    assert torch.ops.aten.where.self not in _targets(gm)


# ── FMA fusion ───────────────────────────────────────────────────────────────

def test_fma_value_runs():
    # a*b + c and 1 - a*b: rewrite to fma; executes close to eager.
    _exec(lambda a, b, c: a * b + c, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b: 1 - a * b, fusions.fuse_mul_add_to_fma, (torch.randn(8), torch.randn(8)))


def test_fma_sub_and_scalar_addend_run():
    _exec(lambda a, b, c: a * b - c, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b, c: c - a * b, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b: a * b + 0.5, fusions.fuse_mul_add_to_fma, (torch.randn(8), torch.randn(8)))


def test_fma_scaled_add_rewrites_and_runs():
    _exec(lambda x, y: torch.add(x, y, alpha=0.5), fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8)))
    gm = make_fx(lambda x, y: torch.add(x, y, alpha=0.5))(torch.randn(8), torch.randn(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_integer_skipped():
    gm = make_fx(lambda a, b, c: a * b + c)(torch.arange(8), torch.arange(8), torch.arange(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0


def test_fma_multi_user_mul_skipped():
    # one mul, two users -> must survive (not consumed by an fma)
    def f(a, b):
        m = a * b
        return m, m + 1
    gm = make_fx(f)(torch.randn(8), torch.randn(8))
    fusions.fuse_mul_add_to_fma(gm.graph)
    assert torch.ops.aten.mul.Tensor in _targets(gm)


def test_fma_rsub_tensor_alpha_is_exact():
    # regression: rsub.Tensor(a*b, other, alpha=3) = other - 3*a*b; alpha was being dropped.
    a = torch.randn(8); b = torch.randn(8); o = torch.tensor(2.0)
    fn = lambda a, b, o: torch.ops.aten.rsub.Tensor(a * b, o, alpha=3.0)
    _exec(fn, fusions.fuse_mul_add_to_fma, (a, b, o), exact=True)  # pass skips it -> bit-exact


def test_fma_rsub_scalar_alpha_is_exact():
    # rsub.Scalar(a*b, other, alpha=3) has the same alpha semantics as rsub.Tensor.
    fn = lambda a, b: torch.rsub(a * b, 2, alpha=3)
    _exec(fn, fusions.fuse_mul_add_to_fma, (torch.tensor(2.0), torch.tensor(3.0)), exact=True)


# ── Integration: beta_debias correctness ─────────────────────────────────────

def test_post_grad_debias_graph():
    from heavyball.utils import beta_debias

    b1, b2, s = torch.tensor(0.9), torch.tensor(0.999), torch.tensor(10)
    gm = make_fx(lambda b1, b2, s: (beta_debias(b1, s), beta_debias(b2, s), 1 - beta_debias(b1, s)))(b1, b2, s)
    fusions.post_grad_custom_pre_pass(gm.graph)
    gm.recompile()
    out = gm(b1, b2, s)
    ref = (beta_debias(b1, s), beta_debias(b2, s), 1 - beta_debias(b1, s))
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_fold_affine_skips_dynamic_alpha():
    g = torch.fx.Graph()
    a, b, alpha = g.placeholder("a"), g.placeholder("b"), g.placeholder("alpha")
    add = g.call_function(torch.ops.aten.add.Tensor, (a, b), {"alpha": alpha})
    g.output(add)
    gm = torch.fx.GraphModule({}, g)
    assert fusions.fold_affine(gm.graph) == 0


def test_beta_zero_stays_exact():
    from heavyball.utils import beta_debias
    for step in (1, 2, 5, 100):
        out = beta_debias(torch.tensor(0.0, dtype=torch.float32), torch.tensor(step, dtype=torch.int64))
        assert float(out) == 0.0, f"beta=0 step={step} got {float(out)}"
