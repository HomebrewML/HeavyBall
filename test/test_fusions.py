import random

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from heavyball import fusions
from heavyball.utils import beta_debias, scalar_guard


def _targets(gm):
    return [node.target for node in gm.graph.nodes if node.op == "call_function"]


def _exec(fn, pass_fn, args):
    gm = make_fx(fn)(*args)
    ref = fn(*args)
    pass_fn(gm.graph)
    gm.recompile()
    out = gm(*args)
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(ref, out, rtol=1e-5, atol=1e-6), f"\n  eager {ref}\n  pass  {out}"


def test_cancel_double_one_minus_rewrites_and_runs():
    _exec(lambda x: (1 - (1 - x)) * 2, fusions.cancel_double_one_minus, (torch.rand(8),))
    gm = make_fx(lambda x: (1 - (1 - x)) * 2)(torch.rand(8))
    assert fusions.cancel_double_one_minus(gm.graph) == 1
    assert all(getattr(t, "overloadpacket", t) not in (torch.ops.aten.sub, torch.ops.aten.rsub) for t in _targets(gm))


def test_cancel_double_one_minus_preserves_output_storage():
    gm = make_fx(lambda x: 1 - (1 - x))(torch.rand(8))
    assert fusions.cancel_double_one_minus(gm.graph) == 0
    gm.recompile()
    x = torch.rand(8)
    assert gm(x).data_ptr() != x.data_ptr()


def test_cancel_double_one_minus_skips_integers():
    gm = make_fx(lambda x: 1 - (1 - x))(torch.arange(8))
    assert fusions.cancel_double_one_minus(gm.graph) == 0


def test_beta_debias_complement_cancels():
    gm = make_fx(lambda b, s: (1 - beta_debias(b, s)) * 2)(torch.tensor(0.9999), torch.tensor(10))
    assert fusions.cancel_double_one_minus(gm.graph) == 1


def test_fold_affine_deep_one_minus():
    _exec(lambda x: (1 - (1 - (1 - (1 - x)))) * 2, fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: (1 - (1 - (1 - (1 - x)))) * 2)(torch.rand(8))
    assert fusions.fold_affine(gm.graph) >= 2
    assert all(getattr(t, "overloadpacket", t) != torch.ops.aten.rsub for t in _targets(gm))


def test_fold_affine_constant_shift():
    _exec(lambda x: 2 - (1 - x), fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: 2 - (1 - x))(torch.rand(8))
    assert fusions.fold_affine(gm.graph) >= 1
    assert torch.ops.aten.add.Tensor in _targets(gm)


def test_fold_affine_negate_distributes():
    _exec(lambda x: -(x * 2 + 1), fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: -(x * 2 + 1))(torch.rand(8))
    fusions.fold_affine(gm.graph)
    assert torch.ops.aten.neg.default not in _targets(gm)


def test_fold_affine_scale_shift_combined():
    _exec(lambda x: (x * 3 + 2) * 4, fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: (x * 3 + 2) * 4)(torch.rand(8))
    n = fusions.fold_affine(gm.graph)
    assert n >= 1


def test_post_grad_affine_fuses_to_fma():
    gm = make_fx(lambda x: (x + 1) * 2)(torch.rand(8))
    assert fusions.post_grad_custom_pre_pass(gm.graph) >= 2
    assert torch.ops.prims.fma.default in _targets(gm)


def test_affine_chain_collapses_before_fma():
    def fn(x):
        return ((x * 3 + 2) * 4 + 5) * 6 + 7

    value = torch.tensor([-160948.484375])
    gm = make_fx(fn)(value)
    assert fusions.post_grad_custom_pre_pass(gm.graph) > 0
    assert _targets(gm).count(torch.ops.prims.fma.default) == 1
    gm.recompile()
    assert torch.equal(gm(value), fn(value.double()).float())
    assert fusions.post_grad_custom_pre_pass(gm.graph) == 0


def test_fma_crosses_functional_fulls_in_lerp_lowering():
    graph = torch.fx.Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    weight = graph.call_function(torch.ops.aten.full.default, ([4], 0.2), {"dtype": torch.float32})
    delta = graph.call_function(torch.ops.aten.sub.Tensor, (y, x))
    product = graph.call_function(torch.ops.aten.mul.Tensor, (weight, delta))
    condition = graph.call_function(torch.ops.aten.full.default, ([4], False), {"dtype": torch.bool})
    base = graph.call_function(torch.ops.aten.where.self, (condition, y, x))
    result = graph.call_function(torch.ops.aten.add.Tensor, (base, product))
    graph.output(result)
    gm = torch.fx.GraphModule({}, graph)
    x_value, y_value = torch.randn(4), torch.randn(4)
    FakeTensorProp(gm).propagate(x_value, y_value)
    expected = gm(x_value, y_value)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    gm.recompile()
    torch.testing.assert_close(gm(x_value, y_value), expected)
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fold_affine_cse_dedup():
    def f(x):
        a = 2 - (1 - x)
        b = 2 - (1 - x)
        return a + b

    _exec(f, fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(f)(torch.rand(8))
    assert fusions.fold_affine(gm.graph) >= 2


def test_fold_affine_skips_already_minimal():
    gm = make_fx(lambda x: x + 1)(torch.rand(8))
    assert fusions.fold_affine(gm.graph) == 0
    gm = make_fx(lambda x: x * 2)(torch.rand(8))
    assert fusions.fold_affine(gm.graph) == 0


def test_fold_affine_skips_single_complement():
    gm = make_fx(lambda x: 1 - x)(torch.rand(8))
    assert fusions.fold_affine(gm.graph) == 0


def test_fold_affine_skips_dynamic_alpha():
    g = torch.fx.Graph()
    a, b, alpha = g.placeholder("a"), g.placeholder("b"), g.placeholder("alpha")
    add = g.call_function(torch.ops.aten.add.Tensor, (a, b), {"alpha": alpha})
    g.output(add)
    gm = torch.fx.GraphModule({}, g)
    assert fusions.fold_affine(gm.graph) == 0


def test_fold_affine_beta_debias_ping_pong():
    b = torch.tensor(0.9)
    s = torch.tensor(10)

    def f(b, s):
        denom = 1 - b ** s
        ratio = (1 - b) / denom
        a = 1 - ratio
        return (1 - a) * 2

    _exec(f, fusions.fold_affine, (b, s))
    gm = make_fx(f)(b, s)
    fusions.fold_affine(gm.graph)
    assert sum(1 for t in _targets(gm) if getattr(t, "overloadpacket", t) == torch.ops.aten.rsub) <= 2


def test_fold_affine_folds_true_division():
    _exec(lambda x: (x * 2 + 1) / 3, fusions.fold_affine, (torch.rand(8),))
    gm = make_fx(lambda x: (x * 2 + 1) / 3)(torch.rand(8))
    fusions.fold_affine(gm.graph)
    assert torch.ops.aten.div.Tensor not in _targets(gm)


def test_fold_affine_preserves_div_rounding_mode():
    x = torch.tensor(3.5)

    def fn(x):
        return torch.div(x, 2, rounding_mode="trunc") + 1 + 1

    gm = make_fx(fn)(x)
    fusions.fold_affine(gm.graph)
    gm.recompile()
    assert torch.equal(gm(x), fn(x))
    assert torch.ops.aten.div.Tensor_mode in _targets(gm)
    assert torch.ops.aten.mul.Tensor not in _targets(gm)


def test_fold_affine_stops_at_aliasing_mutation():
    def fn(x):
        before = x + 1
        x.view_as(x).add_(3)
        return before, x + 1

    gm = make_fx(fn)(torch.tensor([2.0]))
    assert fusions.fold_affine(gm.graph) == 0
    gm.recompile()
    got = gm(torch.tensor([2.0]))
    expected = fn(torch.tensor([2.0]))
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_fold_affine_preserves_identity_before_mutation():
    def fn(x):
        value = 1 - (1 - x)
        value.add_(2)
        return x, value

    gm = make_fx(fn)(torch.tensor([2.0]))
    assert fusions.fold_affine(gm.graph) == 0
    gm.recompile()
    got = gm(torch.tensor([2.0]))
    expected = fn(torch.tensor([2.0]))
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_fold_affine_does_not_cse_mutated_source():
    def fn(x):
        first = x + 1
        second = x + 1
        first.add_(2)
        return second * 2

    gm = make_fx(fn)(torch.tensor([2.0]))
    assert fusions.fold_affine(gm.graph) == 0
    gm.recompile()
    assert torch.equal(gm(torch.tensor([2.0])), fn(torch.tensor([2.0])))


def test_impure_functions_are_affine_barriers():
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.call_function(torch.ops.inductor.accumulate_grad_.default, (value, value))
    after = graph.call_function(torch.ops.aten.add.Tensor, (value, 1))
    graph.output(after)
    assert fusions._epochs(graph)[after] == 1


def test_fold_affine_preserves_unannotated_mutation():
    graph = torch.fx.Graph()
    weight = graph.placeholder("weight")
    indices = graph.placeholder("indices")

    def affine(value):
        value = graph.call_function(torch.ops.aten.mul.Tensor, (value, 2))
        value = graph.call_function(torch.ops.aten.add.Tensor, (value, 1))
        return graph.call_function(torch.ops.aten.mul.Tensor, (value, 2))

    before = affine(weight)
    graph.call_function(torch.ops.aten._no_grad_embedding_renorm_.default, (weight, indices, 2.0, 2.0))
    after = affine(weight)
    graph.output((before, after, weight))
    gm = torch.fx.GraphModule({}, graph)
    weight_value = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    index_value = torch.tensor([0])
    expected = gm(weight_value.clone(), index_value)
    FakeTensorProp(gm).propagate(weight_value, index_value)
    assert fusions.fold_affine(gm.graph) > 0
    gm.recompile()
    got = gm(weight_value.clone(), index_value)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_fold_affine_keeps_zero_return_effects():
    def fn(x, condition):
        torch.ops.aten._assert_async.default(condition)
        return (x + 1) * 2

    gm = make_fx(fn)(torch.tensor([2.0]), torch.tensor(True))
    assert fusions.fold_affine(gm.graph) == 1
    gm.recompile()
    with pytest.raises(RuntimeError):
        gm(torch.tensor([2.0]), torch.tensor(False))


def test_fold_affine_tracks_emitted_nodes_across_mutation():
    def fn(x):
        value = (x + 1) * 2
        value.view_as(value).add_(8)
        return value + 4, value

    gm = make_fx(fn)(torch.tensor([2.0]))
    assert fusions.fold_affine(gm.graph) == 1
    gm.recompile()
    got = gm(torch.tensor([2.0]))
    expected = fn(torch.tensor([2.0]))
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_fold_affine_preserves_untracked_aliases():
    aliases = (
        lambda x: torch.ops.aten._unsafe_view.default(x, [1]),
        lambda x: torch.ops.inductor._reinterpret_tensor.default(x, [1], [1], 0),
        torch.ops.aten.lift.default,
    )
    for alias in aliases:
        def fn(x):
            value = 1 - (1 - x)
            value = alias(value)
            value.add_(2)
            return x, value

        gm = make_fx(fn)(torch.tensor([2.0]))
        assert fusions.fold_affine(gm.graph) == 0
        gm.recompile()
        got = gm(torch.tensor([2.0]))
        expected = fn(torch.tensor([2.0]))
        assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_fold_affine_keeps_unrepresentable_coefficients():
    cases = (
        (lambda x: x * 1e20 * 1e20, torch.tensor([1e-30, 0.0, float("inf")])),
        (lambda x: x * 1e-20 * 1e-30, torch.tensor([1e30, 0.0, float("inf")])),
    )
    for fn, x in cases:
        gm = make_fx(fn)(x)
        assert fusions.fold_affine(gm.graph) == 0
        gm.recompile()
        assert torch.equal(gm(x), fn(x))


def test_fma_handles_keyword_only_aten_arguments():
    graph = torch.fx.Graph()
    x, y, z = graph.placeholder("x"), graph.placeholder("y"), graph.placeholder("z")
    product = graph.call_function(torch.ops.aten.mul.Tensor, (), {"self": x, "other": y})
    result = graph.call_function(torch.ops.aten.add.Tensor, (), {"self": product, "other": z})
    graph.output(result)
    gm = torch.fx.GraphModule({}, graph)
    values = torch.randn(4), torch.randn(4), torch.randn(4)
    FakeTensorProp(gm).propagate(*values)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    gm.recompile()
    torch.testing.assert_close(gm(*values), values[0] * values[1] + values[2])


def test_fma_value_runs():
    _exec(lambda a, b, c: a * b + c, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b: 1 - a * b, fusions.fuse_mul_add_to_fma, (torch.randn(8), torch.randn(8)))


def test_fma_promotes_integral_tensor_addends():
    for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        a, b = torch.randn(8), torch.randn(8)
        c = torch.randint(0, 2, (8,), dtype=dtype)
        gm = make_fx(lambda a, b, c: a * b + c)(a, b, c)
        assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
        gm.recompile()
        torch.testing.assert_close(gm(a, b, c), a * b + c)
        assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_keeps_integral_subtraction_unfused():
    a = torch.full((4,), 2.0)
    b = torch.full((4,), 3.0)
    c = torch.tensor((0, 1, 2, 255), dtype=torch.uint8)
    gm = make_fx(lambda a, b, c: a * b - c)(a, b, c)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
    gm.recompile()
    assert torch.equal(gm(a, b, c), a * b - c)


def test_fma_sub_and_scalar_addend_run():
    _exec(lambda a, b, c: a * b - c, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b, c: c - a * b, fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8), torch.randn(8)))
    _exec(lambda a, b: a * b + 0.5, fusions.fuse_mul_add_to_fma, (torch.randn(8), torch.randn(8)))


def test_fma_negates_promoted_factor():
    for mask in (torch.tensor([True, False]), torch.tensor([3, 4], dtype=torch.uint8)):
        x = torch.tensor([2.0, 3.0])
        c = torch.tensor([7.0, 11.0])
        _exec(lambda mask, x, c: c - mask * x, fusions.fuse_mul_add_to_fma, (mask, x, c))
        gm = make_fx(lambda mask, x, c: c - mask * x)(mask, x, c)
        assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
        assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_negates_scalar_factor_without_a_tensor_negation():
    gm = make_fx(lambda x, y: y - x * 64)(torch.randn(8), torch.randn(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    fma = next(node for node in gm.graph.nodes if node.target == torch.ops.prims.fma.default)
    assert fma.args[1] == -64
    assert torch.ops.aten.neg.default not in _targets(gm)


def test_fma_scaled_add_rewrites_and_runs():
    _exec(lambda x, y: torch.add(x, y, alpha=0.5), fusions.fuse_mul_add_to_fma,
          (torch.randn(8), torch.randn(8)))
    gm = make_fx(lambda x, y: torch.add(x, y, alpha=0.5))(torch.randn(8), torch.randn(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_scaled_add_preserves_promotion():
    cases = (
        (torch.tensor([2, 3], dtype=torch.uint8), torch.tensor([10.0, 20.0]), -2),
        (torch.tensor([1.0009765625, 2.333], dtype=torch.float16), torch.tensor([10.123456, 20.654321]), 0.1),
    )
    for factor, base, alpha in cases:
        gm = make_fx(lambda base, factor: torch.add(base, factor, alpha=alpha))(base, factor)
        assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
        gm.recompile()
        assert torch.equal(gm(base, factor), torch.add(base, factor, alpha=alpha))


def test_fma_integer_skipped():
    gm = make_fx(lambda a, b, c: a * b + c)(torch.arange(8), torch.arange(8), torch.arange(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0


def test_fma_multi_user_mul_skipped():
    def f(a, b):
        m = a * b
        return m, m + 1

    gm = make_fx(f)(torch.randn(8), torch.randn(8))
    fusions.fuse_mul_add_to_fma(gm.graph)
    assert torch.ops.aten.mul.Tensor in _targets(gm)


def test_fma_reused_mul_input_skipped():
    def f(a, b):
        product = a * b
        return product + product

    gm = make_fx(f)(torch.randn(8), torch.randn(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
    assert torch.ops.prims.fma.default not in _targets(gm)


def test_fma_preserves_out_write():
    def fn(a, b, c, out):
        torch.add(a * b, c, out=out)
        return out

    args = torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([-1.0])
    gm = make_fx(fn)(*args)
    assert fusions.post_grad_custom_pre_pass(gm.graph) == 0
    gm.recompile()
    out = torch.tensor([-1.0])
    result = gm(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0]), out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out, torch.tensor([10.0]))


def test_fma_rsub_tensor_alpha_fuses():
    a = torch.randn(8)
    b = torch.randn(8)
    o = torch.tensor(2.0)

    def fn(a, b, o):
        return torch.ops.aten.rsub.Tensor(a * b, o, alpha=3.0)

    gm = make_fx(fn)(a, b, o)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_rsub_scalar_alpha_fuses():
    def fn(a, b):
        return torch.rsub(a * b, 2, alpha=3)

    gm = make_fx(fn)(torch.tensor(2.0), torch.tensor(3.0))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_mixed_dtype_fuses():
    a = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    c = torch.randn(8, dtype=torch.float32)
    gm = make_fx(lambda a, b, c: a * b + c)(a, b, c)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_mixed_dtype_improves_precision():
    a = torch.tensor([1.0009765625], dtype=torch.float16)
    b = torch.tensor([1.0009765625], dtype=torch.float16)
    c = torch.tensor([-1.001953125], dtype=torch.float32)

    def fn(a, b, c):
        return a * b + c

    reference = (a.double() * b.double() + c.double()).float()
    assert not torch.equal(fn(a, b, c), reference)
    assert torch.equal(fusions.compile(fn, fullgraph=True)(a, b, c), reference)


def test_fma_same_dtype_tensor_addend_fuses():
    a = torch.randn(8, dtype=torch.float16)
    b = torch.randn(8, dtype=torch.float16)
    c = torch.randn(8, dtype=torch.float16)
    gm = make_fx(lambda a, b, c: a * b + c)(a, b, c)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1


def test_fma_integral_factor_promotes():
    gm = make_fx(lambda step, lr: 1 - step * lr)(torch.tensor(1, dtype=torch.int64), torch.tensor(0.1))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_scaled_product_fuses():
    gm = make_fx(lambda a, b, c: torch.add(a * b, c, alpha=0.5))(
        torch.randn(8), torch.randn(8), torch.randn(8))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    assert torch.ops.prims.fma.default in _targets(gm)


def test_fma_stops_at_aliasing_mutation():
    def fn(a, b, c):
        product = a * b
        a.view_as(a).add_(1)
        return product + c

    gm = make_fx(fn)(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0]))
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
    gm.recompile()
    got = gm(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0]))
    expected = fn(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0]))
    assert torch.equal(got, expected)


def test_post_grad_debias_graph():
    b1, b2, s = torch.tensor(0.9), torch.tensor(0.999), torch.tensor(10)
    gm = make_fx(lambda b1, b2, s: (beta_debias(b1, s), beta_debias(b2, s), 1 - beta_debias(b1, s)))(b1, b2, s)
    fusions.post_grad_custom_pre_pass(gm.graph)
    gm.recompile()
    out = gm(b1, b2, s)
    ref = (beta_debias(b1, s), beta_debias(b2, s), 1 - beta_debias(b1, s))
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_beta_zero_stays_exact():
    for step in (1, 2, 5, 100):
        out = beta_debias(torch.tensor(0.0, dtype=torch.float32), torch.tensor(step, dtype=torch.int64))
        assert float(out) == 0.0, f"beta=0 step={step} got {float(out)}"


def _affine_programs():
    rng = random.Random(0xF00D)
    kinds = ("add", "sub", "rsub", "mul", "div", "neg")
    values = (-1.5, -0.75, -0.25, 0.125, 0.5, 1.25)
    alphas = (0.25, 0.5, 1.0, 2.0)
    return [tuple((rng.choice(kinds), rng.choice(values), rng.choice(alphas)) for _ in range(6)) for _ in range(8)]


def _apply_affine(x, operations):
    for kind, value, alpha in operations:
        if kind == "add":
            x = torch.add(x, value, alpha=alpha)
        elif kind == "sub":
            x = torch.sub(x, value, alpha=alpha)
        elif kind == "rsub":
            x = torch.rsub(x, value, alpha=alpha)
        elif kind == "mul":
            x = x * value
        elif kind == "div":
            x = x / value
        else:
            x = -x
    return x


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    ((torch.float32, 2e-6, 1e-6), (torch.float16, 5e-3, 6e-3), (torch.bfloat16, 4e-2, 5e-2)),
)
def test_randomized_affine_dags_match_fp64_and_converge(dtype, rtol, atol):
    generator = torch.Generator().manual_seed(0xA11F1E)
    for operations in _affine_programs():
        x = (torch.randn(31, generator=generator) * 0.5).to(dtype)
        gm = make_fx(lambda x: _apply_affine(x, operations))(x)
        assert fusions.fold_affine(gm.graph) > 0
        gm.recompile()
        expected = _apply_affine(x.double(), operations).to(dtype)
        torch.testing.assert_close(gm(x), expected, rtol=rtol, atol=atol)
        assert fusions.fold_affine(gm.graph) == 0


def _apply_fma_dag(a, b, c, d, coefficients):
    alpha, scale, delta, bias = coefficients
    value = torch.add(a * b, c, alpha=alpha)
    return torch.add(value * scale, d, alpha=delta) + bias


def test_randomized_fma_dags_match_fp64_and_converge():
    rng = random.Random(0xF00A)
    coefficients = [tuple(rng.choice((-0.5, 0.5, 1.0, 2.0)) for _ in range(4)) for _ in range(8)]
    generator = torch.Generator().manual_seed(0xF00)
    for values in coefficients:
        args = tuple(torch.randn(16, generator=generator) * 0.5 for _ in range(4))
        gm = make_fx(lambda a, b, c, d: _apply_fma_dag(a, b, c, d, values))(*args)
        assert fusions.post_grad_custom_pre_pass(gm.graph) > 0
        assert torch.ops.prims.fma.default in _targets(gm)
        gm.recompile()
        expected = _apply_fma_dag(*(x.double() for x in args), values).float()
        torch.testing.assert_close(gm(*args), expected, rtol=2e-6, atol=1e-6)
        assert fusions.post_grad_custom_pre_pass(gm.graph) == 0


def test_scalar_guard_promotes_floating_optimizer_scalars_to_fp32():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        reference = torch.ones((), dtype=dtype)
        scalar, tensor_scalar = scalar_guard(0.1, torch.tensor(0.1, dtype=dtype), reference)
        assert scalar.dtype == tensor_scalar.dtype == torch.float32
        assert scalar.ndim == tensor_scalar.ndim == 0


def test_scalar_alpha_dtype_matrix_matches_fp64():
    cases = (
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
    )
    for base_dtype, factor_dtype in cases:
        for alpha in (-2.0, -0.5, 0.125, 2.0):
            base = torch.tensor((0.3125, -1.75, 3.125), dtype=base_dtype)
            factor = torch.tensor((1.0009765625, -0.333, 2.5), dtype=factor_dtype)

            def fn(base, factor):
                return torch.add(base, factor, alpha=alpha)

            expected = (base.double() + factor.double() * alpha).to(fn(base, factor).dtype)
            gm = make_fx(fn)(base, factor)
            fusions.fuse_mul_add_to_fma(gm.graph)
            gm.recompile()
            actual = gm(base, factor)
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("alias", (lambda x: x.view_as(x), lambda x: x.transpose(0, 1)))
def test_post_grad_keeps_affine_epochs_across_alias_writes(alias):
    def fn(x):
        before = torch.add(x * 0.5, 1.0, alpha=0.25)
        alias(x).add_(0.5)
        after = torch.add(x * 0.5, 1.0, alpha=0.25)
        return before, after

    value = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    gm = make_fx(fn)(value.clone())
    fusions.post_grad_custom_pre_pass(gm.graph)
    gm.recompile()
    expected = fn(value.clone())
    actual = gm(value.clone())
    assert all(torch.equal(x, y) for x, y in zip(actual, expected))


def test_compile_prefers_fp32_product_for_mixed_product_fma():
    torch.manual_seed(17)
    a = (torch.randn(4096) * 10).to(torch.float16)
    b = (torch.randn(4096) * 10).to(torch.float16)
    c = torch.randn(4096) * 10
    d = torch.randn(4096) * 10

    def fn(a, b, c, d):
        return c * d + a * b

    reference = (c.double() * d.double() + a.double() * b.double()).float()
    eager_error = (fn(a, b, c, d) - reference).abs().sum()
    gm = make_fx(fn)(a, b, c, d)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
    fma = next(node for node in gm.graph.nodes if node.target == torch.ops.prims.fma.default)
    assert all(arg.meta["val"].dtype == torch.float32 for arg in fma.args[:2])
    compiled_error = (fusions.compile(fn, fullgraph=True)(a, b, c, d) - reference).abs().sum()
    assert compiled_error < eager_error


def test_compiled_adam_like_recurrence_tracks_fp64():
    def step(param, exp_avg, exp_avg_sq, grad, beta1, beta2, lr, eps):
        exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)
        param = param - lr * exp_avg / (exp_avg_sq.sqrt() + eps)
        return param, exp_avg, exp_avg_sq

    gradients = [torch.sin(torch.arange(64, dtype=torch.float32) * 0.17 + index * 0.13) for index in range(128)]
    scalars = tuple(torch.tensor(value, dtype=torch.float32) for value in (0.9, 0.99, 0.01, 1e-8))

    def run(step_fn, dtype):
        param = torch.linspace(-1, 1, 64, dtype=dtype)
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        beta1, beta2, lr, eps = (scalar.to(dtype) for scalar in scalars)
        for grad in gradients:
            param, exp_avg, exp_avg_sq = step_fn(param, exp_avg, exp_avg_sq, grad.to(dtype), beta1, beta2, lr, eps)
        return param, exp_avg, exp_avg_sq

    reference = run(step, torch.float64)
    eager = run(step, torch.float32)
    compiled = run(fusions.compile(step, fullgraph=True), torch.float32)
    eager_error = sum((actual.double() - expected).abs().sum() for actual, expected in zip(eager, reference))
    compiled_error = sum((actual.double() - expected).abs().sum() for actual, expected in zip(compiled, reference))
    assert max((actual.double() - expected).abs().max() for actual, expected in zip(compiled, reference)) < 2e-6
    assert compiled_error <= eager_error + 2e-6
