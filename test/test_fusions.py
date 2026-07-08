import torch
from torch.fx.experimental.proxy_tensor import make_fx

from heavyball import fusions


def _targets(gm):
    return [node.target for node in gm.graph.nodes if node.op == "call_function"]


def test_mul_add_rewrites_to_fma():
    a = torch.randn(8)
    b = torch.randn(8)
    c = torch.randn(8)

    for fn in (lambda a, b, c: a * b + c, lambda a, b, c: c + a * b):
        gm = make_fx(fn)(a, b, c)

        assert fusions.fuse_mul_add_to_fma(gm.graph) == 1

        targets = _targets(gm)
        assert torch.ops.prims.fma.default in targets
        assert torch.ops.aten.add.Tensor not in targets
        assert torch.ops.aten.mul.Tensor not in targets


def test_mul_add_fusion_skips_integer_tensors():
    a = torch.arange(8)
    b = torch.arange(8)
    c = torch.arange(8)
    gm = make_fx(lambda a, b, c: a * b + c)(a, b, c)

    assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
    assert torch.ops.prims.fma.default not in _targets(gm)


def test_mul_sub_rewrites_to_fma():
    a = torch.randn(8)
    b = torch.randn(8)
    c = torch.randn(8)

    for fn in (lambda a, b, c: a * b - c, lambda a, b, c: c - a * b):
        gm = make_fx(fn)(a, b, c)

        assert fusions.fuse_mul_add_to_fma(gm.graph) == 1

        targets = _targets(gm)
        assert torch.ops.prims.fma.default in targets
        assert torch.ops.aten.sub.Tensor not in targets
        assert torch.ops.aten.mul.Tensor not in targets


def test_scaled_add_rewrites_to_fma():
    x = torch.randn(8)
    y = torch.randn(8)

    # add(x, y, alpha=k) -> fma(y, k, x); a scaled add has no separate mul.
    gm = make_fx(lambda x, y: torch.add(x, y, alpha=0.5))(x, y)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1

    targets = _targets(gm)
    assert torch.ops.prims.fma.default in targets
    assert torch.ops.aten.add.Tensor not in targets


def test_rsub_rewrites_to_fma():
    # 1 - a*b (scalar on the left) compiles to aten.rsub, which the pass must fuse.
    a = torch.tensor(0.01)
    b = torch.tensor(1e-3)
    gm = make_fx(lambda a, b: 1 - a * b)(a, b)
    assert fusions.fuse_mul_add_to_fma(gm.graph) == 1

    targets = _targets(gm)
    assert torch.ops.prims.fma.default in targets
    assert torch.ops.aten.rsub.Scalar not in targets


def test_scalar_addend_rewrites_to_fma():
    # a*b +/- scalar: prims.fma accepts a scalar addend, so these fuse too.
    a = torch.randn(8)
    b = torch.randn(8)
    for fn in (lambda a, b: a * b - 1, lambda a, b: a * b + 0.5):
        gm = make_fx(fn)(a, b)
        assert fusions.fuse_mul_add_to_fma(gm.graph) == 1
        assert torch.ops.prims.fma.default in _targets(gm)


def _dtypes(gm):
    return {str(nd.target).split("default")[0].rstrip("."): nd.meta["val"].dtype
            for nd in gm.graph.nodes if nd.op == "call_function"
            and isinstance(nd.meta.get("val"), torch.Tensor)}


def test_scalar_ops_promoted_to_fp64_tensors_stay():
    # t * (1 - beta**step): the scalar chain runs in fp64, the tensor mul stays fp32.
    gm = make_fx(lambda b, s, t: t * (1 - b**s))(
        torch.tensor(0.9), torch.tensor(5), torch.randn(4))
    assert fusions.promote_scalar_ops_to_fp64(gm.graph) >= 1

    dt = _dtypes(gm)
    assert dt["aten.pow.Tensor_Tensor"] == torch.float64   # scalar op -> fp64
    assert dt["aten.rsub.Scalar"] == torch.float64         # scalar op -> fp64
    assert dt["aten.mul.Tensor"] == torch.float32          # tensor op -> unchanged


def test_plain_add_sub_left_alone():
    a = torch.randn(8)
    b = torch.randn(8)

    for fn in (lambda a, b: a + b, lambda a, b: a - b):
        gm = make_fx(fn)(a, b)
        assert fusions.fuse_mul_add_to_fma(gm.graph) == 0
        assert torch.ops.prims.fma.default not in _targets(gm)


def test_beta_zero_stays_exact():
    # beta_debias must return exactly 0 for beta=0 (0**step == 0 for step >= 1,
    # so 1 - (1-0)/(1-0) collapses to 0 in any float dtype).
    from heavyball.utils import beta_debias

    for step in (1, 2, 5, 100):
        out = beta_debias(torch.tensor(0.0, dtype=torch.float32), torch.tensor(step, dtype=torch.int64))
        assert float(out) == 0.0, f"beta=0 step={step} got {float(out)}"
