import os

import pytest
import torch

import heavyball
import heavyball.chainable as C
import heavyball.utils

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
heavyball.utils.compile_mode = None


def _identity_update(state, group, update, grad, param):
    return update


def test_chain_applies_update_on_cpu():
    param = [torch.nn.Parameter(torch.zeros(2))]
    grad = [torch.ones(2)]
    group = {"lr": 0.1, "caution": False, "weight_decay": 0.0}

    with torch.no_grad():
        C.chain(lambda _: {}, group, grad, param, _identity_update)

    assert torch.allclose(param[0].detach(), torch.full((2,), -0.1))


def test_branch_merges_multiple_paths():
    def double(_, __, update, ___, ____):
        return [u * 2 for u in update]

    def negate(_, __, update, ___, ____):
        return [u * -1 for u in update]

    def merge_fn(outputs):
        return [sum(vals) / len(vals) for vals in zip(*outputs)]

    branch = C.Branch([[double], [negate]], merge_fn)

    update = [torch.ones(2)]
    grad = [torch.ones(2)]
    param = [torch.nn.Parameter(torch.ones(2))]

    result = branch(lambda _: {}, {}, update, grad, param)
    expected = torch.full_like(update[0], 0.5)
    assert torch.allclose(result[0], expected)


def test_set_indices_assigns_transform_ids():
    def base(_, __, update, ___, ____, buffer):
        assert buffer is not None
        return update

    zero_guard = C.ZeroGuard(base, ["buffer"])
    assigned = C.set_indices([zero_guard], retain=False)[0]
    assert assigned.transform_idx == 0

    def state_fn(_x):
        return {}

    group = {"storage_dtype": "float32"}
    update = [torch.ones(1)]
    grad = [torch.ones(1)]
    param = [torch.nn.Parameter(torch.ones(1))]

    assigned(state_fn, group, update, grad, param)


# Optimizers whose chains are purely elementwise must NOT need gather
_EXPECT_NO_GATHER = {"SGD", "ForeachAdamW", "ForeachNAdam", "ForeachAdEMAMix", "UnscaledAdamW", "ForeachAdamC",
    "ForeachRMSprop", "ForeachSFAdamW", "ForeachADOPT", "ForeachLaProp", "PaLMForeachSFAdamW", }

# Optimizers whose chains use shape-dependent or global-reduction ops must need gather
_EXPECT_GATHER = {"ForeachSOAP", "ForeachSOAPNAdam", "ForeachSOAPAdEMAMix", "ForeachSOLP", "ForeachMuon", "MuonLaProp",
    "OrthoLaProp", "LaPropOrtho", "ForeachPSGDKron", "ForeachPurePSGD", "ForeachCachedPSGDKron", "ForeachDelayedPSGD",
    "ForeachCachedDelayedPSGDKron", "ForeachCachedNewtonPSGD", "NewtonHybrid2PSGDKron", "ForeachPSGDLRA",
    "ForeachDelayedPSGDLRA", "ForeachNewtonPSGDLRA", "NewtonHybrid2PSGDLRA", "SUDSAdamW", "Scion", "ForeachSignLaProp",
    "MSAMLaProp", "PaLMForeachSOAP", "PrecondScheduleForeachSOAP", "PrecondSchedulePaLMForeachSOAP", }

_SKIP_INSTANTIATE = {"SplitOpt", "SAMWrapper"}

_ALL_OPTS = [n for n in heavyball.__all__ if n not in _SKIP_INSTANTIATE and n in (_EXPECT_NO_GATHER | _EXPECT_GATHER)]


@pytest.mark.parametrize("opt_name", _ALL_OPTS)
def test_needs_gather_flag(opt_name):
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    extra = {"max_lr": 0.0025} if opt_name == "ForeachAdamC" else {}
    opt = getattr(heavyball, opt_name)(params, lr=1e-3, **extra)
    if opt_name in _EXPECT_NO_GATHER:
        assert not opt._needs_gather, f"{opt_name} should be elementwise (no gather needed)"
    elif opt_name in _EXPECT_GATHER:
        assert opt._needs_gather, f"{opt_name} should require full param gather"
