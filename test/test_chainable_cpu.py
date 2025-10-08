import os

import torch

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
