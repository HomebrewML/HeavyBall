"""Regression tests for per-Engine torch.compile recompile budgets."""

from contextlib import contextmanager
from dataclasses import replace

import torch
import torch._dynamo as dyn

import heavyball
import heavyball.core as core


@contextmanager
def _low_recompile_limit(limit=2):
    saved = dyn.config.recompile_limit
    dyn.reset()
    dyn.config.recompile_limit = limit
    try:
        yield
    finally:
        dyn.config.recompile_limit = saved
        dyn.reset()


def test_distinct_shape_optimizers_do_not_share_recompile_budget():
    with _low_recompile_limit():
        optimizers = []
        for index in range(4):
            params = [torch.nn.Parameter(torch.randn(3 + index, 3 + index))]
            optimizer = heavyball.AdamW(params, lr=1e-3)
            params[0].grad.copy_(torch.randn_like(params[0]))
            optimizer.step()
            optimizers.append(optimizer)


def test_distinct_shape_param_groups_do_not_share_recompile_budget():
    with _low_recompile_limit():
        params = [torch.nn.Parameter(torch.randn(3 + index, 3 + index)) for index in range(4)]
        optimizer = heavyball.HeavyBallOptimizer(
            [{"params": [param]} for param in params],
            heavyball.adamw,
            lr=1e-3,
        )
        for param in params:
            param.grad.copy_(torch.randn_like(param))
        optimizer.step()


def test_same_shape_param_groups_share_compiled_step():
    dyn.reset()
    dyn.utils.counters.clear()
    try:
        params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(8)]
        optimizer = heavyball.HeavyBallOptimizer(
            [{"params": [param]} for param in params],
            heavyball.adamw,
            lr=1e-3,
        )
        for param in params:
            param.grad.copy_(torch.randn_like(param))

        optimizer.step()

        assert dyn.utils.counters["stats"]["unique_graphs"] == 1
    finally:
        dyn.reset()
        dyn.utils.counters.clear()


def test_separate_engines_have_distinct_compiled_step_code_names():
    first = heavyball.AdamW([torch.nn.Parameter(torch.randn(3, 3))])
    second = heavyball.AdamW([torch.nn.Parameter(torch.randn(4, 4))])

    first_name = first._engines[0].compiled_step.__wrapped__.__code__.co_name
    second_name = second._engines[0].compiled_step.__wrapped__.__code__.co_name

    assert first_name != second_name


def test_distinct_state_abis_have_distinct_compiled_steps():
    with _low_recompile_limit(1):
        plain_param = torch.nn.Parameter(torch.randn(5))
        ecc_param = torch.nn.Parameter(torch.randn(5))
        plain = heavyball.AdamW([plain_param], lr=1e-3)
        ecc = heavyball.AdamW([ecc_param], lr=1e-3, ecc=8)

        plain_param.grad.copy_(torch.randn_like(plain_param))
        ecc_param.grad.copy_(torch.randn_like(ecc_param))
        plain.step()
        ecc.step()

        plain_code = plain._engines[0].compiled_step.__wrapped__.__code__
        ecc_code = ecc._engines[0].compiled_step.__wrapped__.__code__
        assert plain_code is not ecc_code


def test_identical_optimizers_share_code_object_cache(monkeypatch):
    cache = {}
    monkeypatch.setattr(core, "_STEP_CODE_CACHE", cache)
    first_params = [torch.nn.Parameter(torch.randn(size)) for size in (11, 13)]
    second_params = [torch.nn.Parameter(torch.randn(size)) for size in (11, 13)]

    first = heavyball.AdamW(first_params, recipe=replace(heavyball.adamw), lr=1e-3)
    first_cache_size = len(cache)
    second = heavyball.AdamW(second_params, recipe=replace(heavyball.adamw), lr=1e-3)

    assert first_cache_size == 2
    assert len(cache) == first_cache_size
    first_code = first._engines[0].compiled_step.__wrapped__.__code__
    second_code = second._engines[0].compiled_step.__wrapped__.__code__
    assert first_code is second_code
