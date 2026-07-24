"""Optimizer facade runtime and static autocomplete signatures."""

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

import heavyball
from heavyball import optim


def test_every_facade_signature_exposes_recipe_defaults():
    for name in optim.__all__:
        facade = getattr(heavyball, name)
        if not (isinstance(facade, type) and issubclass(facade, optim.HeavyBallOptimizer) and facade is not optim.HeavyBallOptimizer):
            continue
        signature = inspect.signature(facade)
        assert "params" in signature.parameters
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        for hyper_name, default in optim._recipe_defaults(facade.recipe, {}).items():
            parameter = signature.parameters[hyper_name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default == default
            expected_annotation = (float | None) if default is None else type(default)
            assert parameter.annotation == expected_annotation


def test_representative_facade_hyperparameters_are_explicit():
    assert "shampoo_beta" in inspect.signature(heavyball.SOAP).parameters
    psgd_kron_parameters = inspect.signature(heavyball.PSGDKron).parameters
    assert "precond_lr" in psgd_kron_parameters
    assert "lower_bound_beta" in psgd_kron_parameters
    psgd_parameters = inspect.signature(heavyball.PSGD).parameters
    assert "max_size_triangular" in psgd_parameters
    assert "rank" in psgd_parameters
    assert "beta3" in inspect.signature(heavyball.AdEMAMix).parameters
    assert "lr" in inspect.signature(heavyball.AdamW).parameters


def test_facades_expose_engine_level_opt_ins():
    for name in optim.__all__:
        facade = getattr(heavyball, name)
        if not (isinstance(facade, type) and issubclass(facade, optim.HeavyBallOptimizer) and facade is not optim.HeavyBallOptimizer):
            continue
        parameters = inspect.signature(facade).parameters
        for engine_kwarg in ("storage_dtype", "ecc", "clip_global_norm", "param_keys"):
            assert parameters[engine_kwarg].kind is inspect.Parameter.KEYWORD_ONLY


def test_generated_stub_is_fresh():
    from heavyball import _autocomplete_stub

    stub_path = Path(optim.__file__).with_suffix(".pyi")
    assert _autocomplete_stub.render() == stub_path.read_text(encoding="utf-8")


def test_facade_class_signature_omits_self():
    # The class-level __signature__ must not surface the bound ``self`` (regression: it was inserted,
    # so ``inspect.signature(AdamW)`` started with ``self``).
    for facade in (heavyball.AdamW, heavyball.AdamC, heavyball.SOAP):
        first = next(iter(inspect.signature(facade).parameters))
        assert first == "params", f"{facade.__name__} signature starts with {first!r}, expected 'params'"


def test_facade_positional_float_gives_actionable_error():
    # ``AdamW(params, 1e-3)`` used to bind the float to the internal ``recipe`` and fail with a cryptic
    # ``AttributeError: 'float' object has no attribute 'otherwise'``. It must now raise a clear error.
    param = torch.nn.Parameter(torch.zeros(1))
    try:
        heavyball.AdamW([param], 1e-3)
    except TypeError:
        return
    except AttributeError:
        pytest.fail("positional float was bound to 'recipe' (cryptic AttributeError)")
    pytest.fail("expected an error for a positional float argument")


def test_facade_recipe_attributes_preserve_construction_and_step():
    for facade in (heavyball.AdamW, heavyball.SOAP):
        model = nn.Linear(4, 3)
        optimizer = facade(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
