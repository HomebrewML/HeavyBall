"""Optimizer facade runtime and static autocomplete signatures."""

import inspect
from pathlib import Path

import pytest
import torch

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
