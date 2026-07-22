"""FSDP2 scope guard: reject unsupported recipes before touching the model."""

import pytest

import heavyball

# Representative rejected facades: matrix-preconditioned, matrix-transform, and full-leaf families.
_REJECTED = [
    "WhitenAdamW",
    "CautiousAdamW", "MSAMAdamW", "MSAM",
]


@pytest.mark.parametrize("name", _REJECTED)
def test_fsdp2_rejects_non_shard_separable(name):
    facade = getattr(heavyball, name, None)
    if facade is None or not hasattr(facade, "fsdp2"):
        pytest.skip(f"{name} not exported")
    # object() is never inspected: the recipe scope check fires before binding resolution.
    with pytest.raises(ValueError, match="shard-separable"):
        facade.fsdp2(object())


@pytest.mark.parametrize("name", ["Muon", "PSGDKron", "SOAP", "Shampoo", "LATHER", "PSGDPro"])
def test_whole_scoped_optimizer_passes_the_recipe_scope_guard(name):
    with pytest.raises(TypeError, match="fully_shard"):
        getattr(heavyball, name).fsdp2(object())


def test_fsdp2_rejects_clip_global_norm():
    with pytest.raises(ValueError, match="clip_global_norm"):
        heavyball.AdamW.fsdp2(object(), clip_global_norm=1.0)


def test_fsdp2_rejects_recipe_override():
    with pytest.raises(ValueError, match="recipe"):
        heavyball.AdamW.fsdp2(object(), recipe="something")


def test_fsdp2_rejects_plain_iterable_model():
    # A componentwise recipe passes the scope check, then must reject a non-model argument.
    with pytest.raises((TypeError, ValueError)):
        heavyball.AdamW.fsdp2([__import__("torch").zeros(2, 2)])
