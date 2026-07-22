import heavyball
from heavyball import optim
from heavyball.optim import HeavyBallOptimizer, _recipe_defaults

_COMMON_HYPERPARAMETERS = {"lr", "beta1", "beta2", "eps", "weight_decay"}


def test_every_public_optimizer_facade_has_a_docstring():
    for name in optim.__all__:
        cls = getattr(heavyball, name)
        if not (isinstance(cls, type) and issubclass(cls, HeavyBallOptimizer)):
            continue
        docstring = cls.__doc__
        assert isinstance(docstring, str)
        assert docstring


def test_hyperparameter_glossary_covers_every_optimizer_specific_knob():
    specific = set()
    for name in optim.__all__:
        cls = getattr(heavyball, name)
        if not (isinstance(cls, type) and issubclass(cls, HeavyBallOptimizer) and cls is not HeavyBallOptimizer):
            continue
        specific |= set(_recipe_defaults(cls.recipe, {})) - _COMMON_HYPERPARAMETERS
    documentation = HeavyBallOptimizer.__doc__
    undocumented = sorted(name for name in specific if name not in documentation)
    assert not undocumented, f"undocumented hyperparameters: {undocumented}"
