"""Verifies finding-5: the commit's config enters the recipe fingerprint, backward-compatibly.

make_retraction_commit is a parametrized factory -- its instances share one __name__, so the old
fingerprint (commit.__name__ only) would silently collide their checkpoints. The fix fingerprints the
commit's .config too, while config-less commits keep their bare-name fingerprint so existing
checkpoints still load.
"""

from unittest.mock import patch

import torch

from heavyball import Engine, Recipe, make_retraction_commit, stiefel_projection
from heavyball.transforms import momentum, muon_commit, orthogonalize, sgd_commit


def _commit_fingerprint(recipe):
    param = torch.nn.Parameter(torch.zeros(4, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = Engine([param], recipe, lr=1e-3, beta1=0.9, weight_decay=0.0)
    return next(iter(optimizer._recipe_fingerprints().values()))["commit"]


def test_config_less_commit_keeps_its_bare_name_fingerprint():
    recipe = Recipe(
        chain=(momentum, orthogonalize), commit=sgd_commit, defaults=dict(lr=1e-3, beta1=0.9, weight_decay=0.0)
    )
    assert _commit_fingerprint(recipe) == "sgd_commit"


def test_retraction_commit_config_distinguishes_fingerprints():
    def recipe(commit):
        return Recipe(
            chain=(momentum, orthogonalize), commit=commit, defaults=dict(lr=1e-3, beta1=0.9, weight_decay=0.0)
        )

    over_sgd = _commit_fingerprint(recipe(make_retraction_commit(sgd_commit, stiefel_projection, name="stiefel")))
    over_muon = _commit_fingerprint(recipe(make_retraction_commit(muon_commit, stiefel_projection, name="stiefel")))
    assert isinstance(over_sgd, tuple)  # config-ful commit -> (name, config)
    assert over_sgd != over_muon  # differing wrapped base commit -> distinct fingerprints


def test_projection_identity_distinguishes_same_named_retraction_commits():
    from heavyball.transforms import oblique_normalization, stiefel_projection

    def recipe(commit):
        return Recipe(
            chain=(momentum, orthogonalize), commit=commit, defaults=dict(lr=1e-3, beta1=0.9, weight_decay=0.0)
        )

    # Same human label, DIFFERENT projection function -> the fingerprint must still differ (the .config
    # records the projection's function identity, not only the passed name).
    same_name_stiefel = _commit_fingerprint(recipe(make_retraction_commit(sgd_commit, stiefel_projection, name="proj")))
    same_name_oblique = _commit_fingerprint(recipe(make_retraction_commit(sgd_commit, oblique_normalization, name="proj")))
    assert same_name_stiefel != same_name_oblique
