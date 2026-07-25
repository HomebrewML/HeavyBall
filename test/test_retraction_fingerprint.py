from unittest.mock import patch

import pytest
import torch

from heavyball import Engine, Recipe, make_retraction_commit, stiefel_projection
from heavyball.transforms import (
    momentum,
    muon_commit,
    oblique_normalization,
    orthogonalize,
    sgd_commit,
)


def _engine(recipe):
    param = torch.nn.Parameter(torch.zeros(4, 4))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(
            [param],
            recipe,
            lr=1e-3,
            beta1=0.9,
            weight_decay=0.0,
            param_keys=("p",),
        )


def _recipe(commit):
    return Recipe(
        chain=(momentum, orthogonalize),
        commit=commit,
        defaults=dict(lr=1e-3, beta1=0.9, weight_decay=0.0),
    )


def test_config_less_commit_keeps_its_bare_name_fingerprint():
    assert _engine(_recipe(sgd_commit)).state_dict()["fingerprint"]["p"]["commit"] == "sgd_commit"


@pytest.mark.parametrize(
    ("source_commit", "target_commit"),
    (
        (
            make_retraction_commit(sgd_commit, stiefel_projection, name="stiefel"),
            make_retraction_commit(muon_commit, stiefel_projection, name="stiefel"),
        ),
        (
            make_retraction_commit(sgd_commit, stiefel_projection, name="proj"),
            make_retraction_commit(sgd_commit, oblique_normalization, name="proj"),
        ),
    ),
)
def test_retraction_config_rejects_incompatible_checkpoint(source_commit, target_commit):
    checkpoint = _engine(_recipe(source_commit)).state_dict()

    with pytest.raises(ValueError, match="checkpoint recipe fingerprint does not match"):
        _engine(_recipe(target_commit)).load_state_dict(checkpoint)
