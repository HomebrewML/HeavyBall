"""Checkpoint recipe fingerprints include transform factory configuration."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from heavyball import sgd
from heavyball.core import Engine, Recipe
from heavyball.kron import kron, make_psgd_kron
from heavyball.psgd_pro import make_psgd_pro, psgd_pro
from heavyball.transforms import sgd_commit


def _engine(recipe: Recipe) -> Engine:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine([parameter], recipe, param_keys=("weight",))


def test_kron_factory_config_rejects_incompatible_checkpoint():
    recipe2a = Recipe(chain=(make_psgd_kron(2),), commit=sgd_commit, defaults=kron.defaults)
    recipe2b = Recipe(chain=(make_psgd_kron(2),), commit=sgd_commit, defaults=kron.defaults)
    recipe8 = Recipe(chain=(make_psgd_kron(8),), commit=sgd_commit, defaults=kron.defaults)
    engine2a = _engine(recipe2a)
    engine2b = _engine(recipe2b)
    engine8 = _engine(recipe8)

    checkpoint = engine2a.state_dict()
    engine2b.load_state_dict(checkpoint)
    with pytest.raises(ValueError, match="checkpoint recipe fingerprint does not match"):
        engine8.load_state_dict(checkpoint)


def test_psgd_pro_sqrt_config_rejects_incompatible_checkpoint():
    recipe = Recipe(chain=(make_psgd_pro(),), commit=sgd_commit, defaults=psgd_pro.defaults)
    sqrt_recipe = Recipe(chain=(make_psgd_pro(sqrt=True),), commit=sgd_commit, defaults=psgd_pro.defaults)
    engine = _engine(recipe)
    sqrt_engine = _engine(sqrt_recipe)

    checkpoint = engine.state_dict()
    with pytest.raises(ValueError, match="checkpoint recipe fingerprint does not match"):
        sqrt_engine.load_state_dict(checkpoint)


def test_effective_clip_global_norm_rejects_incompatible_checkpoint():
    recipe = replace(sgd, clip_global_norm=0.1)
    source_param = torch.nn.Parameter(torch.zeros(1))
    target_param = torch.nn.Parameter(torch.zeros(1))
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        source = Engine([source_param], recipe, lr=1.0, weight_decay=0.0, param_keys=("p",))
        target = Engine(
            [target_param], recipe, clip_global_norm=1.0,
            lr=1.0, weight_decay=0.0, param_keys=("p",),
        )

    checkpoint = source.state_dict()
    with pytest.raises(ValueError, match="checkpoint recipe fingerprint does not match"):
        target.load_state_dict(checkpoint)
