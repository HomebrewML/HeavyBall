"""Behavioral coverage of the Engine's input-validation guards: each triggers the real error path."""

from unittest.mock import patch

import pytest
import torch

from heavyball import adamw
from heavyball.core import Engine


def _engine(recipe=adamw, shape=(2, 2), count=1, **hyper):
    params = [torch.nn.Parameter(torch.zeros(*shape)) for _ in range(count)]
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        return Engine(params, recipe, lr=1e-3, **hyper)


def test_set_hyper_rejects_host_only_hyperparameter():
    opt = _engine()
    with pytest.raises(ValueError, match="not a dynamic hyperparameter"):
        opt.set_hyper("preconditioner_update_probability", 0.5)


def test_set_hyper_rejects_unknown_name():
    opt = _engine()
    with pytest.raises(ValueError, match="unknown hyperparameter"):
        opt.set_hyper("definitely_not_a_hyper", 0.5)


def test_set_hyper_updates_a_dynamic_cell():
    opt = _engine()
    namespace = next(iter(opt._hyper_locations.values()))
    opt.set_hyper("lr", 7e-4)
    assert float(namespace.lr) == pytest.approx(7e-4)


def test_step_observed_mapping_must_cover_every_parameter():
    opt = _engine(count=2)
    with pytest.raises(ValueError, match="every trainable parameter"):
        opt.step(observed={opt.params[0]: True})  # one of two params


def test_step_observed_mapping_missing_key_is_rejected():
    opt = _engine(count=1)
    stranger = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="every trainable parameter"):
        opt.step(observed={stranger: True})  # right length, wrong key


def test_step_observed_sequence_must_have_one_value_per_parameter():
    opt = _engine(count=2)
    with pytest.raises(ValueError, match="one value for every trainable parameter"):
        opt.step(observed=[True])  # one value for two params


def test_step_observed_values_must_be_bools():
    opt = _engine(count=1)
    with pytest.raises(TypeError, match="observed values must be host bools"):
        opt.step(observed=[1])  # int, not bool


def test_engine_produce_rejects_an_undeclared_observation_on_an_owned_parameter():
    opt = _engine()
    with pytest.raises(ValueError, match="not a declared observation for this parameter's group"):
        opt.produce(opt.params[0], "sum_grad_squared", torch.zeros(2, 2))


def test_module_produce_rejects_a_parameter_without_the_observation_binding():
    from heavyball.core import produce

    unbound = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="not a declared observation for this parameter$"):
        produce(unbound, "grad", torch.zeros(2, 2))


def test_engine_produce_rejects_a_foreign_parameter():
    opt = _engine()
    foreign = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="parameter is not owned by this Engine"):
        opt.produce(foreign, "sum_grad_squared", torch.zeros(2, 2))
