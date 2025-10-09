import numpy as np
import optuna
import pandas as pd
import torch
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.samplers import RandomSampler
from optuna.trial import TrialState

from heavyball import helpers


def test_bound_to_torch_roundtrip_cpu():
    arr = np.arange(4, dtype=np.float64).reshape(2, 2)
    tensor = helpers.bound_to_torch(arr.tobytes(), arr.shape, "cpu")
    assert torch.allclose(tensor, torch.from_numpy(arr.T))


def test_nextafter_matches_numpy():
    forward = helpers.nextafter(0.5, 1.0)
    backward = helpers.nextafter(1, 0)
    assert forward == np.nextafter(0.5, 1.0)
    assert backward == np.nextafter(1, 0)


def test_untransform_numerical_param_torch_handles_steps():
    dist = FloatDistribution(0.0, 1.0, step=0.1)
    value = torch.tensor(0.46)
    untransformed = helpers._untransform_numerical_param_torch(value, dist, transform_log=False)
    assert torch.isclose(untransformed, torch.tensor(0.5))


def test_simple_api_sampler_suggest_all_returns_expected():
    distributions = {"x": FloatDistribution(0.0, 1.0), "y": IntDistribution(0, 3, step=1)}

    class _Sampler(helpers.SimpleAPIBaseSampler):
        def infer_relative_search_space(self, study, trial):
            return self.search_space

        def sample_relative(self, study, trial, search_space):
            return {}

        def sample_independent(self, study, trial, param_name, param_distribution):
            return trial.params[param_name]

    sampler = _Sampler(distributions)

    class DummyTrial:
        def __init__(self, params):
            self.params = params

        def _suggest(self, name, dist):
            return self.params[name]

    trial = DummyTrial({"x": 0.25, "y": 2})
    suggestions = sampler.suggest_all(trial)
    assert suggestions == {"x": 0.25, "y": 2}


def test_botorch_sampler_sample_relative_smoke(monkeypatch):
    search_space = {"width": FloatDistribution(0.0, 1.0)}
    study = optuna.create_study(direction="minimize", sampler=RandomSampler(seed=0))
    for _ in range(3):
        trial = study.ask()
        width = trial.suggest_float("width", 0.0, 1.0)
        study.tell(trial, width)

    sampler = helpers.BoTorchSampler(search_space, n_startup_trials=1, seed=0, device="cpu")

    def _dummy_candidates(params, values, *_args):
        assert params.shape[1] == 1
        return params.mean(dim=0)

    sampler._candidates_func = _dummy_candidates

    pending = study.ask()
    suggestion = sampler.sample_relative(study, pending, search_space)
    assert "width" in suggestion
    assert 0.0 <= suggestion["width"] <= 1.0


def test_hebo_sampler_observe_and_sample(monkeypatch):
    class DummyHEBO:
        def __init__(self, *_args, **_kwargs):
            self.observed = None

        def suggest(self):
            return pd.DataFrame([{"depth": 0.0}])

        def observe(self, params, values):
            self.observed = (params, values)

    monkeypatch.setattr(helpers, "HEBO", DummyHEBO)

    search_space = {"depth": FloatDistribution(0.0, 1.0)}
    sampler = helpers.HEBOSampler(search_space, seed=1)

    study = optuna.create_study(direction="minimize", sampler=RandomSampler(seed=1))
    trial = study.ask()
    trial.suggest_float("depth", 0.0, 1.0)
    study.tell(trial, 0.2)

    suggestion = sampler.sample_relative(study, study.ask(), search_space)
    assert suggestion["depth"] == 0.0

    completed = study.get_trials(deepcopy=False)[0]
    sampler.after_trial(study, completed, TrialState.COMPLETE, [0.2])
    assert sampler._hebo.observed is not None
