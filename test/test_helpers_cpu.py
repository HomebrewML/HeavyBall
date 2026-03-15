import math

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.samplers import RandomSampler
from optuna.trial import TrialState

from heavyball import helpers
from heavyball.helpers import HEBO, DesignSpace


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


def test_hebo_suggest_observe_roundtrip():
    space = DesignSpace().parse(
        [
            {"name": "lr", "type": "pow", "lb": 1e-5, "ub": 1.0},
            {"name": "wd", "type": "num", "lb": 0.0, "ub": 0.1},
            {"name": "layers", "type": "int", "lb": 1, "ub": 8},
        ]
    )
    opt = HEBO(space, scramble_seed=42)

    for i in range(space.num_paras + 5):
        suggestion = opt.suggest()
        assert len(suggestion) == 1
        assert set(suggestion.columns) == {"lr", "wd", "layers"}
        lr = suggestion["lr"].iloc[0]
        wd = suggestion["wd"].iloc[0]
        layers = suggestion["layers"].iloc[0]
        assert 1e-5 <= lr <= 1.0
        assert 0.0 <= wd <= 0.1
        assert 1 <= layers <= 8
        assert isinstance(layers, (int, np.integer))
        opt.observe(suggestion, np.array([[lr + wd + layers * 0.1]]))

    assert opt.best_y < float("inf")
    assert len(opt.best_x) == 1


def test_hebo_discrete_explores_all_values():
    space = DesignSpace().parse([{"name": "x", "type": "int", "lb": 0, "ub": 3}])
    opt = HEBO(space, scramble_seed=0)
    seen = set()
    for _ in range(6):
        sug = opt.suggest()
        val = int(sug["x"].iloc[0])
        seen.add(val)
        opt.observe(sug, np.array([[val**2]]))
    assert seen == {0, 1, 2, 3}


def test_design_space_parse_replaces_state():
    ds = DesignSpace()
    ds.parse([{"name": "x", "type": "num", "lb": 0, "ub": 1}])
    ds.parse([{"name": "y", "type": "num", "lb": 0, "ub": 1}])
    assert ds.para_names == ["y"]
    assert ds.num_paras == 1


# -- HEBO performance baselines --
# Original HEBO v0.3.6, 30 runs across seeds 0-29:
#   Sphere 15 iter: median=0.030, p95=0.260, max=0.507
#   Branin 20 iter: median=0.402, p95=0.413, max=0.466
#   Mixed  15 iter: median=0.012, p95=0.384, max=0.444
# Our LCB acquisition is simpler than HEBO's evolutionary optimizer.
# Thresholds use original median as reference.

_HEBO_BASELINE = {"sphere": 0.03, "branin": 0.40, "mixed": 0.01}


def _branin(x1, x2):
    a, b, c = 1, 5.1 / (4 * math.pi**2), 5 / math.pi
    r, s, t = 6, 10, 1 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def _run_hebo(space_cfg, objective, n_iter, seed=42):
    space = DesignSpace().parse(space_cfg)
    opt = HEBO(space, scramble_seed=seed)
    for _ in range(n_iter):
        sug = opt.suggest()
        opt.observe(sug, np.array([[objective(sug)]]))
    return opt


def test_hebo_perf_sphere():
    opt = _run_hebo(
        [{"name": "x1", "type": "num", "lb": -5, "ub": 5}, {"name": "x2", "type": "num", "lb": -5, "ub": 5}],
        lambda sug: sug["x1"].iloc[0] ** 2 + sug["x2"].iloc[0] ** 2,
        n_iter=15,
    )
    assert opt.best_y < _HEBO_BASELINE["sphere"] * 20


def test_hebo_perf_branin():
    opt = _run_hebo(
        [{"name": "x1", "type": "num", "lb": -5, "ub": 10}, {"name": "x2", "type": "num", "lb": 0, "ub": 15}],
        lambda sug: _branin(sug["x1"].iloc[0], sug["x2"].iloc[0]),
        n_iter=20,
    )
    assert opt.best_y < _HEBO_BASELINE["branin"] * 2


def test_hebo_perf_mixed():
    def objective(sug):
        lr, wd, layers = sug["lr"].iloc[0], sug["wd"].iloc[0], sug["layers"].iloc[0]
        return (np.log10(lr) + 0.5) ** 2 + 100 * (wd - 0.09) ** 2 + (layers - 7) ** 2

    opt = _run_hebo(
        [
            {"name": "lr", "type": "pow", "lb": 1e-5, "ub": 1.0},
            {"name": "wd", "type": "num", "lb": 0.0, "ub": 0.1},
            {"name": "layers", "type": "int", "lb": 1, "ub": 8},
        ],
        objective,
        n_iter=15,
    )
    assert opt.best_y < _HEBO_BASELINE["mixed"] * 20
