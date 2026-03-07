import copy
import functools
import math
import threading
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy
import numpy as np
import optuna
import optunahub
import pandas as pd
import torch
from optuna._transform import _SearchSpaceTransform as SearchSpaceTransform
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from sklearn.preprocessing import power_transform
from torch import Tensor
from torch.nn import functional as F
from torch.quasirandom import SobolEngine

from heavyball.utils import scalar_guard

_MAXINT32 = (1 << 31) - 1
_SAMPLER_KEY = "auto:sampler"


@contextmanager
def manual_seed(seed: int | None = None) -> Generator[None, None, None]:
    r"""
    Contextmanager for manual setting the torch.random seed.

    Args:
        seed: The seed to set the random number generator to.

    Returns:
        Generator

    Example:
        >>> with manual_seed(1234):
        >>>     X = torch.rand(3)

    copied as-is from https://github.com/meta-pytorch/botorch/blob/a42cd65f9b704cdb6f2ee64db99a022eb15295d5/botorch/utils/sampling.py#L53C1-L75C50 under the MIT License
    """
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


class SimpleAPIBaseSampler(BaseSampler):
    def __init__(
        self,
        search_space: Optional[dict[str, BaseDistribution]] = None,
    ):
        self.search_space = {} if search_space is None else dict(search_space)

    def suggest_all(self, trial: FrozenTrial):
        return {k: trial._suggest(k, dist) for k, dist in self.search_space.items()}


def _get_default_candidates_func(
    n_objectives: int,
    has_constraint: bool,
    consider_running_trials: bool,
) -> Callable[
    [
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        Tensor | None,
    ],
    Tensor,
]:
    """
    The original is available at https://github.com/optuna/optuna-integration/blob/156a8bc081322791015d2beefff9373ed7b24047/optuna_integration/botorch/botorch.py under the MIT License
    """

    # lazy import
    from optuna_integration.botorch import (
        ehvi_candidates_func,
        logei_candidates_func,
        qehvi_candidates_func,
        qei_candidates_func,
        qparego_candidates_func,
    )

    if n_objectives > 3 and not has_constraint and not consider_running_trials:
        return ehvi_candidates_func
    elif n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    elif consider_running_trials:
        return qei_candidates_func
    else:
        return logei_candidates_func


@functools.cache
def bound_to_torch(bound: bytes, shape: tuple, device: str):
    bound = np.frombuffer(bound, dtype=np.float64).reshape(shape)
    bound = np.transpose(bound, (1, 0))
    return torch.from_numpy(bound).to(torch.device(device))


@functools.cache
def nextafter(x: float, y: float) -> Union[float, int]:
    return numpy.nextafter(x, y)


def _untransform_numerical_param_torch(
    trans_param: Union[float, Tensor],
    distribution: BaseDistribution,
    transform_log: bool,
) -> Tensor:
    d = distribution

    if isinstance(d, FloatDistribution):
        if d.log:
            param = trans_param.exp() if transform_log else trans_param
            if d.single():
                return param
            return param.clamp(max=nextafter(d.high, d.high - 1))

        if d.step is not None:
            scaled = ((trans_param - d.low) / d.step).round() * d.step + d.low
            return scaled.clamp(min=d.low, max=d.high)

        if d.single():
            return trans_param

        return trans_param.clamp(max=nextafter(d.high, d.high - 1))

    if not isinstance(d, IntDistribution):
        raise ValueError(f"Unexpected distribution type: {type(d)}")

    if d.log:
        param = trans_param.exp().round() if transform_log else trans_param
    else:
        param = ((trans_param - d.low) / d.step).round() * d.step + d.low
    param = param.clamp(min=d.low, max=d.high)
    return param.to(torch.int64)


@torch.no_grad()
def untransform(self: SearchSpaceTransform, trans_params: Tensor) -> dict[str, Any]:
    assert trans_params.shape == (self._raw_bounds.shape[0],)

    if self._transform_0_1:
        trans_params = self._raw_bounds[:, 0] + trans_params * (self._raw_bounds[:, 1] - self._raw_bounds[:, 0])

    params = {}

    for (name, distribution), encoded_columns in zip(self._search_space.items(), self.column_to_encoded_columns):
        if isinstance(distribution, CategoricalDistribution):
            raise ValueError("We don't support categorical parameters.")
        else:
            param = _untransform_numerical_param_torch(trans_params[encoded_columns], distribution, self._transform_log)

        params[name] = param

    return {n: v.item() for n, v in params.items()}


class BoTorchSampler(SimpleAPIBaseSampler):
    """
    A significantly more efficient implementation of `BoTorchSampler` from Optuna - keeps more on the GPU / in torch
    The original is available at https://github.com/optuna/optuna-integration/blob/156a8bc081322791015d2beefff9373ed7b24047/optuna_integration/botorch/botorch.py under the MIT License
    The original API is kept for backward compatibility, but many arguments are ignored to improve maintainability.
    """

    def __init__(
        self,
        search_space: Optional[dict[str, BaseDistribution]] = None,
        *,
        candidates_func: Optional[Callable[..., Tensor]] = None,
        constraints_func: Optional[Callable[..., Tensor]] = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: Optional[BaseSampler] = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        trial_chunks: int = 128,
    ):
        if constraints_func is not None:
            raise NotImplementedError("constraints_func is currently not supported by BoTorchSampler.")
        if consider_running_trials:
            raise NotImplementedError("consider_running_trials is currently not supported by BoTorchSampler.")
        if candidates_func is not None and not callable(candidates_func):
            raise TypeError("candidates_func must be callable.")
        self._candidates_func = candidates_func
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self.trial_chunks = trial_chunks

        self._study_id: int | None = None
        self.search_space = {} if search_space is None else dict(search_space)
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device or torch.device("cpu")
        self.seen_trials = set()
        self._values = None
        self._params = None
        self._index = 0
        self._bounds_dim: int | None = None

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self.search_space

    @torch.no_grad()
    def _preprocess_trials(
        self, trans: SearchSpaceTransform, study: Study, trials: list[FrozenTrial]
    ) -> Tuple[int, Tensor, Tensor]:
        bounds_dim = trans.bounds.shape[0]
        if self._bounds_dim is not None and self._bounds_dim != bounds_dim:
            self._values = None
            self._params = None
            self._index = 0
            self.seen_trials = set()
        if self._bounds_dim is None:
            self._bounds_dim = bounds_dim

        new_trials = []
        for trial in trials:
            tid: int = trial._trial_id
            if tid not in self.seen_trials:
                self.seen_trials.add(tid)
                new_trials.append(trial)
        trials = new_trials

        n_objectives = len(study.directions)
        if not new_trials:
            if self._values is None or self._params is None:
                empty_values = torch.zeros((0, n_objectives), dtype=torch.float64, device=self._device)
                empty_params = torch.zeros((0, bounds_dim), dtype=torch.float64, device=self._device)
                return n_objectives, empty_values, empty_params
            return n_objectives, self._values[: self._index], self._params[: self._index]

        n_completed_trials = len(trials)
        values: numpy.ndarray = numpy.empty((n_completed_trials, n_objectives), dtype=numpy.float64)
        params: numpy.ndarray = numpy.empty((n_completed_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        for trial_idx, trial in enumerate(trials):
            if trial.state != TrialState.COMPLETE:
                raise ValueError(f"TrialState must be COMPLETE, but {trial.state} was found.")

            params[trial_idx] = trans.transform(trial.params)
            values[trial_idx, :] = np.array(trial.values)

        for obj_idx, direction in enumerate(study.directions):
            if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                values[:, obj_idx] *= -1

        bounds_dim = trans.bounds.shape[0]
        cache_stale = (
            self._values is None
            or self._params is None
            or self._values.size(1) != n_objectives
            or self._params.size(1) != bounds_dim
        )
        if cache_stale:
            self._values = torch.zeros((self.trial_chunks, n_objectives), dtype=torch.float64, device=self._device)
            self._params = torch.zeros((self.trial_chunks, bounds_dim), dtype=torch.float64, device=self._device)
            self._index = 0
            self.seen_trials = set()
            self._bounds_dim = bounds_dim
        spillage = (self._index + n_completed_trials) - self._values.size(0)
        if spillage > 0:
            pad = int(math.ceil(spillage / self.trial_chunks) * self.trial_chunks)
            self._values = F.pad(self._values, (0, 0, 0, pad))
            self._params = F.pad(self._params, (0, 0, 0, pad))
        values_tensor = torch.from_numpy(values).to(self._device)
        params_tensor = torch.from_numpy(params).to(self._device)
        self._values[self._index : self._index + n_completed_trials] = values_tensor
        self._params[self._index : self._index + n_completed_trials] = params_tensor
        self._index += n_completed_trials

        return n_objectives, self._values[: self._index], self._params[: self._index]

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        assert isinstance(search_space, dict)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        n_completed_trials = len(completed_trials)
        if n_completed_trials < self._n_startup_trials:
            return {}

        trans = SearchSpaceTransform(search_space)
        n_objectives, values, params = self._preprocess_trials(trans, study, completed_trials)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=n_objectives, has_constraint=False, consider_running_trials=False
            )

        bounds = bound_to_torch(trans.bounds.tobytes(), trans.bounds.shape, str(self._device))

        with manual_seed(self._seed):
            candidates = self._candidates_func(params, values, None, bounds, None)
            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )
        return untransform(trans, candidates)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


# Minimal Bayesian optimization, based on HEBO (Huawei, MIT License).
# Replaces the `hebo` package which is broken on Python 3.12+ due to its GPy dependency.
# Original: https://github.com/huawei-noah/HEBO


class _Param:
    __slots__ = ('lb', 'name', 'ptype', 'ub')

    def __init__(self, name, ptype, lb, ub):
        self.name, self.ptype, self.lb, self.ub = name, ptype, float(lb), float(ub)

    @property
    def is_log(self):
        return self.ptype in ('pow', 'pow_int')

    @property
    def is_int(self):
        return self.ptype in ('int', 'pow_int')

    @property
    def is_discrete_after_transform(self):
        return self.ptype == 'int'

    @property
    def opt_lb(self):
        return np.log(self.lb) if self.is_log else self.lb

    @property
    def opt_ub(self):
        return np.log(self.ub) if self.is_log else self.ub

    def transform(self, x):
        return np.log(x) if self.is_log else x

    def inverse(self, x):
        v = np.exp(x) if self.is_log else x
        return np.round(v).astype(int) if self.is_int else v


class DesignSpace:
    def __init__(self):
        self.paras = {}
        self.para_names = []

    def parse(self, configs):
        self.paras = {}
        self.para_names = []
        for c in configs:
            p = _Param(c['name'], c['type'], c['lb'], c['ub'])
            self.paras[p.name] = p
            self.para_names.append(p.name)
        return self

    @property
    def num_paras(self):
        return len(self.para_names)

    def opt_lb(self, device='cpu'):
        return torch.tensor([self.paras[n].opt_lb for n in self.para_names], device=device)

    def opt_ub(self, device='cpu'):
        return torch.tensor([self.paras[n].opt_ub for n in self.para_names], device=device)

    def transform(self, df, device='cpu'):
        cols = [self.paras[n].transform(df[n].values.astype(float)) for n in self.para_names]
        data = np.column_stack(cols) if cols else np.zeros((len(df), 0))
        return torch.as_tensor(data, dtype=torch.float32, device=device)

    def inverse_transform(self, x):
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        return pd.DataFrame({n: self.paras[n].inverse(x[:, i]) for i, n in enumerate(self.para_names)})


class _GP(gpytorch.models.ExactGP):
    def __init__(self, X, y, lik, ard_dims):
        super().__init__(X, y, lik)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))


class _Surrogate:
    def __init__(self, num_epochs=100, lr=0.01, noise_lb=8e-4):
        self.num_epochs, self.lr, self.noise_lb = num_epochs, lr, noise_lb

    def fit(self, X, y):
        self._x_min, self._x_max = X.min(0).values, X.max(0).values
        span = (self._x_max - self._x_min).clamp(min=1e-8)
        Xs = 2 * (X - self._x_min) / span - 1

        self._y_mu, self._y_std = y.mean(), y.std(correction=0).clamp(min=1e-8)
        ys = ((y - self._y_mu) / self._y_std).squeeze()

        with torch.device(X.device):
            self.lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(self.noise_lb),
                noise_prior=gpytorch.priors.LogNormalPrior(np.log(0.01), 0.5),
            )
            self.gp = _GP(Xs, ys, self.lik, ard_dims=X.shape[1])
        self.lik.noise = max(1e-2, self.noise_lb)

        self.gp.train()
        self.lik.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.lik, self.gp)
        opt = torch.optim.Adam(self.gp.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            try:
                opt.zero_grad()
                loss = -mll(self.gp(Xs), ys)
                loss.backward()
                opt.step()
            except RuntimeError:
                break

        self.gp.eval()
        self.lik.eval()

    def _scale_x(self, X):
        span = (self._x_max - self._x_min).clamp(min=1e-8)
        return 2 * (X - self._x_min) / span - 1

    @torch.no_grad()
    def predict(self, X):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(float_value=1e-3):
            pred = self.gp(self._scale_x(X))
        mu = pred.mean * self._y_std + self._y_mu
        var = pred.variance * self._y_std ** 2
        return mu, var.clamp(min=1e-8)


class HEBO:
    def __init__(self, space, scramble_seed=None, rand_sample=None, device='cpu'):
        self.space = space
        self.device = torch.device(device)
        self.X = pd.DataFrame(columns=space.para_names)
        self.y = np.zeros((0, 1))
        self.rand_sample = rand_sample if rand_sample is not None else (1 + space.num_paras)
        self._seed = scramble_seed
        self.sobol = SobolEngine(space.num_paras, scramble=True, seed=scramble_seed)

    def _quasi_sample(self, n):
        lb, ub = self.space.opt_lb(), self.space.opt_ub()
        samp = self.sobol.draw(n)
        samp = samp * (ub - lb) + lb
        for i, name in enumerate(self.space.para_names):
            if self.space.paras[name].is_discrete_after_transform:
                samp[:, i] = samp[:, i].round()
        return self.space.inverse_transform(samp)

    def _transform_y(self):
        ys = self.y / max(self.y.std(), 1e-8)
        try:
            if self.y.min() <= 0:
                y = power_transform(ys, method='yeo-johnson')
            else:
                y = power_transform(ys, method='box-cox')
                if y.std() < 0.5:
                    y = power_transform(ys, method='yeo-johnson')
            if y.std() < 0.5:
                raise RuntimeError()
            return torch.as_tensor(y, dtype=torch.float32, device=self.device)
        except (RuntimeError, ValueError):
            return torch.as_tensor(self.y, dtype=torch.float32, device=self.device)

    def suggest(self, n_suggestions=1):
        if self.X.shape[0] < self.rand_sample:
            return self._quasi_sample(n_suggestions)

        try:
            return self._suggest_bo(n_suggestions)
        except (RuntimeError, ValueError):
            return self._quasi_sample(n_suggestions)

    def _suggest_bo(self, n_suggestions):
        dev = self.device
        X = self.space.transform(self.X, device=dev)
        y = self._transform_y()

        model = _Surrogate()
        model.fit(X, y)

        dim = X.shape[1]
        n_obs = max(1, self.X.shape[0])
        kappa = np.sqrt(2.0 * np.log(n_obs * dim))

        lb, ub = self.space.opt_lb(dev).float(), self.space.opt_ub(dev).float()
        n_cand = max(2000, 200 * dim)
        seed = None if self._seed is None else self._seed + n_obs
        cands = torch.empty(n_cand, dim, device=dev)
        SobolEngine(dim, scramble=True, seed=seed).draw(n_cand, out=cands)
        cands = lb + (ub - lb) * cands

        for i, name in enumerate(self.space.para_names):
            if self.space.paras[name].is_discrete_after_transform:
                cands[:, i] = cands[:, i].round()

        mu, var = model.predict(cands)
        lcb = mu - kappa * var.sqrt()

        order = lcb.argsort().cpu().numpy()
        cands_cpu = cands.cpu().numpy()
        X_cpu = X.cpu().numpy()
        selected = []
        seen = {tuple(row) for row in X_cpu}
        for idx in order:
            key = tuple(cands_cpu[idx])
            if key not in seen:
                seen.add(key)
                selected.append(idx)
            if len(selected) >= n_suggestions:
                break

        result = self.space.inverse_transform(cands_cpu[selected])

        while len(result) < n_suggestions:
            extra = self._quasi_sample(n_suggestions - len(result))
            result = pd.concat([result, extra], ignore_index=True)

        return result.head(n_suggestions)

    def observe(self, X, y):
        valid = np.isfinite(y.reshape(-1))
        new_x = X[valid]
        if len(self.X) == 0:
            self.X = new_x.reset_index(drop=True)
        else:
            self.X = pd.concat([self.X, new_x], ignore_index=True)
        self.y = np.vstack([self.y, y[valid].reshape(-1, 1)])

    @property
    def best_x(self):
        return self.X.iloc[[self.y.ravel().argmin()]]

    @property
    def best_y(self):
        return self.y.min()


def _convert_to_hebo_design_space(search_space: dict[str, BaseDistribution]) -> DesignSpace:
    if not search_space:
        raise ValueError("Empty search space.")
    design_space = []
    for name, distribution in search_space.items():
        config: dict[str, Any] = {"name": name}
        if isinstance(distribution, (FloatDistribution, IntDistribution)):
            if not distribution.log and distribution.step is not None:
                config["type"] = "int"
                n_steps = int(np.round((distribution.high - distribution.low) / distribution.step + 1))
                config["lb"] = 0
                config["ub"] = n_steps - 1
            else:
                config["lb"] = distribution.low
                config["ub"] = distribution.high
                if distribution.log:
                    config["type"] = "pow_int" if isinstance(distribution, IntDistribution) else "pow"
                else:
                    assert not isinstance(distribution, IntDistribution)
                    config["type"] = "num"
        else:
            raise NotImplementedError(f"Unsupported distribution: {distribution}")

        design_space.append(config)
    return DesignSpace().parse(design_space)


class HEBOSampler(optunahub.samplers.SimpleBaseSampler, SimpleAPIBaseSampler):
    """
    Simplified version of https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/hebo/sampler.py
    modified under the MIT License
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        *,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        if constant_liar:
            raise NotImplementedError("constant_liar is not supported by HEBOSampler.")
        self._hebo = HEBO(_convert_to_hebo_design_space(search_space), scramble_seed=self._seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._rng = np.random.default_rng(seed)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params = {}
        for name, row in self._hebo.suggest().items():
            if name not in search_space:
                continue

            dist = search_space[name]
            if isinstance(dist, (IntDistribution, FloatDistribution)) and not dist.log and dist.step is not None:
                step_index = row.iloc[0]
                params[name] = dist.low + step_index * dist.step
            else:
                params[name] = row.iloc[0]
        return params

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if values is None:
            return
        sign = 1 if study.direction == StudyDirection.MINIMIZE else -1
        values = np.array([values[0]])
        worst_value = np.nanmax(values) if study.direction == StudyDirection.MINIMIZE else np.nanmin(values)
        nan_padded_values = sign * np.where(np.isnan(values), worst_value, values)[:, np.newaxis]
        params = pd.DataFrame([trial.params])
        for name, dist in trial.distributions.items():
            if isinstance(dist, (IntDistribution, FloatDistribution)) and not dist.log and dist.step is not None:
                params[name] = (params[name] - dist.low) / dist.step

        self._hebo.observe(params, nan_padded_values)

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self.search_space

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)


class FastINGO:
    """
    Taken from https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/implicit_natural_gradient/sampler.py
    under the MIT License
    """

    def __init__(
        self,
        mean: np.ndarray,
        inv_sigma: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        last_n: int = 4096,
        loco_step_size: float = 0.1,
        device: str | None = None,
        batchnorm_decay: float = 0.99,
        score_decay: float = 0.99,
    ) -> None:
        if device is None:
            device = _use_cuda()
        n_dimension = len(mean)
        if population_size is None:
            population_size = 4 + int(np.floor(3 * np.log(n_dimension)))
            population_size = 2 * (population_size // 2)

        self.last_n = last_n
        self.batchnorm_decay = batchnorm_decay
        self.score_decay = score_decay
        self._learning_rate = learning_rate or 1.0 / np.sqrt(n_dimension)
        self._mean = torch.from_numpy(mean).to(device)
        self._sigma = torch.from_numpy(inv_sigma).to(device)
        self._lower = torch.from_numpy(lower).to(device)
        self._upper = torch.from_numpy(upper).to(device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0x123123 if seed is None else seed)
        self.loco_step_size = loco_step_size
        self._population_size = population_size
        self.device = device

        self._ys = None
        self._means = None
        self._z = None
        self._stds = None
        self._g = 0

    @torch.no_grad()
    def _concat(self, name, x):
        item = getattr(self, name, None)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        elif not isinstance(x, torch.Tensor):
            x = scalar_guard(x, self._mean).view(1)
        if item is not None:
            x = torch.cat((item, x), dim=0)[-self.last_n :]
        setattr(self, name, x)

    @property
    def dim(self) -> int:
        return self._mean.shape[0]

    @property
    def generation(self) -> int:
        return self._g

    @property
    def population_size(self) -> int:
        return self._population_size

    @torch.no_grad()
    def ask(self) -> np.ndarray:
        dimension = self._mean.shape[0]
        z = torch.randn(dimension, generator=self.generator, device=self.device, dtype=torch.float64)
        self._concat("_z", z[None])
        self._concat("_means", self._mean[None])
        self._concat("_stds", self._sigma[None])
        x = z / self._sigma.clamp(min=1e-8).sqrt() + self._mean
        return x.clamp(min=self._lower, max=self._upper).cpu().numpy()

    @torch.no_grad()
    def tell(self, y: float) -> None:
        self._g += 1
        self._concat("_ys", y)
        y = self._ys
        if y.numel() <= 2:
            return

        min_y = y.min()
        max_y = y.max()
        if torch.isclose(max_y, min_y, rtol=0.0, atol=1e-12):
            return

        if min_y <= 0:
            y = y + (1e-8 - min_y)
        y = y.clamp_min_(1e-8).log()

        ema = -torch.arange(y.size(0), device=y.device, dtype=y.dtype)
        weight = self.batchnorm_decay**ema
        weight = weight / weight.sum().clamp(min=1e-8)
        y_mean = weight @ y
        y_mean_sq = weight @ y.square()
        y_std = (y_mean_sq - y_mean.square()).clamp(min=1e-8).sqrt()
        score = (y.view(-1, 1) - y_mean) / y_std

        z = self._z
        mean_orig = self._means
        sigma_orig = self._stds
        mean_grad = score * (z / sigma_orig.clamp(min=1e-8).sqrt())
        sigma_grad = -score * z.square() * sigma_orig
        target_mean = mean_orig - mean_grad * self.loco_step_size  # MSE(current, target)
        target_sigma = sigma_orig - sigma_grad * self.loco_step_size

        weight = self.score_decay**ema
        weight = weight / weight.sum().clamp(min=1e-8)
        self._mean, self._sigma = weight @ target_mean, weight @ target_sigma


class ImplicitNaturalGradientSampler(BaseSampler):
    """
    Taken from https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/implicit_natural_gradient/sampler.py
    under the MIT License
    """

    def __init__(
        self,
        search_space: Dict[str, BaseDistribution],
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        lr: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ) -> None:
        self.search_space = search_space
        self._x0 = x0
        self._sigma0 = sigma0
        self._lr = lr
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._optimizer: Optional[FastINGO] = None
        self._seed = seed
        self._population_size = population_size

        self._param_queue: List[Dict[str, Any]] = []

    def _get_optimizer(self) -> FastINGO:
        assert self._optimizer is not None
        return self._optimizer

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._optimizer:
            self._optimizer.generator.seed()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self.search_space.items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def _check_trial_is_generation(self, trial: FrozenTrial) -> bool:
        current_gen = self._get_optimizer().generation
        trial_gen = trial.system_attrs.get("ingo", -1)
        return current_gen == trial_gen

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            self._warn_independent_sampling = False
            return {}

        trans = SearchSpaceTransform(search_space)

        if self._optimizer is None or self._optimizer.dim != len(trans.bounds):
            self._optimizer = self._init_optimizer(trans, population_size=self._population_size)
            self._param_queue.clear()

        solution_trials = [t for t in completed_trials if self._check_trial_is_generation(t)]
        for t in solution_trials:
            self._optimizer.tell(-t.value if study.direction == StudyDirection.MAXIMIZE else t.value)

        study._storage.set_trial_system_attr(trial._trial_id, "ingo", self._get_optimizer().generation)
        return trans.untransform(self._optimizer.ask())

    def _init_optimizer(
        self,
        trans: SearchSpaceTransform,
        population_size: Optional[int] = None,
    ) -> FastINGO:
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._x0 is None:
            mean = lower_bounds + (upper_bounds - lower_bounds) / 2
        else:
            mean = trans.transform(self._x0)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0
        inv_sigma = 1 / sigma0 * np.ones(n_dimension)

        return FastINGO(
            mean=mean,
            inv_sigma=inv_sigma,
            lower=lower_bounds,
            upper=upper_bounds,
            seed=self._seed,
            population_size=population_size,
            learning_rate=self._lr,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


class ThreadLocalSampler(threading.local):
    sampler: BaseSampler | None = None


def init_cmaes(study, seed, trials, search_space):
    trials = copy.deepcopy(trials)
    trials.sort(key=lambda trial: trial.datetime_complete)
    return CmaEsSampler(seed=seed, source_trials=trials, lr_adapt=True)


def init_hebo(study, seed, trials, search_space):
    sampler = HEBOSampler(search_space=search_space, seed=seed)
    for trial in trials:
        sampler.after_trial(study, trial, TrialState.COMPLETE, trial.values)
    return sampler


def _use_cuda():
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_botorch(study, seed, trials, search_space):
    return BoTorchSampler(
        search_space=search_space, seed=seed, device=_use_cuda()
    )  # will automatically pull in latest data


def init_nsgaii(study, seed, trials, search_space):
    module = optunahub.load_module(
        "samplers/nsgaii_with_initial_trials",
    )
    return module.NSGAIIwITSampler(seed=seed)


def init_random(study, seed, trials, search_space):
    return optuna.samplers.RandomSampler(seed=seed)


def init_ingo(study, seed, trials, search_space):
    return ImplicitNaturalGradientSampler(search_space=search_space, seed=seed)


class AutoSampler(BaseSampler):
    def __init__(
        self,
        samplers: Iterable[Tuple[int, Callable]] | None = None,
        search_space: Optional[dict[str, BaseDistribution]] = None,
        *,
        seed: int | None = None,
        constraints_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        if constraints_func is not None:
            raise NotImplementedError("constraints_func is not supported by AutoSampler.")
        if samplers is None:
            if search_space is None:
                raise ValueError("AutoSampler requires a search_space when using the default sampler schedule.")
            samplers = ((0, init_hebo), (100, init_nsgaii))
        self.sampler_indices = np.sort(np.array([x[0] for x in samplers], dtype=np.int32))
        self.samplers = [x[1] for x in sorted(samplers, key=lambda x: x[0])]
        self.search_space = {} if search_space is None else dict(search_space)
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self._thread_local_sampler = ThreadLocalSampler()
        self._constraints_func = constraints_func
        self._completed_trials = 0
        self._current_index = -1

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_thread_local_sampler"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._thread_local_sampler = ThreadLocalSampler()

    @property
    def _sampler(self) -> BaseSampler:
        if self._thread_local_sampler.sampler is None:
            seed_for_random_sampler = self._rng.rng.randint(_MAXINT32)
            self._sampler = RandomSampler(seed=seed_for_random_sampler)

        return self._thread_local_sampler.sampler

    @_sampler.setter
    def _sampler(self, sampler: BaseSampler) -> None:
        self._thread_local_sampler.sampler = sampler

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._sampler.reseed_rng()

    def _update_sampler(self, study: Study):
        if len(study.directions) > 1:
            raise ValueError("Multi-objective optimization is not supported.")

        if isinstance(self._sampler, CmaEsSampler):
            return

        complete_trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        self._completed_trials = max(self._completed_trials, len(complete_trials))
        new_index = (self._completed_trials >= self.sampler_indices).sum() - 1
        if new_index == self._current_index or new_index < 0:
            return
        self._current_index = new_index
        self._sampler = self.samplers[new_index](
            study, self._rng.rng.randint(_MAXINT32), complete_trials, self.search_space
        )

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler.sample_relative(study, trial, search_space or self.search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # NOTE(nabenabe): Sampler must be updated in this method. If, for example, it is updated in
        # infer_relative_search_space, the sampler for before_trial and that for sample_relative,
        # after_trial might be different, meaning that the sampling routine could be incompatible.
        if len(study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)) != 0:
            self._update_sampler(study)

        sampler_name = self._sampler.__class__.__name__
        study._storage.set_trial_system_attr(trial._trial_id, _SAMPLER_KEY, sampler_name)
        self._sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if state not in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
            raise ValueError(f"Unsupported trial state: {state}.")
        self._sampler.after_trial(study, trial, state, values)
