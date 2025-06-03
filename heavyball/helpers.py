from __future__ import annotations

import functools
import math
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy
import numpy as np
import optuna
import optunahub
import pandas as pd
import torch
from botorch.utils.sampling import manual_seed
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna_integration.botorch import (
    ehvi_candidates_func,
    logei_candidates_func,
    qehvi_candidates_func,
    qei_candidates_func,
    qparego_candidates_func,
)
from torch import Tensor
from torch.nn import functional as F

from heavyball.utils import scalar_guard

_MAXINT32 = (1 << 31) - 1
_SAMPLER_KEY = "auto:sampler"


class SimpleAPIBaseSampler(BaseSampler):
    def __init__(
        self,
        search_space: dict[str, BaseDistribution] = None,
    ):
        self.search_space = search_space

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


@functools.lru_cache(maxsize=None)
def bound_to_torch(bound: bytes, shape: tuple, device: str):
    bound = np.frombuffer(bound, dtype=np.float64).reshape(shape)
    bound = np.transpose(bound, (1, 0))
    return torch.from_numpy(bound).to(torch.device(device))


@functools.lru_cache(maxsize=None)
def nextafter(x: Union[float, int], y: Union[float, int]) -> Union[float, int]:
    return numpy.nextafter(x, y)


def _untransform_numerical_param_torch(
    trans_param: Union[float, int, Tensor],
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
def untransform(self: _SearchSpaceTransform, trans_params: Tensor) -> dict[str, Any]:
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
        search_space: dict[str, BaseDistribution] = None,
        *,
        candidates_func: None = None,
        constraints_func: None = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        trial_chunks: int = 128,
    ):
        assert constraints_func is None
        assert candidates_func is None
        assert consider_running_trials is False
        assert independent_sampler is None
        self._candidates_func = None
        self._independent_sampler = RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self.trial_chunks = trial_chunks

        self._study_id: int | None = None
        self.search_space = search_space
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device or torch.device("cpu")
        self.seen_trials = set()
        self._values = None
        self._params = None
        self._index = 0

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self.search_space

    @torch.no_grad()
    def _preprocess_trials(
        self, trans: _SearchSpaceTransform, study: Study, trials: list[FrozenTrial]
    ) -> Tuple[int, Tensor, Tensor]:
        new_trials = []
        for trial in trials:
            tid: int = trial._trial_id
            if tid not in self.seen_trials:
                self.seen_trials.add(tid)
                new_trials.append(trial)
        trials = new_trials

        n_objectives = len(study.directions)
        if not new_trials:
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

        if self._values is None:
            self._values = torch.zeros((self.trial_chunks, n_objectives), dtype=torch.float64, device=self._device)
            self._params = torch.zeros(
                (self.trial_chunks, trans.bounds.shape[0]), dtype=torch.float64, device=self._device
            )
        spillage = (self._index + n_completed_trials) - self._values.size(0)
        if spillage > 0:
            pad = int(math.ceil(spillage / self.trial_chunks) * self.trial_chunks)
            self._values = F.pad(self._values, (0, 0, 0, pad))
            self._params = F.pad(self._params, (0, 0, 0, pad))
        self._values[self._index : self._index + n_completed_trials] = torch.from_numpy(values)
        self._params[self._index : self._index + n_completed_trials] = torch.from_numpy(params)
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

        trans = _SearchSpaceTransform(search_space)
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
        assert constant_liar is False
        assert independent_sampler is None
        self._hebo = HEBO(_convert_to_hebo_design_space(search_space), scramble_seed=self._seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
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
        if self._hebo is None or values is None:
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
        device="cuda",
        batchnorm_decay: float = 0.99,
        score_decay: float = 0.99,
    ) -> None:
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

        y = y + torch.where(y.min() <= 0, 1e-8 - y.min(), 0)
        y = y.log()

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
            self._optimizer._rng.seed()

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

        trans = _SearchSpaceTransform(search_space)

        if self._optimizer is None:
            self._optimizer = self._init_optimizer(trans, population_size=self._population_size)

        if self._optimizer.dim != len(trans.bounds):
            self._warn_independent_sampling = False
            return {}

        solution_trials = [t for t in completed_trials if self._check_trial_is_generation(t)]
        for t in solution_trials:
            self._optimizer.tell(-t.value if study.direction == StudyDirection.MAXIMIZE else t.value)

        study._storage.set_trial_system_attr(trial._trial_id, "ingo", self._get_optimizer().generation)
        return trans.untransform(self._optimizer.ask())

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
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
    trials.sort(key=lambda trial: trial.datetime_complete)
    return CmaEsSampler(seed=seed, source_trials=trials, lr_adapt=True)


def init_hebo(study, seed, trials, search_space):
    sampler = HEBOSampler(search_space=search_space, seed=seed)
    for trial in trials:
        sampler.after_trial(study, trial, TrialState.COMPLETE, trial.values)
    return sampler


def init_botorch(study, seed, trials, search_space):
    return BoTorchSampler(search_space=search_space, seed=seed, device="cuda")  # will automatically pull in latest data


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
        search_space: dict[str, BaseDistribution] = None,
        *,
        seed: int | None = None,
        constraints_func: None = None,
    ) -> None:
        assert constraints_func is None
        if samplers is None:
            samplers = ((0, init_hebo), (100, init_nsgaii))
        self.sampler_indices = np.sort(np.array([x[0] for x in samplers], dtype=np.int32))
        self.samplers = [x[1] for x in sorted(samplers, key=lambda x: x[0])]
        self.search_space = search_space
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
        if new_index == self._current_index:
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
        return self._sampler.sample_relative(study, trial, self.search_space)

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
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._sampler.after_trial(study, trial, state, values)
