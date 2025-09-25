import copy
import functools
import gc
import inspect
import os
import random
import sys
import time
import warnings
from typing import Callable, Optional

import numpy as np
import optuna
import torch
from torch import nn
from torch._dynamo import config

import heavyball.utils
from heavyball import chainable as C
from heavyball.helpers import AutoSampler
from heavyball.utils import PrecondInitError

config.cache_size_limit = 2**16

np.warnings = warnings

base_args = {
    "betas": (0.9, 0.999),
    "precondition_frequency": 16,
    "merge_dims": True,
    "warmup_steps": 100,
    "max_precond_dim": 2**16,
    "beta": 0.9,
    "max_size_triangular": 2**16,
    "split": False,
    "precond_grad_accum": False,
    "momentum_into_precond_update": True,
    "eps": 1e-8,
    "weight_decay": 0,
    "precond_update_power_iterations": 16,
    "dampening": 2**-24,
    "preconditioner_update_probability": 1.0,
    "precond_init_scale": 1.0,
    "store_triu_as_line": False,
    "update_clipping": None,
    "delayed": True,
}


def get_optim(optim: str | C.BaseOpt, params, **kwargs) -> C.BaseOpt:
    args = {**base_args, **kwargs}
    if isinstance(optim, str):
        optim = getattr(heavyball, optim)
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


class FailureCounter:
    def __init__(self, mapping, broadcast: int = 1):
        self.mapping = mapping
        self.broadcast = broadcast
        max_consecutive_failures, minimal_improvement = zip(*mapping.items())
        self.max_consecutive_failures = torch.tensor(max_consecutive_failures, dtype=torch.float64, device="cuda")
        self.minimal_improvement = torch.tensor(minimal_improvement, dtype=torch.float64, device="cuda")
        self.consecutive_failures = torch.zeros(len(minimal_improvement), dtype=torch.int64, device="cuda").repeat(
            broadcast
        )

    def compare(self, inp, other):
        old_state = inp.reshape(1, -1, 1)  # vertical
        new_state = other.reshape(1, 1, -1)  # horizontal

        return new_state * (1 - self.minimal_improvement.reshape(-1, 1, 1)) < old_state

    def new(self):
        return FailureCounter(self.mapping, self.broadcast)

    def __call__(self, comparison, failure_scale: float = 1):
        failed = torch.any(comparison, axis=tuple(range(1, comparison.ndim)))
        self.consecutive_failures.copy_(torch.where(failed, self.consecutive_failures + 1, 0))
        mask = self.consecutive_failures >= (self.max_consecutive_failures.view(-1, 1) * failure_scale).flatten()
        return torch.any(mask)


class Validator:
    ema_index: int = 4
    warmup: int = 256
    ema_patience: float = 2
    ema_start: int = 0

    def __init__(self, ema_mapping, global_mapping, steps, emas: int = 20):
        self.step = 0
        self.emas = emas

        self.ema_states = torch.zeros((self.emas,), dtype=torch.float64, device="cuda")
        es = self.ema_start + 1
        self.update_factor = 2.0 ** (-torch.arange(es, 20 + es, dtype=torch.float64, device="cuda"))
        self.ema_failures = FailureCounter(ema_mapping)
        self.triu_indices = torch.triu_indices(self.emas, self.emas, offset=1)

        self.global_min_loss = torch.tensor((float("inf"),) * steps, dtype=torch.float64, device="cuda")
        self.global_min_failures = FailureCounter({1: 0}, steps)

        self.global_avg_loss = torch.zeros_like(self.global_min_loss)
        self.global_avg_step = torch.zeros_like(self.global_avg_loss)

        self.weighting = torch.arange(1, 1 + self.global_min_loss.size(0), device="cuda")
        self.weighting = self.weighting.clamp(min=self.warmup).view(1, -1)
        self.weighting = functools.reduce(torch.minimum, [self.weighting**p * f for f, p in global_mapping.items()])

        self.seen_until = np.zeros((), dtype=np.int64)  # seen_until has to be shared
        self.global_avg_failures = FailureCounter({1: 0}, steps)

    def new(self):
        new = copy.copy(self)
        new.ema_failures = new.ema_failures.new()
        new.global_min_failures = new.global_min_failures.new()
        new.global_avg_failures = new.global_avg_failures.new()
        new.ema_states = torch.zeros_like(new.ema_states)
        new.step = 0
        return new

    def _update_ema(self, loss):
        self.step += 1
        np.copyto(self.seen_until, np.maximum(self.seen_until, self.step - 1))

        uf = 1 - heavyball.utils.beta_debias(1 - self.update_factor, self.step)
        self.ema_states += uf * (loss - self.ema_states)

    def _global_min(self):
        loss = self.ema_states[self.ema_index]
        comparison = self.global_min_failures.compare(loss, self.global_min_loss)
        global_failed = self.global_min_failures(comparison.view(-1, 1), self.weighting)
        loss_slice = self.global_min_loss[self.step - 1 :]
        loss_slice.copy_(torch.where(torch.logical_and(loss < loss_slice, torch.isfinite(loss)), loss, loss_slice))
        return global_failed

    def _global_avg(self):
        loss = self.ema_states[self.ema_index]

        self.global_avg_step[self.step - 1] += 1
        self.global_avg_loss[self.step - 1].lerp_(loss, 1 / self.global_avg_step[self.step - 1])

        comparison = self.global_avg_failures.compare(loss, self.global_avg_loss).view(-1, 1)
        comparison[self.seen_until - 1 :].fill_(False)
        return self.global_avg_failures(comparison, self.weighting)

    def _local_convergence(self):
        comparison = self.ema_failures.compare(self.ema_states, self.ema_states)
        comparison = comparison[tuple([slice(None), *self.triu_indices])]
        return self.ema_failures(comparison, self.ema_patience)

    def __call__(self, loss):
        self._update_ema(loss)

        outputs = [self._global_min(), self._global_avg(), self._local_convergence()]
        if self.step < self.warmup:
            return torch.zeros_like(outputs[0])
        return functools.reduce(torch.logical_or, outputs)


class Stop(Exception):
    pass


class SkipConfig(ValueError):
    pass


class WinConditionMet(ValueError):
    pass


class Plotter(nn.Module):
    def __init__(
        self,
        objective_fn,
        x_limits=(-5, 5),
        y_limits=(-5, 5),
        resolution=300,
        transform=None,
        inverse_transform=None,
        should_normalize: bool = True,
    ):
        super().__init__()
        self.should_normalize = should_normalize
        self.objective = objective_fn
        self.initial = objective_fn.param.data.clone()
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.resolution = resolution
        self.transform = transform if transform else lambda x: x
        self.inverse_transform = inverse_transform if inverse_transform else lambda x: x

        self.param = objective_fn.param

        with torch.no_grad():
            x = torch.linspace(x_limits[0], x_limits[1], resolution)
            y = torch.linspace(y_limits[0], y_limits[1], resolution)
            self.X, self.Y = torch.meshgrid(x, y, indexing="ij")
            Z = torch.zeros_like(self.X)
            for i in range(resolution):
                for j in range(resolution):
                    objective_fn.param.data[:] = torch.tensor([self.X[i, j].item(), self.Y[i, j].item()], device="cuda")
                    Z[i, j] = self.transform(objective_fn())
            objective_fn.param.data[:] = self.initial
        self.Z = Z

        self.trajectory = [self.initial.detach().cpu().numpy()]

    def forward(self, *args):
        value = self.objective(*args)
        with torch.no_grad():
            self.trajectory.append(self.param.cpu().detach().numpy())
        return self.transform(value)

    def plot(self, title=None, save_path=None):
        """Create contour plot with optimization trajectory.

        Args:
            title: Optional title for the plot
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        z = self.Z
        if self.should_normalize:
            z = z - z.min()
            z = z / z.max()
            z = z + 1e-8
        plt.contourf(self.X.numpy(), self.Y.numpy(), z.log().numpy(), levels=1000)

        # Plot trajectory
        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], "r.-", label="Optimization path")
        plt.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

        plt.colorbar(label=f"Log({'Normalized' * self.should_normalize}ObjectiveValue)")
        plt.xlabel("x")
        plt.ylabel("y")
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.close()


class MultiPlotter:
    def __init__(self, plotters):
        if not plotters:
            raise ValueError("MultiPlotter requires at least one Plotter instance.")
        self.plotters = plotters
        self._reference = plotters[0][1]

    def plot(self, title=None, save_path=None):
        import matplotlib.colors
        import matplotlib.pyplot as plt

        ref = self._reference
        fig, ax = plt.subplots(figsize=(10, 8))

        z = ref.Z
        if ref.should_normalize:
            z = z - z.min()
            z = z / z.max()
            z = z + 1e-8

        contour = ax.contourf(ref.X.numpy(), ref.Y.numpy(), z.log().numpy(), levels=1000)

        color_cycle = plt.rcParams.get("axes.prop_cycle")
        if color_cycle is not None:
            colors = list(color_cycle.by_key().get("color", []))
        else:
            colors = []
        if not colors:
            colors = list(matplotlib.colors.TABLEAU_COLORS.values())

        for idx, (label, plotter) in enumerate(self.plotters):
            color = colors[idx % len(colors)]
            trajectory = np.array(plotter.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], ".-", color=color, label=label)
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color=color,
                edgecolors="black",
                linewidths=0.8,
                s=70,
                marker="o",
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=color,
                linewidths=1.6,
                s=90,
                marker="x",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True)

        label = "Log(ObjectiveValue)"
        if ref.should_normalize:
            label = "Log(NormalizedObjectiveValue)"
        fig.colorbar(contour, ax=ax, label=label)

        if save_path:
            fig.savefig(save_path)
        plt.close(fig)


class Objective:
    def __init__(
        self,
        failure_threshold,
        model,
        opt: str,
        steps,
        group,
        data,
        loss_fn,
        win_condition,
        weight_decay,
        warmup_trials,
        eval_callback,
        ema_index: int = 0,
        **kwargs,
    ):
        self.failure_threshold = failure_threshold
        self.model = model.cuda()
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.RNNBase):
                mod.flatten_parameters()
        self.opt = opt
        self.steps = steps
        self.group = group
        self.data = data
        self.loss_fn = loss_fn
        self._win_condition = win_condition
        self.weight_decay = weight_decay
        self.warmup_trials = int(warmup_trials)
        self.kwargs = kwargs
        self.ema_index = ema_index
        self.eval_callback = eval_callback
        self.hyperparam_log_history: list[np.ndarray] = []

        # up to 32768 consecutive times can the new loss be (1 - 1e-7)x larger than the preceding loss
        self.validator = Validator(
            {  # We can't check for improvement, as it's not guaranteed - "1% in 1k steps" may not happen
                1024: -1e-4,  # 1.1x over 1024 steps
                256: -1e-3,  # 1.25x over 256 steps
                64: -0.01,  # 2x over 64 steps
                16: -0.1,  # 5.4x over 16 steps
                8: -0.33,  # 24x over 8 steps
                4: -0.5,  # 16x over 4 steps
                2: -0.75,  # 16x over 2 steps
                1: -0.99,  # 100x over 1 step
            },
            {100: 2, 1600: 1},  # step_count^2 * 1 | step_count ^ 1.5 * 4 | step_count ^ 1 * 16
            steps,
        )
        self.m = None
        self.attempt = 0
        self.best_loss = None
        self.best_at = 0
        self.avg = None
        self.use_cudnn = True
        self.set_precond_init_scale = False
        self.end_time = int(os.environ.get("HEAVYBALL_BENCHMARK_TIMEOUT", 3600 * 4)) + time.time()
        self._last_loss = None

    def win_condition(self, loss=None):
        if loss is not None:
            self._last_loss = loss
        return self._win_condition(self.m, self._last_loss)[0]

    def _inner(
        self,
        params,
        model_override=None,
        *,
        allow_validator_stop: bool = True,
        allow_win_condition: bool = True,
    ):
        input_kwargs = locals()
        input_kwargs.pop("self")
        params = tuple(float(p) for p in params)
        params_dict = self._format_params(params)
        model_template = self.model if model_override is None else model_override
        self.m = self._clone_model(model_template)
        o = get_optim(
            self.opt,
            self.m.parameters(),
            **params_dict,
            weight_decay=self.weight_decay,
            **self.kwargs,
        )
        torch_hist = torch.empty(self.group, dtype=torch.float64, device="cuda")
        validator = self.validator.new()

        for i in range(self.steps // self.group):
            if hasattr(o, "train"):
                o.train()

            if not hasattr(self, "test_accuracies"):
                self.callback_results = []

            if self.eval_callback is not None:
                test_accuracy = self.eval_callback(self.m)
                self.callback_results.append(test_accuracy)

            for j in range(self.group):
                inp, tgt = self.data()

                def _closure():
                    loss = self.m() if inp is None else self.m(inp)
                    if self.loss_fn is not None:
                        loss = self.loss_fn(loss, tgt)
                    loss.backward()
                    return loss

                try:
                    loss = o.step(_closure)
                except PrecondInitError:
                    self.set_precond_init_scale = True
                    return self._inner(**input_kwargs)

                o.zero_grad()

                with torch.no_grad():
                    torch_hist[j] = loss.detach()
            if hasattr(o, "eval"):
                o.eval()
            with torch.no_grad():
                for loss in torch_hist:
                    loss_cpu = loss.item()
                    win_condition_reached = allow_win_condition and self.win_condition(loss_cpu)
                    if not np.isfinite(loss_cpu) or win_condition_reached:
                        return validator.ema_states.min().item(), self.m, loss_cpu
                    validator_triggered = validator(loss).item()
                    if allow_validator_stop and validator_triggered:
                        return validator.ema_states.min().item(), self.m, loss_cpu
        return validator.ema_states.min().item(), self.m, loss.item()

    def objective(self, params):
        self.attempt += 1
        target, _, loss = self._inner(params)
        params_log10 = np.log10(np.array(params, dtype=np.float64))
        self.hyperparam_log_history.append(params_log10)
        if self.best_loss is None or loss < self.best_loss or not np.isfinite(self.best_loss):
            self.best_loss = loss
            self.best_at = self.attempt
            self.avg = np.log(np.array(params))
        if self.best_at * 8 < self.attempt and self.attempt - self.best_at > self.warmup_trials:  # no improvements
            raise Stop
        if time.time() > self.end_time:  # timeout
            raise Stop
        return target

    def _format_params(self, params: tuple[float, float, float, float]) -> dict[str, float]:
        params_dict = {
            "lr": params[0],
            "betas": (1 - params[1], 1 - params[2]),
            "beta": 1 - params[1],
            "shampoo_beta": 1 - params[3],
            "sam_step_size": params[3],
            "eps": 1e-8,
            "precond_lr": params[3],
        }
        if self.set_precond_init_scale:
            params_dict["precond_init_scale"] = 0.1
        return params_dict

    def _clone_model(self, model_template):
        cloned = copy.deepcopy(model_template)
        if hasattr(cloned, "cuda"):
            cloned = cloned.cuda()
        if hasattr(cloned, "trajectory"):
            initial = getattr(cloned, "initial", None)
            param_tensor = getattr(cloned, "param", None)
            if initial is not None:
                cloned.trajectory = [initial.detach().cpu().numpy()]
            elif param_tensor is not None:
                cloned.trajectory = [param_tensor.detach().cpu().numpy()]
            else:
                cloned.trajectory = []
        return cloned

    def _replay_params(self, params, model_override=None, *, callback_period: int | None = None):
        params = tuple(float(p) for p in params)
        params_dict = self._format_params(params)
        model_template = self.model if model_override is None else model_override
        replay_model = self._clone_model(model_template)
        optimizer = get_optim(
            self.opt,
            replay_model.parameters(),
            **params_dict,
            weight_decay=self.weight_decay,
            **self.kwargs,
        )

        total_steps = self.steps
        period = callback_period if callback_period is not None else max(1, self.group)

        for step_idx in range(total_steps):
            if hasattr(optimizer, "train"):
                optimizer.train()

            if not hasattr(self, "callback_results"):
                self.callback_results = []
            if self.eval_callback is not None and step_idx % period == 0:
                test_accuracy = self.eval_callback(replay_model)
                self.callback_results.append(test_accuracy)

            inp, tgt = self.data()

            def _closure():
                loss = replay_model() if inp is None else replay_model(inp)
                if self.loss_fn is not None:
                    loss = self.loss_fn(loss, tgt)
                loss.backward()
                return loss

            try:
                optimizer.step(_closure)
            except PrecondInitError:
                self.set_precond_init_scale = True
                params_dict = self._format_params(params)
                optimizer = get_optim(
                    self.opt,
                    replay_model.parameters(),
                    **params_dict,
                    weight_decay=self.weight_decay,
                    **self.kwargs,
                )
                optimizer.step(_closure)

            optimizer.zero_grad()
            if hasattr(optimizer, "eval"):
                optimizer.eval()

        return replay_model

    def evaluate_params(self, params, model_override=None):
        params = tuple(float(p) for p in params)
        original_group = self.group
        try:
            self.group = 1
            return self._replay_params(
                params,
                model_override=model_override,
                callback_period=original_group,
            )
        finally:
            self.group = original_group

    def compute_hparam_log_ema(self, beta: float = 0.9):
        if not 0 < beta < 1:
            raise ValueError("EMA beta must be in (0, 1).")
        if not self.hyperparam_log_history:
            return None
        ema = self.hyperparam_log_history[0].copy()
        for log_vals in self.hyperparam_log_history[1:]:
            ema = beta * ema + (1 - beta) * log_vals
        return ema

    def get_best(self):
        original_group = self.group
        try:
            self.group = 1
            return self._replay_params(tuple(np.exp(self.avg)), callback_period=original_group)
        finally:
            self.group = original_group


def loss_win_condition(target):
    def win(_model, loss: float):
        return loss <= target, {}

    return win


def param_norm_win_condition(target, offset):
    target = torch.full((), target, device="cuda")

    def win(model, loss):
        with torch.no_grad():
            norm = model.param.add(offset).square().mean().sqrt()
            return (norm < target).item(), {}

    return win


def param0_win_condition(target):
    target = torch.full((), target, device="cuda")

    def win(model, loss):
        with torch.no_grad():
            return (model.param[0] < target).item(), {}

    return win


def set_seed(seed: int = 0x1239121):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanup():
    gc.enable()
    gc.collect()
    gc.disable()
    with torch.cuda.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _none_data():
    return None, None


def trial(
    model,
    data,
    loss_fn,
    win_condition,
    steps,
    opt,
    weight_decay,
    trials=10,
    failure_threshold=3,
    group=256,
    return_best: bool = False,
    warmup_trial_pct: float = 1,
    random_trials: int = 10,
    eval_callback: Optional[Callable] = None,  # evaluate_test_accuracy(dataloader)
    ema_beta: float = 0.9,
):
    if data is None:
        data = _none_data
    group = min(group, steps)
    heavyball.utils.set_torch(einsum_strategy="heavyball")

    if not 0 < ema_beta < 1:
        raise ValueError("ema_beta must be in (0, 1).")

    opt_list = opt if isinstance(opt, list) else [opt]
    if not opt_list:
        raise ValueError("At least one optimizer must be provided.")

    plotter_template = model if isinstance(model, Plotter) else None
    if plotter_template is not None:
        base_model_template = copy.deepcopy(plotter_template.objective)
    else:
        base_model_template = copy.deepcopy(model)

    heavyball.utils._ignore_warning("logei_candidates_func is experimental")
    heavyball.utils._ignore_warning("BoTorchSampler is experimental")
    heavyball.utils._ignore_warning("It will be set to log2(param_count). This requires `params` to be of type list.")
    heavyball.utils._ignore_warning("rank was set to")
    heavyball.utils._ignore_warning(
        "The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior."
    )

    results = []
    baseline_cautioning = getattr(heavyball.utils, "_compilable_cautioning", None)

    for opt_entry in opt_list:
        cleanup()
        set_seed()

        kwargs = {"caution": False, "mars": False}
        restore_caution = False

        if isinstance(opt_entry, str):
            opt_label = opt_entry
            opt_name = opt_entry
            if opt_name.startswith("cautious-"):
                opt_name = opt_name[len("cautious-") :]
                kwargs["caution"] = True
            if opt_name.startswith("unscaled_cautious-"):
                opt_name = opt_name[len("unscaled_cautious-") :]
                heavyball.utils.disable_caution_scaling()
                kwargs["caution"] = True
                restore_caution = True
            if opt_name.startswith("mars-"):
                opt_name = opt_name[len("mars-") :]
                kwargs["mars"] = True
            if opt_name.startswith("unscaled-"):
                opt_name = opt_name[len("unscaled-") :]
                kwargs["unscaled"] = True
            if opt_name.startswith("adaptive-"):
                opt_name = opt_name[len("adaptive-") :]
                kwargs["adaptive"] = True
            if opt_name.startswith("ortho-"):
                opt_name = opt_name[len("ortho-") :]
                kwargs["ortho_method"] = "newtonschulz-graft"
            opt_callable = getattr(heavyball, opt_name)
        else:
            opt_callable = opt_entry
            opt_label = getattr(opt_callable, "__name__", repr(opt_callable))

        did_win = False

        def _win_condition(*args):
            nonlocal did_win
            win_state, out = win_condition(*args)
            did_win |= win_state
            return did_win, out

        base_model = copy.deepcopy(base_model_template)
        obj = Objective(
            failure_threshold,
            base_model,
            opt_callable,
            steps,
            group,
            data,
            loss_fn,
            _win_condition,
            weight_decay,
            max(trials * warmup_trial_pct, 1 + random_trials),
            eval_callback,
            **kwargs,
        )

        torch.cuda.synchronize()
        stdout, sys.stdout = sys.stdout, sys.stderr

        set_seed()
        start_time = time.time()
        winning_params = {
            "lr": float("nan"),
            "1mbeta1": float("nan"),
            "1mbeta2": float("nan"),
            "1mshampoo_beta": float("nan"),
        }
        prev_best = float("inf")
        callback_results = None
        try:
            sampler = AutoSampler(
                seed=0x123125,
                search_space={
                    "lr": optuna.distributions.FloatDistribution(1e-7, 100, log=True),
                    "1mbeta1": optuna.distributions.FloatDistribution(1e-5, 1, log=True),
                    "1mbeta2": optuna.distributions.FloatDistribution(1e-7, 1, log=True),
                    "1mshampoo_beta": optuna.distributions.FloatDistribution(1e-7, 1, log=True),
                },
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _optuna_objective(trial):
                set_seed(0x12312)
                lr = trial.suggest_float("lr", 1e-7, 100, log=True)
                one_minus_beta1 = trial.suggest_float("1mbeta1", 1e-5, 1, log=True)
                one_minus_beta2 = trial.suggest_float("1mbeta2", 1e-7, 1, log=True)
                one_minus_shampoo_beta = trial.suggest_float("1mshampoo_beta", 1e-7, 1, log=True)
                out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
                if out < prev_best:
                    winning_params.update({
                        "lr": lr,
                        "1mbeta1": one_minus_beta1,
                        "1mbeta2": one_minus_beta2,
                        "1mshampoo_beta": one_minus_shampoo_beta,
                    })
                if obj.win_condition(out):
                    raise WinConditionMet
                return out

            set_seed()
            try:
                study.optimize(_optuna_objective, n_trials=trials)
            except WinConditionMet:
                pass
            callback_results = getattr(obj, "callback_results", None)
        finally:
            sys.stdout = stdout
            if restore_caution and baseline_cautioning is not None:
                heavyball.utils._compilable_cautioning = baseline_cautioning

        torch.cuda.synchronize()
        end_time = time.time()

        ema_logs = obj.compute_hparam_log_ema(beta=ema_beta)
        ema_params = np.power(10.0, ema_logs) if ema_logs is not None else None

        def _fmt(value, precision=5):
            return f"{value:.{precision}f}" if np.isfinite(value) else "nan"

        ema_msg = ""
        if ema_params is not None:
            ema_lr, ema_1mb1, ema_1mb2, ema_1msh = ema_params
            ema_msg = (
                f" | EMAβ={ema_beta:.2f}(lr={ema_lr:.5f}, betas=({1 - ema_1mb1:.3f}, {1 - ema_1mb2:.4f}), "
                f"shampoo_beta={1 - ema_1msh:.3f})"
            )

        callback_msg = ""
        if callback_results:
            callback_msg = f" | Callback Results: {callback_results}"

        print(
            f"[{opt_label}] Took: {end_time - start_time} | Attempt: {obj.attempt} | "
            f"{opt_label}(lr={_fmt(winning_params['lr'])}, betas=({1 - winning_params['1mbeta1']:.3f}, {1 - winning_params['1mbeta2']:.4f}), "
            f"shampoo_beta={1 - winning_params['1mshampoo_beta']:.3f}){ema_msg} | Best Loss: {obj.best_loss}{callback_msg}"
        )

        if return_best:
            if ema_params is None:
                model_result = obj.get_best()
            else:
                model_override = plotter_template if plotter_template is not None else None
                model_result = obj.evaluate_params(ema_params, model_override=model_override)
            if isinstance(model_result, Plotter):
                print(f"[{opt_label}] Replay trajectory length: {len(model_result.trajectory)}")
            results.append({
                "label": opt_label,
                "model": model_result,
                "best_loss": getattr(obj, "best_loss", None),
                "callback_results": callback_results,
            })

    if return_best:
        labelled_models = [(res["label"], res["model"]) for res in results if res["model"] is not None]
        if not labelled_models:
            return None
        if len(labelled_models) == 1:
            return labelled_models[0][1]
        if all(isinstance(model, Plotter) for _, model in labelled_models):
            return MultiPlotter(labelled_models)
        return labelled_models[0][1]


def evaluate_test_accuracy(test_loader):
    def _fn(model):
        # Save the current training state
        was_training = model.training

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                output = model(data)

                # Handle different output shapes
                if output.dim() > 2:  # Sequence modeling: [batch, seq_len, vocab_size]
                    pred = output.argmax(dim=-1)  # [batch, seq_len]
                    pred_flat = pred.view(-1)
                    target_flat = target.view(-1)
                    correct += pred_flat.eq(target_flat).sum().item()
                    total += target_flat.numel()
                else:  # Regular classification: [batch, num_classes]
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.numel()

        # Restore the original training state
        model.train(was_training)
        return correct / total

    return _fn
