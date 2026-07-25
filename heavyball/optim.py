"""PyTorch optimizer facades over HeavyBall's slab-native recipes."""

import copy
import inspect
import math
import os
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from . import (
    adamc,
    adamuon,
    adamw,
    ademamix,
    adopt,
    aurora,
    cautious_adamw,
    heavy_kl_shampoo_adamw,
    heavy_kl_soap_adamw,
    hyperball_adamw,
    kl_shampoo_adamw,
    kl_soap_adamw,
    kron_adamw,
    laprop,
    laprop_ortho,
    lather_adamw,
    lion,
    mars_adamw,
    msam_laprop,
    muon,
    muon_laprop,
    nadam,
    normuon,
    oblique,
    ortho_laprop,
    orthograd_adamw,
    polargrad,
    psgd,
    psgd_lra_adamw,
    psgd_nfactor_adamw,
    psgd_pro_adamw,
    qsgd_adamw,
    rmsprop,
    scion,
    sf_adamw,
    sgd,
    shampoo_adamw,
    sign_laprop,
    signsgd,
    soap_adamw,
    soap_ademamix_adamw,
    soap_nadam_adamw,
    solp_adamw,
    spel,
    suds_adamw,
    truegrad_adam,
    truegrad_laprop,
    truegrad_nadam,
    truegrad_rmsprop,
    unscaled_adamw,
    whiten_adamw,
)
from .core import FSDP2Binding, Recipe, Route, build, fsdp2_bindings, fsdp2_recipe_scope_supported


_DCP_FORMAT = 1


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


def _dtensor_schema(value: Tensor) -> tuple[tuple[int, ...], str, tuple[tuple[str, int | None], ...]]:
    from torch.distributed.tensor import DTensor

    placements = ()
    if isinstance(value, DTensor):
        placements = tuple(
            (type(placement).__name__, getattr(placement, "dim", None))
            for placement in value.placements
        )
    return tuple(value.shape), str(value.dtype), placements


def _recipe_defaults(recipe: Recipe | Route, overrides: Mapping[str, object]) -> dict[str, object]:
    if isinstance(recipe, Recipe):
        defaults = dict(recipe.defaults)
    else:
        defaults = _recipe_defaults(recipe.otherwise, {})
        defaults.update(_recipe_defaults(recipe.then, {}))
    defaults.update(overrides)
    return defaults


_ENGINE_HYPERPARAMETERS: tuple[tuple[str, str, object], ...] = (
    ("storage_dtype", "torch.dtype | str | None", None),
    ("ecc", "int | str | None", None),
    ("clip_global_norm", "float | None", None),
    ("param_keys", "Sequence[str] | None", None),
)

_ENGINE_HYPERPARAMETER_NAMES = frozenset(name for name, _, _ in _ENGINE_HYPERPARAMETERS)
_OPTIMIZER_WIDE_HYPERPARAMETERS = (
    "clip_global_norm",
    "storage_dtype",
    "ecc",
    "preconditioner_update_probability",
)
_ENGINE_HYPERPARAMETER_DEFAULTS = {
    name: default for name, _, default in _ENGINE_HYPERPARAMETERS
}


def _facade_hyperparameters(recipe: Recipe | Route) -> tuple[tuple[str, object, object], ...]:
    recipe_hyperparameters = (
        (name, (float | None) if default is None else type(default), default)
        for name, default in _recipe_defaults(recipe, {}).items()
    )
    return (*recipe_hyperparameters, *_ENGINE_HYPERPARAMETERS)


def _same_hyper_value(left: object, right: object) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            tensor = left if isinstance(left, Tensor) else right
            other = right if isinstance(left, Tensor) else left
            return tensor.numel() == 1 and bool(tensor.detach().cpu() == other)
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    try:
        equal = left == right
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(equal) if isinstance(equal, bool) else False


@dataclass(frozen=True)
class _TensorHyperSnapshot:
    value: Tensor
    version: int


def _snapshot_hyper(value: object) -> object:
    return _TensorHyperSnapshot(value, value._version) if isinstance(value, Tensor) else value


def _canonical_optimizer_wide_value(name: str, value: object) -> object:
    if name == "storage_dtype" and isinstance(value, str):
        return getattr(torch, value.removeprefix("torch."), value)
    if name == "ecc":
        return {"bf16+8": 8, "bf16+16": 16}.get(value, value)
    return value


def _effective_group_hypers(
    param_groups: list[dict[str, object]], defaults: Mapping[str, object]
) -> list[dict[str, object]]:
    effective = []
    for param_group in param_groups:
        values = dict(defaults)
        values.update({name: value for name, value in param_group.items() if name != "params"})
        effective.append(values)
    return effective


def _optimizer_wide_hypers(effective_groups: list[dict[str, object]]) -> dict[str, object]:
    resolved = {}
    for name in _OPTIMIZER_WIDE_HYPERPARAMETERS:
        values = [
            (
                group_id,
                group[name] if name in group else _ENGINE_HYPERPARAMETER_DEFAULTS[name],
            )
            for group_id, group in enumerate(effective_groups)
            if name in group or name in _ENGINE_HYPERPARAMETER_DEFAULTS
        ]
        if not values:
            continue
        first_group_id, first_value = values[0]
        first_canonical = _canonical_optimizer_wide_value(name, first_value)
        for group_id, value in values[1:]:
            if not _same_hyper_value(first_canonical, _canonical_optimizer_wide_value(name, value)):
                raise ValueError(
                    f"parameter groups specify conflicting optimizer-wide values for {name!r}: "
                    f"group {first_group_id} has {first_value!r}, group {group_id} has {value!r}"
                )
        resolved[name] = first_value
    return resolved


def _validate_param_group_options(
    recipe: Recipe | Route, param_groups: list[dict[str, object]], constructor_hyper: Mapping[str, object]
) -> None:
    recipe_names = set(_recipe_defaults(recipe, {}))
    allowed = recipe_names | _ENGINE_HYPERPARAMETER_NAMES
    unknown = set(constructor_hyper) - allowed
    unknown.update(
        name
        for param_group in param_groups
        for name in param_group
        if name not in allowed and name != "params"
    )
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise ValueError(f"unknown hyperparameter override(s): {names}")
    if any("param_keys" in group for group in param_groups):
        raise ValueError("'param_keys' must be supplied once to the optimizer constructor, not per group")


def _real_hyper_value(
    optimizer_name: str,
    name: str,
    value: object,
    *,
    allow_none: bool = False,
    allow_callable: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if callable(value) and allow_callable:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.dtype is torch.bool or value.is_complex():
            raise TypeError(
                f"{optimizer_name} hyperparameter {name!r}={value!r} must be a real scalar"
            )
        scalar = value.detach().item()
    else:
        scalar = value
    if not isinstance(scalar, Real) or isinstance(scalar, bool):
        raise TypeError(
            f"{optimizer_name} hyperparameter {name!r}={value!r} must be a real scalar"
        )
    try:
        numeric = float(scalar)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{optimizer_name} hyperparameter {name!r}={value!r} must be finite"
        ) from error
    if not math.isfinite(numeric):
        raise ValueError(
            f"{optimizer_name} hyperparameter {name!r}={value!r} must be finite"
        )
    return numeric


def _validate_hyperparameter_domains(
    optimizer_name: str,
    recipe: Recipe | Route,
    param_groups: list[dict[str, object]],
    defaults: Mapping[str, object],
    *,
    allow_callable_cadence: bool = True,
    allow_inherited_max_lr: bool = False,
) -> None:
    recipe_names = set(_recipe_defaults(recipe, {}))
    for effective in _effective_group_hypers(param_groups, defaults):
        for name in recipe_names:
            if name not in effective:
                continue
            value = effective[name]
            numeric = _real_hyper_value(
                optimizer_name,
                name,
                value,
                allow_none=name == "max_lr" and allow_inherited_max_lr,
                allow_callable=(
                    name == "preconditioner_update_probability"
                    and allow_callable_cadence
                ),
            )
            if numeric is None:
                continue
            if name == "eps" and numeric <= 0:
                domain = "greater than 0"
            elif name in {"beta1", "beta2"} and not 0 <= numeric < 1:
                domain = "in [0, 1)"
            elif name in {"lr", "weight_decay", "max_lr"} and numeric < 0:
                domain = "greater than or equal to 0"
            else:
                continue
            raise ValueError(
                f"{optimizer_name} hyperparameter {name!r}={value!r} must be {domain}"
            )

        clip_global_norm = effective.get("clip_global_norm")
        if clip_global_norm is not None:
            _real_hyper_value(
                optimizer_name,
                "clip_global_norm",
                clip_global_norm,
            )


def _validate_engine_hyperparameter_types(
    optimizer_name: str,
    param_groups: list[dict[str, object]],
) -> None:
    for group in param_groups:
        storage_dtype = group.get("storage_dtype")
        resolved_storage_dtype = storage_dtype
        if isinstance(storage_dtype, str):
            resolved_storage_dtype = getattr(
                torch,
                storage_dtype.removeprefix("torch."),
                storage_dtype,
            )
        ecc = group.get("ecc")
        if ecc is not None and (
            type(ecc) not in (int, str)
            or ecc not in (8, 16, "bf16+8", "bf16+16")
        ):
            raise ValueError(
                f"{optimizer_name} hyperparameter 'ecc'={ecc!r}: "
                'ecc must be None, 8, 16, "bf16+8", or "bf16+16"'
            )
        if (
            ecc is not None
            and resolved_storage_dtype is not None
            and resolved_storage_dtype is not torch.bfloat16
        ):
            raise ValueError(
                f"{optimizer_name} hyperparameters 'ecc'={ecc!r}, "
                f"'storage_dtype'={storage_dtype!r}: ecc requires storage_dtype "
                "to be None or torch.bfloat16"
            )
        if resolved_storage_dtype is not None and resolved_storage_dtype is not torch.bfloat16:
            raise ValueError(
                f"{optimizer_name} hyperparameter 'storage_dtype'={storage_dtype!r}: "
                "storage_dtype must be None or torch.bfloat16"
            )
        param_keys = group.get("param_keys")
        if param_keys is not None and (
            isinstance(param_keys, (str, bytes))
            or not isinstance(param_keys, Sequence)
            or any(not isinstance(key, str) for key in param_keys)
        ):
            raise TypeError(
                f"{optimizer_name} hyperparameter 'param_keys'={param_keys!r} "
                "must be a sequence of strings or None"
            )


def _recipe_has_observations(recipe: Recipe | Route) -> bool:
    if isinstance(recipe, Recipe):
        return bool(recipe.observations)
    return _recipe_has_observations(recipe.then) or _recipe_has_observations(recipe.otherwise)


def _build_group_aware_engine(
    param_groups: list[dict[str, object]],
    recipe: Recipe | Route,
    defaults: Mapping[str, object],
    *,
    bindings: Mapping[int, FSDP2Binding] | None,
    rng_seed: int,
):
    effective_groups = _effective_group_hypers(param_groups, defaults)
    optimizer_wide = _optimizer_wide_hypers(effective_groups)
    recipe_names = set(_recipe_defaults(recipe, {}))
    dynamic_group_hypers = {
        group_id: {
            name: value
            for name, value in effective.items()
            if name in recipe_names and name != "preconditioner_update_probability"
        }
        for group_id, effective in enumerate(effective_groups)
    }
    flat_params = [param for group in param_groups for param in group["params"]]
    param_group_ids = [
        group_id
        for group_id, group in enumerate(param_groups)
        for _ in group["params"]
    ]
    build_hyper = {}
    if "preconditioner_update_probability" in optimizer_wide:
        build_hyper["preconditioner_update_probability"] = optimizer_wide[
            "preconditioner_update_probability"
        ]
    engine = build(
        flat_params,
        recipe,
        param_keys=defaults.get("param_keys"),
        storage_dtype=optimizer_wide.get("storage_dtype"),
        ecc=optimizer_wide.get("ecc"),
        clip_global_norm=optimizer_wide.get("clip_global_norm"),
        bindings=bindings,
        _rng_seed=rng_seed,
        _leaf_indices=range(len(flat_params)),
        param_group_ids=param_group_ids,
        param_group_hypers=dynamic_group_hypers,
        **build_hyper,
    )
    return engine, len(flat_params)


def _materialize_params(
    params: Iterable[Tensor],
) -> list[dict[str, object]]:
    supplied = list(params)
    if not supplied or not any(isinstance(value, Mapping) for value in supplied):
        return [{"params": supplied}]
    if not all(isinstance(value, Mapping) for value in supplied):
        raise TypeError("params must be an iterable of parameters or parameter groups")
    groups = []
    for supplied_group in supplied:
        group = dict(supplied_group)
        if "params" not in group:
            raise ValueError("parameter group must contain 'params'")
        group_params = group["params"]
        group["params"] = [group_params] if isinstance(group_params, Tensor) else list(group_params)
        groups.append(group)
    return groups


class HeavyBallOptimizer(torch.optim.Optimizer):
    """A ``torch.optim.Optimizer``-style facade over HeavyBall Engines.

    Construction slab-backs the supplied parameters, so construct this optimizer
    before wrapping the model with DDP or ``torch.compile``. For a supported
    componentwise recipe, call ``Optimizer.fsdp2(model, ...)`` after ``fully_shard(model)``.

    The interface deliberately differs from ``torch.optim``. Parameters and gradients are slab-bound
    into persistent buffers, so call ``optimizer.zero_grad(set_to_none=False)`` (the default), never
    reassign ``p.data``/``p.grad`` (write them in place), and do not change parameter storage after
    construction (e.g. ``model.to(...)``). DDP must use ``gradient_as_bucket_view=False``. By default
    every optimized parameter advances on every step -- weight decay, moments, and the step clock
    update even when it received no gradient -- but callers can pass ``observed=`` to mark inactive
    parameters. This mask controls only HeavyBall state advancement; it does not change DDP's
    autograd-graph unused-parameter detection or collectives. Conditional DDP graphs still require
    ``find_unused_parameters=True``, and conditional FSDP2 graphs require
    ``set_reduce_scatter_unused_params(True)``. Defaults differ from similarly named ``torch.optim``
    optimizers, and hyperparameters are keyword-only.

    Low-precision optimizer state (opt-in per optimizer; compute always promotes to
    fp32, so accuracy is preserved and only storage shrinks):

    - ``storage_dtype=torch.bfloat16``: state in bfloat16. Half the state memory and,
      from the lower memory bandwidth, measured faster than ``torch.optim.AdamW(fused=True)``
      (up to ~24% on a 50M-parameter model; see ``benchmarks/precision_speed.py``). The
      trajectory stays within ~1e-4 of fp32.
    - ``ecc=8``: bfloat16 plus an int8 holding the top 8 of the 16 discarded mantissa bits
      (the last 8 are stochastically rounded), 3 bytes per element (0.75x fp32), ~256x tighter
      than plain bfloat16 (near fp16 precision).
    - ``ecc=16``: an int16 holding all 16 discarded mantissa bits, 4 bytes per element -- the
      narrow/correction pair reconstructs the fp32 value *bit-exactly* at the same memory as
      fp32, while the stored narrow stays bfloat16. Use plain fp32 unless you need that bf16 view.
    - Default (``storage_dtype=None``, ``ecc=None``): fp32 state.

    Boolean flags and integer counters always keep their natural dtype. Every facade's
    hyperparameters are explicit in its signature (``inspect.signature`` or IDE autocomplete);
    beyond the usual ``lr`` / ``beta1`` / ``beta2`` / ``eps`` / ``weight_decay`` the
    optimizer-specific knobs are:

    - ``preconditioner_update_probability``: per-step probability of rebuilding a preconditioner
      basis (SOAP, Shampoo, PSGD, KL, whitening).
    - ``shampoo_beta``: EMA decay for the SOAP/Shampoo Gram (preconditioner) factors.
    - ``max_precond_dim``: axes larger than this use a diagonal rather than a full preconditioner
      (SOAP, KL).
    - ``precond_lr``, ``lower_bound_beta``, ``dampening``, ``max_size_triangular``: PSGD's
      preconditioner learning rate, its running spectral-lower-bound EMA (the step-size floor), the
      damping added to the whitening probe, and the largest axis kept as a full triangular factor
      (larger axes use a diagonal factor).
    - ``rank``: the low-rank approximation rank for PSGD-LRA's ``(I + UV^T) diag(d)`` factors.
    - ``sam_step_size``: MSAM's sharpness-aware perturbation radius.
    - ``weight_lr_power``, ``r``: schedule-free's learning-rate weighting exponent and age warmup power.
    - ``alpha``, ``beta3`` (with ``alpha_warmup`` / ``beta3_warmup``): AdEMAMix's slow-EMA mixing
      weight and decay, and their warmups.
    - ``momentum_decay``: the decay rate of NAdam's scheduled Nesterov-momentum coefficient.
    - ``mars_gamma``: MARS's variance-reduction strength.
    - ``init_factor``: the initial value of KL's eigenvalue estimates.
    - ``scale``: Scion's linear-minimization-oracle scale.
    - ``max_lr``: AdamC's reference learning rate; its weight decay is scaled by ``lr / max_lr``.
    - ``caution`` / ``cautious_weight_decay``: enable the cautious update mask and its weight-decay
      coefficient.
    """

    recipe: Recipe | Route | None = None

    @classmethod
    def fsdp2(cls, model, **hyper) -> "HeavyBallOptimizer":
        """Build over the persistent local shards of a fully_shard'd model."""

        if "recipe" in hyper:
            raise ValueError("fsdp2() does not accept a recipe override; use an allowed optimizer facade")
        recipe = cls.recipe
        schedule_free = recipe is sf_adamw
        if _recipe_has_observations(recipe):
            raise ValueError(
                f"{cls.__name__}.fsdp2(): observation-bearing recipes are not supported "
                "under FSDP2 yet"
            )
        if not fsdp2_recipe_scope_supported(recipe):
            raise ValueError(
                f"{cls.__name__}.fsdp2() requires every callable to be shard-separable or whole-scoped"
            )
        if hyper.get("clip_global_norm") is not None:
            raise ValueError(
                "fsdp2() does not support clip_global_norm because it requires a cross-rank full-model reduction"
            )
        if schedule_free:
            caution = hyper.get("caution", sf_adamw.defaults["caution"])
            cautious_weight_decay = hyper.get(
                "cautious_weight_decay", sf_adamw.defaults["cautious_weight_decay"]
            )
            if not cls._fsdp2_disabled(caution) or not cls._fsdp2_disabled(cautious_weight_decay):
                raise ValueError(
                    f"{cls.__name__}.fsdp2() requires caution=0 and cautious_weight_decay=0 because cautious "
                    "normalization reduces over the full logical parameter."
                )
        resolved_bindings = fsdp2_bindings(model)
        binding_by_param_id = {id(binding.param): binding for binding in resolved_bindings}
        if hyper.get("param_keys") is None:
            names_by_param_id = {id(param): name for name, param in model.named_parameters()}
            try:
                hyper["param_keys"] = tuple(names_by_param_id[id(binding.param)] for binding in resolved_bindings)
            except KeyError:
                FSDP2Binding._incompatible("a fully-sharded parameter is absent from model.named_parameters()")
        return cls(
            (binding.param for binding in resolved_bindings),
            _fsdp2_bindings=binding_by_param_id,
            **hyper,
        )

    @staticmethod
    def _fsdp2_disabled(value) -> bool:
        if isinstance(value, Tensor):
            return value.numel() == 1 and bool((value.detach() == 0).item())
        try:
            return bool(value == 0)
        except (TypeError, ValueError):
            return False

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if cls.recipe is not None:
            cls.__signature__ = inspect.Signature(
                [
                    inspect.Parameter(
                        "params", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Iterable[Tensor]
                    ),
                    *(
                        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
                        for name, annotation, default in _facade_hyperparameters(cls.recipe)
                    ),
                ],
                return_annotation=None,
            )

    def __init__(
        self, params: Iterable[Tensor], recipe: Recipe | Route | None = None,
        _fsdp2_bindings: Mapping[int, FSDP2Binding] | None = None,
        **hyper,
    ) -> None:
        if recipe is None:
            recipe = type(self).recipe
        elif not isinstance(recipe, (Recipe, Route)):
            raise TypeError(
                f"{type(self).__name__}() second positional argument is the internal 'recipe' "
                f"(expected Recipe/Route, got {type(recipe).__name__}). Pass hyperparameters such as "
                f"the learning rate by keyword: {type(self).__name__}(params, lr=...)."
            )
        torch_params = _materialize_params(params)
        _validate_param_group_options(recipe, torch_params, hyper)
        defaults = _recipe_defaults(recipe, hyper)
        _validate_hyperparameter_domains(
            type(self).__name__,
            recipe,
            torch_params,
            defaults,
            allow_inherited_max_lr=True,
        )
        _validate_engine_hyperparameter_types(
            type(self).__name__,
            _effective_group_hypers(torch_params, defaults),
        )
        self._recipe = recipe
        self._fsdp2_mode = _fsdp2_bindings is not None
        self._rng_seed = torch.initial_seed()
        engine, leaf_count = _build_group_aware_engine(
            torch_params,
            recipe,
            defaults,
            bindings=_fsdp2_bindings,
            rng_seed=self._rng_seed,
        )
        self._engine = engine
        self._engines = [engine]
        self._engine_by_param_id = {id(param): engine for param in engine.params}
        self._rng_seed = engine._rng_seed
        self._rng_next_leaf_index = leaf_count
        self._hyper_names = frozenset(
            name for namespace in engine._hyper_locations.values() for name in vars(namespace)
        )
        self._initializing = True
        try:
            super().__init__(torch_params, defaults=defaults)
        finally:
            self._initializing = False
        self._resolve_inherited_hypers()
        self._synced_hypers = [
            {
                name: _snapshot_hyper(param_group[name])
                for name in self._hyper_names
                if name in param_group
            }
            for param_group in self.param_groups
        ]

    def _resolve_inherited_hypers(self) -> None:
        for group in self.param_groups:
            if "max_lr" in group and group["max_lr"] is None:
                lr = group["lr"]
                # Cloning prevents in-place LR schedules from aliasing AdamC's inherited maximum.
                group["max_lr"] = lr.detach().clone() if isinstance(lr, Tensor) else lr

    def add_param_group(self, param_group: dict[str, object]) -> None:
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
            return
        if self._fsdp2_mode:
            raise ValueError("fsdp2() optimizers cannot add parameters after their FSDP storage is bound")
        materialized_group = _materialize_params([param_group])[0]
        _validate_param_group_options(self._recipe, [materialized_group], {})
        _validate_hyperparameter_domains(
            type(self).__name__,
            self._recipe,
            [materialized_group],
            self.defaults,
            allow_inherited_max_lr=True,
        )
        _validate_engine_hyperparameter_types(
            type(self).__name__,
            _effective_group_hypers([materialized_group], self.defaults),
        )
        if self.defaults.get("param_keys") is not None:
            raise ValueError("cannot add a parameter group when param_keys fixed the Engine parameter schema")
        candidate_groups = [dict(group) for group in self.param_groups]
        candidate_groups.append(materialized_group)
        old_engine = self._engine
        old_state = old_engine.state_dict()
        old_values = {id(param): param.detach().clone() for param in old_engine.params}
        old_grads = {
            id(param): None if param.grad is None else param.grad.detach().clone()
            for param in old_engine.params
        }
        parameter_snapshots = tuple(
            (param, param.data, param.grad, dict(vars(param)))
            for group in candidate_groups
            for param in group["params"]
            if isinstance(param, Tensor)
        )
        original_param_groups = list(self.param_groups)
        try:
            engine, leaf_count = _build_group_aware_engine(
                candidate_groups,
                self._recipe,
                self.defaults,
                bindings=None,
                rng_seed=self._rng_seed,
            )

            migrated = engine.state_dict()
            for name in ("age", "state", "param_init_pending"):
                migrated[name].update(old_state[name])
            if "corrections" in old_state:
                migrated["corrections"].update(old_state["corrections"])
            if len(engine._steps_by_device) == 1:
                migrated["step"] = old_state["step"]
            else:
                if set(old_state["step"]) == {"global"}:
                    old_device = str(next(iter(old_engine._steps_by_device)))
                    migrated["step"][old_device] = old_state["step"]["global"]
                else:
                    migrated["step"].update(old_state["step"])
            migrated["train_mode"] = old_state["train_mode"]
            if "cadence" in old_state:
                migrated["cadence"] = old_state["cadence"]
            engine.load_state_dict(migrated)
            with torch.no_grad():
                for param in old_engine.params:
                    param.copy_(old_values[id(param)])
                    if old_grads[id(param)] is not None:
                        param.grad.copy_(old_grads[id(param)])

            super().add_param_group(materialized_group)
            self._resolve_inherited_hypers()
            hyper_names = frozenset(
                name for namespace in engine._hyper_locations.values() for name in vars(namespace)
            )
            synced_hypers = [
                {
                    name: _snapshot_hyper(group[name])
                    for name in hyper_names
                    if name in group
                }
                for group in self.param_groups
            ]
            engines = [engine]
            engine_by_param_id = {id(param): engine for param in engine.params}
        except BaseException:
            self.param_groups[:] = original_param_groups
            with torch.no_grad():
                for param, data, grad, attributes in parameter_snapshots:
                    param.data = data
                    param.grad = grad
                    vars(param).clear()
                    vars(param).update(attributes)
            raise

        self._engine = engine
        self._engines = engines
        self._engine_by_param_id = engine_by_param_id
        self._rng_seed = engine._rng_seed
        self._rng_next_leaf_index = leaf_count
        self._hyper_names = hyper_names
        self._synced_hypers = synced_hypers

    @torch.no_grad()
    def _sync_hypers_from_groups(self, *, force: bool) -> None:
        changed = []
        for group_id, (param_group, synced) in enumerate(
            zip(self.param_groups, self._synced_hypers, strict=True)
        ):
            for name in self._hyper_names:
                if name not in param_group:
                    continue
                value = param_group[name]
                snapshot = synced.get(name)
                if isinstance(value, Tensor):
                    unchanged = (
                        isinstance(snapshot, _TensorHyperSnapshot)
                        and snapshot.value is value
                        and snapshot.version == value._version
                    )
                else:
                    unchanged = (
                        not isinstance(snapshot, _TensorHyperSnapshot)
                        and _same_hyper_value(value, snapshot)
                    )
                if force or not unchanged:
                    changed.append((group_id, name, value, synced))
        if not changed:
            return

        _validate_hyperparameter_domains(
            type(self).__name__,
            self._recipe,
            self.param_groups,
            self.defaults,
        )
        _validate_engine_hyperparameter_types(type(self).__name__, self.param_groups)
        if self._fsdp2_mode and self._recipe is sf_adamw:
            for _, name, value, _ in changed:
                if name in {"caution", "cautious_weight_decay"} and not self._fsdp2_disabled(value):
                    raise ValueError(
                        f"{type(self).__name__}.fsdp2() requires caution=0 and "
                        "cautious_weight_decay=0 because cautious normalization reduces over "
                        "the full logical parameter."
                    )
        for group_id, name, value, synced in changed:
            self._engine.set_hyper(name, value, group_id=group_id)
            synced[name] = _snapshot_hyper(value)

    def step(
        self,
        closure: Callable[[], Tensor] | None = None,
        observed: Sequence[bool] | Mapping[Tensor, bool] | None = None,
    ) -> Tensor | None:
        """Run one optimizer step and return the closure loss, if any.

        ``observed`` accepts one host bool per trainable parameter, either as a sequence in
        construction order or as a mapping containing every parameter. Marking a parameter
        ``False`` prevents its HeavyBall parameter update, moments, decay, and clock from advancing.
        It does not suppress DDP communication or mark a parameter unused in the autograd graph:
        conditional DDP graphs still need ``find_unused_parameters=True``. Conditional FSDP2 graphs
        need ``set_reduce_scatter_unused_params(True)``. Omitting the mask preserves HeavyBall's
        default that every parameter is observed.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_hypers_from_groups(force=False)
        try:
            self._engine.step(observed=observed)
        except ValueError as error:
            message = str(error)
            if "gradient for parameter" in message and "no longer slab-bound" in message:
                raise ValueError(
                    f"{message} When using DistributedDataParallel, set "
                    "gradient_as_bucket_view=False."
                ) from error
            raise
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False) -> None:
        for engine in self._engines:
            engine.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def produce(self, param: Tensor, name: str, value: Tensor) -> None:
        engine = self._engine_by_param_id.get(id(param))
        if engine is None:
            raise ValueError("parameter is not owned by this Engine")
        engine.produce(param, name, value)

    def train(self, mode: bool = True) -> "HeavyBallOptimizer":
        for engine in self._engines:
            engine.train(mode)
        return self

    def eval(self) -> "HeavyBallOptimizer":
        for engine in self._engines:
            engine.eval()
        return self

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["engines"] = [engine.state_dict() for engine in self._engines]
        return state_dict

    def _dcp_state_dict(self, *, staging: bool) -> tuple[dict, list[tuple[Tensor, Tensor]]]:
        """Expose the logical FSDP2 state to DCP, optionally through empty staging tensors."""

        from torch.distributed.tensor import DTensor, Shard

        if not self._fsdp2_mode:
            raise ValueError("dcp_save()/dcp_load() require an optimizer constructed with fsdp2()")

        copies: list[tuple[Tensor, Tensor]] = []

        def stage(target: Tensor) -> Tensor:
            if not staging:
                return target
            if isinstance(target, DTensor):
                staged = DTensor.from_local(
                    torch.empty_like(target.to_local()),
                    target.device_mesh,
                    target.placements,
                    run_check=False,
                    shape=target.shape,
                    stride=target.stride(),
                )
            else:
                staged = torch.empty_like(target)
            copies.append((target, staged))
            return staged

        def slab_dtensor(group, slab: Tensor) -> DTensor:
            if isinstance(slab, DTensor):
                return slab
            reference = group.params[0]
            if not isinstance(reference, DTensor):
                raise ValueError("DCP checkpointing requires FSDP2 DTensor parameters")
            local_rows = reference.to_local().shape[0]
            if (
                slab.ndim != reference.ndim + 1
                or slab.shape[0] != len(group.params)
                or slab.shape[1] < local_rows
                or tuple(slab.shape[2:]) != tuple(reference.shape[1:])
            ):
                raise ValueError("FSDP2 shard-separable state does not match its parameter row shard")
            global_shape = (slab.shape[0], *reference.shape)
            placements = tuple(
                Shard(placement.dim + 1) if isinstance(placement, Shard) else placement
                for placement in reference.placements
            )
            return DTensor.from_local(
                slab.narrow(1, 0, local_rows),
                reference.device_mesh,
                placements,
                run_check=False,
                shape=global_shape,
                stride=_contiguous_stride(global_shape),
            )

        optimizer_groups = tuple(
            {
                "param_count": len(param_group["params"]),
                "values": {name: value for name, value in param_group.items() if name != "params"},
            }
            for param_group in self.param_groups
        )
        metadata = {"format": _DCP_FORMAT, "param_groups": optimizer_groups, "engines": []}
        state_dict = {"metadata": metadata, "engines": {}}

        for engine_index, engine in enumerate(self._engines):
            engine_key = str(engine_index)
            model = {
                engine._param_keys[id(param)]: stage(param)
                for param in engine.params
            }
            groups = {}
            group_schemas = []
            for group_index, group in enumerate(engine.groups):
                group_key = str(group_index)

                def slots(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
                    return {name: stage(slab_dtensor(group, value)) for name, value in values.items()}

                states = {
                    str(transform_index): slots(values)
                    for transform_index, values in enumerate(group.states)
                }
                corrections = {
                    str(transform_index): slots(values)
                    for transform_index, values in enumerate(group.state_corrections)
                }
                commit = slots(group.commit_state)
                commit_corrections = slots(group.commit_corrections)
                group_state = {
                    "age": stage(group.age),
                    "leaf_indices": stage(group.leaf_indices),
                    "states": states,
                    "corrections": corrections,
                    "commit": commit,
                    "commit_corrections": commit_corrections,
                }
                groups[group_key] = group_state
                group_schemas.append({
                    "param_keys": tuple(engine._param_keys[id(param)] for param in group.params),
                    "age": _dtensor_schema(group_state["age"]),
                    "leaf_indices": _dtensor_schema(group_state["leaf_indices"]),
                    "states": tuple(
                        tuple((name, _dtensor_schema(value)) for name, value in values.items())
                        for values in states.values()
                    ),
                    "corrections": tuple(
                        tuple((name, _dtensor_schema(value)) for name, value in values.items())
                        for values in corrections.values()
                    ),
                    "commit": tuple((name, _dtensor_schema(value)) for name, value in commit.items()),
                    "commit_corrections": tuple(
                        (name, _dtensor_schema(value)) for name, value in commit_corrections.items()
                    ),
                })

            steps = {str(index): stage(value) for index, value in enumerate(engine._steps)}
            rng_seeds = {str(index): stage(value) for index, value in enumerate(engine._rng_seeds.values())}
            hyper_items = tuple(engine._hyper_locations.items())
            hypers = {
                str(index): {name: stage(value) for name, value in vars(namespace).items()}
                for index, (_, namespace) in enumerate(hyper_items)
            }
            cadence = None
            if engine._cadence is not None:
                probability = engine._cadence.probability
                if callable(probability):
                    raise ValueError("cannot checkpoint a callable cadence probability schedule")
                cadence = {
                    "probability": probability,
                    "step": engine._cadence.step,
                    "cumulative": engine._cadence.cumulative,
                    "compensation": engine._cadence.compensation,
                }
            rng_leaf_indices = {
                engine._param_keys[id(param)]: int(group.leaf_indices[index].detach().cpu())
                for group in engine.groups
                for index, param in enumerate(group.params)
            }
            ecc = None if engine.ecc is None else (8 if engine.ecc is torch.int8 else 16)
            engine_metadata = {
                "fingerprint": engine._recipe_fingerprints(),
                "param_keys": tuple(engine.param_keys),
                "model_schema": tuple((name, _dtensor_schema(value)) for name, value in model.items()),
                "group_schemas": tuple(group_schemas),
                "step_schema": tuple(_dtensor_schema(value) for value in steps.values()),
                "rng_schema": tuple(_dtensor_schema(value) for value in rng_seeds.values()),
                "hyper_keys": tuple(key for key, _ in hyper_items),
                "hyper_schema": tuple(
                    tuple((name, _dtensor_schema(value)) for name, value in values.items())
                    for values in hypers.values()
                ),
                "ecc": ecc,
                "train_mode": engine._train_mode,
                "cadence": cadence,
                "rng": {"seed": engine._rng_seed, "leaf_indices": rng_leaf_indices},
                "param_init_pending": dict(engine._deferred_param_init_pending),
            }
            metadata["engines"].append(engine_metadata)
            state_dict["engines"][engine_key] = {
                "model": model,
                "groups": groups,
                "steps": steps,
                "rng_seeds": rng_seeds,
                "hypers": hypers,
            }
        metadata["engines"] = tuple(metadata["engines"])
        return state_dict, copies

    def dcp_save(self, checkpoint_dir: str | os.PathLike[str]) -> None:
        """Save FSDP2 model and optimizer state with distributed-checkpoint resharding metadata."""

        import torch.distributed.checkpoint as dcp

        state_dict, _ = self._dcp_state_dict(staging=False)
        dcp.save(state_dict, checkpoint_id=checkpoint_dir)

    @torch.no_grad()
    def dcp_load(self, checkpoint_dir: str | os.PathLike[str], *, trusted: bool = False) -> None:
        """Load and reshard an FSDP2 model and optimizer checkpoint onto the current world."""

        if not trusted:
            import warnings
            warnings.warn(
                "DCP checkpoint loading deserializes untrusted data and may execute arbitrary code. "
                "Only load checkpoints from sources you trust. Pass trusted=True to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        import torch.distributed.checkpoint as dcp
        from torch.distributed.tensor import DTensor

        state_dict, copies = self._dcp_state_dict(staging=True)
        expected_metadata = copy.deepcopy(state_dict["metadata"])
        dcp.load(state_dict, checkpoint_id=checkpoint_dir)
        saved_metadata = state_dict.get("metadata")
        self._validate_dcp_metadata(saved_metadata, expected_metadata, state_dict)
        for target, staged in copies:
            if isinstance(target, DTensor):
                target.to_local().copy_(staged.to_local())
            else:
                target.copy_(staged)
        self._restore_dcp_host_state(saved_metadata)

    def _validate_dcp_metadata(self, saved: object, expected: dict, state_dict: dict) -> None:
        if not isinstance(saved, Mapping) or saved.get("format") != _DCP_FORMAT:
            raise ValueError("expected a HeavyBall format-1 DCP checkpoint")
        saved_engines = saved.get("engines")
        expected_engines = expected["engines"]
        if not isinstance(saved_engines, (list, tuple)) or len(saved_engines) != len(expected_engines):
            raise ValueError("DCP checkpoint Engine state does not match this optimizer")
        for engine_index, (saved_engine, expected_engine) in enumerate(
            zip(saved_engines, expected_engines, strict=True)
        ):
            if not isinstance(saved_engine, Mapping):
                raise ValueError("DCP checkpoint Engine state does not match this optimizer")
            saved_static = dict(saved_engine)
            expected_static = dict(expected_engine)
            for name in ("train_mode", "cadence", "rng", "param_init_pending"):
                saved_static.pop(name, None)
                expected_static.pop(name, None)
            if saved_static != expected_static:
                raise ValueError("DCP checkpoint schema or recipe does not match this optimizer")
            if not isinstance(saved_engine.get("train_mode"), bool):
                raise ValueError("DCP checkpoint train mode must be a bool")
            self._validate_dcp_cadence(saved_engine.get("cadence"), self._engines[engine_index])
            saved_param_init_pending = saved_engine.get("param_init_pending")
            expected_param_init_pending = expected_engine["param_init_pending"]
            if saved_param_init_pending is not None and (
                not isinstance(saved_param_init_pending, Mapping)
                or set(saved_param_init_pending) != set(expected_param_init_pending)
                or any(not isinstance(value, bool) for value in saved_param_init_pending.values())
            ):
                raise ValueError(
                    "DCP checkpoint deferred parameter initialization does not match this optimizer"
                )
            self._engines[engine_index]._validate_deferred_param_init_pending(
                {
                    param_key: False
                    for param_key in expected_param_init_pending
                }
                if saved_param_init_pending is None
                else saved_param_init_pending
            )
            rng = saved_engine.get("rng")
            if not isinstance(rng, Mapping) or set(rng) != {"seed", "leaf_indices"}:
                raise ValueError("DCP checkpoint RNG state does not match this optimizer")
            seed = rng["seed"]
            leaf_indices = rng["leaf_indices"]
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < (1 << 64)
                or not isinstance(leaf_indices, Mapping)
                or set(leaf_indices) != set(self._engines[engine_index]._param_locations)
            ):
                raise ValueError("DCP checkpoint RNG state does not match this optimizer")
            leaf_values = tuple(leaf_indices.values())
            if any(
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << 63)
                for value in leaf_values
            ) or len(set(leaf_values)) != len(leaf_values):
                raise ValueError("DCP checkpoint RNG leaf indices must be unique non-negative int64 values")
            loaded_engine = state_dict["engines"][str(engine_index)]
            seed_words = (seed & 0xFFFFFFFF, seed >> 32)
            for value in loaded_engine["rng_seeds"].values():
                if tuple(int(word) for word in value.detach().cpu()) != seed_words:
                    raise ValueError("DCP checkpoint RNG tensor does not match its fingerprint")
            loaded_steps = tuple(loaded_engine["steps"].values())
            step_by_target_id = {}
            for target, value in zip(
                self._engines[engine_index]._steps,
                loaded_steps,
                strict=True,
            ):
                if value.ndim != 0:
                    raise ValueError(
                        "DCP checkpoint steps must be non-negative scalar counters"
                    )
                loaded_step = int(value.detach().cpu())
                if not 0 <= loaded_step < (1 << 63):
                    raise ValueError(
                        "DCP checkpoint steps must be non-negative scalar counters"
                    )
                step_by_target_id[id(target)] = loaded_step
            for group_index, group in enumerate(self._engines[engine_index].groups):
                loaded_group = loaded_engine["groups"][str(group_index)]
                loaded_leaves = tuple(int(value) for value in loaded_group["leaf_indices"].detach().cpu())
                param_keys = tuple(self._engines[engine_index]._param_keys[id(param)] for param in group.params)
                if loaded_leaves != tuple(leaf_indices[key] for key in param_keys):
                    raise ValueError("DCP checkpoint RNG tensor does not match its fingerprint")
                loaded_age = loaded_group["age"]
                group_step = step_by_target_id.get(id(group.step))
                if group_step is None:
                    raise ValueError(
                        "DCP checkpoint step counters do not match their parameter groups"
                    )
                if bool(
                    ((loaded_age < 0) | (loaded_age >= group_step)).any().detach().cpu()
                ):
                    raise ValueError(
                        "DCP checkpoint ages must be non-negative and smaller than their step counter"
                    )
        if len({engine["rng"]["seed"] for engine in saved_engines}) != 1:
            raise ValueError("DCP checkpoint Engine RNG seeds do not match")

        saved_groups = saved.get("param_groups")
        expected_groups = expected["param_groups"]
        if not isinstance(saved_groups, (list, tuple)) or len(saved_groups) != len(expected_groups):
            raise ValueError("DCP checkpoint parameter groups do not match this optimizer")
        for saved_group, expected_group in zip(saved_groups, expected_groups, strict=True):
            if (
                not isinstance(saved_group, Mapping)
                or saved_group.get("param_count") != expected_group["param_count"]
                or not isinstance(saved_group.get("values"), Mapping)
                or set(saved_group["values"]) != set(expected_group["values"])
            ):
                raise ValueError("DCP checkpoint parameter groups do not match this optimizer")
        saved_value_groups = [dict(saved_group["values"]) for saved_group in saved_groups]
        _validate_hyperparameter_domains(
            type(self).__name__,
            self._recipe,
            saved_value_groups,
            self.defaults,
            allow_callable_cadence=False,
        )
        _validate_engine_hyperparameter_types(type(self).__name__, saved_value_groups)

    @staticmethod
    def _validate_dcp_cadence(saved: object, engine) -> None:
        if (saved is None) != (engine._cadence is None):
            raise ValueError("DCP checkpoint cadence presence does not match this optimizer")
        if saved is None:
            return
        if not isinstance(saved, Mapping) or set(saved) != {
            "probability", "step", "cumulative", "compensation"
        }:
            raise ValueError("DCP checkpoint cadence state does not match this optimizer")
        if callable(engine._cadence.probability):
            raise ValueError("cannot load a checkpoint into a callable cadence probability schedule")
        probability = saved["probability"]
        step = saved["step"]
        cumulative = saved["cumulative"]
        compensation = saved["compensation"]
        if (
            not isinstance(probability, Real)
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0 <= probability <= 1
            or not isinstance(step, int)
            or isinstance(step, bool)
            or not 0 <= step < (1 << 63)
            or not isinstance(cumulative, Real)
            or isinstance(cumulative, bool)
            or not math.isfinite(float(cumulative))
            or not isinstance(compensation, Real)
            or isinstance(compensation, bool)
            or not math.isfinite(float(compensation))
        ):
            raise ValueError("DCP checkpoint cadence state does not match this optimizer")

    def _restore_dcp_host_state(self, metadata: Mapping) -> None:
        saved_groups = metadata["param_groups"]
        for param_group, saved_group in zip(self.param_groups, saved_groups, strict=True):
            param_group.update(copy.deepcopy(saved_group["values"]))
        self._sync_hypers_from_groups(force=True)
        seeds = set()
        for engine, saved_engine in zip(self._engines, metadata["engines"], strict=True):
            engine._train_mode = saved_engine["train_mode"]
            engine._rng_seed = saved_engine["rng"]["seed"]
            saved_param_init_pending = saved_engine.get("param_init_pending")
            engine._deferred_param_init_pending = (
                {
                    param_key: False
                    for param_key in engine._deferred_param_init_pending
                }
                if saved_param_init_pending is None
                else dict(saved_param_init_pending)
            )
            seeds.add(engine._rng_seed)
            if engine._cadence is not None:
                cadence = saved_engine["cadence"]
                engine._cadence.probability = cadence["probability"]
                engine._cadence.step = cadence["step"]
                engine._cadence.cumulative = cadence["cumulative"]
                engine._cadence.compensation = cadence["compensation"]
        if len(seeds) != 1:
            raise ValueError("DCP checkpoint Engine RNG seeds do not match")
        self._rng_seed = next(iter(seeds))

    def load_state_dict(self, state_dict: Mapping) -> None:
        if "engines" not in state_dict:
            raise ValueError(
                "expected a HeavyBallOptimizer 4.0 state dict with Engine state; HeavyBall 3.x "
                "optimizer checkpoints are not migratable to 4.0 because the Engine state structure "
                "changed. Restart training or warmstart from model weights only."
            )
        torch_state_dict = dict(state_dict)
        engine_states = torch_state_dict.pop("engines")
        if not isinstance(engine_states, (list, tuple)) or len(engine_states) != len(self._engines):
            raise ValueError("checkpoint Engine state does not match this optimizer")
        for engine, engine_state in zip(self._engines, engine_states, strict=True):
            if not isinstance(engine_state, Mapping):
                raise ValueError("checkpoint Engine state does not match this optimizer")
            saved_ecc = engine_state.get("ecc")
            current_ecc = None if engine.ecc is None else (8 if engine.ecc is torch.int8 else 16)
            if saved_ecc != current_ecc:
                raise ValueError("checkpoint ECC configuration does not match this optimizer")
        # Snapshot both layers because either half of checkpoint commit can fail.
        engine_snapshots = [engine.state_dict() for engine in self._engines]
        group_snapshots = [dict(group) for group in self.param_groups]
        state_snapshot = copy.deepcopy(self.state)
        synced_snapshot = [dict(synced) for synced in self._synced_hypers]
        rng_seed_snapshot = self._rng_seed
        try:
            for index, engine in enumerate(self._engines):
                engine.load_state_dict(engine_states[index])
            super().load_state_dict(torch_state_dict)
            self._sync_hypers_from_groups(force=True)
            seeds = {engine._rng_seed for engine in self._engines}
            if len(seeds) != 1:
                raise ValueError("checkpoint Engine RNG seeds do not match")
            self._rng_seed = next(iter(seeds))
        except BaseException:
            for engine, snapshot in zip(self._engines, engine_snapshots, strict=True):
                engine.load_state_dict(snapshot)
            for group, snapshot in zip(self.param_groups, group_snapshots, strict=True):
                group.clear()
                group.update(snapshot)
            self.state = state_snapshot
            self._synced_hypers = synced_snapshot
            self._rng_seed = rng_seed_snapshot
            self._sync_hypers_from_groups(force=True)
            raise


class AdamW(HeavyBallOptimizer):
    """Adam with decoupled weight decay."""

    recipe = adamw


class AdamC(HeavyBallOptimizer):
    """AdamW whose weight decay is scaled by ``lr / max_lr``, so the effective decay follows the
    learning-rate schedule.

    ``max_lr`` is the reference learning rate at which ``weight_decay`` is specified; omit it to
    inherit the (per-group) construction ``lr`` so the effective decay stays constant as ``lr``
    schedules.
    """

    recipe = adamc


class MARSAdamW(HeavyBallOptimizer):
    """MARS variance-reduced gradient correction, clipped to unit L2 norm per leaf, feeding AdamW
    (MARS, Algorithm 2)."""

    recipe = mars_adamw


class OrthoGradAdamW(HeavyBallOptimizer):
    """AdamW on the gradient projected orthogonal to the parameter, with its norm grafted back
    (Grokking at the Edge of Numerical Stability, eq. 11)."""

    recipe = orthograd_adamw


class CautiousAdamW(HeavyBallOptimizer):
    """AdamW that keeps only update entries sharing the gradient's sign (cautious masking), with the
    kept entries rescaled by numel/kept (the cautious-optimizer normalization)."""

    recipe = cautious_adamw


class HyperBallAdamW(HeavyBallOptimizer):
    """Hyperball (arXiv:2606.16899, Algorithm 1): holds each matrix parameter's norm at its initial
    value instead of using weight decay; AdamW on other parameters."""

    recipe = hyperball_adamw


class RMSprop(HeavyBallOptimizer):
    """RMSprop: debiased second-moment (root-mean-square) gradient normalization."""

    recipe = rmsprop


class Lion(HeavyBallOptimizer):
    """Lion: a two-rate momentum EMA reduced to its sign, with decoupled weight decay."""

    recipe = lion


class LaProp(HeavyBallOptimizer):
    """LaProp: normalize the gradient by its second moment before taking the first-moment EMA."""

    recipe = laprop


class OrthoLaProp(HeavyBallOptimizer):
    """LaProp on the gradient first projected orthogonal to the parameter."""

    recipe = ortho_laprop


class LaPropOrtho(HeavyBallOptimizer):
    """LaProp, then the resulting update projected orthogonal to the parameter."""

    recipe = laprop_ortho


class SignLaProp(HeavyBallOptimizer):
    """LaProp followed by a sign step grafted back to each leaf's update norm."""

    recipe = sign_laprop


class NAdam(HeavyBallOptimizer):
    """NAdam: Adam with Nesterov-accelerated momentum."""

    recipe = nadam


class AdEMAMix(HeavyBallOptimizer):
    """AdEMAMix: Adam with an additional slow gradient EMA mixed into the first moment, on a warmup
    schedule."""

    recipe = ademamix


class ADOPT(HeavyBallOptimizer):
    """ADOPT (paper Algorithm 2): normalizes by the previous step's second moment with raw,
    un-debiased EMAs, seeding the variance from the first gradient."""

    recipe = adopt


class UnscaledAdamW(HeavyBallOptimizer):
    """AdamW that accumulates the first-moment EMA in second-moment-normalized coordinates rather
    than on the raw gradient."""

    recipe = unscaled_adamw


class SignSGD(HeavyBallOptimizer):
    """Sign-of-gradient descent."""

    recipe = signsgd


class SGD(HeavyBallOptimizer):
    """Stochastic gradient descent: a stateless raw-gradient step with decoupled weight decay.

    Unlike ``torch.optim.SGD``, this exposes no momentum or Nesterov state. For heavy-ball momentum
    use ``torch.optim.SGD(momentum=...)`` or a custom Recipe built on ``heavyball.momentum``.
    """

    recipe = sgd


class Scion(HeavyBallOptimizer):
    """Scion (arXiv:2502.07529): momentum passed through a norm-constrained linear-minimization
    oracle (spectral for matrices, RMS for vectors).

    A fresh Scion optimizer reinitializes its parameters in place -- matrices to a seeded orthogonal
    frame, vectors to zero -- on the FIRST ``step()`` (not at construction), because the oracle assumes
    a norm-constrained starting point. Because that gradient was computed at the pre-initialization
    values, the first step performs ONLY the reinitialization and applies no update (it does not
    consume the gradient); training effectively begins on the second step. Constructing Scion does not
    mutate parameters, but the first ``step()`` overwrites their current values, so loading model or
    pretrained weights before the first step does NOT preserve them. Only resuming from a Scion
    checkpoint skips the reinitialization (the checkpoint records that it has already occurred).
    """

    recipe = scion


class SFAdamW(HeavyBallOptimizer):
    """Schedule-free AdamW (Defazio et al.): an averaged evaluation iterate that removes the
    learning-rate schedule.

    Switch to the evaluation iterate with ``optimizer.eval()`` for validation/inference and back to
    the training iterate with ``optimizer.train()`` before resuming training.
    """

    recipe = sf_adamw


class ScheduleFree(SFAdamW):
    """Deprecated alias for :class:`SFAdamW`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "ScheduleFree is deprecated; use SFAdamW instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class MSAM(HeavyBallOptimizer):
    """Momentum-SAM (arXiv:2401.12033): a sharpness-aware perturbation along the momentum direction,
    normalized per leaf.

    Uses distinct train/eval iterates: call ``optimizer.eval()`` for validation/inference and
    ``optimizer.train()`` before resuming training.
    """

    recipe = msam_laprop


class AdaMuon(HeavyBallOptimizer):
    """AdaMuon: apply Muon's polar iteration to the sign of its momentum, divide by the raw
    elementwise second-moment RMS using the same momentum coefficient, and align the result to RMS
    0.2 for 2D weights; use AdamW on other parameters."""

    recipe = adamuon


class Aurora(HeavyBallOptimizer):
    """Aurora: apply two damped leverage-balancing/polar iterations to heavy-ball momentum for
    tall 2D weights, with ``beta2`` as the damping coefficient; use Muon for other matrices and
    AdamW for non-matrix parameters."""

    recipe = aurora


class Muon(HeavyBallOptimizer):
    """Muon: heavy-ball momentum orthogonalized by a five-step Newton-Schulz iteration, for 2D
    weights; AdamW on other parameters."""

    recipe = muon


class SpEL(HeavyBallOptimizer):
    """SpEL: project EMA momentum into the Stiefel tangent space, apply the stochastic matrix-sign
    linear minimization oracle, and retract the update onto the manifold for 2D weights; use AdamW
    on other parameters."""

    recipe = spel


class PolarGrad(HeavyBallOptimizer):
    """PolarGrad: heavy-ball momentum's polar direction scaled by its nuclear norm, for 2D weights;
    AdamW on other parameters."""

    recipe = polargrad


class MuonLaProp(HeavyBallOptimizer):
    """Muon orthogonalization applied to a LaProp-normalized update, for 2D weights; AdamW on other
    parameters."""

    recipe = muon_laprop


class NorMuon(HeavyBallOptimizer):
    """NorMuon: orthogonalize EMA momentum, normalize it by a raw row-wise second moment, and set
    its RMS to 0.2 for 2D weights; use AdamW on other parameters."""

    recipe = normuon


class Oblique(HeavyBallOptimizer):
    """Row-normalized Adam on the oblique manifold for 2D weights; AdamW on other parameters."""

    recipe = oblique


class Whitening(HeavyBallOptimizer):
    """Whiten each square-matrix parameter's gradient to unit covariance by left-multiplying it by
    the inverse spectral root of its running Gram (Shampoo-style factored spectral root).

    Non-square and non-matrix parameters use AdamW.
    """

    recipe = whiten_adamw


class WhitenAdamW(Whitening):
    """Deprecated alias for :class:`Whitening`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "WhitenAdamW is deprecated; use Whitening instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class SOAP(HeavyBallOptimizer):
    """SOAP: Adam run inside Shampoo's preconditioner eigenbasis (Gram factors refreshed on the
    ``preconditioner_update_probability`` cadence); AdamW on non-matrix parameters."""

    recipe = soap_adamw


class SOLP(HeavyBallOptimizer):
    """SOAP with a LaProp inner update in place of Adam; AdamW on non-matrix parameters."""

    recipe = solp_adamw


class SOAPNAdam(HeavyBallOptimizer):
    """SOAP with a NAdam inner update; AdamW on non-matrix parameters."""

    recipe = soap_nadam_adamw


class SOAPAdEMAMix(HeavyBallOptimizer):
    """SOAP with an AdEMAMix inner update, rotating both moments with the basis; AdamW on non-matrix
    parameters."""

    recipe = soap_ademamix_adamw


class HeavySOAP(HeavyBallOptimizer):
    """HeavySOAP: SOAP with eigenvalue-sorted QR and Hadamard-square second-moment transport
    (Nestler, HomebrewML). In heavyball4 these improvements are the default SOAP behavior;
    this facade preserves the heavyball 3.x name."""

    recipe = soap_adamw


class HeavySOLP(HeavyBallOptimizer):
    """HeavySOLP: SOLP with eigenvalue-sorted QR and Hadamard-square second-moment transport."""

    recipe = solp_adamw


class HeavySOAPNAdam(HeavyBallOptimizer):
    """HeavySOAPNAdam: SOAPNAdam with eigenvalue-sorted QR and Hadamard-square second-moment transport."""

    recipe = soap_nadam_adamw


class HeavySOAPAdEMAMix(HeavyBallOptimizer):
    """HeavySOAPAdEMAMix: SOAPAdEMAMix with eigenvalue-sorted QR and Hadamard-square second-moment transport."""

    recipe = soap_ademamix_adamw


class Shampoo(HeavyBallOptimizer):
    """Shampoo: a two-sided Kronecker-factored preconditioner (inverse fourth root), rebuilt on the
    refresh cadence; AdamW on non-matrix parameters."""

    recipe = shampoo_adamw


class KLSOAP(HeavyBallOptimizer):
    """KL-SOAP (Lin et al., arXiv:2509.03378): SOAP-style projected Adam whose Kronecker factors are
    estimated after whitening the opposite factor by its eigenvalue EMA; AdamW on non-matrix
    parameters."""

    recipe = kl_soap_adamw


class KLShampoo(HeavyBallOptimizer):
    """KL-Shampoo (Lin et al., arXiv:2509.03378): Shampoo-style momentum in the KL-corrected
    eigensystem; AdamW on non-matrix parameters."""

    recipe = kl_shampoo_adamw


class HeavyKLSOAP(HeavyBallOptimizer):
    """Heavy KL-SOAP: KL-SOAP with Moore-Penrose pseudoinverse eigenvalue scaling and
    eigenvalue-sorted QR basis refresh; AdamW on non-matrix parameters."""

    recipe = heavy_kl_soap_adamw


class HeavyKLShampoo(HeavyBallOptimizer):
    """Heavy KL-Shampoo: KL-Shampoo with Moore-Penrose pseudoinverse eigenvalue scaling and
    eigenvalue-sorted QR basis refresh; AdamW on non-matrix parameters."""

    recipe = heavy_kl_shampoo_adamw


class PSGD(HeavyBallOptimizer):
    """Recommended general PSGD entry point. Multi-dimensional leaves with an axis larger than 2048
    use PSGD-LRA, other matrix-mergeable leaves use PSGD-Kron, and vectors or remaining small leaves
    use N-factor PSGD."""

    recipe = psgd


class PSGDKron(HeavyBallOptimizer):
    """Explicit PSGD-Kron override (use :class:`PSGD` for general use): Li's online
    gradient-whitening Kronecker preconditioner (P = QᵀQ), with triangular factors up to
    ``max_size_triangular`` and diagonal factors beyond; AdamW on non-matrix parameters. The
    preconditioned update carries no momentum; ``LATHER`` runs Adam (with momentum) in the same
    whitening basis."""

    recipe = kron_adamw


class PSGDLRA(HeavyBallOptimizer):
    """Explicit PSGD-LRA override (use :class:`PSGD` for general use): a diagonal + low-rank
    preconditioner fit by Lie-group gradient descent."""

    recipe = psgd_lra_adamw


class PSGDNfactor(HeavyBallOptimizer):
    """Explicit N-factor PSGD override (use :class:`PSGD` for general use): one Q factor per axis for
    higher-rank parameters; AdamW on other parameters."""

    recipe = psgd_nfactor_adamw


class PSGDPro(HeavyBallOptimizer):
    """Explicit PSGD-PRO override (use :class:`PSGD` for general use): Li's full per-dimension Q
    factors fit by a stochastic Procrustes update and applied as P = QᵀQ; AdamW on non-matrix
    parameters. The preconditioned update carries no momentum."""

    recipe = psgd_pro_adamw


class QSGD(HeavyBallOptimizer):
    """Explicit QSGD override (use :class:`PSGD` for general use): Li's PSGD-PRO preconditioner
    applied once per axis (Q, not QᵀQ); AdamW on non-matrix parameters. The preconditioned update
    carries no momentum."""

    recipe = qsgd_adamw


class LATHER(HeavyBallOptimizer):
    """LATHER: Adam whose moments live in PSGD-Kron's triangular-factor eigenbasis, rotated into
    refreshed coordinates on the cadence; AdamW on non-matrix parameters."""

    recipe = lather_adamw


class SUDSAdamW(HeavyBallOptimizer):
    """SUDS: Adam in a rank-1 Fisher (Householder) basis maintained per leaf by an Oja
    power-iteration update."""

    recipe = suds_adamw


class TrueGradAdam(HeavyBallOptimizer):
    """TrueGrad Adam: Adam whose second-moment EMA uses per-sample gradient-squared observations
    instead of the stochastic gradient squared.

    Requires per-sample gradient observations: call ``heavyball.register_truegrad(model)`` before the
    first forward pass (only module types supported by that helper can produce them).
    """

    recipe = truegrad_adam


class TrueGradRMSprop(HeavyBallOptimizer):
    """TrueGrad RMSprop: RMSprop whose second-moment EMA uses per-sample gradient-squared
    observations.

    Requires per-sample gradient observations: call ``heavyball.register_truegrad(model)`` before the
    first forward pass (only module types supported by that helper can produce them).
    """

    recipe = truegrad_rmsprop


class TrueGradLaProp(HeavyBallOptimizer):
    """TrueGrad LaProp: LaProp whose second-moment normalization uses per-sample gradient-squared
    observations.

    Requires per-sample gradient observations: call ``heavyball.register_truegrad(model)`` before the
    first forward pass (only module types supported by that helper can produce them).
    """

    recipe = truegrad_laprop


class TrueGradNAdam(HeavyBallOptimizer):
    """TrueGrad NAdam: NAdam whose second-moment EMA uses per-sample gradient-squared
    observations.

    Requires per-sample gradient observations: call ``heavyball.register_truegrad(model)`` before the
    first forward pass (only module types supported by that helper can produce them).
    """

    recipe = truegrad_nadam


class SplitOpt(torch.optim.Optimizer):
    """Delegates different parameter groups to different HeavyBall optimizers.

        opt = SplitOpt([
            {'params': matrices, 'optimizer': Muon, 'lr': 0.02},
            {'params': vectors, 'optimizer': AdamW, 'lr': 0.001},
        ])
    """

    def __init__(self, specs):
        normalized, all_params = [], []
        for spec in specs:
            spec = dict(spec)
            params = list(spec.pop("params"))
            if not params:
                continue
            optimizer_cls = spec.pop("optimizer")
            if not (isinstance(optimizer_cls, type) and issubclass(optimizer_cls, HeavyBallOptimizer)):
                raise TypeError(f"SplitOpt requires HeavyBallOptimizer subclasses, got {optimizer_cls}")
            normalized.append((optimizer_cls, params, spec))
            all_params.extend(params)
        if not normalized:
            raise ValueError("No optimizers created")
        if len({id(p) for p in all_params}) != len(all_params):
            raise ValueError("A parameter cannot belong to multiple SplitOpt optimizers")
        self.optimizers = [cls(params, **kwargs) for cls, params, kwargs in normalized]
        self._params_by_optimizer = tuple(
            tuple(param for group in optimizer.param_groups for param in group["params"])
            for optimizer in self.optimizers
        )
        self._initializing = True
        try:
            super().__init__(all_params, {})
        finally:
            self._initializing = False
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def step(self, closure=None, observed=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if observed is None:
            child_observed = (None,) * len(self.optimizers)
        elif isinstance(observed, Mapping):
            all_params = tuple(
                param for params in self._params_by_optimizer for param in params
            )
            if len(observed) != len(all_params):
                raise ValueError("observed must contain every trainable parameter")
            try:
                values = tuple(observed[param] for param in all_params)
            except KeyError as error:
                raise ValueError(
                    "observed must contain every trainable parameter"
                ) from error
            if any(not isinstance(value, bool) for value in values):
                raise TypeError("observed values must be host bools")
            child_observed = tuple(
                {param: observed[param] for param in params}
                for params in self._params_by_optimizer
            )
        else:
            values = tuple(observed)
            expected = sum(len(params) for params in self._params_by_optimizer)
            if len(values) != expected:
                raise ValueError(
                    "observed must contain one value for every trainable parameter"
                )
            if any(not isinstance(value, bool) for value in values):
                raise TypeError("observed values must be host bools")
            child_values = []
            offset = 0
            for params in self._params_by_optimizer:
                child_values.append(values[offset : offset + len(params)])
                offset += len(params)
            child_observed = tuple(child_values)
        for opt, mask in zip(self.optimizers, child_observed, strict=True):
            opt.step(observed=mask)
        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def train(self, mode=True):
        for opt in self.optimizers:
            opt.train(mode)
        return self

    def eval(self):
        for opt in self.optimizers:
            opt.eval()
        return self

    def state_dict(self):
        return {
            "classes": [f"{type(opt).__module__}.{type(opt).__qualname__}" for opt in self.optimizers],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError(f"Expected {len(self.optimizers)} optimizer states, got {len(states)}")
        expected = [f"{type(opt).__module__}.{type(opt).__qualname__}" for opt in self.optimizers]
        if state_dict["classes"] != expected:
            raise ValueError(f"Expected optimizer classes {expected}, got {state_dict['classes']}")
        snapshots = [opt.state_dict() for opt in self.optimizers]
        try:
            for opt, state in zip(self.optimizers, states, strict=True):
                opt.load_state_dict(state)
        except BaseException:
            for opt, snapshot in zip(self.optimizers, snapshots, strict=True):
                opt.load_state_dict(snapshot)
            raise

    def add_param_group(self, param_group):
        if self._initializing:
            return super().add_param_group(param_group)
        raise RuntimeError("SplitOpt does not support add_param_group; recreate with all parameter groups")


__all__ = [
    "ADOPT",
    "AdEMAMix",
    "AdaMuon",
    "AdamC",
    "AdamW",
    "Aurora",
    "CautiousAdamW",
    "HeavyBallOptimizer",
    "HeavyKLSOAP",
    "HeavyKLShampoo",
    "HeavySOAP",
    "HeavySOAPAdEMAMix",
    "HeavySOAPNAdam",
    "HeavySOLP",
    "HyperBallAdamW",
    "KLSOAP",
    "KLShampoo",
    "LATHER",
    "LaProp",
    "LaPropOrtho",
    "Lion",
    "MARSAdamW",
    "MSAM",
    "Muon",
    "MuonLaProp",
    "NAdam",
    "NorMuon",
    "Oblique",
    "OrthoGradAdamW",
    "OrthoLaProp",
    "PSGD",
    "PSGDKron",
    "PSGDLRA",
    "PSGDNfactor",
    "PSGDPro",
    "PolarGrad",
    "QSGD",
    "RMSprop",
    "SFAdamW",
    "SGD",
    "SOAP",
    "SOAPAdEMAMix",
    "SOAPNAdam",
    "SOLP",
    "SpEL",
    "SUDSAdamW",
    "ScheduleFree",
    "Scion",
    "Shampoo",
    "SignLaProp",
    "SignSGD",
    "SplitOpt",
    "TrueGradAdam",
    "TrueGradLaProp",
    "TrueGradNAdam",
    "TrueGradRMSprop",
    "UnscaledAdamW",
    "WhitenAdamW",
    "Whitening",
]
