"""PyTorch optimizer facades over HeavyBall's slab-native recipes."""

import copy
import inspect
import os
from typing import Callable, Iterable, Mapping

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


def _recipe_hyperparameters(recipe: Recipe | Route) -> tuple[tuple[str, type, object], ...]:
    return tuple((name, type(default), default) for name, default in _recipe_defaults(recipe, {}).items())


# Engine-level knobs every facade accepts through ``**hyper`` (build/Engine, not recipe defaults).
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
    return (*_recipe_hyperparameters(recipe), *_ENGINE_HYPERPARAMETERS)


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
    """A ``torch.optim.Optimizer`` facade over HeavyBall Engines.

    Construction slab-backs the supplied parameters, so construct this optimizer
    before wrapping the model with DDP or ``torch.compile``. For a supported
    componentwise recipe, call ``Optimizer.fsdp2(model, ...)`` after ``fully_shard(model)``.

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
                    "ScheduleFree.fsdp2() requires caution=0 and cautious_weight_decay=0 because cautious "
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
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
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
        recipe = recipe if recipe is not None else type(self).recipe
        torch_params = _materialize_params(params)
        _validate_param_group_options(recipe, torch_params, hyper)
        defaults = _recipe_defaults(recipe, hyper)
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
        self._synced_hypers = [
            {name: param_group[name] for name in self._hyper_names if name in param_group}
            for param_group in self.param_groups
        ]

    def add_param_group(self, param_group: dict[str, object]) -> None:
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
            return
        if self._fsdp2_mode:
            raise ValueError("fsdp2() optimizers cannot add parameters after their FSDP storage is bound")
        materialized_group = _materialize_params([param_group])[0]
        _validate_param_group_options(self._recipe, [materialized_group], {})
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
        engine, leaf_count = _build_group_aware_engine(
            candidate_groups,
            self._recipe,
            self.defaults,
            bindings=None,
            rng_seed=self._rng_seed,
        )

        migrated = engine.state_dict()
        for name in ("age", "state"):
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
        self._engine = engine
        self._engines = [engine]
        self._engine_by_param_id = {id(param): engine for param in engine.params}
        self._rng_seed = engine._rng_seed
        self._rng_next_leaf_index = leaf_count
        self._hyper_names = frozenset(
            name for namespace in engine._hyper_locations.values() for name in vars(namespace)
        )
        self._synced_hypers = [
            {name: group[name] for name in self._hyper_names if name in group}
            for group in self.param_groups
        ]

    @torch.no_grad()
    def _sync_hypers_from_groups(self, *, force: bool) -> None:
        for group_id, (param_group, synced) in enumerate(
            zip(self.param_groups, self._synced_hypers, strict=True)
        ):
            for name in self._hyper_names:
                if name not in param_group:
                    continue
                value = param_group[name]
                if force or not _same_hyper_value(value, synced.get(name)):
                    self._engine.set_hyper(name, value, group_id=group_id)
                    synced[name] = value

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._fsdp2_mode and self._recipe is sf_adamw:
            for param_group in self.param_groups:
                if not self._fsdp2_disabled(param_group["caution"]) or not self._fsdp2_disabled(
                    param_group["cautious_weight_decay"]
                ):
                    raise ValueError(
                        "ScheduleFree.fsdp2() requires caution=0 and cautious_weight_decay=0 because cautious "
                        "normalization reduces over the full logical parameter."
                    )
        self._sync_hypers_from_groups(force=False)
        self._engine.step()
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
    def dcp_load(self, checkpoint_dir: str | os.PathLike[str]) -> None:
        """Load and reshard an FSDP2 model and optimizer checkpoint onto the current world."""

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
            for name in ("train_mode", "cadence", "rng"):
                saved_static.pop(name, None)
                expected_static.pop(name, None)
            if saved_static != expected_static:
                raise ValueError("DCP checkpoint schema or recipe does not match this optimizer")
            if not isinstance(saved_engine.get("train_mode"), bool):
                raise ValueError("DCP checkpoint train mode must be a bool")
            self._validate_dcp_cadence(saved_engine.get("cadence"), self._engines[engine_index])
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
            for group_index, group in enumerate(self._engines[engine_index].groups):
                loaded_group = loaded_engine["groups"][str(group_index)]
                loaded_leaves = tuple(int(value) for value in loaded_group["leaf_indices"].detach().cpu())
                param_keys = tuple(self._engines[engine_index]._param_keys[id(param)] for param in group.params)
                if loaded_leaves != tuple(leaf_indices[key] for key in param_keys):
                    raise ValueError("DCP checkpoint RNG tensor does not match its fingerprint")
                if bool((loaded_group["age"] < 0).any().detach().cpu()):
                    raise ValueError("DCP checkpoint ages must be non-negative")
            for value in loaded_engine["steps"].values():
                if value.ndim != 0 or int(value.detach().cpu()) < 0:
                    raise ValueError("DCP checkpoint steps must be non-negative scalar counters")
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
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0 <= probability <= 1
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(saved["cumulative"], (int, float))
            or not isinstance(saved["compensation"], (int, float))
        ):
            raise ValueError("DCP checkpoint cadence state does not match this optimizer")

    def _restore_dcp_host_state(self, metadata: Mapping) -> None:
        saved_groups = metadata["param_groups"]
        for param_group, saved_group in zip(self.param_groups, saved_groups, strict=True):
            param_group.update(copy.deepcopy(saved_group["values"]))
        # Engine hyper tensors were already restored above; adopt the public group values (the source of
        # truth for scheduler-driven hypers) so a value changed after the last step is not kept stale.
        self._sync_hypers_from_groups(force=True)
        seeds = set()
        for engine, saved_engine in zip(self._engines, metadata["engines"], strict=True):
            engine._train_mode = saved_engine["train_mode"]
            engine._rng_seed = saved_engine["rng"]["seed"]
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
            raise ValueError("expected a HeavyBallOptimizer state dict with Engine state")
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
        super().load_state_dict(torch_state_dict)
        for index, engine in enumerate(self._engines):
            engine.load_state_dict(engine_states[index])
        # Push the restored public group hypers into the Engine cells. A scheduler value set AFTER the
        # last step leaves the Engine cell stale at checkpoint time; the public group is the source of
        # truth, so resume must adopt it (else the P==C sync cache below would skip the update forever).
        self._sync_hypers_from_groups(force=True)
        seeds = {engine._rng_seed for engine in self._engines}
        if len(seeds) != 1:
            raise ValueError("checkpoint Engine RNG seeds do not match")
        self._rng_seed = next(iter(seeds))


class AdamW(HeavyBallOptimizer):
    """Adam with decoupled weight decay."""

    recipe = adamw


class AdamC(HeavyBallOptimizer):
    """AdamW whose weight decay is scaled by ``lr / max_lr``, so the effective decay follows the
    learning-rate schedule."""

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
    """Stochastic gradient descent (raw-gradient step) with decoupled weight decay."""

    recipe = sgd


class Scion(HeavyBallOptimizer):
    """Scion (arXiv:2502.07529): momentum passed through a norm-constrained linear-minimization
    oracle (spectral for matrices, RMS for vectors).

    Construction REINITIALIZES the model in place -- matrices to a random orthogonal frame, vectors
    to zero -- because the oracle assumes a norm-constrained starting point. Construct Scion *before*
    loading a checkpoint or pretrained weights, or those weights are silently overwritten.
    """

    recipe = scion


class ScheduleFree(HeavyBallOptimizer):
    """Schedule-free AdamW (Defazio et al.): an averaged evaluation iterate that removes the
    learning-rate schedule."""

    recipe = sf_adamw


class MSAM(HeavyBallOptimizer):
    """Momentum-SAM (arXiv:2401.12033): a sharpness-aware perturbation along the momentum direction,
    normalized per leaf."""

    recipe = msam_laprop


class AdaMuon(HeavyBallOptimizer):
    """AdaMuon: Muon's orthogonalized direction with elementwise second-moment normalization
    (RMSprop), for 2D weights; AdamW on other parameters."""

    recipe = adamuon


class Aurora(HeavyBallOptimizer):
    """Aurora: heavy-ball momentum with a leverage-balanced polar direction, for 2D weights;
    AdamW on other parameters."""

    recipe = aurora


class Muon(HeavyBallOptimizer):
    """Muon: heavy-ball momentum orthogonalized by a five-step Newton-Schulz iteration, for 2D
    weights; AdamW on other parameters."""

    recipe = muon


class SpEL(HeavyBallOptimizer):
    """SpEL: Muon's orthogonalized direction followed by a Stiefel-manifold retraction, for 2D
    weights; AdamW on other parameters."""

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
    """NorMuon: Muon's orthogonalized direction with row/column second-moment normalization and
    Frobenius-norm preservation, for 2D weights; AdamW on other parameters."""

    recipe = normuon


class Oblique(HeavyBallOptimizer):
    """Row-normalized Adam on the oblique manifold for 2D weights; AdamW on other parameters."""

    recipe = oblique


class Whitening(HeavyBallOptimizer):
    """Left-whitening preconditioner for square 2D weights (Q = GG^-1/2 from the gradient Gram,
    rebuilt on refresh); AdamW on other parameters."""

    recipe = whiten_adamw


class WhitenAdamW(HeavyBallOptimizer):
    """Left-whitening preconditioner for square 2D weights (Q = GG^-1/2 from the gradient Gram,
    rebuilt on the refresh cadence); AdamW on other parameters."""

    recipe = whiten_adamw


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


class PSGDKron(HeavyBallOptimizer):
    """PSGD-Kron (Li's psgd_torch): an online gradient-whitening Kronecker preconditioner (P = QᵀQ);
    triangular factors up to ``max_size_triangular``, diagonal beyond; AdamW on non-matrix
    parameters. The preconditioned update carries no momentum; ``LATHER`` runs Adam (with momentum)
    in the same whitening basis."""

    recipe = kron_adamw


class PSGDLRA(HeavyBallOptimizer):
    """PSGD-LRA: diagonal + low-rank preconditioner fit by Lie-group gradient descent."""

    recipe = psgd_lra_adamw


class PSGDNfactor(HeavyBallOptimizer):
    """N-factor PSGD-PRO: one Q factor per axis for higher-rank parameters; AdamW on other parameters."""

    recipe = psgd_nfactor_adamw


class PSGDPro(HeavyBallOptimizer):
    """PSGD-PRO (Li's psgd_torch): full per-dimension Q factors fit by a stochastic Procrustes update
    and applied as P = QᵀQ; AdamW on non-matrix parameters. The preconditioned update carries
    no momentum."""

    recipe = psgd_pro_adamw


class QSGD(HeavyBallOptimizer):
    """QSGD (Li's psgd_torch): the PSGD-PRO preconditioner applied once per axis (Q, not QᵀQ); AdamW
    on non-matrix parameters. The preconditioned update carries no momentum."""

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
    instead of the stochastic gradient squared."""

    recipe = truegrad_adam


class TrueGradRMSprop(HeavyBallOptimizer):
    """TrueGrad RMSprop: RMSprop whose second-moment EMA uses per-sample gradient-squared
    observations."""

    recipe = truegrad_rmsprop


class TrueGradLaProp(HeavyBallOptimizer):
    """TrueGrad LaProp: LaProp whose second-moment normalization uses per-sample gradient-squared
    observations."""

    recipe = truegrad_laprop


class TrueGradNAdam(HeavyBallOptimizer):
    """TrueGrad NAdam: NAdam whose second-moment EMA uses per-sample gradient-squared
    observations."""

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
        self._initializing = True
        try:
            super().__init__(all_params, {})
        finally:
            self._initializing = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

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
        for opt, state in zip(self.optimizers, states):
            opt.load_state_dict(state)

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
    "PSGDKron",
    "PSGDLRA",
    "PSGDNfactor",
    "PSGDPro",
    "PolarGrad",
    "QSGD",
    "RMSprop",
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
