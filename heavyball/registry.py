"""Public optimizer discovery and state-size estimation helpers."""

import gc
import inspect
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from . import optim
from .core import Recipe, Route, fsdp2_recipe_scope_supported
from .optim import HeavyBallOptimizer


def _facades() -> dict[str, type[HeavyBallOptimizer]]:
    return {
        name: facade
        for name in optim.__all__
        if (
            isinstance((facade := getattr(optim, name)), type)
            and issubclass(facade, HeavyBallOptimizer)
            and facade is not HeavyBallOptimizer
        )
    }


def _alias_target(facade: type[HeavyBallOptimizer]) -> str | None:
    doc = inspect.getdoc(facade) or ""
    if not doc.startswith("Deprecated alias for"):
        return None
    for base in facade.__bases__:
        if issubclass(base, HeavyBallOptimizer) and base is not HeavyBallOptimizer:
            return base.__name__
    return None


def list_optimizers(include_aliases: bool = False) -> list[str]:
    """Return the public HeavyBall optimizer facade names."""

    facades = _facades()
    return sorted(
        name
        for name, facade in facades.items()
        if include_aliases or _alias_target(facade) is None
    )


def _callable_name(function) -> str:
    return getattr(function, "__name__", type(function).__name__)


def _recipe_structure(recipe: Recipe | Route) -> dict[str, object]:
    if isinstance(recipe, Route):
        return {
            "when": _callable_name(recipe.when),
            "then": _recipe_structure(recipe.then),
            "otherwise": _recipe_structure(recipe.otherwise),
        }
    return {
        "transforms": [_callable_name(transform) for transform in recipe.chain],
        "commit": _callable_name(recipe.commit),
    }


_ALL_SHAPES = (
    "ADOPT",
    "AdEMAMix",
    "AdamC",
    "AdamW",
    "CautiousAdamW",
    "LaProp",
    "LaPropOrtho",
    "Lion",
    "MARSAdamW",
    "MSAM",
    "NAdam",
    "OrthoGradAdamW",
    "OrthoLaProp",
    "RMSprop",
    "SFAdamW",
    "SGD",
    "SUDSAdamW",
    "Scion",
    "SignLaProp",
    "SignSGD",
    "TrueGradAdam",
    "TrueGradLaProp",
    "TrueGradNAdam",
    "TrueGradRMSprop",
    "UnscaledAdamW",
)
_MATRIX_ONLY = (
    "AdaMuon",
    "Aurora",
    "Muon",
    "MuonLaProp",
    "NorMuon",
    "Oblique",
    "PolarGrad",
    "SpEL",
)
_MERGED_MATRIX = (
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
    "PSGDKron",
    "PSGDPro",
    "QSGD",
    "SOAP",
    "SOAPAdEMAMix",
    "SOAPNAdam",
    "SOLP",
    "Shampoo",
)

_ROUTING = {
    **dict.fromkeys(_ALL_SHAPES, "the same recipe is used for every parameter shape"),
    **dict.fromkeys(
        _MATRIX_ONLY,
        "exactly 2-D parameters use the named matrix recipe; every other shape uses AdamW",
    ),
    **dict.fromkeys(
        _MERGED_MATRIX,
        "matrix-mergeable parameters use the named matrix recipe; degenerate/non-matrix leaves use AdamW",
    ),
    "PSGD": (
        "multidimensional with an axis > 2048 -> PSGD-LRA; matrix-mergeable -> "
        "PSGD-Kron; otherwise (including vectors) -> N-factor PSGD"
    ),
    "PSGDKron": "matrix-mergeable -> PSGD-Kron; otherwise -> AdamW",
    "PSGDLRA": "non-scalar with more than one element -> PSGD-LRA; otherwise -> AdamW",
    "PSGDNfactor": (
        "three or more non-singleton axes -> N-factor PSGD; otherwise matrix-mergeable "
        "-> PSGD-PRO; otherwise -> AdamW"
    ),
    "PSGDPro": "matrix-mergeable -> PSGD-PRO; otherwise -> AdamW",
    "QSGD": "matrix-mergeable -> QSGD (one Q application); otherwise -> AdamW",
    "Whitening": "square 2-D parameters use whitening; every other shape uses AdamW",
}

_STANDARD_LIFECYCLE = (
    "Every observed step advances normally; no bootstrap-only step or train/eval iterate swap."
)
_LIFECYCLE = {name: _STANDARD_LIFECYCLE for name in _ROUTING}
_LIFECYCLE.update(
    {
        "ADOPT": (
            "The first observed step only seeds the second moment; it does not update the "
            "parameter. Optimization starts on the second observed step."
        ),
        "SUDSAdamW": (
            "The first observed step only bootstraps the Fisher/Householder direction; it does "
            "not update the parameter. Optimization starts on the second observed step."
        ),
        "Scion": (
            "The first step reinitializes matrices to seeded orthogonal frames and vectors to "
            "zero, discards the gradient computed at the old values, and does not advance the "
            "clock. A resumed Scion checkpoint records that this has happened."
        ),
        "SFAdamW": (
            "Call eval() for the averaged evaluation iterate and train() before resuming "
            "training."
        ),
        "MSAM": (
            "Call eval() for its evaluation representation and train() before resuming training."
        ),
    }
)

_STANDARD_COMPILE = (
    "The normal full-step graph is compiled lazily on first use with fullgraph=True, static "
    "shapes, and max-autotune; changing a dynamic group hyper fills scalar cells without recompiling."
)
_REFRESH_COMPILE = {
    "HeavyKLSOAP",
    "HeavyKLShampoo",
    "HeavySOAP",
    "HeavySOAPAdEMAMix",
    "HeavySOAPNAdam",
    "HeavySOLP",
    "KLSOAP",
    "KLShampoo",
    "LATHER",
    "PSGD",
    "PSGDKron",
    "PSGDLRA",
    "PSGDNfactor",
    "PSGDPro",
    "QSGD",
    "SOAP",
    "SOAPAdEMAMix",
    "SOAPNAdam",
    "SOLP",
    "Shampoo",
    "Whitening",
}
_COMPILE_BEHAVIOR = {name: _STANDARD_COMPILE for name in _ROUTING}
for _name in _REFRESH_COMPILE:
    _COMPILE_BEHAVIOR[_name] = (
        f"{_STANDARD_COMPILE} The normal and preconditioner-refresh paths are distinct lazy "
        "compiled graphs."
    )
for _name in ("SFAdamW", "MSAM"):
    _COMPILE_BEHAVIOR[_name] = (
        f"{_STANDARD_COMPILE} The train/eval representation swap is a separate lazy compiled graph."
    )
_COMPILE_BEHAVIOR["Scion"] = (
    "The first-step host-side reinitialization returns before the optimizer graph runs; the normal "
    "full-step graph compiles lazily on the first real update."
)

_SLAB_CONTRACT = {
    "storage": (
        "Parameters and gradients are persistent slab views; do not reassign p.data or p.grad or "
        "change parameter storage after construction."
    ),
    "zero_grad": "Use zero_grad(set_to_none=False); set_to_none=True is unsupported.",
    "observed": (
        "By default every leaf advances. observed=False suppresses that leaf's HeavyBall update, "
        "moments, decay, and clock only; it does not change distributed unused-parameter detection."
    ),
}

_DEFAULT_GOTCHA = (
    "Requires the persistent slab/zero_grad/observed contract reported in storage_contract."
)
_GOTCHAS = {name: _DEFAULT_GOTCHA for name in _ROUTING}
_GOTCHAS.update(
    {
        "ADOPT": "Its first observed gradient is bootstrap-only; no parameter update occurs.",
        "AdamC": (
            "Omitted max_lr is captured from each group's construction lr; later schedulers change "
            "lr but not that reference."
        ),
        "MSAM": "Validation/inference requires eval(), followed by train() before training resumes.",
        "SFAdamW": "Validation/inference requires eval(), followed by train() before training resumes.",
        "SGD": "This facade is stateless raw SGD; it does not expose momentum or Nesterov options.",
        "SUDSAdamW": "Its first observed gradient is bootstrap-only; no parameter update occurs.",
        "Scion": (
            "A fresh optimizer overwrites parameters on its first step and discards that step's "
            "gradient; load pretrained weights before construction only if this overwrite is intended."
        ),
        "TrueGradAdam": "register_truegrad(model) is required before the first forward; FSDP2 is unsupported.",
        "TrueGradLaProp": "register_truegrad(model) is required before the first forward; FSDP2 is unsupported.",
        "TrueGradNAdam": "register_truegrad(model) is required before the first forward; FSDP2 is unsupported.",
        "TrueGradRMSprop": "register_truegrad(model) is required before the first forward; FSDP2 is unsupported.",
    }
)

_OPTIMIZER_METADATA = {
    name: {
        "routing": _ROUTING[name],
        "lifecycle": _LIFECYCLE[name],
        "compile_behavior": _COMPILE_BEHAVIOR[name],
        "gotcha": _GOTCHAS[name],
    }
    for name in _ROUTING
}


def _recipe_has_observations(recipe: Recipe | Route) -> bool:
    if isinstance(recipe, Recipe):
        return bool(recipe.observations)
    return _recipe_has_observations(recipe.then) or _recipe_has_observations(recipe.otherwise)


def _fsdp2_supported(recipe: Recipe | Route) -> bool:
    return not _recipe_has_observations(recipe) and fsdp2_recipe_scope_supported(recipe)


def _distributed_limitations(
    name: str,
    recipe: Recipe | Route,
    fsdp2_supported: bool,
) -> tuple[str, ...]:
    limitations = [
        (
            "DDP requires gradient_as_bucket_view=False. Conditional graphs still require "
            "find_unused_parameters=True in addition to observed=."
        ),
        (
            "FSDP2 requires fully_shard() before facade.fsdp2(), rejects clip_global_norm, and "
            "conditional graphs require set_reduce_scatter_unused_params(True) plus observed=."
        ),
    ]
    if _recipe_has_observations(recipe):
        limitations.append("FSDP2 does not support observation-bearing (TrueGrad) recipes yet.")
    elif not fsdp2_supported:
        limitations.append(
            "FSDP2 is unavailable because at least one callable is neither shard-separable nor whole-scoped."
        )
    if name == "SFAdamW":
        limitations.append(
            "FSDP2 requires caution=0 and cautious_weight_decay=0 for SFAdamW."
        )
    return tuple(limitations)


def describe(name: str) -> dict[str, object]:
    """Describe one optimizer facade without constructing it."""

    if not isinstance(name, str):
        raise TypeError("name must be a string")
    facades = _facades()
    try:
        requested = facades[name]
    except KeyError as error:
        choices = ", ".join(list_optimizers(include_aliases=True))
        raise ValueError(f"unknown optimizer {name!r}; choose one of: {choices}") from error

    alias_target = _alias_target(requested)
    canonical_name = name if alias_target is None else alias_target
    facade = facades[canonical_name]
    aliases = sorted(
        alias_name
        for alias_name, alias in facades.items()
        if _alias_target(alias) == canonical_name
    )
    algorithm = " ".join((inspect.getdoc(requested) or requested.__name__).split())
    boundary = algorithm.find(". ")
    if boundary >= 0:
        algorithm = algorithm[: boundary + 1]
    try:
        metadata = _OPTIMIZER_METADATA[canonical_name]
    except KeyError as error:
        raise RuntimeError(
            f"optimizer discovery metadata is missing {canonical_name!r}"
        ) from error
    defaults = {
        parameter_name: parameter.default
        for parameter_name, parameter in inspect.signature(facade).parameters.items()
        if parameter_name != "params" and parameter.default is not inspect.Parameter.empty
    }
    fsdp2_supported = _fsdp2_supported(facade.recipe)
    return {
        "name": name,
        "canonical_name": canonical_name,
        "algorithm": algorithm,
        "recipe": _recipe_structure(facade.recipe),
        "signature_defaults": defaults,
        "supported_dtypes": {
            "parameters": (
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            ),
            "state_storage": (
                "fp32/fp64",
                "bfloat16",
                "bfloat16+ECC8",
                "bfloat16+ECC16",
                "natural bool/int slots",
            ),
        },
        "distributed_modes": {
            "single_process": True,
            "DDP": True,
            "FSDP2": fsdp2_supported,
        },
        "routing": metadata["routing"],
        "lifecycle": metadata["lifecycle"],
        "storage_contract": dict(_SLAB_CONTRACT),
        "compile_behavior": metadata["compile_behavior"],
        "distributed_limitations": _distributed_limitations(
            canonical_name,
            facade.recipe,
            fsdp2_supported,
        ),
        "aliases": aliases,
        "gotcha": metadata["gotcha"],
    }


def _cpu_parameter(parameter: Tensor) -> torch.nn.Parameter:
    if not isinstance(parameter, Tensor):
        raise TypeError("params must contain tensors or parameter-group mappings")
    return torch.nn.Parameter(
        torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="cpu"),
        requires_grad=parameter.requires_grad,
    )


def _cpu_params(
    params: Iterable[Tensor] | Iterable[Mapping[str, object]],
) -> list[torch.nn.Parameter] | list[dict[str, object]]:
    supplied = list(params)
    if not any(isinstance(value, Mapping) for value in supplied):
        return [_cpu_parameter(parameter) for parameter in supplied]
    if not all(isinstance(value, Mapping) for value in supplied):
        raise TypeError("params must be all tensors or all parameter-group mappings")

    groups = []
    for supplied_group in supplied:
        group = dict(supplied_group)
        if "params" not in group:
            raise ValueError("parameter group must contain 'params'")
        group_params = group["params"]
        materialized = [group_params] if isinstance(group_params, Tensor) else list(group_params)
        group["params"] = [_cpu_parameter(parameter) for parameter in materialized]
        groups.append(group)
    return groups


def _resolve_facade(
    optimizer: str | type[HeavyBallOptimizer],
) -> type[HeavyBallOptimizer]:
    facades = _facades()
    if isinstance(optimizer, str):
        try:
            facade = facades[optimizer]
        except KeyError as error:
            raise ValueError(f"unknown optimizer {optimizer!r}") from error
    else:
        facade = optimizer
    if (
        not isinstance(facade, type)
        or not issubclass(facade, HeavyBallOptimizer)
        or facade is HeavyBallOptimizer
    ):
        raise TypeError("optimizer must be a HeavyBall optimizer facade or its public name")
    alias_target = _alias_target(facade)
    return facades[alias_target] if alias_target is not None else facade


def estimate_state_bytes(
    params: Iterable[Tensor] | Iterable[Mapping[str, object]],
    optimizer: str | type[HeavyBallOptimizer],
    *,
    storage_dtype: torch.dtype | str | None = None,
    ecc: int | str | None = None,
) -> int:
    """Estimate live transform/commit state storage, including ECC correction slabs.

    HeavyBall declares its full persistent state during construction, so this builds an isolated
    fake-CPU facade over shape/dtype-equivalent parameters and counts the same slabs as the benchmark.
    Fake tensors preserve routing, shapes, and dtypes without allocating parameter-sized storage.
    This never runs ``step()`` or mutates the supplied parameters.
    """

    facade = _resolve_facade(optimizer)
    cpu_params = []
    candidate = None
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            cpu_params = _cpu_params(params)
            candidate = facade(cpu_params, storage_dtype=storage_dtype, ecc=ecc)
            return sum(
                slab.numel() * slab.element_size()
                for engine in candidate._engines
                for group in engine.groups
                for slots in (
                    *group.states,
                    group.commit_state,
                    *group.state_corrections,
                    group.commit_corrections,
                )
                for slab in slots.values()
            )
    finally:
        candidate = None
        cpu_params.clear()
        gc.collect()


__all__ = ["describe", "estimate_state_bytes", "list_optimizers"]
