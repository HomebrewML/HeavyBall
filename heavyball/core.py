"""Static slab storage and one-whole-step compilation for HeavyBall 4.0."""

import math
import types
from collections import OrderedDict, namedtuple
from contextlib import ExitStack
from dataclasses import InitVar, dataclass, field
from numbers import Real
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ._inductor_config import STEP_COMPILE_OPTIONS
from .codecs import decode, encode
from .numerics import _wide, broadcast_leaf, stochastic_copy_
from .transforms import _STATE_PHILOX_ROUNDS, SHARD, Tempo, Whole

Transform = Callable[[Tensor, SimpleNamespace, Tensor, dict[str, Tensor], Tempo], tuple[Tensor, dict[str, Tensor], Tensor]]
Commit = Callable[[Tensor, Tensor, dict[str, Tensor], Tempo], tuple[Tensor, dict[str, Tensor]]]
EvalSwap = Callable[[Tensor, dict[str, Tensor], SimpleNamespace, bool], tuple[Tensor, dict[str, Tensor]]]

_HOST_ONLY_HYPERPARAMETERS = frozenset(("preconditioner_update_probability",))
_COMPILE_CACHE_MAX_SIZE = 128
_STEP_CODE_CACHE: OrderedDict[object, object] = OrderedDict()
_FSDP2_COMPILE_IDENTITIES: OrderedDict[tuple[object, ...], object] = OrderedDict()
_STEP_CODE_SERIAL = 0
_RNG_CHECKPOINT_KEY = ("heavyball_rng",)


def _bounded_cache_get_or_create(cache, key, factory):
    try:
        value = cache[key]
    except KeyError:
        value = factory()
        if len(cache) >= _COMPILE_CACHE_MAX_SIZE:
            if isinstance(cache, OrderedDict):
                cache.popitem(last=False)
            else:
                cache.pop(next(iter(cache)))
        cache[key] = value
    else:
        if isinstance(cache, OrderedDict):
            cache.move_to_end(key)
    return value


def clear_cache() -> None:
    """Clear HeavyBall's process-local compile identity caches."""

    _STEP_CODE_CACHE.clear()
    _FSDP2_COMPILE_IDENTITIES.clear()


def _keyed_compile_fn(fn, key):
    def create_code():
        global _STEP_CODE_SERIAL

        serial = _STEP_CODE_SERIAL
        _STEP_CODE_SERIAL += 1
        return fn.__code__.replace(co_name=f"{fn.__code__.co_name}__hb{serial}")

    code = _bounded_cache_get_or_create(_STEP_CODE_CACHE, key, create_code)
    return types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)


def _narrowed_dtype(dtype, storage_dtype):
    """Narrow floating state to storage_dtype for memory, but keep deliberately higher-precision (fp64)
    state -- e.g. PSGD/LATHER running_lower_bound stability scalars -- at its own dtype."""
    if storage_dtype is not None and dtype.is_floating_point and dtype is not torch.float64:
        return storage_dtype
    return dtype


def _slab_dtype(value: Tensor, storage_dtype: "torch.dtype | None") -> torch.dtype:
    """Narrow only floating-point state; flags and counters keep their natural dtype."""

    return _narrowed_dtype(value.dtype, storage_dtype)


@dataclass
class RefreshCadence:
    """Host-side cumulative-probability refresh selector, matching legacy cadence."""

    probability: Real | Callable[[int], Real] = 1.0
    step: int = 0
    cumulative: float = 0.0
    compensation: float = 0.0

    def next_step_type(self) -> str:
        self.step += 1
        probability = self.probability(self.step) if callable(self.probability) else self.probability
        if not isinstance(probability, Real) or not 0 <= probability <= 1:
            raise ValueError("preconditioner_update_probability must be a number in [0, 1]")
        adjusted = float(probability) - self.compensation
        previous = self.cumulative
        self.cumulative = previous + adjusted
        self.compensation = (self.cumulative - previous) - adjusted
        return "refresh" if int(self.cumulative) > int(previous) else "normal"


@dataclass(frozen=True)
class Recipe:
    chain: tuple[Transform, ...]
    commit: Commit
    defaults: Mapping[str, float]
    observations: tuple[str, ...] = ()
    clip_global_norm: float | None = None


@dataclass(frozen=True)
class ParamInfo:
    """Immutable parameter metadata available to build-time route predicates."""

    param: InitVar[Tensor]
    shape: tuple[int, ...] = field(init=False)
    ndim: int = field(init=False)
    dtype: torch.dtype = field(init=False)

    def __post_init__(self, param: Tensor) -> None:
        object.__setattr__(self, "shape", tuple(param.shape))
        object.__setattr__(self, "ndim", param.ndim)
        object.__setattr__(self, "dtype", param.dtype)


@dataclass(frozen=True)
class Route:
    when: Callable[[ParamInfo], bool]
    then: 'Recipe | Route'
    otherwise: 'Recipe | Route'


def _resolve_route(node: 'Recipe | Route', param_info: ParamInfo) -> Recipe:
    if isinstance(node, Route):
        matches = node.when(param_info)
        if not isinstance(matches, bool):
            raise TypeError("Route predicates must return a host bool")
        return _resolve_route(node.then if matches else node.otherwise, param_info)
    return node


def _collect_route_recipes(node: 'Recipe | Route') -> tuple[Recipe, ...]:
    if isinstance(node, Route):
        return (*_collect_route_recipes(node.then), *_collect_route_recipes(node.otherwise))
    return (node,)


@dataclass
class Group:
    params: tuple[Tensor, ...]
    param_slab: Tensor
    grad_slab: Tensor
    observed: Tensor
    age: Tensor
    base_seed: Tensor
    leaf_indices: Tensor
    element_offset: int
    states: tuple[dict[str, Tensor], ...]
    state_corrections: tuple[dict[str, Tensor], ...]
    state_element_offsets: tuple[dict[str, int], ...]
    state_owner_ranges: tuple[tuple[int, int] | None, ...]
    commit_state: dict[str, Tensor]
    commit_corrections: dict[str, Tensor]
    commit_element_offsets: dict[str, int]
    observations: SimpleNamespace
    hyper: SimpleNamespace
    step: Tensor
    recipe: Recipe
    param_group_id: int
    observed_cache: tuple[bool, ...] | None = None


def _callable_plan_abi(function: Callable) -> object:
    identity = getattr(function, "__name__", None)
    if identity is None:
        identity = id(function)
    config = getattr(function, "config", None)
    normalized = None if config is None else tuple(sorted((name, repr(value)) for name, value in config.items()))
    return identity if normalized is None else (identity, normalized)


def _plan_abi(
    groups: Sequence[Group],
    *,
    clip_norm: Tensor | None = None,
    eval_swap: bool = False,
    compile_scope: object | None = None,
) -> tuple[object, ...]:
    aliases: dict[int, int] = {}

    def tensor(value: Tensor) -> tuple[object, ...]:
        alias = aliases.setdefault(id(value), len(aliases))
        value_type = type(value)
        return (
            alias,
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            tuple(value.stride()),
            str(value.layout),
            f"{value_type.__module__}.{value_type.__qualname__}",
            value.requires_grad,
        )

    def namespace(items) -> tuple[tuple[object, ...], ...]:
        return tuple((name, *tensor(value)) for name, value in items)

    def step_group(group: Group) -> tuple[object, ...]:
        return (
            ("param_slab", tensor(group.param_slab)),
            ("observations", namespace(vars(group.observations).items())),
            ("observed", tensor(group.observed)),
            ("age", tensor(group.age)),
            ("base_seed", tensor(group.base_seed)),
            ("leaf_indices", tensor(group.leaf_indices)),
            ("element_offset", group.element_offset),
            ("states", tuple(namespace(state.items()) for state in group.states)),
            ("state_corrections", tuple(namespace(state.items()) for state in group.state_corrections)),
            (
                "state_element_offsets",
                tuple(tuple(offsets.items()) for offsets in group.state_element_offsets),
            ),
            ("commit_state", namespace(group.commit_state.items())),
            ("commit_corrections", namespace(group.commit_corrections.items())),
            ("commit_element_offsets", tuple(group.commit_element_offsets.items())),
            ("hyper", namespace(vars(group.hyper).items())),
            ("step", tensor(group.step)),
            ("chain", tuple(_callable_plan_abi(transform) for transform in group.recipe.chain)),
            ("commit", _callable_plan_abi(group.recipe.commit)),
        )

    def swap_group(group: Group) -> tuple[object, ...]:
        return (
            ("param_slab", tensor(group.param_slab)),
            ("commit_state", namespace(group.commit_state.items())),
            ("commit_corrections", namespace(group.commit_corrections.items())),
            ("hyper", namespace(vars(group.hyper).items())),
            ("commit_element_offsets", tuple(group.commit_element_offsets.items())),
            ("step", tensor(group.step)),
            ("age", tensor(group.age)),
            ("base_seed", tensor(group.base_seed)),
            ("leaf_indices", tensor(group.leaf_indices)),
            ("element_offset", group.element_offset),
            ("eval_swap", _callable_plan_abi(group.recipe.commit.eval_swap)),
        )

    selected_groups = (
        tuple(group for group in groups if getattr(group.recipe.commit, "eval_swap", None) is not None)
        if eval_swap else tuple(groups)
    )
    return (
        ("compile_scope", compile_scope),
        ("clip_norm", None if clip_norm is None else tensor(clip_norm)),
        ("groups", tuple((swap_group if eval_swap else step_group)(group) for group in selected_groups)),
    )


@dataclass
class _ObservationBinding:
    declared: frozenset[str]
    shape: tuple[int, ...]
    produced: set[str] = field(default_factory=set)


_OBSERVATION_BINDING = "_heavyball_observation_binding"


@dataclass(frozen=True)
class FSDP2RegroupPlan:
    """Static balanced packing metadata for one FSDP2 whole-leaf bucket."""

    param_keys: tuple[str, ...]
    global_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    world_size: int
    mesh_ranks: tuple[int, ...]
    placements: tuple[tuple[str, int], ...]
    owner_ranges: tuple[tuple[int, int], ...]
    pack_indices: tuple[int, ...]
    unpack_indices: tuple[int, ...]
    collective_order: tuple[int, int]
    specialization: tuple[object, ...] = ()

    @classmethod
    def create(
        cls,
        param_keys: Sequence[str],
        global_shape: Sequence[int],
        storage_shape: Sequence[int],
        world_size: int,
        *,
        mesh_ranks: Sequence[int] | None = None,
        placements: Sequence[tuple[str, int]] = (("Shard", 0),),
        collective_index: int = 0,
        specialization: tuple[object, ...] = (),
    ) -> "FSDP2RegroupPlan":
        keys = tuple(param_keys)
        logical = tuple(global_shape)
        storage = tuple(storage_shape)
        if not keys:
            raise ValueError("an FSDP2 regroup plan requires at least one parameter")
        if world_size < 1:
            raise ValueError("an FSDP2 regroup plan requires a positive world size")
        if not logical or len(storage) != len(logical):
            raise ValueError("FSDP2 regroup requires non-scalar parameters with matching storage rank")
        padded_rows = math.ceil(logical[0] / world_size)
        if storage != (padded_rows, *logical[1:]):
            raise ValueError(
                "FSDP2 local storage must be the globally-derived padded dim-0 shard"
            )
        count = len(keys)
        from torch.distributed.tensor import Shard

        owner_counts_and_offsets = tuple(
            Shard.local_shard_size_and_offset(count, world_size, rank)
            for rank in range(world_size)
        )
        owner_counts = tuple(owner_count for owner_count, _ in owner_counts_and_offsets)
        owner_ranges = tuple(
            (offset, offset + owner_count)
            for owner_count, offset in owner_counts_and_offsets
        )
        max_owned = max(owner_counts)
        sentinel = count
        packed = []
        for start, stop in owner_ranges:
            packed.extend(range(start, stop))
            packed.extend((sentinel,) * (max_owned - (stop - start)))
        unpacked = [0] * count
        for packed_index, param_index in enumerate(packed):
            if param_index != sentinel:
                unpacked[param_index] = packed_index
        ranks = tuple(range(world_size)) if mesh_ranks is None else tuple(mesh_ranks)
        if len(ranks) != world_size:
            raise ValueError("FSDP2 mesh rank count does not match its world size")
        return cls(
            keys,
            logical,
            storage,
            world_size,
            ranks,
            tuple(placements),
            tuple(owner_ranges),
            tuple(packed),
            tuple(unpacked),
            (2 * collective_index, 2 * collective_index + 1),
            tuple(specialization),
        )

    @property
    def max_owned(self) -> int:
        return len(self.pack_indices) // self.world_size

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "fsdp2-regroup-v2",
            self.global_shape,
            self.storage_shape,
            self.world_size,
            self.mesh_ranks,
            self.placements,
            self.owner_ranges,
            self.pack_indices,
            self.unpack_indices,
            self.specialization,
        )


def _plain_view_matches(actual: Tensor, expected: Tensor) -> bool:
    return (
        actual.data_ptr() == expected.data_ptr()
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.storage_offset() == expected.storage_offset()
        and actual.dtype == expected.dtype
        and actual.device == expected.device
    )


def _storage_view_matches(actual: Tensor, expected: Tensor) -> bool:
    return (
        actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.storage_offset() == expected.storage_offset()
        and actual.dtype == expected.dtype
        and actual.device == expected.device
    )


def _storage_view_identity(value: object) -> tuple[object, ...] | None:
    """Return the storage and view metadata whose change requires binding validation."""

    if not isinstance(value, Tensor):
        return None
    return (
        value.untyped_storage()._cdata,
        tuple(value.shape),
        value.stride(),
        value.storage_offset(),
        value.dtype,
        value.device,
    )


def _dense_storage_span(value: Tensor) -> tuple[int, int] | None:
    if value.numel() == 0:
        start = value.storage_offset() * value.element_size()
        return start, start
    expected_stride = 1
    for stride, size in sorted(
        (stride, size) for size, stride in zip(value.shape, value.stride(), strict=True) if size > 1
    ):
        if stride != expected_stride:
            return None
        expected_stride *= size
    start = value.storage_offset() * value.element_size()
    return start, start + value.numel() * value.element_size()


def _storage_element_starts(value: Tensor) -> tuple[int, ...]:
    offsets = {value.storage_offset()}
    for size, stride in zip(value.shape, value.stride(), strict=True):
        offsets = {offset + index * stride for offset in offsets for index in range(size)}
    element_size = value.element_size()
    return tuple(sorted(offset * element_size for offset in offsets))


def _storage_views_overlap(first: Tensor, second: Tensor) -> bool:
    if first.numel() == 0 or second.numel() == 0:
        return False
    first_span = _dense_storage_span(first)
    second_span = _dense_storage_span(second)
    if first_span is not None and second_span is not None:
        return first_span[0] < second_span[1] and second_span[0] < first_span[1]
    if first_span is not None:
        return any(
            start < first_span[1] and first_span[0] < start + second.element_size()
            for start in _storage_element_starts(second)
        )
    if second_span is not None:
        return any(
            start < second_span[1] and second_span[0] < start + first.element_size()
            for start in _storage_element_starts(first)
        )

    first_starts = _storage_element_starts(first)
    second_starts = _storage_element_starts(second)
    first_index = second_index = 0
    while first_index < len(first_starts) and second_index < len(second_starts):
        first_start = first_starts[first_index]
        second_start = second_starts[second_index]
        if first_start < second_start + second.element_size() and second_start < first_start + first.element_size():
            return True
        if first_start + first.element_size() <= second_start:
            first_index += 1
        else:
            second_index += 1
    return False


@dataclass(frozen=True)
class PlainBinding:
    """Own the ordinary parameter/gradient aliases at the slab boundary."""

    param: Tensor

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return tuple(self.param.shape)

    def validate_param(self) -> None:
        if not self.param.data.is_contiguous():
            raise ValueError(
                f"parameter of shape {tuple(self.param.shape)} has non-contiguous strides "
                f"{tuple(self.param.stride())}; heavyball needs C-contiguous parameters. Make it contiguous "
                "before constructing the optimizer, e.g. "
                "nn.Parameter(p.detach().contiguous(), requires_grad=p.requires_grad)."
            )

    def snapshot(self) -> tuple[Tensor, Tensor | None]:
        return self.param.data, self.param.grad

    def restore(self, snapshot: tuple[Tensor, Tensor | None]) -> None:
        self.param.data, self.param.grad = snapshot

    def bind_param(self, row: Tensor) -> None:
        self.validate_param()
        row.copy_(self.param.detach())
        self.param.data = row

    def bind_grad(self, row: Tensor) -> None:
        self.param.grad = row

    def initializer_reference(self, row: Tensor) -> Tensor:
        del row
        return self.param

    def storage_identity(self) -> tuple[object, ...]:
        return _storage_view_identity(self.param), _storage_view_identity(self.param.grad)

    def validate(self, param_row: Tensor, grad_row: Tensor) -> None:
        if self.param.grad is None or not _plain_view_matches(self.param.grad, grad_row):
            raise ValueError(
                f"gradient for parameter {tuple(self.param.shape)} is no longer slab-bound. Write gradients in "
                "place (loss.backward() or p.grad.copy_(g)), not by reassigning p.grad; this also happens if the "
                "parameter's storage changed after construction (e.g. model.to(...) after building the optimizer) "
                "-- place final device/dtype before constructing heavyball."
            )
        if not _plain_view_matches(self.param.data, param_row):
            raise ValueError(
                f"weights for parameter {tuple(self.param.shape)} are no longer slab-bound. Update them in place "
                "(p.data.copy_(w)), not by reassigning p.data; this also happens if the parameter's storage "
                "changed after construction (e.g. model.to(...) after building the optimizer) -- place final "
                "device/dtype before constructing heavyball."
            )


class FSDP2Binding:
    """Own the version-sensitive aliases from an FSDPParam to local slab rows."""

    _FSDP_PARAM_FIELDS = (
        "_sharded_param_data",
        "fsdp_placement",
        "padded_sharded_param_size",
        "sharded_param",
        "sharded_size",
        "sharded_state",
    )

    def __init__(self, fsdp_param) -> None:
        from torch.distributed.tensor import DTensor, Replicate, Shard

        missing = [name for name in self._FSDP_PARAM_FIELDS if not hasattr(fsdp_param, name)]
        if missing:
            self._incompatible(f"FSDPParam is missing {', '.join(missing)}")
        param = fsdp_param.sharded_param
        missing = [name for name in ("_local_tensor", "device_mesh", "placements") if not hasattr(param, name)]
        if missing:
            self._incompatible(f"the sharded DTensor parameter is missing {', '.join(missing)}")
        if not isinstance(param, DTensor):
            self._incompatible(f"sharded_param has type {type(param).__name__}, not DTensor")
        if not isinstance(fsdp_param._sharded_param_data, Tensor):
            self._incompatible("_sharded_param_data is not a Tensor")
        if not isinstance(param._local_tensor, Tensor):
            self._incompatible("DTensor _local_tensor is not a Tensor")
        if getattr(fsdp_param.sharded_state, "name", None) != "SHARDED":
            raise ValueError("HeavyBall fsdp2() must be constructed while the model parameters are resharded")
        placements = tuple(param.placements)
        mesh = param.device_mesh
        mesh_missing = [
            name for name in ("get_group", "get_local_rank", "mesh", "ndim", "size") if not hasattr(mesh, name)
        ]
        if mesh_missing:
            self._incompatible(f"DeviceMesh is missing {', '.join(mesh_missing)}")
        one_dimensional = (
            mesh.ndim == 1
            and len(placements) == 1
            and isinstance(placements[0], Shard)
            and placements[0].dim == 0
        )
        hsdp = (
            mesh.ndim == 2
            and getattr(mesh, "mesh_dim_names", None) == ("replicate", "shard")
            and len(placements) == 2
            and isinstance(placements[0], Replicate)
            and isinstance(placements[1], Shard)
            and placements[1].dim == 0
        )
        if (
            not (one_dimensional or hsdp)
            or not isinstance(fsdp_param.fsdp_placement, Shard)
            or fsdp_param.fsdp_placement.dim != 0
        ):
            raise ValueError(
                "HeavyBall fsdp2() requires a 1D FSDP mesh or a 2D replicate x shard HSDP mesh "
                "sharded on parameter dim 0"
            )
        storage_shape = tuple(fsdp_param.padded_sharded_param_size)
        valid_shape = tuple(fsdp_param.sharded_size)
        global_shape = tuple(param.shape)
        if not global_shape:
            raise ValueError("HeavyBall fsdp2() does not support scalar parameters sharded on dim 0")
        try:
            shard_mesh = mesh if one_dimensional else mesh["shard"]
            world_size = shard_mesh.size()
            mesh_rank = shard_mesh.get_local_rank()
            mesh_ranks = tuple(int(rank) for rank in shard_mesh.mesh.reshape(-1).tolist())
            process_group = shard_mesh.get_group()
            replicate_process_group = None if one_dimensional else mesh["replicate"].get_group()
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            self._incompatible(f"could not resolve the DeviceMesh layout ({error})")
        if world_size < 1 or not 0 <= mesh_rank < world_size or len(mesh_ranks) != world_size:
            self._incompatible("DeviceMesh size, ranks, and local rank are inconsistent")
        padded_rows = math.ceil(global_shape[0] / world_size)
        expected_valid_length = min(max(global_shape[0] - mesh_rank * padded_rows, 0), padded_rows)
        if (
            len(global_shape) != param.ndim
            or len(storage_shape) != param.ndim
            or len(valid_shape) != param.ndim
            or storage_shape != (padded_rows, *global_shape[1:])
            or storage_shape[1:] != valid_shape[1:]
            or valid_shape[0] > storage_shape[0]
            or valid_shape[0] != expected_valid_length
            or tuple(param._local_tensor.shape) != valid_shape
        ):
            self._incompatible(
                "FSDPParam padded/local shapes do not match the expected dim-0 padded shard layout"
            )
        self.fsdp_param = fsdp_param
        self.param = param
        self.global_shape = global_shape
        self.storage_shape = storage_shape
        self.valid_length = valid_shape[0]
        self.mesh = mesh
        self.shard_mesh = shard_mesh
        self.mesh_rank = mesh_rank
        self.mesh_ranks = mesh_ranks
        self.world_size = world_size
        self.process_group = process_group
        self.replicate_process_group = replicate_process_group
        self.placements = placements
        self.placement_identity = tuple(
            (type(placement).__name__, getattr(placement, "dim", -1)) for placement in placements
        )
        self.sharded_state = fsdp_param.sharded_state
        self._dtensor_type = DTensor

    @staticmethod
    def _incompatible(detail: str) -> None:
        raise RuntimeError(
            "HeavyBall fsdp2() is incompatible with this PyTorch FSDP2 implementation: "
            f"{detail}. Expected the torch 2.13 FSDPParam storage aliases; no copy fallback is available."
        )

    @classmethod
    def functional_all_to_all_single(cls):
        try:
            from torch.distributed import _functional_collectives as funcol
        except (ImportError, RuntimeError) as error:
            cls._incompatible(f"functional collectives could not be imported ({error})")
        collective = getattr(funcol, "all_to_all_single", None)
        if not callable(collective):
            cls._incompatible("torch.distributed._functional_collectives.all_to_all_single is unavailable")
        return collective

    def bucket_identity(self, matrix_plan: tuple[object, ...]) -> tuple[object, ...]:
        return (
            self.global_shape,
            self.storage_shape,
            self.world_size,
            self.mesh_ranks,
            id(self.mesh),
            id(self.process_group),
            self.placement_identity,
            matrix_plan,
        )

    def _valid_view(self, row: Tensor) -> Tensor:
        return row.narrow(0, 0, self.valid_length)

    def snapshot(self) -> tuple[Tensor, Tensor, Tensor | None]:
        return self.fsdp_param._sharded_param_data, self.param._local_tensor, self.param.grad

    def restore(self, snapshot: tuple[Tensor, Tensor, Tensor | None]) -> None:
        sharded_data, local_tensor, grad = snapshot
        self.fsdp_param._sharded_param_data = sharded_data
        self.param._local_tensor = local_tensor
        self.param.grad = grad

    def bind_param(self, row: Tensor) -> None:
        valid = self._valid_view(row)
        row.zero_()
        valid.copy_(self.param._local_tensor.detach())
        try:
            self.fsdp_param._sharded_param_data = row.reshape(-1)
            self.param._local_tensor = valid
        except (AttributeError, RuntimeError, TypeError) as error:
            self._incompatible(f"could not repoint the FSDPParam storage aliases ({error})")

    def bind_grad(self, row: Tensor) -> None:
        valid = self._valid_view(row)
        try:
            self.param.grad = self._dtensor_type.from_local(
                valid,
                self.mesh,
                self.placements,
                run_check=False,
                shape=self.param.shape,
                stride=self.param.stride(),
            )
        except (AttributeError, RuntimeError, TypeError) as error:
            self._incompatible(f"could not prebind the sharded DTensor gradient ({error})")

    def initializer_reference(self, row: Tensor) -> Tensor:
        return row

    def storage_identity(self) -> tuple[object, ...]:
        grad = self.param.grad
        return (
            id(self.fsdp_param.sharded_state),
            id(self.fsdp_param.sharded_param),
            _storage_view_identity(self.fsdp_param._sharded_param_data),
            _storage_view_identity(self.param._local_tensor),
            type(grad),
            id(getattr(grad, "device_mesh", None)),
            tuple(
                (type(placement), getattr(placement, "dim", None))
                for placement in getattr(grad, "placements", ())
            ),
            _storage_view_identity(getattr(grad, "_local_tensor", None)),
        )

    def validate(self, param_row: Tensor, grad_row: Tensor) -> None:
        valid_param = self._valid_view(param_row)
        valid_grad = self._valid_view(grad_row)
        if self.fsdp_param.sharded_state is not self.sharded_state:
            raise ValueError(
                f"FSDP2 parameter {tuple(self.param.shape)} is not resharded; HeavyBall requires "
                "set_reshard_after_backward(True) before optimizer.step()."
            )
        if (
            self.fsdp_param.sharded_param is not self.param
            or not isinstance(self.fsdp_param._sharded_param_data, Tensor)
            or not _storage_view_matches(self.fsdp_param._sharded_param_data, param_row.reshape(-1))
            or not isinstance(self.param._local_tensor, Tensor)
            or not _storage_view_matches(self.param._local_tensor, valid_param)
        ):
            raise ValueError(
                f"weights for FSDP2 parameter {tuple(self.param.shape)} are no longer slab-bound; "
                "recreate the optimizer after assigning model state or converting its dtype/device."
            )
        grad = self.param.grad
        if (
            grad is None
            or not isinstance(grad, self._dtensor_type)
            or grad.device_mesh != self.mesh
            or tuple(grad.placements) != self.placements
            or not isinstance(grad._local_tensor, Tensor)
            or not _storage_view_matches(grad._local_tensor, valid_grad)
        ):
            raise ValueError(
                f"gradient for FSDP2 parameter {tuple(self.param.shape)} is no longer slab-bound; "
                "use optimizer.zero_grad() and do not reassign or clear the DTensor gradient."
            )


def fsdp2_bindings(model) -> tuple[FSDP2Binding, ...]:
    """Resolve every FSDPParam from a fully-sharded model and fail closed on API drift."""

    from torch.distributed.fsdp import fully_shard

    if not hasattr(model, "modules") or not callable(model.modules):
        raise TypeError("fsdp2() requires the fully_shard'd model, not a parameter iterable")
    fsdp_params = []
    seen = set()
    found_state = False
    for module in model.modules():
        state = fully_shard.state(module)
        if state is None:
            continue
        found_state = True
        if not hasattr(state, "_fsdp_param_groups"):
            FSDP2Binding._incompatible("FSDPState is missing _fsdp_param_groups")
        for group in state._fsdp_param_groups:
            if not hasattr(group, "fsdp_params"):
                FSDP2Binding._incompatible("FSDPParamGroup is missing fsdp_params")
            for fsdp_param in group.fsdp_params:
                if id(fsdp_param) not in seen:
                    seen.add(id(fsdp_param))
                    fsdp_params.append(fsdp_param)
    if not found_state or not fsdp_params:
        raise ValueError("fsdp2() requires a model already wrapped by torch.distributed.fsdp.fully_shard")
    bindings = tuple(FSDP2Binding(fsdp_param) for fsdp_param in fsdp_params)
    by_param_id = {id(binding.param): binding for binding in bindings}
    unmanaged = [
        name for name, param in model.named_parameters() if param.requires_grad and id(param) not in by_param_id
    ]
    if unmanaged:
        names = ", ".join(repr(name) for name in unmanaged)
        raise ValueError(f"fsdp2() requires every trainable model parameter to be fully sharded; unmanaged: {names}")
    return bindings


def _recipe_manifest(recipe: Recipe) -> tuple[object, ...]:
    def transform_identity(transform) -> tuple[object, ...]:
        config = getattr(transform, "config", None)
        normalized = () if config is None else tuple(sorted((name, repr(value)) for name, value in config.items()))
        return transform.__name__, normalized

    return (
        tuple(transform_identity(transform) for transform in recipe.chain),
        transform_identity(recipe.commit),
        tuple(sorted((name, repr(value)) for name, value in recipe.defaults.items())),
        tuple(recipe.observations),
        recipe.clip_global_norm,
    )


def _validate_fsdp2_manifest(
    assignments: Sequence[tuple[Tensor, Recipe]],
    binding_by_param_id: Mapping[int, FSDP2Binding],
    param_keys: Mapping[int, str],
    buckets: Mapping[tuple[object, ...], Sequence[Tensor]],
    bucket_recipes: Mapping[tuple[object, ...], Recipe],
    regroup_plans: Mapping[tuple[object, ...], tuple[FSDP2RegroupPlan, ...]],
) -> tuple[object, ...]:
    import torch.distributed as dist

    bindings = tuple(binding_by_param_id[id(param)] for param, _ in assignments)
    first = bindings[0]
    if any(
        binding.mesh_ranks != first.mesh_ranks
        or binding.world_size != first.world_size
        or binding.process_group is not first.process_group
        for binding in bindings[1:]
    ):
        raise ValueError("HeavyBall fsdp2() requires every parameter to use one common DeviceMesh process group")
    if not dist.is_available() or not dist.is_initialized():
        FSDP2Binding._incompatible("torch.distributed is not initialized")
    parameter_manifest = tuple(
        (
            param_keys[id(param)],
            _recipe_manifest(recipe),
            binding_by_param_id[id(param)].global_shape,
            binding_by_param_id[id(param)].storage_shape,
            binding_by_param_id[id(param)].mesh_ranks,
            binding_by_param_id[id(param)].placement_identity,
        )
        for param, recipe in assignments
    )
    bucket_manifest = tuple(
        (
            tuple(param_keys[id(param)] for param in leaves),
            _recipe_manifest(bucket_recipes[key]),
            binding_by_param_id[id(leaves[0])].global_shape,
            tuple(plan.owner_ranges for plan in regroup_plans.get(key, ())),
            tuple(plan.collective_order for plan in regroup_plans.get(key, ())),
            tuple(plan.specialization for plan in regroup_plans.get(key, ())),
            tuple(("all_to_all_single", "all_to_all_single") for _ in regroup_plans.get(key, ())),
        )
        for key, leaves in buckets.items()
    )
    manifest = ("heavyball-fsdp2-construction-v1", parameter_manifest, bucket_manifest)
    gathered = [None] * first.world_size
    try:
        dist.all_gather_object(gathered, manifest, group=first.process_group)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        FSDP2Binding._incompatible(f"could not validate the cross-rank construction manifest ({error})")
    disagreements = tuple(rank for rank, candidate in enumerate(gathered) if candidate != manifest)
    if disagreements:
        raise ValueError(
            "HeavyBall fsdp2() construction manifest differs across mesh ranks "
            f"{disagreements}; parameters were not storage-bound"
        )
    return manifest


def _callable_distributed_scope(fn: Callable) -> str | Whole:
    scope = getattr(fn, "distributed_scope", SHARD)
    if scope == SHARD or isinstance(scope, Whole):
        return scope
    raise ValueError(f"{fn.__name__}.distributed_scope must be SHARD or Whole(...)")


def _callable_identity(fn: Callable) -> tuple[object, ...]:
    config = getattr(fn, "config", None)
    normalized = () if config is None else tuple(sorted((name, repr(value)) for name, value in config.items()))
    scope = _callable_distributed_scope(fn)
    scope_identity = "shard" if scope == SHARD else ("whole", scope.inputs)
    return fn.__name__, normalized, scope_identity


@dataclass(frozen=True)
class _FSDP2WholeSegment:
    start: int
    stop: int
    indexed_transforms: tuple[tuple[int, Transform], ...]
    terminal_commit: bool = False

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "fsdp2-whole-segment-v1",
            self.start,
            self.stop,
            tuple((index, _callable_identity(transform)) for index, transform in self.indexed_transforms),
            self.terminal_commit,
        )


def _fsdp2_scope_segments(recipe: Recipe) -> tuple[_FSDP2WholeSegment, ...]:
    segments = []
    index = 0
    while index < len(recipe.chain):
        if _callable_distributed_scope(recipe.chain[index]) == SHARD:
            index += 1
            continue
        start = index
        while index < len(recipe.chain) and _callable_distributed_scope(recipe.chain[index]) != SHARD:
            index += 1
        segments.append(
            _FSDP2WholeSegment(
                start,
                index,
                tuple((value, recipe.chain[value]) for value in range(start, index)),
            )
        )
    if _callable_distributed_scope(recipe.commit) != SHARD:
        if segments and segments[-1].stop == len(recipe.chain):
            previous = segments[-1]
            segments[-1] = _FSDP2WholeSegment(
                previous.start, previous.stop, previous.indexed_transforms, terminal_commit=True
            )
        else:
            segments.append(_FSDP2WholeSegment(len(recipe.chain), len(recipe.chain), (), terminal_commit=True))
    return tuple(segments)


def _fsdp2_scope_identity(recipe: Recipe) -> tuple[object, ...]:
    return (
        "fsdp2-callable-scope-v1",
        tuple((index, _callable_identity(transform)) for index, transform in enumerate(recipe.chain)),
        ("commit", _callable_identity(recipe.commit)),
        tuple(segment.identity for segment in _fsdp2_scope_segments(recipe)),
    )


def fsdp2_recipe_scope_supported(recipe_or_route: Recipe | Route) -> bool:
    for recipe in _collect_route_recipes(recipe_or_route):
        for fn in (*recipe.chain, recipe.commit):
            scope = _callable_distributed_scope(fn)
            if scope == SHARD and not getattr(fn, "distributed_shard_separable", True):
                return False
    return True


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(tuple(shape)):
        strides.append(stride)
        stride *= max(size, 1)
    return tuple(reversed(strides))


class _FSDP2RegroupKernel:
    """One generic owner regroup/scatter pair for arbitrary whole-segment payloads."""

    def __init__(self, plan: FSDP2RegroupPlan, binding: FSDP2Binding) -> None:
        self.plan = plan
        self.binding = binding
        self.collective = FSDP2Binding.functional_all_to_all_single()
        self.world_size = plan.world_size
        self.mesh = binding.shard_mesh
        self.padded_rows = plan.storage_shape[0]
        self.logical_rows = plan.global_shape[0]
        self.tail_shape = plan.storage_shape[1:]
        self.storage_numel = math.prod(plan.storage_shape)
        self.max_owned = plan.max_owned
        self.owner_start, self.owner_stop = plan.owner_ranges[binding.mesh_rank]
        self.owner_count = self.owner_stop - self.owner_start
        self.permute = (1, 0, *range(2, 2 + len(plan.storage_shape)))
        self.pack_indices = torch.tensor(plan.pack_indices, dtype=torch.int64, device=binding.param.device)
        self.unpack_indices = torch.tensor(plan.unpack_indices, dtype=torch.int64, device=binding.param.device)

    def regroup(self, *slabs: Tensor) -> Tensor:
        payload = torch.cat(tuple(_wide(slab).reshape(slab.shape[0], -1) for slab in slabs), dim=1)
        padded = torch.cat((payload, torch.zeros_like(payload.narrow(0, 0, 1))), dim=0)
        return self.collective(padded.index_select(0, self.pack_indices), None, None, self.mesh)

    def owner_view(self, received: Tensor, payload_index: int) -> Tensor:
        if not self.owner_count:
            return received.new_empty((0, *self.plan.global_shape))
        piece = received.narrow(1, payload_index * self.storage_numel, self.storage_numel)
        return (
            piece.reshape(self.world_size, self.max_owned, *self.plan.storage_shape)
            .permute(self.permute)
            .reshape(self.max_owned, self.world_size * self.padded_rows, *self.tail_shape)
            .narrow(0, 0, self.owner_count)
            .narrow(1, 0, self.logical_rows)
        )

    def scatter(self, whole: Tensor, live: Tensor) -> tuple[Tensor, Tensor]:
        padded = whole.new_zeros(
            (self.max_owned, self.world_size * self.padded_rows, *self.tail_shape)
        )
        if self.owner_count:
            padded.narrow(0, 0, self.owner_count).narrow(1, 0, self.logical_rows).copy_(whole)
        inverse = (
            padded.reshape(self.max_owned, self.world_size, self.padded_rows, *self.tail_shape)
            .permute(self.permute)
            .reshape(self.world_size * self.max_owned, self.storage_numel)
        )
        padded_live = live.new_zeros((self.max_owned,))
        if self.owner_count:
            padded_live.narrow(0, 0, self.owner_count).copy_(live)
        live_payload = (
            padded_live.reshape(1, self.max_owned)
            .expand(self.world_size, self.max_owned)
            .reshape(self.world_size * self.max_owned, 1)
            .to(dtype=inverse.dtype)
        )
        received = self.collective(torch.cat((inverse, live_payload), dim=1), None, None, self.mesh)
        local = received.index_select(0, self.unpack_indices)
        update = local.narrow(1, 0, self.storage_numel).reshape(
            len(self.plan.param_keys), *self.plan.storage_shape
        )
        return update, local.select(1, self.storage_numel) != 0


def _make_owner_dtensor(local: Tensor, binding: FSDP2Binding, global_shape: tuple[int, ...]) -> Tensor:
    from torch.distributed.tensor import DTensor, Shard

    return DTensor.from_local(
        local,
        binding.shard_mesh,
        (Shard(0),),
        run_check=False,
        shape=global_shape,
        stride=_contiguous_stride(global_shape),
    )


def _local_checkpoint_leaf(
    slab: Tensor,
    index: int,
    owner_range: tuple[int, int] | None = None,
) -> Tensor | None:
    """Return the directly-owned leaf view for a checkpoint slot, if this rank owns it."""

    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(slab, DTensor):
        return slab[index]
    if (
        slab.device_mesh.ndim != 1
        or len(slab.placements) != 1
        or not isinstance(slab.placements[0], Shard)
        or slab.placements[0].dim != 0
    ):
        raise ValueError("owner-whole checkpoint state must use a one-dimensional Shard(0) DTensor")
    if owner_range is None:
        raise ValueError("owner-whole checkpoint state is missing its owner range")
    owner_start, owner_stop = owner_range
    local = slab.to_local()
    if not 0 <= owner_start <= owner_stop <= slab.shape[0] or local.shape[0] != owner_stop - owner_start:
        raise ValueError("owner-whole checkpoint state does not match its DTensor owner shard")
    if owner_start <= index < owner_stop:
        return local[index - owner_start]
    return None


def _checkpoint_leaf(slab: Tensor, index: int, owner_range: tuple[int, int] | None = None) -> Tensor:
    leaf = _local_checkpoint_leaf(slab, index, owner_range)
    if leaf is None:
        return torch.empty((0, *slab.shape[1:]), dtype=slab.dtype)
    return leaf.detach().cpu().clone()


def _checkpoint_source(value: object) -> object:
    """Unwrap replicated DTensor rows emitted by the former owner-whole checkpoint path."""

    from torch.distributed.tensor import DTensor, Replicate

    if isinstance(value, DTensor):
        if any(not isinstance(placement, Replicate) for placement in value.placements):
            return None
        return value.to_local()
    return value


def _validate_owner_initial_states(
    initial_states: tuple[dict[str, Tensor], ...],
    binding: FSDP2Binding,
) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    import torch.distributed as dist

    if initial_states:
        first = initial_states[0]
        for candidate in initial_states[1:]:
            if candidate.keys() != first.keys():
                raise ValueError("owner initializer returned incompatible state keys within a bucket")
            for name, value in candidate.items():
                if value.shape != first[name].shape or value.dtype != first[name].dtype:
                    raise ValueError("owner initializer returned incompatible state shape or dtype within a bucket")
        local_schema = tuple(sorted((name, tuple(value.shape), value.dtype) for name, value in first.items()))
    else:
        local_schema = None
    gathered = [None] * binding.world_size
    try:
        dist.all_gather_object(gathered, local_schema, group=binding.process_group)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        FSDP2Binding._incompatible(f"could not validate the owner-state slot schema ({error})")
    schemas = tuple(schema for schema in gathered if schema is not None)
    if not schemas or any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("owner initializer returned incompatible state schema across ranks")
    return schemas[0]


def _make_fsdp2_whole_segment_transform(
    recipe: Recipe,
    segment: _FSDP2WholeSegment,
    plan: FSDP2RegroupPlan,
    binding: FSDP2Binding,
) -> Transform:
    from torch.distributed.tensor import DTensor

    kernel = _FSDP2RegroupKernel(plan, binding)
    input_names = ["update"]
    for _, transform in segment.indexed_transforms:
        scope = _callable_distributed_scope(transform)
        for name in scope.inputs:
            if name not in input_names:
                input_names.append(name)
    if segment.terminal_commit:
        for name in ("param", "obs.grad"):
            if name not in input_names:
                input_names.append(name)
    observation_names = tuple(
        name.removeprefix("obs.") for name in input_names if name.startswith("obs.")
    )
    owner_observation_type = namedtuple("OwnerObservations", observation_names)

    def resolve_input(name: str, update: Tensor, obs: SimpleNamespace, param: Tensor) -> Tensor:
        if name == "update":
            return update
        if name == "param":
            return param
        if name.startswith("obs."):
            return getattr(obs, name.removeprefix("obs."))
        raise ValueError(f"unsupported whole-segment input {name!r}")

    def wrap_candidate(target: Tensor, local: Tensor) -> Tensor:
        return DTensor.from_local(
            local,
            target.device_mesh,
            target.placements,
            run_check=False,
            shape=target.shape,
            stride=target.stride(),
        )

    def fsdp2_whole_segment(update, obs, param, state, tempo):
        slabs = tuple(resolve_input(name, update, obs, param) for name in input_names)
        received = kernel.regroup(*slabs)
        whole_inputs = {
            name: kernel.owner_view(received, index) for index, name in enumerate(input_names)
        }
        transformed = whole_inputs["update"]
        owner_age = tempo.age.narrow(0, kernel.owner_start, kernel.owner_count)
        owner_live = tempo.live.narrow(0, kernel.owner_start, kernel.owner_count)
        if not kernel.owner_count:
            scattered, scattered_live = kernel.scatter(transformed, owner_live)
            return scattered, state, scattered_live
        owner_leaf_indices = tempo.leaf_indices.narrow(
            0, kernel.owner_start, kernel.owner_count
        )
        owner_observations = owner_observation_type(
            *(whole_inputs[f"obs.{name}"] for name in observation_names)
        )
        owner_tempo = tempo._replace(
            age=owner_age,
            live=owner_live,
            raw_grad=whole_inputs.get("obs.grad"),
            logical_shape=plan.global_shape,
            leaf_indices=owner_leaf_indices,
            element_offset=0,
        )
        candidates = {}
        for transform_index, transform in segment.indexed_transforms:
            inbound = owner_live
            prefix = f"{transform_index}:"
            local_state = {
                name.removeprefix(prefix): value.to_local()
                for name, value in state.items()
                if name.startswith(prefix)
            }
            transformed, candidate, transform_live = transform(
                transformed,
                owner_observations,
                whole_inputs.get("param"),
                local_state,
                owner_tempo._replace(live=inbound),
            )
            for name, previous in local_state.items():
                value = candidate[name] * torch.ones_like(candidate[name])
                value = torch.where(broadcast_leaf(inbound, value), value, previous)
                candidates[f"{transform_index}:{name}"] = wrap_candidate(
                    state[f"{transform_index}:{name}"], value
                )
            owner_live = inbound & transform_live
        if segment.terminal_commit:
            commit_state = {
                name.removeprefix("commit:"): value.to_local()
                for name, value in state.items()
                if name.startswith("commit:")
            }
            owner_tempo = owner_tempo._replace(live=owner_live)
            whole_param = whole_inputs["param"]
            new_param, commit_candidate = recipe.commit(
                whole_param, transformed, commit_state, owner_tempo
            )
            for name, previous in commit_state.items():
                value = commit_candidate[name] * torch.ones_like(commit_candidate[name])
                value = torch.where(broadcast_leaf(owner_live, value), value, previous)
                candidates[f"commit:{name}"] = wrap_candidate(state[f"commit:{name}"], value)
            transformed = torch.where(broadcast_leaf(owner_live, new_param), new_param, whole_param)
        scattered, scattered_live = kernel.scatter(transformed, owner_live)
        return scattered, candidates, scattered_live

    def state_init_group(
        slab: Tensor,
        hyper: SimpleNamespace,
        storage_dtype: torch.dtype | None,
        correction_dtype: torch.dtype | None,
        base_seed: Tensor,
        leaf_indices: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        from torch.distributed import _functional_collectives as funcol

        received = funcol.wait_tensor(kernel.regroup(slab))
        whole = kernel.owner_view(received, 0)
        owner_leaf_indices = leaf_indices.narrow(0, kernel.owner_start, kernel.owner_count)
        owner_age = torch.zeros(kernel.owner_count, dtype=torch.int64, device=slab.device)
        owner_live = torch.ones(kernel.owner_count, dtype=torch.bool, device=slab.device)
        init_tempo = Tempo(
            torch.zeros((), dtype=torch.int64, device=slab.device),
            owner_age,
            owner_live,
            hyper,
            False,
            base_seed=base_seed,
            leaf_indices=owner_leaf_indices,
        )
        states = {}
        corrections = {}
        initializers = [
            (str(index), transform.init, getattr(transform, "state_init_hyper", ()))
            for index, transform in segment.indexed_transforms
        ]
        if segment.terminal_commit:
            initializers.append(("commit", recipe.commit.init, getattr(recipe.commit, "state_init_hyper", ())))
        for namespace, initializer, hyper_names in initializers:
            state_hyper = {name: getattr(hyper, name) for name in hyper_names}
            initial = tuple(initializer(whole[index], **state_hyper) for index in range(kernel.owner_count))
            schema = _validate_owner_initial_states(initial, binding)
            for name, shape, dtype in schema:
                qualified = f"{namespace}:{name}"
                target_dtype = _narrowed_dtype(dtype, storage_dtype)
                local = torch.empty(
                    (kernel.owner_count, *shape), dtype=target_dtype, device=slab.device
                )
                correction = (
                    torch.empty((kernel.owner_count, *shape), dtype=correction_dtype, device=slab.device)
                    if correction_dtype is not None
                    and dtype.is_floating_point
                    and _narrowed_dtype(dtype, storage_dtype) == storage_dtype
                    else None
                )
                for owner_index, values in enumerate(initial):
                    value = values[name]
                    leaf_tempo = init_tempo._replace(
                        age=owner_age[owner_index : owner_index + 1],
                        live=owner_live[owner_index : owner_index + 1],
                        leaf_indices=owner_leaf_indices[owner_index : owner_index + 1],
                    )
                    if correction is not None:
                        random = (
                            None
                            if correction_dtype is torch.int16
                            else leaf_tempo.random_like(value.unsqueeze(0))[0]
                        )
                        narrow, residual = encode(
                            value, torch.bfloat16, correction_dtype, random=random
                        )
                        local[owner_index].copy_(narrow)
                        correction[owner_index].copy_(residual)
                    else:
                        stochastic_copy_(
                            local[owner_index].unsqueeze(0), value.unsqueeze(0), leaf_tempo
                        )
                global_shape = (len(plan.param_keys), *shape)
                states[qualified] = _make_owner_dtensor(local, binding, global_shape)
                if correction is not None:
                    corrections[qualified] = _make_owner_dtensor(correction, binding, global_shape)
        return states, corrections

    def empty_init(ref_leaf: Tensor) -> dict[str, Tensor]:
        del ref_leaf
        return {}

    fsdp2_whole_segment.init = empty_init
    fsdp2_whole_segment.state_init_group = state_init_group
    fsdp2_whole_segment.checkpoint_owner_ranges = plan.owner_ranges
    param_initializers = tuple(
        (transform, transform.param_init)
        for _, transform in segment.indexed_transforms
        if hasattr(transform, "param_init")
    )
    if param_initializers:
        @torch.no_grad()
        def param_init_group(slab: Tensor, seeds: Sequence[int], **init_hyper) -> None:
            received = kernel.regroup(slab)
            whole = kernel.owner_view(received, 0)
            for transform, initializer in param_initializers:
                names = getattr(transform, "param_init_hyper", ())
                values = {name: init_hyper[name] for name in names}
                for index in range(kernel.owner_count):
                    initializer(
                        whole[index], seed=seeds[kernel.owner_start + index], **values
                    )
            initialized, _ = kernel.scatter(
                whole, torch.ones(kernel.owner_count, dtype=torch.bool, device=slab.device)
            )
            slab.copy_(initialized)

        fsdp2_whole_segment.param_init_group = param_init_group
        fsdp2_whole_segment.param_init_hyper = tuple(dict.fromkeys(
            name
            for transform, _ in param_initializers
            for name in getattr(transform, "param_init_hyper", ())
        ))
        fsdp2_whole_segment.param_init_deferred = all(
            getattr(transform, "param_init_deferred", False)
            for transform, _ in param_initializers
        )
    names = "_".join(transform.__name__ for _, transform in segment.indexed_transforms)
    if segment.terminal_commit:
        names = f"{names + '_' if names else ''}{recipe.commit.__name__}"
    fsdp2_whole_segment.__name__ = f"fsdp2_whole_segment_{names}"
    fsdp2_whole_segment.config = {"plan": plan.identity, "segment": segment.identity}
    return fsdp2_whole_segment


def _make_fsdp2_shard_transform(transform: Transform, logical_shape: tuple[int, ...]) -> Transform:
    def fsdp2_shard_transform(update, obs, param, state, tempo):
        return transform(update, obs, param, state, tempo._replace(logical_shape=logical_shape))

    for name in (
        "init",
        "state_init_hyper",
        "state_init_seeded",
        "param_init",
        "param_init_group",
        "param_init_hyper",
        "param_init_deferred",
    ):
        if hasattr(transform, name):
            setattr(fsdp2_shard_transform, name, getattr(transform, name))
    fsdp2_shard_transform.__name__ = f"fsdp2_shard_{transform.__name__}"
    return fsdp2_shard_transform


def _make_fsdp2_shard_commit(commit: Commit, logical_shape: tuple[int, ...]) -> Commit:
    def fsdp2_shard_commit(param, update, state, tempo):
        return commit(param, update, state, tempo._replace(logical_shape=logical_shape))

    for name in ("init", "state_init_hyper", "eval_swap"):
        if hasattr(commit, name):
            setattr(fsdp2_shard_commit, name, getattr(commit, name))
    fsdp2_shard_commit.__name__ = f"fsdp2_shard_{commit.__name__}"
    return fsdp2_shard_commit


def _make_fsdp2_terminal_commit() -> Commit:
    def fsdp2_terminal_commit(param, update, state, tempo):
        del param, state, tempo
        return update, {}

    def empty_init(ref_leaf: Tensor) -> dict[str, Tensor]:
        del ref_leaf
        return {}

    fsdp2_terminal_commit.init = empty_init
    return fsdp2_terminal_commit


def _specialize_fsdp2_recipe(
    recipe: Recipe,
    segments: tuple[_FSDP2WholeSegment, ...],
    plans: tuple[FSDP2RegroupPlan, ...],
    binding: FSDP2Binding,
) -> Recipe:
    segment_by_start = {segment.start: (segment, plan) for segment, plan in zip(segments, plans, strict=True)}
    chain = []
    index = 0
    while index < len(recipe.chain):
        if index in segment_by_start:
            segment, plan = segment_by_start[index]
            chain.append(_make_fsdp2_whole_segment_transform(recipe, segment, plan, binding))
            index = segment.stop
        else:
            chain.append(_make_fsdp2_shard_transform(recipe.chain[index], binding.global_shape))
            index += 1
    if len(recipe.chain) in segment_by_start and segment_by_start[len(recipe.chain)][0].start == len(recipe.chain):
        segment, plan = segment_by_start[len(recipe.chain)]
        chain.append(_make_fsdp2_whole_segment_transform(recipe, segment, plan, binding))
    terminal = bool(segments and segments[-1].terminal_commit)
    commit = _make_fsdp2_terminal_commit() if terminal else _make_fsdp2_shard_commit(
        recipe.commit, binding.global_shape
    )
    return Recipe(
        chain=tuple(chain),
        commit=commit,
        defaults=recipe.defaults,
        observations=recipe.observations,
        clip_global_norm=recipe.clip_global_norm,
    )


def _scalar(value, reference: Tensor) -> Tensor:
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("hyperparameters must be 0-d tensors or Python scalars")
        return value.detach().to(device=reference.device, dtype=dtype).reshape(())
    try:
        scalar = torch.tensor(value, device=reference.device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"hyperparameter values must be numbers or 0-d tensors, got {type(value).__name__}"
        ) from error
    if scalar.numel() != 1:
        raise ValueError("hyperparameters must be 0-d tensors or Python scalars")
    return scalar


@torch.no_grad()
def produce(param: Tensor, name: str, value: Tensor) -> None:
    """Accumulate one declared observation into its slab-bound parameter view."""

    binding = getattr(param, _OBSERVATION_BINDING, None)
    if binding is None or name not in binding.declared:
        raise ValueError(f"{name!r} is not a declared observation for this parameter")
    target = getattr(param, name)
    if tuple(value.shape) != binding.shape:
        raise ValueError(
            f"observation {name!r} value shape {tuple(value.shape)} does not match bound shape {binding.shape}"
        )
    target.add_(value)
    binding.produced.add(name)


class Engine:
    """Owns permanently slab-backed leaves and one compiled whole-step per step type."""

    def __init__(self, params: Iterable[Tensor], recipe_or_route: Recipe | Route, *,
                 param_keys: Sequence[str] | None = None, clip_global_norm: float | None = None,
                 storage_dtype: torch.dtype | str | None = None,
                 ecc: int | str | None = None,
                 bindings: Mapping[int, PlainBinding | FSDP2Binding] | None = None,
                 _rng_seed: int | None = None,
                 _leaf_indices: Sequence[int] | None = None,
                 param_group_ids: Sequence[int] | None = None,
                 param_group_hypers: Mapping[int, Mapping[str, object]] | None = None,
                 **overrides) -> None:
        self.recipe = recipe_or_route
        ecc_dtypes = {
            8: torch.int8,
            16: torch.int16,
            "bf16+8": torch.int8,
            "bf16+16": torch.int16,
        }
        if ecc is not None and (type(ecc) not in (int, str) or ecc not in ecc_dtypes):
            raise ValueError('ecc must be None, 8, 16, "bf16+8", or "bf16+16"')
        correction_dtype = None if ecc is None else ecc_dtypes[ecc]
        if isinstance(storage_dtype, str):
            storage_dtype = getattr(torch, storage_dtype.removeprefix("torch."), storage_dtype)
        if correction_dtype is not None and storage_dtype is not None and storage_dtype is not torch.bfloat16:
            raise ValueError("ecc requires storage_dtype to be None or torch.bfloat16")
        if storage_dtype is not None and storage_dtype is not torch.bfloat16:
            raise ValueError("storage_dtype must be None or torch.bfloat16")
        if correction_dtype is not None:
            storage_dtype = torch.bfloat16
        self.storage_dtype = storage_dtype
        self.ecc = correction_dtype
        self._rng_seed = torch.initial_seed() if _rng_seed is None else _rng_seed
        if (
            not isinstance(self._rng_seed, int)
            or isinstance(self._rng_seed, bool)
            or not 0 <= self._rng_seed < (1 << 64)
        ):
            raise ValueError("_rng_seed must be an unsigned 64-bit int")
        supplied_params = tuple(params)
        if param_group_ids is None:
            supplied_param_group_ids = (0,) * len(supplied_params)
        else:
            supplied_param_group_ids = tuple(param_group_ids)
            if len(supplied_param_group_ids) != len(supplied_params):
                raise ValueError("param_group_ids must contain one id for every supplied parameter")
            if any(
                not isinstance(group_id, int) or isinstance(group_id, bool) or group_id < 0
                for group_id in supplied_param_group_ids
            ):
                raise ValueError("param_group_ids must contain non-negative integers")
        supplied_group_ids = set(supplied_param_group_ids)
        if param_group_hypers is None:
            param_group_hypers = {}
        elif not isinstance(param_group_hypers, Mapping):
            raise TypeError("param_group_hypers must map parameter-group ids to hyperparameter mappings")
        invalid_hyper_group_ids = set(param_group_hypers) - supplied_group_ids
        if invalid_hyper_group_ids:
            raise ValueError("param_group_hypers contains an id absent from param_group_ids")
        if any(
            not isinstance(group_hypers, Mapping) for group_hypers in param_group_hypers.values()
        ):
            raise TypeError("param_group_hypers values must be hyperparameter mappings")
        param_group_hypers = {
            group_id: dict(group_hypers) for group_id, group_hypers in param_group_hypers.items()
        }
        if _leaf_indices is None:
            supplied_leaf_indices = tuple(range(len(supplied_params)))
        else:
            supplied_leaf_indices = tuple(_leaf_indices)
            if len(supplied_leaf_indices) != len(supplied_params):
                raise ValueError("_leaf_indices must contain one index for every supplied parameter")
            if any(
                not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < (1 << 63)
                for index in supplied_leaf_indices
            ):
                raise ValueError("_leaf_indices must contain non-negative int64 indices")
            if len(set(supplied_leaf_indices)) != len(supplied_leaf_indices):
                raise ValueError("_leaf_indices must be unique")
        if param_keys is None:
            supplied_keys = tuple(str(index) for index in range(len(supplied_params)))
        else:
            supplied_keys = tuple(param_keys)
            if len(supplied_keys) != len(supplied_params):
                raise ValueError("param_keys must contain one key for every supplied parameter")
            if any(not isinstance(key, str) for key in supplied_keys):
                raise TypeError("param_keys must contain strings")
        if any(not isinstance(param, Tensor) for param in supplied_params):
            raise TypeError("Engine can optimize tensors only")
        if any(not param.is_leaf for param in supplied_params):
            raise ValueError("Engine can optimize leaf tensors only")
        if len({id(param) for param in supplied_params}) != len(supplied_params):
            raise ValueError("a parameter may appear only once in an Engine")
        keyed_params = tuple(
            (param, key) for param, key in zip(supplied_params, supplied_keys, strict=True) if param.requires_grad
        )
        self.params = tuple(param for param, _ in keyed_params)
        self.param_keys = tuple(key for _, key in keyed_params)
        if not self.params:
            raise ValueError("Engine requires at least one trainable parameter")
        if any(not param.is_floating_point() for param in self.params):
            raise TypeError("Engine supports floating-point parameters only")
        if len(set(self.param_keys)) != len(self.param_keys):
            raise ValueError("param_keys must be unique")
        if bindings is None:
            binding_by_param_id = {id(param): PlainBinding(param) for param in self.params}
        else:
            missing_bindings = [param for param in self.params if id(param) not in bindings]
            if missing_bindings:
                raise ValueError("a storage binding is required for every trainable parameter")
            binding_by_param_id = {id(param): bindings[id(param)] for param in self.params}
        if any(binding.param is not param for param in self.params for binding in (binding_by_param_id[id(param)],)):
            raise ValueError("storage bindings must reference their supplied parameters")
        fsdp2_mode = all(isinstance(binding_by_param_id[id(param)], FSDP2Binding) for param in self.params)
        if any(isinstance(binding_by_param_id[id(param)], FSDP2Binding) for param in self.params) != fsdp2_mode:
            raise ValueError("an Engine cannot mix plain and FSDP2 storage bindings")
        if not fsdp2_mode and torch.distributed.is_available() and torch.distributed.is_initialized():
            from torch.distributed.tensor import DTensor

            if any(isinstance(param, DTensor) for param in self.params):
                raise ValueError(
                    "an FSDP2 DTensor parameter was passed to the plain optimizer constructor; for a "
                    "fully_shard(model) model use the optimizer's .fsdp2(model) classmethod instead of "
                    "constructing it from model.parameters()."
                )
        if not fsdp2_mode:
            storage_groups: dict[torch.UntypedStorage, list[Tensor]] = {}
            for param in supplied_params:
                storage_groups.setdefault(param.untyped_storage(), []).append(param)
            for storage_params in storage_groups.values():
                if any(
                    _storage_views_overlap(current, nxt)
                    for index, current in enumerate(storage_params)
                    for nxt in storage_params[index + 1:]
                ):
                    raise ValueError(
                        "two parameters share overlapping storage (weight tying via separate "
                        "nn.Parameter objects, or a parameter and a view of another); heavyball binds "
                        "each parameter into its own slab row, which silently breaks the shared storage. "
                        "Use a single nn.Parameter for tied weights instead of separate ones."
                    )
            for param in self.params:
                binding_by_param_id[id(param)].validate_param()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            binding0 = binding_by_param_id[id(self.params[0])] if fsdp2_mode else None
            process_group = binding0.process_group if binding0 is not None else None
            replicate_group = binding0.replicate_process_group if binding0 is not None else None
            backend = torch.distributed.get_backend(process_group)
            seed_device = self.params[0].device if backend in ("nccl", "xccl") else torch.device("cpu")
            seed_words = torch.tensor(
                (self._rng_seed & 0xFFFFFFFF, self._rng_seed >> 32),
                dtype=torch.int64,
                device=seed_device,
            )
            if process_group is None:
                torch.distributed.broadcast(seed_words, src=0)
            else:
                torch.distributed.broadcast(seed_words, group=process_group, group_src=0)
                if replicate_group is not None:
                    # HSDP replicas must share the seed or stochastic steps silently diverge.
                    torch.distributed.broadcast(seed_words, group=replicate_group, group_src=0)
            self._rng_seed = int(seed_words[0]) | (int(seed_words[1]) << 32)
        self._bindings = binding_by_param_id
        self._param_keys = {id(param): key for param, key in keyed_params}
        # Construction-order seeds preserve checkpoint compatibility across regrouping.
        param_init_seeds = {id(param): index for index, param in enumerate(supplied_params)}
        param_rng_indices = {
            id(param): index for param, index in zip(supplied_params, supplied_leaf_indices, strict=True)
        }
        param_group_ids_by_param = {
            id(param): group_id
            for param, group_id in zip(supplied_params, supplied_param_group_ids, strict=True)
        }

        if isinstance(recipe_or_route, (Route, Recipe)):
            recipe_order = _collect_route_recipes(recipe_or_route)
            assignments = []
            for param in self.params:
                assignments.append((param, _resolve_route(recipe_or_route, ParamInfo(param))))
        else:
            raise TypeError("Engine requires a Recipe or Route")
        self._recipe_indices = {}
        self._recipe_bases = {}
        for recipe in recipe_order:
            self._recipe_indices.setdefault(id(recipe), len(self._recipe_indices))
            self._recipe_bases[id(recipe)] = recipe

        recipes = []
        for _, recipe in assignments:
            if all(recipe is not known for known in recipes):
                recipes.append(recipe)
        known_overrides = {name for recipe in recipe_order for name in recipe.defaults}
        unknown_overrides = sorted(
            (set(overrides) | {name for values in param_group_hypers.values() for name in values})
            - known_overrides
        )
        if unknown_overrides:
            names = ", ".join(repr(name) for name in unknown_overrides)
            raise ValueError(f"unknown hyperparameter override(s): {names}")
        cadence_defaults = [
            recipe.defaults["preconditioner_update_probability"]
            for recipe in recipes
            if "preconditioner_update_probability" in recipe.defaults
        ]
        if cadence_defaults:
            probability = overrides.get("preconditioner_update_probability", max(cadence_defaults))
            self._cadence: RefreshCadence | None = RefreshCadence(probability)
        else:
            self._cadence = None
        recipe_clip = recipes[0].clip_global_norm
        if clip_global_norm is None and any(recipe.clip_global_norm != recipe_clip for recipe in recipes[1:]):
            raise ValueError("routed recipes must agree on clip_global_norm")
        self.clip_global_norm = recipe_clip if clip_global_norm is None else clip_global_norm
        if self.clip_global_norm is not None and len({param.device for param in self.params}) != 1:
            raise ValueError("global clipping requires parameters on one device")

        buckets: dict[tuple[object, ...], list[Tensor]] = {}
        bucket_recipes: dict[tuple[object, ...], Recipe] = {}
        for param, recipe in assignments:
            binding = binding_by_param_id[id(param)]
            storage_shape = binding.storage_shape
            param_group_id = param_group_ids_by_param[id(param)]
            if isinstance(binding, FSDP2Binding):
                fsdp2_bucket = binding.bucket_identity(_fsdp2_scope_identity(recipe))
            else:
                fsdp2_bucket = ()
            key = (
                id(recipe), param_group_id, storage_shape, param.dtype, param.device, storage_dtype, correction_dtype,
                fsdp2_bucket,
            )
            buckets.setdefault(key, []).append(param)
            bucket_recipes[key] = recipe

        base_bucket_recipes = dict(bucket_recipes)
        regroup_plans: dict[tuple[object, ...], tuple[FSDP2RegroupPlan, ...]] = {}
        collective_index = 0
        for key, leaves in buckets.items():
            recipe = base_bucket_recipes[key]
            segments = _fsdp2_scope_segments(recipe)
            if not segments:
                continue
            reference_binding = binding_by_param_id[id(leaves[0])]
            if not isinstance(reference_binding, FSDP2Binding):
                continue
            if any(
                binding_by_param_id[id(param)].global_shape != reference_binding.global_shape
                or binding_by_param_id[id(param)].storage_shape != reference_binding.storage_shape
                or binding_by_param_id[id(param)].process_group is not reference_binding.process_group
                or binding_by_param_id[id(param)].placement_identity != reference_binding.placement_identity
                for param in leaves[1:]
            ):
                raise ValueError("FSDP2 whole-segment buckets require one logical shape, mesh, and placement")
            plans = tuple(
                FSDP2RegroupPlan.create(
                    tuple(self._param_keys[id(param)] for param in leaves),
                    reference_binding.global_shape,
                    reference_binding.storage_shape,
                    reference_binding.world_size,
                    mesh_ranks=reference_binding.mesh_ranks,
                    placements=reference_binding.placement_identity,
                    collective_index=collective_index + index,
                    specialization=segment.identity,
                )
                for index, segment in enumerate(segments)
            )
            regroup_plans[key] = plans
            collective_index += len(plans)

        if fsdp2_mode:
            self._fsdp2_manifest = _validate_fsdp2_manifest(
                assignments,
                binding_by_param_id,
                self._param_keys,
                buckets,
                base_bucket_recipes,
                regroup_plans,
            )
            compile_buckets = tuple(
                (
                    tuple(self._param_keys[id(param)] for param in leaves),
                    binding_by_param_id[id(leaves[0])].global_shape,
                    binding_by_param_id[id(leaves[0])].storage_shape,
                    binding_by_param_id[id(leaves[0])].mesh_ranks,
                    id(binding_by_param_id[id(leaves[0])].mesh),
                    id(binding_by_param_id[id(leaves[0])].process_group),
                    binding_by_param_id[id(leaves[0])].mesh_rank,
                    binding_by_param_id[id(leaves[0])].placement_identity,
                    tuple(plan.identity for plan in regroup_plans.get(key, ())),
                )
                for key, leaves in buckets.items()
            )
            compile_identity = (id(recipe_or_route), compile_buckets)
            self.recipe = _bounded_cache_get_or_create(
                _FSDP2_COMPILE_IDENTITIES, compile_identity, object
            )

        for key, leaves in buckets.items():
            reference_binding = binding_by_param_id[id(leaves[0])]
            if not isinstance(reference_binding, FSDP2Binding):
                continue
            base_recipe = base_bucket_recipes[key]
            specialized = _specialize_fsdp2_recipe(
                base_recipe,
                _fsdp2_scope_segments(base_recipe),
                regroup_plans.get(key, ()),
                reference_binding,
            )
            bucket_recipes[key] = specialized
            self._recipe_bases[id(specialized)] = base_recipe

        steps: dict[torch.device, Tensor] = {}
        rng_seeds: dict[torch.device, Tensor] = {}
        hypers: dict[tuple[int, int, torch.device, torch.dtype], SimpleNamespace] = {}
        clip_norms: dict[torch.device, Tensor] = {}
        self.groups = []
        deferred_param_initializers = []
        missing = object()
        original_bindings = tuple(
            (
                param,
                binding_by_param_id[id(param)],
                binding_by_param_id[id(param)].snapshot(),
                {
                    name: getattr(param, name, missing)
                    for name in (_OBSERVATION_BINDING, *recipe.observations)
                },
            )
            for param, recipe in assignments
        )

        def restore_param(param, binding, snapshot, attributes):
            binding.restore(snapshot)
            for name, value in attributes.items():
                if value is missing:
                    if hasattr(param, name):
                        delattr(param, name)
                else:
                    setattr(param, name, value)

        with torch.no_grad(), ExitStack() as rollback:
            for binding in original_bindings:
                rollback.callback(restore_param, *binding)
            for key, leaves in buckets.items():
                recipe = bucket_recipes[key]
                base_recipe = base_bucket_recipes[key]
                device = leaves[0].device
                reference = leaves[0]
                param_group_id = param_group_ids_by_param[id(reference)]
                if device not in steps:
                    steps[device] = torch.ones((), dtype=torch.long, device=device)
                    rng_seeds[device] = torch.tensor(
                        (self._rng_seed & 0xFFFFFFFF, self._rng_seed >> 32),
                        dtype=torch.int64,
                        device=device,
                    )
                    if self.clip_global_norm is not None:
                        clip_norms[device] = _scalar(self.clip_global_norm, reference)
                hyper_key = (id(base_recipe), param_group_id, device, reference.dtype)
                if hyper_key not in hypers:
                    group_overrides = param_group_hypers.get(param_group_id, {})
                    values = {
                        name: group_overrides.get(name, overrides.get(name, value))
                        for name, value in recipe.defaults.items()
                        if name not in _HOST_ONLY_HYPERPARAMETERS
                    }
                    if "max_lr" in values and values["max_lr"] is None and "lr" in values:
                        values["max_lr"] = values["lr"]
                    hypers[hyper_key] = SimpleNamespace(
                        **{name: _scalar(value, reference) for name, value in values.items()}
                    )
                hyper, step = hypers[hyper_key], steps[device]
                reference_binding = binding_by_param_id[id(reference)]
                storage_shape = reference_binding.storage_shape
                slab = torch.empty((len(leaves), *storage_shape), dtype=reference.dtype, device=device)
                for index, param in enumerate(leaves):
                    binding_by_param_id[id(param)].bind_param(slab[index])
                for chain_transform in recipe.chain:
                    group_init_fn = getattr(chain_transform, "param_init_group", None)
                    init_fn = getattr(chain_transform, "param_init", None)
                    if group_init_fn is None and init_fn is None:
                        continue
                    init_hyper = {
                        name: getattr(hyper, name)
                        for name in getattr(chain_transform, "param_init_hyper", ())
                    }
                    seeds = tuple(param_init_seeds[id(param)] for param in leaves)
                    if getattr(chain_transform, "param_init_deferred", False):
                        deferred_param_initializers.append(
                            (
                                group_init_fn,
                                init_fn,
                                slab,
                                seeds,
                                init_hyper,
                                tuple(self._param_keys[id(param)] for param in leaves),
                            )
                        )
                        continue
                    if group_init_fn is not None:
                        group_init_fn(
                            slab,
                            seeds=seeds,
                            **init_hyper,
                        )
                    elif init_fn is not None:
                        for index, seed in enumerate(seeds):
                            init_fn(slab[index], seed=seed, **init_hyper)
                grad_slab = torch.zeros_like(slab)
                observed = torch.ones(len(leaves), dtype=torch.bool, device=device)
                age = torch.zeros(len(leaves), dtype=torch.int64, device=device)
                leaf_indices = torch.tensor(
                    tuple(param_rng_indices[id(param)] for param in leaves),
                    dtype=torch.int64,
                    device=device,
                )
                if isinstance(reference_binding, FSDP2Binding):
                    element_offset = (
                        reference_binding.mesh_rank
                        * storage_shape[0]
                        * math.prod(reference_binding.global_shape[1:])
                    )
                else:
                    element_offset = 0
                init_tempo = Tempo(
                    step,
                    age,
                    observed,
                    hyper,
                    False,
                    base_seed=rng_seeds[device],
                    leaf_indices=leaf_indices,
                    element_offset=element_offset,
                )

                def rng_element_offset(
                    value: Tensor,
                    *,
                    shard_offset: int = element_offset,
                    shard_shape: tuple[int, ...] = storage_shape,
                ) -> int:
                    return shard_offset if tuple(value.shape[1:]) == shard_shape else 0

                for index, param in enumerate(leaves):
                    binding_by_param_id[id(param)].bind_grad(grad_slab[index])
                initializer_references = tuple(
                    binding_by_param_id[id(param)].initializer_reference(slab[index])
                    for index, param in enumerate(leaves)
                )
                states = []
                state_corrections = []
                state_element_offsets = []
                state_owner_ranges = []
                for transform in recipe.chain:
                    group_state_init = getattr(transform, "state_init_group", None)
                    if group_state_init is not None:
                        state_slabs, correction_slabs = group_state_init(
                            slab, hyper, storage_dtype, correction_dtype,
                            rng_seeds[device], leaf_indices,
                        )
                    else:
                        state_hyper = {
                            name: getattr(hyper, name) for name in getattr(transform, "state_init_hyper", ())
                        }
                        if getattr(transform, "state_init_seeded", False):
                            initial_states = tuple(
                                transform.init(
                                    leaf,
                                    seed=(self._rng_seed + param_rng_indices[id(param)]) % (1 << 64),
                                    **state_hyper,
                                )
                                for param, leaf in zip(leaves, initializer_references, strict=True)
                            )
                        else:
                            initial_states = tuple(
                                transform.init(leaf, **state_hyper)
                                for leaf in initializer_references
                            )
                        first_state = initial_states[0]
                        for initial_state in initial_states[1:]:
                            if initial_state.keys() != first_state.keys():
                                raise ValueError("transform initializer returned incompatible state keys within a bucket")
                            for name, value in initial_state.items():
                                first_value = first_state[name]
                                if value.shape != first_value.shape or value.dtype != first_value.dtype:
                                    raise ValueError(
                                        "transform initializer returned incompatible state shape or dtype within a bucket"
                                    )
                        state_slabs = {
                            name: value.new_empty((len(leaves), *value.shape), dtype=_slab_dtype(value, storage_dtype))
                            for name, value in first_state.items()
                        }
                        correction_slabs = {
                            name: value.new_empty((len(leaves), *value.shape), dtype=correction_dtype)
                            for name, value in first_state.items()
                            if correction_dtype is not None
                            and value.is_floating_point()
                            and _narrowed_dtype(value.dtype, storage_dtype) == storage_dtype
                        }
                        for index, initial_state in enumerate(initial_states):
                            for name, value in initial_state.items():
                                leaf_tempo = init_tempo._replace(
                                    age=age[index : index + 1],
                                    live=observed[index : index + 1],
                                    leaf_indices=leaf_indices[index : index + 1],
                                )
                                if name in correction_slabs:
                                    random = (
                                        None
                                        if correction_dtype is torch.int16
                                        else leaf_tempo._replace(
                                            element_offset=rng_element_offset(state_slabs[name])
                                        ).random_like(value.unsqueeze(0))[0]
                                    )
                                    narrow, correction = encode(
                                        value, torch.bfloat16, correction_dtype, random=random
                                    )
                                    state_slabs[name][index].copy_(narrow)
                                    correction_slabs[name][index].copy_(correction)
                                else:
                                    stochastic_copy_(
                                        state_slabs[name][index].unsqueeze(0),
                                        value.unsqueeze(0),
                                        leaf_tempo._replace(
                                            element_offset=rng_element_offset(state_slabs[name])
                                        ),
                                    )
                    states.append(state_slabs)
                    state_corrections.append(correction_slabs)
                    state_element_offsets.append(
                        {name: rng_element_offset(value) for name, value in state_slabs.items()}
                    )
                    owner_ranges = getattr(transform, "checkpoint_owner_ranges", None)
                    state_owner_ranges.append(
                        None if owner_ranges is None else owner_ranges[reference_binding.mesh_rank]
                    )
                states = tuple(states)
                state_corrections = tuple(state_corrections)
                state_element_offsets = tuple(state_element_offsets)
                state_owner_ranges = tuple(state_owner_ranges)
                commit_init = getattr(recipe.commit, "init", None)
                initial_commit_states = (
                    tuple(commit_init(leaf) for leaf in initializer_references) if commit_init is not None else ({},)
                )
                commit_state = {
                    name: value.new_empty((len(leaves), *value.shape), dtype=_slab_dtype(value, storage_dtype))
                    for name, value in initial_commit_states[0].items()
                }
                commit_corrections = {
                    name: value.new_empty((len(leaves), *value.shape), dtype=correction_dtype)
                    for name, value in initial_commit_states[0].items()
                    if correction_dtype is not None
                    and value.is_floating_point()
                    and _narrowed_dtype(value.dtype, storage_dtype) == storage_dtype
                }
                for index, initial_state in enumerate(initial_commit_states):
                    for name, value in initial_state.items():
                        leaf_tempo = init_tempo._replace(
                            age=age[index : index + 1],
                            live=observed[index : index + 1],
                            leaf_indices=leaf_indices[index : index + 1],
                        )
                        if name in commit_corrections:
                            random = (
                                None
                                if correction_dtype is torch.int16
                                else leaf_tempo._replace(
                                    element_offset=rng_element_offset(commit_state[name])
                                ).random_like(value.unsqueeze(0))[0]
                            )
                            narrow, correction = encode(
                                value, torch.bfloat16, correction_dtype, random=random
                            )
                            commit_state[name][index].copy_(narrow)
                            commit_corrections[name][index].copy_(correction)
                        else:
                            stochastic_copy_(
                                commit_state[name][index].unsqueeze(0),
                                value.unsqueeze(0),
                                leaf_tempo._replace(
                                    element_offset=rng_element_offset(commit_state[name])
                                ),
                            )
                commit_element_offsets = {
                    name: rng_element_offset(value) for name, value in commit_state.items()
                }
                observations = SimpleNamespace(grad=grad_slab)
                for param in leaves:
                    setattr(
                        param,
                        _OBSERVATION_BINDING,
                        _ObservationBinding(frozenset(recipe.observations), tuple(param.shape)),
                    )
                for name in recipe.observations:
                    observation_slab = torch.zeros_like(_wide(grad_slab))
                    setattr(observations, name, observation_slab)
                    for index, param in enumerate(leaves):
                        setattr(param, name, observation_slab[index])
                self.groups.append(
                    Group(
                        tuple(leaves), slab, grad_slab, observed, age, rng_seeds[device], leaf_indices,
                        element_offset, states, state_corrections, state_element_offsets, state_owner_ranges,
                        commit_state, commit_corrections, commit_element_offsets, observations, hyper, step, recipe,
                        param_group_id,
                    )
                )
            rollback = rollback.pop_all()

        with rollback:
            self.groups = tuple(self.groups)
            self._steps_by_device = steps
            self._steps = tuple(steps.values())
            self._rng_seeds = rng_seeds
            self._clip_norms = tuple(clip_norms.values())
            self._deferred_param_initializers = tuple(deferred_param_initializers)
            self._deferred_param_init_pending = {
                param_key: True
                for initializer in deferred_param_initializers
                for param_key in initializer[-1]
            }
            self._binding_identities = {
                id(param): self._bindings[id(param)].storage_identity()
                for param in self.params
            }
            self.hyper, self.step_count = self.groups[0].hyper, self._steps[0]
            from torch.distributed.tensor import DTensor

            self.state = {
                param: {
                    name: slab[index]
                    for slots in group.states
                    for name, slab in slots.items()
                    if not isinstance(slab, DTensor)
                }
                for group in self.groups
                for index, param in enumerate(group.params)
            }
            self._param_locations = {
                self._param_keys[id(param)]: (group, index)
                for group in self.groups
                for index, param in enumerate(group.params)
            }
            self._observation_groups = {
                id(param): group
                for group in self.groups
                for param in group.params
            }
            self._hyper_locations = {
                self._checkpoint_hyper_key(
                    group.recipe, group.param_group_id, group.param_slab.device, group.param_slab.dtype
                ): group.hyper
                for group in self.groups
            }
            self._train_mode: bool = True
            self.compiled_steps = {
                "normal": self._compile_normal(),
                "refresh": self._compile_refresh(),
            }
            self.compiled_step = self.compiled_steps["normal"]
            self.compiled_eval_swaps = {
                True: self._compile_eval_swap(entering_train=True),
                False: self._compile_eval_swap(entering_train=False),
            }
            rollback.pop_all()

    def _checkpoint_hyper_key(
        self, recipe: Recipe, param_group_id: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[int, int | str]:
        base_recipe = self._recipe_bases[id(recipe)]
        index = self._recipe_indices[id(base_recipe)]
        locations = {
            (group.param_slab.device, group.param_slab.dtype)
            for group in self.groups
            if group.param_group_id == param_group_id
            and self._recipe_bases[id(group.recipe)] is base_recipe
        }
        if len(locations) == 1:
            return param_group_id, index
        devices, dtypes = map(set, zip(*locations, strict=True))
        location = f"{index}{f'@{device}' if len(devices) > 1 else ''}{f'@{dtype}' if len(dtypes) > 1 else ''}"
        return param_group_id, location

    def _recipe_fingerprint(self, recipe: Recipe) -> dict[str, object]:
        def config_of(transform: Transform) -> tuple[tuple[str, object], ...] | None:
            config = getattr(transform, "config", None)
            return tuple(sorted(config.items())) if config is not None else None

        recipe = self._recipe_bases.get(id(recipe), recipe)
        commit_config = config_of(recipe.commit)
        return {
            "chain": tuple((transform.__name__, config_of(transform)) for transform in recipe.chain),
            "commit": recipe.commit.__name__ if commit_config is None else (recipe.commit.__name__, commit_config),
            "observations": tuple(recipe.observations),
            "clip_global_norm": None if not self._clip_norms else float(self._clip_norms[0].detach().cpu()),
        }

    def _recipe_fingerprints(self) -> dict[str, dict[str, object]]:
        return {
            self._param_keys[id(param)]: self._recipe_fingerprint(group.recipe)
            for group in self.groups
            for param in group.params
        }

    def _compile_normal(self):
        return self._compile_step(refresh=False)

    def _compile_refresh(self):
        return self._compile_step(refresh=True)

    def _compile_step(self, *, refresh: bool):
        def shares_param_noise(group: Group, target: Tensor) -> bool:
            return (
                target.dtype is torch.bfloat16
                and target.shape == group.param_slab.shape
                and getattr(target, "placements", None)
                == getattr(group.param_slab, "placements", None)
            )

        def needs_param_noise(group: Group) -> bool:
            targets = [group.param_slab]
            targets.extend(
                value
                for state, corrections in zip(
                    group.states, group.state_corrections, strict=True
                )
                for name, value in state.items()
                if name not in corrections
            )
            targets.extend(
                value
                for name, value in group.commit_state.items()
                if name not in group.commit_corrections
            )
            return any(shares_param_noise(group, target) for target in targets)

        plans = tuple(
            (
                group.param_slab, group.observations, group.observed, group.age,
                group.base_seed, group.leaf_indices, group.element_offset, group.states, group.state_corrections,
                group.state_element_offsets, group.commit_state, group.commit_corrections,
                group.commit_element_offsets, group.hyper, group.step,
                group.recipe.chain, group.recipe.commit, needs_param_noise(group),
            )
            for group in self.groups
        )
        clip_norm = self._clip_norms[0] if self._clip_norms else None
        steps = self._steps

        def whole_step():
            pending = []
            for (
                param_slab, obs, observed, age, base_seed, leaf_indices, element_offset, states,
                state_corrections, state_element_offsets, commit_state, commit_corrections,
                commit_element_offsets, hyper, step, chain, commit, use_param_noise,
            ) in plans:
                age_now = age + observed.to(torch.int64)
                update = obs.grad
                live = observed
                tempo = Tempo(
                    step,
                    age_now,
                    live,
                    hyper,
                    refresh,
                    base_seed=base_seed,
                    leaf_indices=leaf_indices,
                    element_offset=element_offset,
                )
                candidates = []
                decoded_states = []
                for transform, state, corrections in zip(chain, states, state_corrections, strict=True):
                    inbound = live
                    decoded_state = {
                        name: decode(value, corrections[name], corrections[name].dtype)
                        if name in corrections else value
                        for name, value in state.items()
                    } if corrections else state
                    update, candidate, transform_live = transform(
                        update, obs, param_slab, decoded_state, tempo._replace(live=inbound)
                    )
                    candidates.append((candidate, inbound))
                    decoded_states.append(decoded_state)
                    live = inbound & transform_live
                pending.append(
                    (
                        param_slab, obs.grad, update, observed, age, age_now, tempo, states,
                        state_corrections, state_element_offsets, decoded_states, candidates, live,
                        commit_state, commit_corrections, commit_element_offsets, hyper, step, commit,
                        use_param_noise,
                    )
                )

            if clip_norm is not None:
                squared = [
                    torch.where(
                        broadcast_leaf(live, update), update, torch.zeros_like(update)
                    ).double().square().sum()
                    for _, _, update, _, _, _, _, _, _, _, _, _, live, *_ in pending
                ]
                total = squared[0]
                for value in squared[1:]:
                    total = total + value
                scale = (clip_norm / (total.sqrt() + 1e-6)).clamp(max=1.0)

            for _, _, _, _, _, _, _, states, _, _, _, candidates, *_ in pending:
                for index, (state, (candidate, inbound)) in enumerate(zip(states, candidates, strict=True)):
                    candidates[index] = (
                        {name: candidate[name] * torch.ones_like(candidate[name]) for name in state}, inbound
                    )

            committed = []
            for (
                param_slab, raw_grad, update, _observed, age, age_now, tempo, states,
                state_corrections, state_element_offsets, decoded_states, candidates, live,
                commit_state, commit_corrections, commit_element_offsets, _hyper, _step, commit,
                use_param_noise,
            ) in pending:
                if clip_norm is not None:
                    update = update * scale
                decoded_commit_state = {
                    name: decode(value, commit_corrections[name], commit_corrections[name].dtype)
                    if name in commit_corrections else value
                    for name, value in commit_state.items()
                } if commit_corrections else commit_state
                new_param, candidate_commit_state = commit(
                    param_slab,
                    update,
                    decoded_commit_state,
                    tempo._replace(live=live, raw_grad=raw_grad),
                )
                candidate_commit_state = {
                    name: candidate_commit_state[name] * torch.ones_like(candidate_commit_state[name])
                    for name in commit_state
                }
                committed.append(
                    (
                        param_slab, new_param, age, age_now, states, state_corrections,
                        state_element_offsets, decoded_states, candidates, live, commit_state,
                        commit_corrections, commit_element_offsets, decoded_commit_state,
                        candidate_commit_state, tempo, use_param_noise,
                    )
                )

            for (
                param_slab, new_param, age, age_now, states, state_corrections,
                state_element_offsets, decoded_states, candidates, live, commit_state,
                commit_corrections, commit_element_offsets, decoded_commit_state,
                candidate_commit_state, tempo, use_param_noise,
            ) in committed:
                param_noise = tempo.random_like(param_slab) if use_param_noise else None
                for state, corrections, element_offsets, decoded_state, (candidate, inbound) in zip(
                    states, state_corrections, state_element_offsets, decoded_states, candidates, strict=True
                ):
                    for name in state:
                        state_tempo = tempo._replace(
                            element_offset=element_offsets[name], rounds=_STATE_PHILOX_ROUNDS
                        )
                        if name in corrections:
                            active = broadcast_leaf(inbound, candidate[name])
                            new = torch.where(active, candidate[name], decoded_state[name])
                            narrow, correction = encode(
                                new,
                                torch.bfloat16,
                                corrections[name].dtype,
                                random=(
                                    None
                                    if corrections[name].dtype is torch.int16
                                    else state_tempo.random_like(new)
                                ),
                            )
                            state[name].copy_(torch.where(active, narrow, state[name]))
                            corrections[name].copy_(torch.where(active, correction, corrections[name]))
                        else:
                            stochastic_copy_(state[name], torch.where(
                                broadcast_leaf(inbound, candidate[name]), candidate[name], state[name]
                            ), state_tempo, shared_noise=param_noise)
                for name in commit_state:
                    commit_tempo = tempo._replace(
                        element_offset=commit_element_offsets[name], rounds=_STATE_PHILOX_ROUNDS
                    )
                    if name in commit_corrections:
                        active = broadcast_leaf(live, candidate_commit_state[name])
                        new = torch.where(active, candidate_commit_state[name], decoded_commit_state[name])
                        narrow, correction = encode(
                            new,
                            torch.bfloat16,
                            commit_corrections[name].dtype,
                            random=(
                                None
                                if commit_corrections[name].dtype is torch.int16
                                else commit_tempo.random_like(new)
                            ),
                        )
                        commit_state[name].copy_(torch.where(active, narrow, commit_state[name]))
                        commit_corrections[name].copy_(torch.where(active, correction, commit_corrections[name]))
                    else:
                        stochastic_copy_(commit_state[name], torch.where(
                            broadcast_leaf(live, candidate_commit_state[name]), candidate_commit_state[name],
                            commit_state[name]
                        ), commit_tempo, shared_noise=param_noise)
                stochastic_copy_(
                    param_slab,
                    torch.where(broadcast_leaf(live, new_param), new_param, param_slab),
                    tempo,
                    shared_noise=param_noise,
                )
                age.copy_(age_now)
            for step in steps:
                step.add_(1)

        compile_scope = id(self.recipe) if hasattr(self, "_fsdp2_manifest") else None
        plan_abi = _plan_abi(self.groups, clip_norm=clip_norm, compile_scope=compile_scope)
        whole_step = _keyed_compile_fn(whole_step, (plan_abi, "step", refresh))
        return torch.compile(whole_step, fullgraph=True, dynamic=False, options=STEP_COMPILE_OPTIONS)

    def _compile_eval_swap(self, *, entering_train: bool):
        plans = tuple(
            (
                group.param_slab, group.commit_state, group.commit_corrections, group.hyper,
                group.commit_element_offsets, group.step, group.age, group.base_seed,
                group.leaf_indices, group.element_offset, eval_swap,
            )
            for group in self.groups
            if (eval_swap := getattr(group.recipe.commit, "eval_swap", None)) is not None
        )
        if not plans:
            return None

        def whole_swap():
            for (
                param_slab, commit_state, commit_corrections, hyper, commit_element_offsets, step, age,
                base_seed, leaf_indices, element_offset, eval_swap,
            ) in plans:
                tempo = Tempo(
                    step,
                    age,
                    torch.ones_like(age, dtype=torch.bool),
                    hyper,
                    False,
                    base_seed=base_seed,
                    leaf_indices=leaf_indices,
                    element_offset=element_offset,
                )
                decoded_commit_state = {
                    name: decode(value, commit_corrections[name], commit_corrections[name].dtype)
                    if name in commit_corrections else value
                    for name, value in commit_state.items()
                } if commit_corrections else commit_state
                new_param, new_state = eval_swap(param_slab, decoded_commit_state, hyper, entering_train)
                unchanged_state = {
                    name: new_state[name] is decoded_commit_state[name] for name in commit_state
                } if commit_corrections else {}
                # Materialization preserves literal state/parameter swaps in eager and compiled modes.
                new_state = {
                    name: new_state[name] * torch.ones_like(new_state[name]) for name in commit_state
                }
                stochastic_copy_(param_slab, new_param, tempo)
                for name in commit_state:
                    commit_tempo = tempo._replace(
                        element_offset=commit_element_offsets[name], rounds=_STATE_PHILOX_ROUNDS
                    )
                    if name in commit_corrections:
                        if unchanged_state[name]:
                            continue
                        narrow, correction = encode(
                            new_state[name],
                            torch.bfloat16,
                            commit_corrections[name].dtype,
                            random=(
                                None
                                if commit_corrections[name].dtype is torch.int16
                                else commit_tempo.random_like(new_state[name])
                            ),
                        )
                        commit_state[name].copy_(narrow)
                        commit_corrections[name].copy_(correction)
                    else:
                        stochastic_copy_(commit_state[name], new_state[name], commit_tempo)

        compile_scope = id(self.recipe) if hasattr(self, "_fsdp2_manifest") else None
        plan_abi = _plan_abi(self.groups, eval_swap=True, compile_scope=compile_scope)
        whole_swap = _keyed_compile_fn(whole_swap, (plan_abi, "eval_swap", entering_train))
        return torch.compile(whole_swap, fullgraph=True, dynamic=False, options=STEP_COMPILE_OPTIONS)

    @torch.no_grad()
    def set_hyper(self, name: str, value, *, group_id: int | None = None) -> None:
        """Fill a dynamic hyperparameter cell in matching semantic-group namespaces.

        The cells are device scalars captured by the compiled step, so filling
        them in place changes its next execution without recompiling. Omitting
        ``group_id`` retains the direct-Engine behavior of updating every group.
        """

        if name in _HOST_ONLY_HYPERPARAMETERS:
            raise ValueError(f"{name!r} is not a dynamic hyperparameter; set it when building the Engine")
        found = False
        namespaces = {
            id(group.hyper): group.hyper
            for group in self.groups
            if group_id is None or group.param_group_id == group_id
        }
        for namespace in namespaces.values():
            if hasattr(namespace, name):
                target = getattr(namespace, name)
                target.copy_(_scalar(value, target))
                found = True
        if not found:
            raise ValueError(f"unknown hyperparameter {name!r}")

    @torch.no_grad()
    def produce(self, param: Tensor, name: str, value: Tensor) -> None:
        group = self._observation_groups.get(id(param))
        if group is None:
            raise ValueError("parameter is not owned by this Engine")
        if name not in group.recipe.observations:
            raise ValueError(f"{name!r} is not a declared observation for this parameter's group")
        produce(param, name, value)

    @torch.no_grad()
    def _run_deferred_param_init(self) -> bool:
        if not any(self._deferred_param_init_pending.values()):
            return False
        import warnings
        warnings.warn(
            "Deferred parameter initialization is running (e.g., Scion reinitializes parameters to "
            "seeded orthogonal frames on the first step). This overwrites the current parameter values.",
            UserWarning,
            stacklevel=2,
        )
        initialized = []
        for group_init_fn, init_fn, slab, seeds, init_hyper, param_keys in self._deferred_param_initializers:
            pending = tuple(
                index for index, key in enumerate(param_keys)
                if self._deferred_param_init_pending[key]
            )
            if group_init_fn is not None:
                if pending and len(pending) != len(seeds):
                    raise ValueError("grouped deferred parameter initialization state is inconsistent")
                if pending:
                    group_init_fn(slab, seeds=seeds, **init_hyper)
            else:
                for index in pending:
                    init_fn(slab[index], seed=seeds[index], **init_hyper)
            initialized.extend(param_keys[index] for index in pending)
        for param_key in initialized:
            self._deferred_param_init_pending[param_key] = False
        return True

    def _validate_deferred_param_init_pending(self, pending: Mapping[str, bool]) -> None:
        for group_init_fn, _, _, _, _, param_keys in self._deferred_param_initializers:
            if group_init_fn is not None and len({pending[param_key] for param_key in param_keys}) > 1:
                raise ValueError("checkpoint grouped deferred parameter initialization is inconsistent")

    def _validate_bindings(self, *, force: bool) -> None:
        for group in self.groups:
            for index, param in enumerate(group.params):
                binding = self._bindings[id(param)]
                identity = binding.storage_identity()
                if force or identity != self._binding_identities[id(param)]:
                    binding.validate(group.param_slab[index], group.grad_slab[index])
                    self._binding_identities[id(param)] = identity

    @torch.no_grad()
    def step(self, *, step_type: str | None = None,
             observed: Sequence[bool] | Mapping[Tensor, bool] | None = None) -> None:
        """Run the host-selected step type; grads are already in their persistent slabs.

        By design every parameter is observed (updated) every step: heavyball does NOT detect
        per-parameter gradient presence, because grads live in persistent slab rows and are never
        None. An inactive parameter therefore still advances (weight decay, moment, clock), which
        diverges from torch.optim's skip-grad-None. Conditional / MoE / frozen-then-unfrozen callers
        must supply a gradient for every optimized parameter or accept that divergence. This is a
        deliberate tradeoff (zero step-time cost, no implicit backward hooks), not an oversight.
        """

        self._validate_bindings(force=observed is not None)
        if observed is None:
            by_param = None
        elif isinstance(observed, Mapping):
            if len(observed) != len(self.params):
                raise ValueError("observed must contain every trainable parameter")
            try:
                values = tuple(observed[param] for param in self.params)
            except KeyError as error:
                raise ValueError("observed must contain every trainable parameter") from error
        else:
            values = tuple(observed)
            if len(values) != len(self.params):
                raise ValueError("observed must contain one value for every trainable parameter")
        if observed is not None:
            if any(not isinstance(value, bool) for value in values):
                raise TypeError("observed values must be host bools")
            by_param = {
                id(param): value
                for param, value in zip(self.params, values, strict=True)
            }
        for group in self.groups:
            if not group.recipe.observations:
                continue
            for param in group.params:
                if by_param is not None and not by_param[id(param)]:
                    continue
                produced = getattr(param, _OBSERVATION_BINDING).produced
                for name in group.recipe.observations:
                    if name not in produced:
                        raise ValueError(
                            f"observation {name!r} for parameter {tuple(param.shape)} was not produced this step; "
                            "attach a producer (heavyball.register_truegrad) before stepping."
                        )
        if step_type is None:
            step_type = self._cadence.next_step_type() if self._cadence is not None else "normal"
        compiled_step = self.compiled_steps[step_type]
        if by_param is None:
            for group in self.groups:
                if group.observed_cache is not None:
                    group.observed.fill_(True)
                    group.observed_cache = None
        else:
            for group in self.groups:
                group_values = tuple(by_param[id(param)] for param in group.params)
                if group_values != group.observed_cache:
                    group.observed.copy_(torch.tensor(
                        group_values, dtype=torch.bool, device=group.param_slab.device
                    ))
                    group.observed_cache = group_values
        if self._run_deferred_param_init():
            self._bump_versions()
            return
        compiled_step()
        self._bump_versions()

    @torch.no_grad()
    def zero_grad(self, *, set_to_none: bool = False) -> None:
        if set_to_none:
            raise ValueError(
                "heavyball requires persistent gradient buffers; call optimizer.zero_grad(set_to_none=False) "
                "(the default)."
            )
        for group in self.groups:
            for slab in vars(group.observations).values():
                slab.zero_()
            if not group.recipe.observations:
                continue
            for param in group.params:
                getattr(param, _OBSERVATION_BINDING).produced.clear()

    @torch.no_grad()
    def _bump_versions(self) -> None:
        """Single version-bump path for every Parameter mutation (ordinary step and eval/train swap). The
        commit rewrote every parameter, so any retained autograd graph through them must be invalidated,
        matching torch.optim's in-place mutation contract. Used by Engine.step and train()."""
        torch.autograd.graph.increment_version(self.params)

    def train(self, mode: bool = True) -> "Engine":
        """Select a commit's train or evaluation parameter representation."""

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if mode != self._train_mode:
            compiled_swap = self.compiled_eval_swaps[mode]
            if compiled_swap is not None:
                compiled_swap()
                self._bump_versions()
            self._train_mode = mode
        return self

    def eval(self) -> "Engine":
        return self.train(False)

    @torch.no_grad()
    def state_dict(self) -> dict:
        """Return logical state keyed by parameter key, transform/commit bucket, and slot name."""

        state = {}
        corrections = {}
        age = {}
        for group in self.groups:
            for index, param in enumerate(group.params):
                param_key = self._param_keys[id(param)]
                age[param_key] = group.age[index].detach().cpu().clone()
                param_state = {
                    transform_index: {
                        name: _checkpoint_leaf(
                            slab, index, group.state_owner_ranges[transform_index]
                        )
                        for name, slab in slots.items()
                    }
                    for transform_index, slots in enumerate(group.states)
                }
                param_state["commit"] = {
                    name: slab[index].detach().cpu().clone() for name, slab in group.commit_state.items()
                }
                state[param_key] = param_state
                if self.ecc is not None:
                    param_corrections = {
                        transform_index: {
                            name: _checkpoint_leaf(
                                slab, index, group.state_owner_ranges[transform_index]
                            )
                            for name, slab in slots.items()
                        }
                        for transform_index, slots in enumerate(group.state_corrections)
                    }
                    param_corrections["commit"] = {
                        name: slab[index].detach().cpu().clone()
                        for name, slab in group.commit_corrections.items()
                    }
                    corrections[param_key] = param_corrections
        if len(self._steps_by_device) == 1:
            step = {"global": int(self._steps[0].detach().cpu())}
        else:
            step = {str(device): int(value.detach().cpu()) for device, value in self._steps_by_device.items()}
        hyper = {
            key: {name: value.detach().cpu().clone() for name, value in vars(values).items()}
            for key, values in self._hyper_locations.items()
        }
        fingerprint = self._recipe_fingerprints()
        fingerprint[_RNG_CHECKPOINT_KEY] = {
            "seed": self._rng_seed,
            "leaf_indices": {
                self._param_keys[id(param)]: int(group.leaf_indices[index].detach().cpu())
                for group in self.groups
                for index, param in enumerate(group.params)
            },
        }
        state_dict = {
            "format": 4 if self.ecc is not None else 3,
            "train_mode": self._train_mode,
            "step": step,
            "age": age,
            "param_init_pending": dict(self._deferred_param_init_pending),
            "hyper": hyper,
            "state": state,
            "fingerprint": fingerprint,
        }
        if self._cadence is not None:
            probability = self._cadence.probability
            if callable(probability):
                raise ValueError("cannot checkpoint a callable cadence probability schedule")
            if not isinstance(probability, Real) or isinstance(probability, bool):
                raise ValueError("cadence probability must be a number")
            if not 0 <= float(probability) <= 1:
                raise ValueError("cadence probability must be in [0, 1]")
            state_dict["cadence"] = {
                "probability": float(probability),
                "step": self._cadence.step,
                "cumulative": self._cadence.cumulative,
                "compensation": self._cadence.compensation,
            }
        if self.ecc is not None:
            state_dict["ecc"] = 8 if self.ecc is torch.int8 else 16
            state_dict["corrections"] = corrections
        return state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping) -> None:
        """Restore logical state into this Engine's already-allocated physical slabs."""

        if not isinstance(state_dict, Mapping) or state_dict.get("format") not in (2, 3, 4):
            raise ValueError("expected a format-2, format-3, or format-4 Engine state dict")
        checkpoint_format = state_dict["format"]
        saved_fingerprint = state_dict.get("fingerprint")
        if not isinstance(saved_fingerprint, Mapping):
            raise ValueError("checkpoint recipe fingerprint does not match this Engine")
        saved_fingerprint = dict(saved_fingerprint)
        saved_rng = saved_fingerprint.pop(_RNG_CHECKPOINT_KEY, None)
        current_fingerprint = self._recipe_fingerprints()
        if saved_fingerprint != current_fingerprint:
            raise ValueError(
                "checkpoint recipe fingerprint does not match this Engine "
                f"(saved {saved_fingerprint!r}, current {current_fingerprint!r})"
            )
        saved_train_mode = state_dict.get("train_mode", True)
        if not isinstance(saved_train_mode, bool):
            raise ValueError("checkpoint train mode must be a bool")
        has_saved_cadence = "cadence" in state_dict
        saved_cadence = state_dict.get("cadence")
        cadence_state = None
        if checkpoint_format in (3, 4) and (self._cadence is not None) != has_saved_cadence:
            raise ValueError("checkpoint cadence presence does not match this Engine")
        if self._cadence is not None and saved_cadence is not None:
            cadence_names = {"probability", "step", "cumulative", "compensation"}
            if not isinstance(saved_cadence, Mapping) or set(saved_cadence) != cadence_names:
                raise ValueError("checkpoint cadence state does not match this Engine")
            cadence_probability = saved_cadence["probability"]
            cadence_step = saved_cadence["step"]
            cadence_cumulative = saved_cadence["cumulative"]
            cadence_compensation = saved_cadence["compensation"]
            if callable(self._cadence.probability):
                raise ValueError("cannot load a checkpoint into a callable cadence probability schedule")
            if not isinstance(cadence_probability, Real) or isinstance(cadence_probability, bool):
                raise ValueError("checkpoint cadence probability must be a number")
            if not 0 <= float(cadence_probability) <= 1:
                raise ValueError("checkpoint cadence probability must be in [0, 1]")
            if not isinstance(cadence_step, int) or isinstance(cadence_step, bool) or cadence_step < 0:
                raise ValueError("checkpoint cadence step must be a non-negative int")
            if not isinstance(cadence_cumulative, Real) or not isinstance(cadence_compensation, Real):
                raise ValueError("checkpoint cadence accumulators must be numbers")
            cadence_state = (
                float(cadence_probability), cadence_step,
                float(cadence_cumulative), float(cadence_compensation),
            )
        saved_age = state_dict.get("age")
        if not isinstance(saved_age, Mapping) or set(saved_age) != set(self._param_locations):
            raise ValueError("checkpoint ages do not match this Engine")
        saved_param_init_pending = state_dict.get("param_init_pending")
        if saved_param_init_pending is None:
            restored_param_init_pending = {
                param_key: False for param_key in self._deferred_param_init_pending
            }
        elif (
            not isinstance(saved_param_init_pending, Mapping)
            or set(saved_param_init_pending) != set(self._deferred_param_init_pending)
            or any(not isinstance(value, bool) for value in saved_param_init_pending.values())
        ):
            raise ValueError("checkpoint deferred parameter initialization does not match this Engine")
        else:
            restored_param_init_pending = dict(saved_param_init_pending)
        self._validate_deferred_param_init_pending(restored_param_init_pending)
        copies = []
        fills = []
        restored_rng_seed = None
        if saved_rng is not None:
            if not isinstance(saved_rng, Mapping) or set(saved_rng) != {"seed", "leaf_indices"}:
                raise ValueError("checkpoint RNG state does not match this Engine")
            restored_rng_seed = saved_rng["seed"]
            saved_leaf_indices = saved_rng["leaf_indices"]
            if (
                not isinstance(restored_rng_seed, int)
                or isinstance(restored_rng_seed, bool)
                or not 0 <= restored_rng_seed < (1 << 64)
            ):
                raise ValueError("checkpoint RNG seed must be an unsigned 64-bit int")
            if not isinstance(saved_leaf_indices, Mapping) or set(saved_leaf_indices) != set(self._param_locations):
                raise ValueError("checkpoint RNG leaf indices do not match this Engine")
            leaf_values = tuple(saved_leaf_indices.values())
            if any(
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << 63)
                for value in leaf_values
            ) or len(set(leaf_values)) != len(leaf_values):
                raise ValueError("checkpoint RNG leaf indices must be unique non-negative int64 values")
            seed_words = torch.tensor(
                (restored_rng_seed & 0xFFFFFFFF, restored_rng_seed >> 32), dtype=torch.int64
            )
            for target in self._rng_seeds.values():
                copies.append((target, seed_words))
            for param_key, (group, index) in self._param_locations.items():
                copies.append((group.leaf_indices[index], torch.tensor(saved_leaf_indices[param_key])))
        for param_key, (group, index) in self._param_locations.items():
            value = saved_age[param_key]
            target = group.age[index]
            if not isinstance(value, Tensor) or tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"checkpoint age has an incompatible shape for parameter {param_key!r}")
            copies.append((target, value))

        saved_state = state_dict.get("state")
        if not isinstance(saved_state, Mapping) or set(saved_state) != set(self._param_locations):
            raise ValueError("checkpoint parameter keys do not match this Engine")
        saved_ecc = state_dict.get("ecc")
        current_ecc = None if self.ecc is None else (8 if self.ecc is torch.int8 else 16)
        if (checkpoint_format == 4) != (saved_ecc is not None):
            raise ValueError("checkpoint format does not match its ECC configuration")
        if saved_ecc != current_ecc:
            raise ValueError("checkpoint ECC configuration does not match this Engine")
        saved_corrections = state_dict.get("corrections")
        if self.ecc is not None and (
            not isinstance(saved_corrections, Mapping)
            or set(saved_corrections) != set(self._param_locations)
        ):
            raise ValueError("checkpoint correction parameter keys do not match this Engine")
        for param_key, (group, index) in self._param_locations.items():
            saved_transforms = saved_state[param_key]
            transform_indices = set(range(len(group.states)))
            if not isinstance(saved_transforms, Mapping):
                raise ValueError(f"checkpoint transforms do not match parameter {param_key!r}")
            saved_indices = set(saved_transforms)
            if saved_indices == transform_indices and not group.commit_state:
                saved_commit_state = {}
            elif saved_indices == transform_indices | {"commit"}:
                saved_commit_state = saved_transforms["commit"]
            else:
                raise ValueError(f"checkpoint transforms do not match parameter {param_key!r}")
            for transform_index, slots in enumerate(group.states):
                saved_slots = saved_transforms[transform_index]
                if not isinstance(saved_slots, Mapping) or set(saved_slots) != set(slots):
                    raise ValueError(f"checkpoint slots do not match parameter {param_key!r}")
                for name, slab in slots.items():
                    owner_range = group.state_owner_ranges[transform_index]
                    target = _local_checkpoint_leaf(
                        slab, index, owner_range
                    )
                    if target is None:
                        continue
                    value = saved_slots[name]
                    if owner_range is not None:
                        value = _checkpoint_source(value)
                    if (
                        not isinstance(value, Tensor)
                        or tuple(value.shape) != tuple(target.shape)
                        or (name in group.state_corrections[transform_index] and value.dtype is not target.dtype)
                    ):
                        raise ValueError(f"checkpoint slot {name!r} has an incompatible shape or dtype")
                    copies.append((target, value))
            if not isinstance(saved_commit_state, Mapping) or set(saved_commit_state) != set(group.commit_state):
                raise ValueError(f"checkpoint commit slots do not match parameter {param_key!r}")
            for name, slab in group.commit_state.items():
                value = saved_commit_state[name]
                target = slab[index]
                if (
                    not isinstance(value, Tensor)
                    or tuple(value.shape) != tuple(target.shape)
                    or (name in group.commit_corrections and value.dtype is not target.dtype)
                ):
                    raise ValueError(f"checkpoint commit slot {name!r} has an incompatible shape or dtype")
                copies.append((target, value))
            if self.ecc is not None:
                saved_param_corrections = saved_corrections[param_key]
                if not isinstance(saved_param_corrections, Mapping) or set(saved_param_corrections) != (
                    transform_indices | {"commit"}
                ):
                    raise ValueError(f"checkpoint corrections do not match parameter {param_key!r}")
                for transform_index, slots in enumerate(group.state_corrections):
                    saved_slots = saved_param_corrections[transform_index]
                    if not isinstance(saved_slots, Mapping) or set(saved_slots) != set(slots):
                        raise ValueError(f"checkpoint correction slots do not match parameter {param_key!r}")
                    for name, slab in slots.items():
                        owner_range = group.state_owner_ranges[transform_index]
                        target = _local_checkpoint_leaf(
                            slab, index, owner_range
                        )
                        if target is None:
                            continue
                        value = saved_slots[name]
                        if owner_range is not None:
                            value = _checkpoint_source(value)
                        if (
                            not isinstance(value, Tensor)
                            or tuple(value.shape) != tuple(target.shape)
                            or value.dtype is not target.dtype
                        ):
                            raise ValueError(f"checkpoint correction slot {name!r} is incompatible")
                        copies.append((target, value))
                saved_commit_corrections = saved_param_corrections["commit"]
                if (
                    not isinstance(saved_commit_corrections, Mapping)
                    or set(saved_commit_corrections) != set(group.commit_corrections)
                ):
                    raise ValueError(f"checkpoint commit corrections do not match parameter {param_key!r}")
                for name, slab in group.commit_corrections.items():
                    value = saved_commit_corrections[name]
                    target = slab[index]
                    if (
                        not isinstance(value, Tensor)
                        or tuple(value.shape) != tuple(target.shape)
                        or value.dtype is not target.dtype
                    ):
                        raise ValueError(f"checkpoint commit correction slot {name!r} is incompatible")
                    copies.append((target, value))

        saved_hyper = state_dict.get("hyper")
        if not isinstance(saved_hyper, Mapping) or set(saved_hyper) != set(self._hyper_locations):
            raise ValueError("checkpoint hyperparameters do not match this Engine")
        for key, values in self._hyper_locations.items():
            saved_values = saved_hyper[key]
            if not isinstance(saved_values, Mapping) or set(saved_values) != set(vars(values)):
                raise ValueError("checkpoint hyperparameter names do not match this Engine")
            for name, target in vars(values).items():
                value = saved_values[name]
                if isinstance(value, Tensor):
                    if value.ndim != 0:
                        raise ValueError(f"checkpoint hyperparameter {name!r} must be scalar")
                    copies.append((target, value))
                else:
                    fills.append((target, value))

        saved_steps = state_dict.get("step")
        if not isinstance(saved_steps, Mapping):
            raise ValueError("checkpoint step counters do not match this Engine")
        if set(saved_steps) == {"global"}:
            for target in self._steps:
                fills.append((target, saved_steps["global"]))
        elif set(saved_steps) == {str(device) for device in self._steps_by_device}:
            for device, target in self._steps_by_device.items():
                fills.append((target, saved_steps[str(device)]))
        else:
            raise ValueError("checkpoint step counters do not match this Engine")

        staged_copies = []
        for target, value in copies:
            staged_copies.append((target, torch.empty_like(target).copy_(value)))

        # Pre-casting every fill makes checkpoint loads atomic on invalid counters.
        staged_fills = []
        for target, value in fills:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError("checkpoint fill value is not a number")
            if not math.isfinite(float(value)):
                raise ValueError("checkpoint fill value must be finite")
            if not torch.is_floating_point(target) and (value != int(value) or value < 0):
                raise ValueError("checkpoint fill value must be a non-negative integer for a counter")
            try:
                staged_fills.append((target, target.new_full((), value)))
            except (OverflowError, RuntimeError, ValueError) as error:
                raise ValueError(f"checkpoint fill value is out of range for {target.dtype}") from error
        for target, staged in staged_copies:
            target.copy_(staged)
        for target, staged in staged_fills:
            target.copy_(staged)
        if cadence_state is not None:
            (
                self._cadence.probability, self._cadence.step,
                self._cadence.cumulative, self._cadence.compensation,
            ) = cadence_state
        if restored_rng_seed is not None:
            self._rng_seed = restored_rng_seed
        self._deferred_param_init_pending = restored_param_init_pending
        self._train_mode = saved_train_mode


def build(params: Iterable[Tensor], recipe_or_route: Recipe | Route, *,
          param_keys: Sequence[str] | None = None, storage_dtype: torch.dtype | str | None = None,
          ecc: int | str | None = None,
          bindings: Mapping[int, PlainBinding | FSDP2Binding] | None = None,
          _rng_seed: int | None = None, _leaf_indices: Sequence[int] | None = None,
          param_group_ids: Sequence[int] | None = None,
          param_group_hypers: Mapping[int, Mapping[str, object]] | None = None,
          **hyper) -> Engine:
    return Engine(
        params, recipe_or_route, param_keys=param_keys, storage_dtype=storage_dtype, ecc=ecc,
        bindings=bindings, _rng_seed=_rng_seed, _leaf_indices=_leaf_indices,
        param_group_ids=param_group_ids, param_group_hypers=param_group_hypers, **hyper
    )
