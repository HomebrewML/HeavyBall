#!/usr/bin/env python3
"""
Utility to migrate HeavyBall 1.x optimizer state dicts to the 2.0.0 layout.

The script rewrites per-parameter state keys to the new transform-indexed names,
reshapes state storage so each parameter-view owns its own dictionary, and
injects the HeavyBall-specific metadata block expected by 2.0.0 optimizers.
"""

from __future__ import annotations

import functools
import importlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import typer


@dataclass
class TransformMapping:
    old_key: str
    new_key: str
    transform_idx: int


def _load_optimizer_class(qualified_name: str):
    if "." in qualified_name:
        module_name, class_name = qualified_name.rsplit(".", 1)
    else:
        module_name, class_name = "heavyball", qualified_name
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Optimizer class '{qualified_name}' not found") from exc


def _guess_tensor_meta(state_entry: Dict[str, Any]) -> Tuple[Tuple[int, ...], torch.dtype]:
    for value in state_entry.values():
        tensor = None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            tensor = value[0]
        if tensor is not None:
            return tuple(tensor.shape), tensor.dtype
    return (1,), torch.float32


def _build_dummy_parameters(state: Dict[int, Dict[str, Any]], param_groups: Sequence[Dict[str, Any]]):
    max_pid = -1
    for group in param_groups:
        max_pid = max(max_pid, *group.get("params", []))
    params: Dict[int, torch.nn.Parameter] = {}
    for pid in range(max_pid + 1):
        shape, dtype = _guess_tensor_meta(state.get(pid, {}))
        if not shape:
            shape = (1,)
        tensor = torch.zeros(shape, dtype=dtype)
        params[pid] = torch.nn.Parameter(tensor)
    return params


def _normalise_group_options(group: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for key, value in group.items():
        if key == "params":
            continue
        if isinstance(value, list) and key in {"betas", "weight_decay_steps"}:
            options[key] = tuple(value)
        else:
            options[key] = value
    return options


def _instantiate_optimizer(opt_class, state_dict: Dict[str, Any]):
    param_groups = state_dict["param_groups"]
    params = _build_dummy_parameters(state_dict["state"], param_groups)
    groups: List[Dict[str, Any]] = []
    for group in param_groups:
        options = _normalise_group_options(group)
        mapped_params = [params[pid] for pid in group.get("params", [])]
        groups.append({"params": mapped_params, **options})
    if not groups:
        raise ValueError("No parameter groups found in optimizer state")
    optimizer = opt_class(groups)
    return optimizer, params


def _collect_transform_mappings(optimizer) -> List[TransformMapping]:
    from heavyball import chainable as C

    def walk(queue: Iterable[Any]):
        stack = list(queue)
        while stack:
            current = stack.pop()
            if isinstance(current, C.FunctionTransform):
                yield current
                stack.append(current.fn)
            elif isinstance(current, functools.partial):  # type: ignore[name-defined]
                stack.append(current.func)
            elif isinstance(current, C.Branch):
                for branch in current.branches:
                    stack.extend(branch)
            elif isinstance(current, (list, tuple)):
                stack.extend(current)

    mappings: List[TransformMapping] = []
    seen = set()
    for transform in walk(getattr(optimizer, "_fns", [])):
        names = getattr(transform, "names", [])
        for name in names:
            if not isinstance(name, str):
                continue
            old_key = f"{transform.fn_name}_{name}"
            new_key = transform.val_name(name)
            key = (old_key, new_key, transform.transform_idx)
            if key in seen:
                continue
            seen.add(key)
            mappings.append(TransformMapping(old_key, new_key, transform.transform_idx))
    return mappings


def _ensure_set(value: Any) -> set:
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return set(value)
    if value is None:
        return set()
    return {value}


def _assign_value(
    target: Dict[int, Dict[str, Any]], key: str, value: Any, mark_initialized: Dict[int, set], idx: int | None
):
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
        for view_idx, tensor in enumerate(value):
            bucket = target.setdefault(view_idx, {})
            bucket[key] = tensor
            if idx is not None:
                mark_initialized.setdefault(view_idx, set()).add(idx)
    else:
        bucket = target.setdefault(0, {})
        bucket[key] = value
        if idx is not None:
            mark_initialized.setdefault(0, set()).add(idx)


def _migrate_single_state(entry: Dict[str, Any], mappings: List[TransformMapping]) -> Dict[int, Dict[str, Any]]:
    migrated: Dict[int, Dict[str, Any]] = {}
    mark_init: Dict[int, set] = {}
    remaining = dict(entry)

    for mapping in mappings:
        if mapping.old_key in remaining:
            value = remaining.pop(mapping.old_key)
            _assign_value(migrated, mapping.new_key, value, mark_init, mapping.transform_idx)

    for key, value in remaining.items():
        if key == "is_initialized":
            existing = _ensure_set(value)
            if existing:
                mark_init.setdefault(0, set()).update(existing)
            continue
        _assign_value(migrated, key, value, mark_init, None)

    for idx, bucket in migrated.items():
        initialized = mark_init.get(idx, set())
        if initialized:
            bucket["is_initialized"] = sorted(initialized)
        elif "is_initialized" in bucket:
            bucket["is_initialized"] = sorted(_ensure_set(bucket["is_initialized"]))
    return migrated


def migrate_state_dict(old_state: Dict[str, Any], optimizer_class: str) -> Dict[str, Any]:
    opt_cls = _load_optimizer_class(optimizer_class)
    optimizer, _ = _instantiate_optimizer(opt_cls, old_state)
    template = optimizer.state_dict()
    mappings = _collect_transform_mappings(optimizer)

    new_state: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for pid, entry in old_state["state"].items():
        new_state[pid] = _migrate_single_state(entry, mappings)

    migrated = dict(template)
    migrated["param_groups"] = old_state["param_groups"]
    migrated_state: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for pid, bucket in new_state.items():
        migrated_state[pid] = {int(idx): dict(values) for idx, values in bucket.items()}
    migrated["state"] = migrated_state

    heavyball_meta = migrated.setdefault("heavyball", template.get("heavyball", {}))
    if "inner_group" not in heavyball_meta:
        heavyball_meta["inner_group"] = {"stochastic_schedule": None}
    if "stochastic_schedule" not in heavyball_meta:
        heavyball_meta["stochastic_schedule"] = None
    if "precond_rng" not in heavyball_meta:
        heavyball_meta["precond_rng"] = pickle.dumps(random.Random(0x12312))
    return migrated


def _resolve_state_container(root: Dict[str, Any], key_path: Sequence[str]) -> Dict[str, Any]:
    container = root
    for key in key_path:
        if key not in container:
            raise KeyError(f"Key '{'.'.join(key_path)}' not found in checkpoint")
        container = container[key]
    if not isinstance(container, dict) or "state" not in container or "param_groups" not in container:
        raise ValueError(f"Target at '{'.'.join(key_path)}' is not an optimizer state dict")
    return container


app = typer.Typer(help="Utilities for migrating HeavyBall optimizer checkpoints.")


@app.command(help="Migrate a HeavyBall optimizer state dict to the 2.0.0 layout.")
def migrate(
    checkpoint: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the checkpoint (.pt/.pth) containing the optimizer state",
    ),
    optimizer_class: str = typer.Argument(
        ...,
        help="Optimizer class to instantiate (e.g., heavyball.ForeachAdamW)",
    ),
    state_key: str = typer.Option(
        "optimizer",
        "--state-key",
        help="Dot-separated key to the optimizer state inside the checkpoint",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for the migrated checkpoint (defaults to overwriting the input)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Perform migration without writing the result"),
) -> None:
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    key_path = [k for k in state_key.split(".") if k]
    state_container = _resolve_state_container(checkpoint_data, key_path)

    migrated = migrate_state_dict(state_container, optimizer_class)
    state_container.clear()
    state_container.update(migrated)

    if dry_run:
        typer.echo("Dry run complete; no file written.")
        return

    output_path = output or checkpoint
    torch.save(checkpoint_data, output_path)
    typer.echo(f"Migrated checkpoint written to {output_path}")


if __name__ == "__main__":
    app()
