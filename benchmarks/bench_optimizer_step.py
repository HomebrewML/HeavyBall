import inspect
from enum import StrEnum
from math import prod
from typing import Annotated

import numpy as np
import torch
import typer

import heavyball

DEFAULT_SHAPES = ((2048, 2048),) * 32
app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


class DType(StrEnum):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


class Library(StrEnum):
    heavyball = "heavyball"
    torch = "torch"


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(map(int, text.lower().replace("x", " ").split()))
    except ValueError as e:
        raise typer.BadParameter(f"invalid shape: {text!r}", param_hint="--shape") from e
    if not shape:
        raise typer.BadParameter(f"invalid shape: {text!r}", param_hint="--shape")
    return shape


def optimizer_kwargs(
    optimizer_cls, library: Library, fused: bool | None, update_precond: bool | None
) -> dict:
    kwargs = {}
    if fused is not None and library is Library.torch:
        kwargs["fused"] = fused
    if update_precond is not None and library is Library.heavyball:
        if "preconditioner_update_probability" not in inspect.signature(optimizer_cls).parameters:
            raise ValueError(f"--update-precond is unsupported by {optimizer_cls.__name__}")
        kwargs["preconditioner_update_probability"] = float(update_precond)
    return kwargs


def make_step(
    optimizer: str,
    library: Library,
    dtype: DType,
    shapes: tuple[tuple[int, ...], ...],
    fused: bool | None,
    update_precond: bool | None,
    seed: int,
):
    module = heavyball if library is Library.heavyball else torch.optim
    optimizer_cls = getattr(module, optimizer)
    kwargs = optimizer_kwargs(optimizer_cls, library, fused, update_precond)
    torch_dtype = getattr(torch, dtype)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    params = []
    for dims in shapes:
        param = torch.nn.Parameter(torch.randn(dims, device="cuda", dtype=torch_dtype, generator=generator))
        param.grad = torch.randn(dims, device="cuda", dtype=torch_dtype, generator=generator)
        params.append(param)
    return optimizer_cls(params, **kwargs).step


def warm(step, steps: int) -> None:
    for _ in range(steps):
        step()


def time_step(step, steps: int) -> float:
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / steps


def benchmark(
    optimizer: str,
    library: Library,
    dtype: DType,
    shape: list[tuple[int, ...]] | None,
    fused: bool | None,
    update_precond: bool | None,
    steps: int,
    warmup: int,
    windows: int,
    seed: int,
) -> None:
    shapes = DEFAULT_SHAPES if shape is None else tuple(shape)
    step = make_step(optimizer, library, dtype, shapes, fused, update_precond, seed)
    warm(step, warmup)
    times = [time_step(step, steps) for _ in range(windows)]

    print(f"{library}: {len(shapes)} tensors, {sum(prod(s) for s in shapes)} total params")
    print(f"Median Time: {np.median(times):.3f}µs")


@app.command()
def main(
    optimizer: Annotated[str, typer.Option()] = "AdamW",
    library: Annotated[Library, typer.Option()] = Library.heavyball,
    dtype: Annotated[DType, typer.Option()] = DType.float32,
    shape: Annotated[list[str] | None, typer.Option()] = None,
    fused: Annotated[bool, typer.Option("--fused")] = False,
    no_fused: Annotated[bool, typer.Option("--no-fused")] = False,
    update_precond: Annotated[bool, typer.Option("--update-precond")] = False,
    no_update_precond: Annotated[bool, typer.Option("--no-update-precond")] = False,
    steps: Annotated[int, typer.Option()] = 300,
    warmup: Annotated[int, typer.Option()] = 20,
    windows: Annotated[int, typer.Option()] = 6,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    shapes = None if shape is None else [parse_shape(value) for value in shape]
    if fused and no_fused:
        raise typer.BadParameter("--fused and --no-fused are mutually exclusive")
    if update_precond and no_update_precond:
        raise typer.BadParameter("--update-precond and --no-update-precond are mutually exclusive")
    fused_value = True if fused else False if no_fused else None
    update_precond_value = True if update_precond else False if no_update_precond else None
    if update_precond_value is not None and library is Library.heavyball:
        try:
            optimizer_kwargs(getattr(heavyball, optimizer), library, fused_value, update_precond_value)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--update-precond") from error
    benchmark(
        optimizer,
        library,
        dtype,
        shapes,
        fused_value,
        update_precond_value,
        steps,
        warmup,
        windows,
        seed,
    )


if __name__ == "__main__":
    app()
