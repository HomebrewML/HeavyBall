import argparse
import contextlib
import importlib.util
import inspect
import sys
import types
from enum import StrEnum
from math import prod
from pathlib import Path

import numpy as np
import torch

import heavyball
import heavyball.utils

DEFAULT_SHAPES = ((2048, 2048),) * 32
DEFAULT_FUSIONS = heavyball.utils.fusions


class DType(StrEnum):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


class Library(StrEnum):
    heavyball = "heavyball"
    torch = "torch"


class Compiler(StrEnum):
    heavyball = "heavyball"
    inductor = "inductor"


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(map(int, text.lower().replace("x", " ").split()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}") from e
    if not shape:
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}")
    return shape


def compile_inductor(fn, **kwargs):
    return torch.compile(fn, backend="inductor", **kwargs)


def compiler_module(compiler: Compiler, fusions_path: Path | None):
    if fusions_path:
        if compiler is not Compiler.heavyball:
            raise ValueError("--fusions-path requires --compiler heavyball")
        spec = importlib.util.spec_from_file_location("heavyball._bench_fusions", fusions_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load {fusions_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    if compiler is Compiler.inductor:
        return types.SimpleNamespace(compile=compile_inductor)
    return DEFAULT_FUSIONS


@contextlib.contextmanager
def compiler_context(compiler: Compiler, fusions_path: Path | None):
    prior = heavyball.utils.fusions
    heavyball.utils.fusions = compiler_module(compiler, fusions_path)
    try:
        yield
    finally:
        heavyball.utils.fusions = prior


def optimizer_kwargs(optimizer_cls, library: Library, compile_step: bool, fused: bool | None, update_precond: bool | None) -> dict:
    kwargs = {"compile_step": compile_step, "consume_grad": False} if library is Library.heavyball else {}
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
    compile_step: bool,
    fused: bool | None,
    update_precond: bool | None,
    seed: int,
):
    module = heavyball if library is Library.heavyball else torch.optim
    optimizer_cls = getattr(module, optimizer)
    kwargs = optimizer_kwargs(optimizer_cls, library, compile_step, fused, update_precond)
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
    compiler: Compiler,
    dtype: DType,
    shape: list[tuple[int, ...]] | None,
    fusions_path: Path | None,
    compile_step: bool,
    fused: bool | None,
    update_precond: bool | None,
    steps: int,
    warmup: int,
    windows: int,
    seed: int,
) -> None:
    shapes = DEFAULT_SHAPES if shape is None else tuple(shape)
    with compiler_context(compiler, fusions_path) if library is Library.heavyball else contextlib.nullcontext():
        step = make_step(optimizer, library, dtype, shapes, compile_step, fused, update_precond, seed)
        warm(step, warmup)
        times = [time_step(step, steps) for _ in range(windows)]

    print(f"{library}/{compiler}: {len(shapes)} tensors, {sum(prod(s) for s in shapes)} total params")
    print(f"Median Time: {np.median(times):.3f}µs")


def compare(
    optimizer: str,
    compiler: Compiler,
    dtype: DType,
    shapes: tuple[tuple[int, ...], ...],
    fusions_path: Path | None,
    fused: bool | None,
    update_precond: bool | None,
    steps: int,
    warmup: int,
    windows: int,
    seed: int,
) -> None:
    with compiler_context(Compiler.inductor, None):
        baseline = make_step(optimizer, Library.heavyball, dtype, shapes, True, fused, update_precond, seed)
        warm(baseline, warmup)
    with compiler_context(compiler, fusions_path):
        candidate = make_step(optimizer, Library.heavyball, dtype, shapes, True, fused, update_precond, seed)
        warm(candidate, warmup)
    baseline_times = []
    candidate_times = []
    for index in range(windows):
        variants = ((baseline, baseline_times), (candidate, candidate_times))
        if index % 2:
            variants = variants[::-1]
        for step, times in variants:
            times.append(time_step(step, steps))
    baseline_time, candidate_time = np.median(baseline_times), np.median(candidate_times)
    print(f"{optimizer} comparison: {len(shapes)} tensors, {sum(prod(s) for s in shapes)} total params")
    print(f"Stock Inductor: {baseline_time:.3f}µs")
    print(f"HeavyBall: {candidate_time:.3f}µs")
    print(f"Speedup: {baseline_time / candidate_time:.4f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--library", choices=tuple(Library), default=Library.heavyball)
    parser.add_argument("--compiler", choices=tuple(Compiler), default=Compiler.heavyball)
    parser.add_argument("--dtype", choices=tuple(DType), default=DType.float32)
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument("--fusions-path", type=Path)
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument("--compare", action="store_true")
    fused = parser.add_mutually_exclusive_group()
    fused.add_argument("--fused", dest="fused", action="store_true")
    fused.add_argument("--no-fused", dest="fused", action="store_false")
    parser.set_defaults(fused=None)
    update_precond = parser.add_mutually_exclusive_group()
    update_precond.add_argument("--update-precond", dest="update_precond", action="store_true")
    update_precond.add_argument("--no-update-precond", dest="update_precond", action="store_false")
    parser.set_defaults(update_precond=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    library, compiler, dtype = Library(args.library), Compiler(args.compiler), DType(args.dtype)
    if args.fusions_path and library is not Library.heavyball:
        parser.error("--fusions-path requires --library heavyball")
    if args.update_precond is not None and library is Library.heavyball:
        try:
            optimizer_kwargs(getattr(heavyball, args.optimizer), library, args.compile_step, args.fused, args.update_precond)
        except ValueError as error:
            parser.error(str(error))
    if args.compare:
        if library is not Library.heavyball:
            parser.error("--compare requires --library heavyball")
        compare(
            args.optimizer,
            compiler,
            dtype,
            DEFAULT_SHAPES if args.shape is None else tuple(args.shape),
            args.fusions_path,
            args.fused,
            args.update_precond,
            args.steps,
            args.warmup,
            args.windows,
            args.seed,
        )
    else:
        benchmark(
            args.optimizer,
            library,
            compiler,
            dtype,
            args.shape,
            args.fusions_path,
            args.compile_step,
            args.fused,
            args.update_precond,
            args.steps,
            args.warmup,
            args.windows,
            args.seed,
        )


if __name__ == "__main__":
    main()
