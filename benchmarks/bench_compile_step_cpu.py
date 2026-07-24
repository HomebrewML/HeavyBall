"""Cold/steady CPU benchmark for HeavyBall's compiled whole-step artifacts.

Each benchmark invocation owns a fresh Inductor cache. Run one optimizer per
process so the reported cold step cannot reuse another optimizer's graph:

    CUDA_VISIBLE_DEVICES="" python benchmarks/bench_compile_step_cpu.py --optimizer AdamW

Use ``--compile-mode max-autotune-no-cudagraphs --disable-cse`` to reproduce
the former compile policy, or ``--disable-cse`` to isolate the FX pass.
"""

import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer


class Optimizer(StrEnum):
    AdamW = "AdamW"
    SOAP = "SOAP"
    PSGDKron = "PSGDKron"
    Muon = "Muon"


class CompileMode(StrEnum):
    current = "current"
    default = "default"
    reduce_overhead = "reduce-overhead"
    max_autotune_no_cudagraphs = "max-autotune-no-cudagraphs"


@dataclass(frozen=True)
class Options:
    optimizer: Optimizer
    compile_mode: CompileMode
    warmup: int
    steps: int
    trajectory_steps: int
    trajectory_out: Path | None
    trajectory_only: bool
    group_fusion: bool
    disable_cse: bool
    workflow: bool
    seed: int


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

CACHE_ROOT = (
    Path(tempfile.mkdtemp(prefix="heavyball-compile-step-cpu-"))
    if __name__ == "__main__"
    else None
)
if CACHE_ROOT is not None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_ROOT)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch._inductor.metrics as inductor_metrics  # noqa: E402

import heavyball  # noqa: E402
import heavyball.core as core  # noqa: E402

# Four matrix weights and transformer-style vector parameters. Repeated vector
# shapes exercise slab bucketing as well as the matrix routes.
SHAPES = (
    (24, 8),
    (24,),
    (8, 8),
    (8,),
    (32, 8),
    (32,),
    (8, 32),
    (8,),
    (8,),
    (8,),
)


def _recompile_events() -> int:
    return sum(int(value) for value in torch._dynamo.utils.counters.get("recompiles", {}).values())


def _counter_snapshot() -> dict[str, dict[str, int]]:
    return {
        category: {name: int(value) for name, value in values.items()}
        for category, values in torch._dynamo.utils.counters.items()
        if values
    }


def _parameters(seed: int) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=torch.float32))
        for shape in SHAPES
    ]


def _install_compile_override(mode: str, group_fusion: bool, disable_cse: bool) -> None:
    if mode == "current" and not group_fusion and not disable_cse:
        return
    original_compile = torch.compile

    def configured_compile(function, **kwargs):
        if mode == "default" or group_fusion:
            kwargs.pop("mode", None)
        elif mode != "current":
            kwargs["mode"] = mode
        if group_fusion:
            kwargs["options"] = {"group_fusion": True}
        elif disable_cse:
            kwargs.pop("options", None)
        return original_compile(function, **kwargs)

    core.torch.compile = configured_compile


def _make_optimizer(params: list[torch.nn.Parameter], optimizer: Optimizer):
    optimizer_type = getattr(heavyball, optimizer)
    kwargs = {"lr": 1e-3}
    if optimizer is Optimizer.SOAP:
        kwargs["preconditioner_update_probability"] = 0.5
    elif optimizer is Optimizer.PSGDKron:
        kwargs["preconditioner_update_probability"] = 1.0
        kwargs["max_size_triangular"] = 64
    return optimizer_type(params, **kwargs)


def _install_constant_gradients(params: list[torch.nn.Parameter], seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for param in params:
        param.grad.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))


def _trajectory(params: list[torch.nn.Parameter], optimizer, steps: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(steps):
        for param in params:
            param.grad.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))
        optimizer.step()
        values.append(torch.cat([param.detach().reshape(-1).cpu() for param in params]))
    return values


def _workflow(args: Options) -> dict[str, object]:
    if args.optimizer is not Optimizer.AdamW:
        raise ValueError("--workflow currently measures AdamW")
    params = _parameters(args.seed)
    optimizer = _make_optimizer(params, args.optimizer)
    _install_constant_gradients(params, args.seed + 1)

    stages = {}

    def step(name: str) -> None:
        started = time.perf_counter()
        optimizer.step()
        stages[name] = {
            "seconds": time.perf_counter() - started,
            "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
            "recompiles": _recompile_events(),
        }

    step("cold")
    step("repeat")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler.step()
    step("scheduler_lr")

    optimizer.eval()
    optimizer.train()
    step("train_eval")

    optimizer.zero_grad()
    scaler = torch.amp.GradScaler("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = sum(param.square().mean() for param in params)
    scaler.scale(loss).backward()
    started = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    stages["amp"] = {
        "seconds": time.perf_counter() - started,
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "recompiles": _recompile_events(),
    }

    added = torch.nn.Parameter(torch.randn(7))
    started = time.perf_counter()
    optimizer.add_param_group({"params": [added]})
    stages["add_param_group_build"] = {
        "seconds": time.perf_counter() - started,
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "recompiles": _recompile_events(),
    }
    for param in (*params, added):
        param.grad.fill_(0.25)
    step("add_param_group_first_step")
    return {
        "optimizer": args.optimizer,
        "compile_mode": args.compile_mode,
        "cse": not args.disable_cse,
        "loadavg": os.getloadavg(),
        "stages": stages,
        "counters": _counter_snapshot(),
    }


def _run(args: Options) -> None:
    if args.warmup < 0 or args.steps <= 0 or args.trajectory_steps < 0:
        raise ValueError("warmup/trajectory-steps must be non-negative and steps must be positive")
    if (args.trajectory_out is None) != (args.trajectory_steps == 0):
        raise ValueError("--trajectory-out and a positive --trajectory-steps must be supplied together")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    inductor_metrics.reset()
    _install_compile_override(args.compile_mode, args.group_fusion, args.disable_cse)

    if args.workflow:
        print(json.dumps(_workflow(args), sort_keys=True), flush=True)
        return

    if args.trajectory_only:
        if not args.trajectory_steps or args.trajectory_out is None:
            raise ValueError("--trajectory-only requires --trajectory-steps and --trajectory-out")
        trajectory_params = _parameters(args.seed)
        trajectory_optimizer = _make_optimizer(trajectory_params, args.optimizer)
        trajectory = _trajectory(
            trajectory_params,
            trajectory_optimizer,
            args.trajectory_steps,
            args.seed + 2,
        )
        torch.save(
            {
                "optimizer": args.optimizer,
                "seed": args.seed,
                "shapes": SHAPES,
                "trajectory": trajectory,
            },
            args.trajectory_out,
        )
        return

    params = _parameters(args.seed)
    optimizer = _make_optimizer(params, args.optimizer)
    _install_constant_gradients(params, args.seed + 1)

    load_start = os.getloadavg()
    started = time.perf_counter()
    optimizer.step()
    first_step_s = time.perf_counter() - started
    first_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    first_kernels = int(inductor_metrics.generated_kernel_count)
    first_recompiles = _recompile_events()

    for _ in range(args.warmup):
        optimizer.step()

    elapsed_ns = []
    for _ in range(args.steps):
        started_ns = time.perf_counter_ns()
        optimizer.step()
        elapsed_ns.append(time.perf_counter_ns() - started_ns)

    result = {
        "optimizer": args.optimizer,
        "compile_mode": args.compile_mode,
        "group_fusion": args.group_fusion,
        "cse": not args.disable_cse,
        "core_file": str(Path(core.__file__).resolve()),
        "torch": torch.__version__,
        "shapes": [list(shape) for shape in SHAPES],
        "threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "steps": args.steps,
        "first_step_s": first_step_s,
        "steady_median_us": statistics.median(elapsed_ns) / 1e3,
        "steady_mean_us": statistics.mean(elapsed_ns) / 1e3,
        "unique_graphs_first": first_graphs,
        "unique_graphs_final": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "recompiles_first": first_recompiles,
        "recompiles_final": _recompile_events(),
        "kernels_first": first_kernels,
        "kernels_final": int(inductor_metrics.generated_kernel_count),
        "ir_nodes_pre_fusion_final": int(inductor_metrics.ir_nodes_pre_fusion),
        "compiled_artifacts": sorted(optimizer._engine.compiled_steps),
        "loadavg_start": load_start,
        "loadavg_end": os.getloadavg(),
        "counters": _counter_snapshot(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)

    if args.trajectory_steps:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        trajectory_params = _parameters(args.seed)
        trajectory_optimizer = _make_optimizer(trajectory_params, args.optimizer)
        trajectory = _trajectory(
            trajectory_params,
            trajectory_optimizer,
            args.trajectory_steps,
            args.seed + 2,
        )
        torch.save(
            {
                "optimizer": args.optimizer,
                "seed": args.seed,
                "shapes": SHAPES,
                "trajectory": trajectory,
            },
            args.trajectory_out,
        )


@app.command()
def main(
    optimizer: Annotated[Optimizer, typer.Option()],
    compile_mode: Annotated[CompileMode, typer.Option()] = CompileMode.current,
    warmup: Annotated[int, typer.Option()] = 20,
    steps: Annotated[int, typer.Option()] = 50,
    trajectory_steps: Annotated[int, typer.Option()] = 0,
    trajectory_out: Annotated[Path | None, typer.Option()] = None,
    trajectory_only: Annotated[bool, typer.Option("--trajectory-only")] = False,
    group_fusion: Annotated[bool, typer.Option("--group-fusion")] = False,
    disable_cse: Annotated[bool, typer.Option("--disable-cse")] = False,
    workflow: Annotated[bool, typer.Option("--workflow")] = False,
    seed: Annotated[int, typer.Option()] = 20260724,
) -> None:
    _run(
        Options(
            optimizer=optimizer,
            compile_mode=compile_mode,
            warmup=warmup,
            steps=steps,
            trajectory_steps=trajectory_steps,
            trajectory_out=trajectory_out,
            trajectory_only=trajectory_only,
            group_fusion=group_fusion,
            disable_cse=disable_cse,
            workflow=workflow,
            seed=seed,
        )
    )


if __name__ == "__main__":
    try:
        app()
    finally:
        if CACHE_ROOT is not None:
            shutil.rmtree(CACHE_ROOT, ignore_errors=True)
