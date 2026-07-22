import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from heavyball import fusions


@dataclass(frozen=True)
class FusionCase:
    name: str
    fn: Callable[..., Any]
    make_args: Callable[[], tuple[Any, ...]]
    reference: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ErrorMetrics:
    invalid: int
    elements: int
    sum_abs: float
    max_abs: float


@dataclass(frozen=True)
class GraphMetrics:
    nodes: int
    fma_nodes: int
    cast_nodes: int
    rewrites: int


@dataclass(frozen=True)
class KernelMetrics:
    generated: int
    sources: int
    triton: int
    load_sites: int
    store_sites: int
    explicit_triton_fmas: int
    fp32_cast_sites: int


@dataclass(frozen=True)
class TimingMetrics:
    median_us: float
    min_us: float
    max_us: float


@dataclass(frozen=True)
class VariantResult:
    graph: GraphMetrics
    kernel: KernelMetrics
    error: ErrorMetrics
    timing: TimingMetrics


@dataclass(frozen=True)
class CaseResult:
    name: str
    baseline: VariantResult
    candidate: VariantResult
    precision_ok: bool
    pointwise_ok: bool
    precision_better: bool
    speedup: float
    measured_win: bool
    accepted: bool


def _fp64(value: Any) -> Any:
    return value.double() if isinstance(value, torch.Tensor) and value.dtype.is_floating_point else value


def _reference_like(value: Any, target: Any) -> Any:
    if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor) and target.dtype.is_floating_point:
        return value.to(dtype=target.dtype)
    return value


def _tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    return pytree.tree_map(fn, value)


def fp64_reference(case: FusionCase, args: tuple[Any, ...] | None = None) -> Any:
    args = case.make_args() if args is None else args
    output_args = _tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value, args)
    reference_args = _tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value, args)
    output = case.fn(*output_args)
    high_precision = case.reference(*reference_args) if case.reference else case.fn(*_tree_map(_fp64, reference_args))
    return pytree.tree_map(_reference_like, high_precision, output)


def error_metrics(value: Any, reference: Any) -> ErrorMetrics:
    values, value_spec = pytree.tree_flatten(value)
    references, reference_spec = pytree.tree_flatten(reference)
    if value_spec != reference_spec or len(values) != len(references):
        return ErrorMetrics(1, 0, math.inf, math.inf)

    invalid = elements = 0
    sum_abs = max_abs = 0.0
    for actual, expected in zip(values, references, strict=True):
        if isinstance(actual, torch.Tensor) != isinstance(expected, torch.Tensor):
            invalid += 1
            continue
        if not isinstance(actual, torch.Tensor):
            invalid += int(actual != expected)
            continue
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            invalid += 1
            continue
        if not actual.dtype.is_floating_point:
            invalid += int(not torch.equal(actual, expected))
            elements += actual.numel()
            continue
        matching_nan = torch.isnan(actual) & torch.isnan(expected)
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        matching_inf = torch.isinf(actual) & (actual == expected)
        zero_sign_mismatch = (
            finite & (actual == 0) & (expected == 0) & (torch.signbit(actual) != torch.signbit(expected))
        )
        valid = matching_nan | (finite & ~zero_sign_mismatch) | matching_inf
        invalid += (~valid).sum().item()
        elements += finite.sum().item()
        if finite.any():
            error = (actual[finite].double() - expected[finite].double()).abs()
            sum_abs += error.sum().item()
            max_abs = max(max_abs, error.max().item())
    return ErrorMetrics(invalid, elements, sum_abs, max_abs)


def _same_signature(candidate: Any, baseline: Any) -> bool:
    candidates, candidate_spec = pytree.tree_flatten(candidate)
    baselines, baseline_spec = pytree.tree_flatten(baseline)
    if candidate_spec != baseline_spec or len(candidates) != len(baselines):
        return False
    for value, expected in zip(candidates, baselines, strict=True):
        if isinstance(value, torch.Tensor) != isinstance(expected, torch.Tensor):
            return False
        if isinstance(value, torch.Tensor) and (value.shape != expected.shape or value.dtype != expected.dtype):
            return False
    return True


def precision_pareto(candidate: Any, baseline: Any, reference: Any) -> bool:
    candidates, candidate_spec = pytree.tree_flatten(candidate)
    baselines, baseline_spec = pytree.tree_flatten(baseline)
    references, reference_spec = pytree.tree_flatten(reference)
    if candidate_spec != baseline_spec or candidate_spec != reference_spec:
        return False
    for value, plain, expected in zip(candidates, baselines, references, strict=True):
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(plain, torch.Tensor)
            or not isinstance(expected, torch.Tensor)
        ):
            if value != expected:
                return False
            continue
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or plain.shape != expected.shape
            or plain.dtype != expected.dtype
        ):
            return False
        if not value.dtype.is_floating_point:
            if not torch.equal(value, expected):
                return False
            continue
        finite = torch.isfinite(expected)
        infinite = torch.isinf(expected)
        nan = torch.isnan(expected)
        if (nan & ~torch.isnan(value)).any():
            return False
        if (infinite & (value != expected)).any():
            return False
        zero = finite & (value == 0) & (expected == 0)
        if (zero & (torch.signbit(value) != torch.signbit(expected))).any():
            return False
        if ((~torch.isfinite(value)) & finite).any():
            return False
        comparable = finite & torch.isfinite(value) & torch.isfinite(plain)
        if (
            comparable.any()
            and (
                (value[comparable].double() - expected[comparable].double()).abs()
                > (plain[comparable].double() - expected[comparable].double()).abs()
            ).any()
        ):
            return False
    return True


def _graph_metrics(case: FusionCase, enabled: bool, args: tuple[Any, ...] | None = None) -> tuple[GraphMetrics, str]:
    graph_module = make_fx(case.fn)(*(case.make_args() if args is None else args))
    pass_fn = getattr(fusions, "post_grad_custom_pre_pass", None) if enabled else None
    rewrites = pass_fn(graph_module.graph) if pass_fn else 0
    targets = [str(node.target) for node in graph_module.graph.nodes if node.op == "call_function"]
    return (
        GraphMetrics(
            nodes=len(tuple(graph_module.graph.nodes)),
            fma_nodes=sum("fma" in target for target in targets),
            cast_nodes=sum("_to_copy" in target or "convert_element_type" in target for target in targets),
            rewrites=rewrites,
        ),
        str(graph_module.graph),
    )


def _kernel_metrics(sources: list[str], generated: int) -> KernelMetrics:
    source = "\n".join(sources)
    return KernelMetrics(
        generated=generated,
        sources=len(sources),
        triton=source.count("async_compile.triton("),
        load_sites=source.count("tl.load("),
        store_sites=source.count("tl.store("),
        explicit_triton_fmas=source.count("tl.fma("),
        fp32_cast_sites=source.count(".to(tl.float32)"),
    )


def _capture(compiled: Callable[..., Any], args: tuple[Any, ...]) -> tuple[Any, KernelMetrics, list[str]]:
    from torch._inductor import metrics
    from torch._inductor.utils import run_and_get_code

    metrics.reset()
    output, sources = run_and_get_code(compiled, *args)
    return output, _kernel_metrics(sources, metrics.generated_kernel_count), sources


def _device(args: tuple[Any, ...]) -> torch.device:
    tensors = [value for value in pytree.tree_leaves(args) if isinstance(value, torch.Tensor)]
    return tensors[0].device if tensors else torch.device("cpu")


def _time_window(compiled: Callable[..., Any], args: tuple[Any, ...], device: torch.device, iterations: int) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            compiled(*args)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / iterations
    start = time.perf_counter()
    for _ in range(iterations):
        compiled(*args)
    return (time.perf_counter() - start) * 1e6 / iterations


def _timing_metrics(times: list[float]) -> TimingMetrics:
    return TimingMetrics(statistics.median(times), min(times), max(times))


def _pair_latency(
    baseline: Callable[..., Any],
    candidate: Callable[..., Any],
    args: tuple[Any, ...],
    warmup: int,
    iterations: int,
    windows: int,
) -> tuple[TimingMetrics, TimingMetrics]:
    for _ in range(warmup):
        baseline(*args)
        candidate(*args)
    device = _device(args)
    baseline_times = []
    candidate_times = []
    for index in range(windows):
        variants = ((baseline, baseline_times), (candidate, candidate_times))
        if index % 2:
            variants = variants[::-1]
        for compiled, times in variants:
            times.append(_time_window(compiled, args, device, iterations))
    return _timing_metrics(baseline_times), _timing_metrics(candidate_times)


def _compile(fn: Callable[..., Any], enabled: bool) -> Callable[..., Any]:
    return fusions.compile(fn, fullgraph=True) if enabled else torch.compile(fn, backend="inductor", fullgraph=True)


def _dump_sources(directory: Path | None, case: str, variant: str, sources: list[str]) -> None:
    if directory is None:
        return
    target = directory / case / variant
    target.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources):
        (target / f"graph_{index:02d}.py").write_text(source)


def _dump_graph(directory: Path | None, case: str, variant: str, graph: str) -> None:
    if directory is None:
        return
    target = directory / case / variant
    target.mkdir(parents=True, exist_ok=True)
    (target / "graph.txt").write_text(graph + "\n")


def evaluate(
    case: FusionCase,
    warmup: int = 20,
    iterations: int = 200,
    windows: int = 11,
    dump_dir: Path | None = None,
    min_speedup: float = 1.02,
) -> CaseResult:
    args = case.make_args()
    reference = fp64_reference(case, args)
    baseline_graph, baseline_graph_source = _graph_metrics(case, False, args)
    candidate_graph, candidate_graph_source = _graph_metrics(case, True, args)
    _dump_graph(dump_dir, case.name, "baseline", baseline_graph_source)
    _dump_graph(dump_dir, case.name, "candidate", candidate_graph_source)
    baseline_compiled = _compile(case.fn, False)
    baseline_output, baseline_kernel, baseline_sources = _capture(baseline_compiled, args)
    _dump_sources(dump_dir, case.name, "baseline", baseline_sources)
    candidate_compiled = _compile(case.fn, True)
    candidate_output, candidate_kernel, candidate_sources = _capture(candidate_compiled, args)
    _dump_sources(dump_dir, case.name, "candidate", candidate_sources)
    baseline_timing, candidate_timing = _pair_latency(
        baseline_compiled, candidate_compiled, args, warmup, iterations, windows
    )
    baseline = VariantResult(
        baseline_graph,
        baseline_kernel,
        error_metrics(baseline_output, reference),
        baseline_timing,
    )
    candidate = VariantResult(
        candidate_graph,
        candidate_kernel,
        error_metrics(candidate_output, reference),
        candidate_timing,
    )
    same_signature = _same_signature(candidate_output, baseline_output)
    pointwise_ok = same_signature and precision_pareto(candidate_output, baseline_output, reference)
    precision_ok = (
        same_signature
        and candidate.error.invalid <= baseline.error.invalid
        and candidate.error.sum_abs <= baseline.error.sum_abs
    )
    is_better = candidate.error.invalid < baseline.error.invalid or (
        candidate.error.invalid == baseline.error.invalid and candidate.error.sum_abs < baseline.error.sum_abs
    )
    speedup = baseline.timing.median_us / candidate.timing.median_us
    measured_win = speedup >= min_speedup
    no_op = sorted(candidate_sources) == sorted(baseline_sources)
    accepted = precision_ok and ((measured_win and not no_op) or (is_better and speedup >= 1))
    return CaseResult(
        case.name,
        baseline,
        candidate,
        precision_ok,
        pointwise_ok,
        is_better,
        speedup,
        measured_win,
        accepted,
    )


def _affine_integer(x):
    return ((x * 3 + 2) * 4 + 5) * 6 + 7


def _affine_rounding(x):
    x = -x
    x = 0.03 - x
    x = x - (-0.1)
    x = x / 0.1
    x = x - 0.03
    return x + 0.1


def _fma(a, b, c):
    return a * b + c


def _lerp(x, y):
    return torch.lerp(x, y, 0.125)


def _mixed_products(a, b, c, d):
    return a * b + c * d


def _scalar_alpha(base, factor):
    return torch.add(base, factor, alpha=0.1)


def _adam(param, exp_avg, exp_avg_sq, grad, beta1, beta2, lr, eps):
    exp_avg = exp_avg * beta1 + grad * (1 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)
    return param - lr * exp_avg / exp_avg_sq.sqrt().clamp_min(eps), exp_avg, exp_avg_sq


def _factory(*values: Any) -> Callable[[], tuple[Any, ...]]:
    def make_args() -> tuple[Any, ...]:
        return tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in values)

    return make_args


def corpus(size: int, dtype: torch.dtype, device: torch.device, seed: int = 0) -> tuple[FusionCase, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def random(dtype_: torch.dtype = dtype):
        return torch.randn(size, generator=generator, device=device, dtype=dtype_)

    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, device=device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)

    affine_input = random()
    affine_input[0] = 0.3339076638221741

    return (
        FusionCase("affine_integer", _affine_integer, _factory(random())),
        FusionCase("affine_rounding", _affine_rounding, _factory(affine_input)),
        FusionCase("fma", _fma, _factory(random(), random(), random())),
        FusionCase("lerp", _lerp, _factory(random(), random())),
        FusionCase(
            "mixed_products",
            _mixed_products,
            _factory(random(dtype), random(dtype), random(torch.float32), random(torch.float32)),
        ),
        FusionCase("scalar_alpha", _scalar_alpha, _factory(random(torch.float32), random(dtype))),
        FusionCase(
            "adam",
            _adam,
            _factory(
                random(), random(), random().abs(), random(), scalar(0.9), scalar(0.99), scalar(0.01), scalar(1e-8)
            ),
        ),
    )


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _device_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        result.update(
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        )
    return result


def _load_fusions(path: Path | None) -> None:
    if path is None:
        return
    spec = importlib.util.spec_from_file_location("heavyball._bench_fusions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    global fusions
    fusions = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=("float16", "bfloat16", "float32", "float64"))
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--windows", type=int, default=11)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fusions-path", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-win", action="store_true")
    args = parser.parse_args()
    _load_fusions(args.fusions_path)
    device, dtype = _device_arg(args.device), _dtype(args.dtype)
    cases = corpus(args.size, dtype, device, args.seed)
    selected = {name for name in args.case} if args.case else None
    cases = tuple(case for case in cases if selected is None or case.name in selected)
    unknown = selected.difference(case.name for case in cases) if selected else set()
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
    results = [
        evaluate(case, args.warmup, args.iterations, args.windows, args.dump_dir, args.min_speedup) for case in cases
    ]
    payload = {
        "environment": _environment(device),
        "config": {
            "dtype": args.dtype,
            "reference": "float64" if dtype != torch.float64 else "input precision",
            "size": args.size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "windows": args.windows,
            "min_speedup": args.min_speedup,
            "fusions_path": str(args.fusions_path) if args.fusions_path else None,
        },
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.strict and any(not result.pointwise_ok for result in results):
        raise SystemExit("A strict precision case regressed")
    if args.require_win and (
        not any(result.accepted for result in results) or any(not result.precision_ok for result in results)
    ):
        raise SystemExit("No accepted hillclimb win")


if __name__ == "__main__":
    main()
