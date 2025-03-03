# main.py
import multiprocessing
import re
import traceback
from datetime import datetime

import numpy as np
import typer

app = typer.Typer()


def last_match(pattern, text):
    matches = re.findall(pattern, text)
    if not matches:
        return None
    last = matches[-1]
    return float(last)


def run_benchmark(script, opt, steps, dtype, trials):
    base = {"name": script.replace(".py", ""), "opt": opt}

    import io
    import pathlib
    import sys
    import time

    sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    start_time = time.time()
    try:
        module_name = script.replace(".py", "")
        module = __import__(module_name)

        # Build arguments
        arguments = {
            "method": ["qr"],
            "dtype": [dtype],
            "steps": steps,
            "weight_decay": 0,
            "opt": [opt],
            "trials": trials,
            "win_condition_multiplier": 1.0,
        }
        # Run the main function
        module.main(**arguments)
    except Exception:
        output = sys.stdout.getvalue()
        error = traceback.format_exc()
    else:
        output = sys.stdout.getvalue()
        error = ""
    finally:
        sys.stdout = stdout

    # Parse output
    success = "Successfully found the minimum." in output

    runtime = last_match(r"Took: ([0-9.]+)", output)
    loss = last_match(r"Best Loss: ([0-9.e\-+]+)", output)
    attempts = int(last_match(r"Attempt: ([0-9]+)", output) or trials)

    total_runtime = time.time() - start_time

    return {
        **base,
        "success": success,
        "runtime": float(runtime or total_runtime),
        "loss": float(loss) if loss else float("inf"),
        "attempts": attempts,
        "error": error if error else "",
    }


def opt_to_config(opt):
    caution = "No"
    mars = "No"
    if opt.startswith("cautious-"):
        opt = opt[len("cautious-") :]
        caution = "Yes"
    if opt.startswith("unscaled_cautious-"):
        opt = opt[len("unscaled_cautious-") :]
        caution = "Unscaled"
    if opt.startswith("mars-"):
        opt = opt[len("mars-") :]
        mars = "Yes"
    return opt, caution, mars


def write_progress(results, opt, output):
    with open(output, "w") as f:
        f.write(
            f"# Benchmark Results\nGenerated: {datetime.now()}\nLast updated: {datetime.now()}\n\n"
        )
        f.write("## Summary (In Progress)\n\n")
        f.write("| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |\n")
        f.write("|-----------|---|---|---------|----------|------|\n")

        for o in opt:
            opt_results = [r for r in results if r["opt"] == o]
            if not opt_results:
                continue
            success = sum(r["success"] for r in opt_results)
            runtime = (
                np.mean([r["runtime"] for r in opt_results if r["success"]]) if success else 0
            )
            attempts = (
                np.mean([r["attempts"] for r in opt_results if r["success"]]) if success else 0
            )

            o, caution, mars = opt_to_config(o)
            f.write(
                f"| {o} | {caution} | {mars} | {success}/{len(opt_results)} | {runtime:.2f}s | {attempts:.1f} |\n"
            )

        f.write("\n## Details\n\n")
        f.write(
            "| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | \n"
        )
        f.write("|-----------|-----------|---------|---|---|----------|------|---|\n")

        for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
            mark = "✓" if r["success"] else "✗"
            runtime = f"{r['runtime']:.2f}s"
            loss = f"{r['loss']:.2e}"
            attempts = f"{r['attempts']:d}"

            opt, caution, mars = opt_to_config(r["opt"])
            f.write(
                f"| {r['name']} | {opt} | {caution} | {mars} | {mark} | {runtime} | {loss} | {attempts} | \n"
            )

        if any(not r["success"] for r in results):
            f.write("\n## Errors\n\n")
            for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
                if not r["success"] and r["error"]:
                    f.write(f"\n### {r['name']} - {r['opt']}\n```\n{r['error']}\n```\n")


@app.command()
def main(
    opt: list[str] = typer.Option([], help="Optimizers"),
    steps: int = 100_000,
    timeout: int = 3600 * 4,
    output: str = "benchmark_results.md",
    trials: int = 1000,
    dtype: str = "float32",
    parallelism: int = 16,
    caution: bool = False,
    mars: bool = False,
    unscaled_caution: bool = False,
):
    benchmarks = [
        "beale.py",
        "rosenbrock.py",
        "rastrigin.py",
        "quadratic_varying_scale.py",
        "quadratic_varying_target.py",
        "noisy_matmul.py",
        "xor_sequence.py",
        "xor_digit.py",
        "xor_spot.py",
        "saddle_point.py",
        "saddle_point_0init.py",
        "discontinuous_gradient.py",
        "wide_linear.py",
        "minimax.py",
        "plateau_navigation.py",
        "scale_invariant.py",
        "momentum_utilization.py",
        "batch_size_scaling.py",
        "sparse_gradient.py",
        "layer_wise_scale.py",
        "parameter_scale.py",
        "gradient_delay.py",
        "gradient_noise_scale.py",
        "adversarial_gradient.py",
        "dynamic_landscape.py",
        "exploding_gradient.py",
    ]

    if mars:
        opt = ["mars-" + o for o in opt]
    if caution:
        opt = ["cautious-" + o for o in opt]
    if unscaled_caution:
        opt = ["unscaled_cautious-" + o for o in opt]

    # Create task queue and result queue
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Create all tasks
    total_tasks = len(benchmarks) * len(opt)
    for script in benchmarks:
        for o in opt:
            task_queue.put((script, o, steps, dtype, trials))

    def worker(task_queue, result_queue):
        while True:
            try:
                script, o, steps, dtype, trials = task_queue.get()
                try:
                    result = run_benchmark(script, o, steps, dtype, trials)
                except Exception as exc:
                    result = {
                        "name": script.replace(".py", ""),
                        "opt": o,
                        "success": False,
                        "runtime": None,
                        "attempts": 0,
                        "loss": float("inf"),
                        "error": str(exc),
                    }
                result_queue.put(result)
            except Exception:
                break

    # Start worker processes
    processes = []
    for _ in range(min(parallelism, total_tasks)):
        p = multiprocessing.Process(target=worker, args=(task_queue, result_queue), daemon=True)
        p.start()
        processes.append(p)

    # Collect results
    results = []
    completed = 0

    try:
        while completed < total_tasks:
            result = result_queue.get()
            results.append(result)
            completed += 1
            print(
                f"Progress: [{completed}/{total_tasks}] {result['name']}.py - {result['opt']}: {'✓' if result['success'] else '✗'}"
            )
            write_progress(results, opt, output)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    finally:
        # Clean up
        for p in processes:
            p.terminate()

        # Save final results
        if results:
            write_progress(results, opt, output)


if __name__ == "__main__":
    app()
