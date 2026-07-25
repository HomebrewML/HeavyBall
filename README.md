# HeavyBall 4.0

HeavyBall is a compile-first optimizer library for PyTorch. Its public optimizer classes are
`torch.optim.Optimizer`-style facades over slab-backed, full-step compiled Engines.

> **4.0 is a prerelease.** Install the current 4.0 development branch from source:
>
> ```bash
> python -m pip install "git+https://github.com/HomebrewML/HeavyBall.git@hb4.0.0-dev"
> ```
>
> The package requires Python 3.11 or newer and PyTorch `>=2.13.0,<3.0.0`. Benchmark results for
> this release must be generated on a supported PyTorch version; PyTorch 2.12 results are not 4.0
> release evidence.

## Quickstart

This example defines one model, moves it to its final device, constructs one optimizer, and trains
for a few steps:

```python
import torch
import heavyball

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Linear(8, 16),
    torch.nn.GELU(),
    torch.nn.Linear(16, 1),
).to(device)
optimizer = heavyball.AdamW(model.parameters(), lr=1e-3)

for _ in range(3):
    inputs = torch.randn(32, 8, device=device)
    targets = inputs.sum(dim=1, keepdim=True)

    optimizer.zero_grad()  # set_to_none=False is HeavyBall's required default
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

print(f"loss: {loss.detach().item():.4f}")
```

HeavyBall binds parameter and gradient storage when the optimizer is constructed. Put the model on
its final device and dtype first; construct HeavyBall after that placement and before wrapping the
model with DDP or `torch.compile`.

## Choose an optimizer

“Reliability posture” below is selection guidance, not a convergence or quality ranking. Start with
the least specialized algorithm that meets the experiment's needs, then measure on the real model.

| Optimizer | Reliability posture | Persistent state | Model shape | Compile/work cost | Distributed support |
| --- | --- | --- | --- | --- | --- |
| `SGD` | Minimal baseline | Stateless | Any floating-point parameter | Small elementwise graph | DDP; FSDP2/HSDP |
| `AdamW` | General baseline | Two moments | Any floating-point parameter | Elementwise graph | DDP; FSDP2/HSDP |
| `Muon` | Matrix-specialized | Momentum on 2-D weights; AdamW state elsewhere | 2-D weights, with AdamW routing for other leaves | Five-step Newton–Schulz matrix path | DDP; FSDP2/HSDP |
| `SOAP` | Matrix-preconditioned | Adam moments plus Gram/basis state | Matrix-mergeable weights, with AdamW routing elsewhere | Eigendecomposition refresh path | DDP; FSDP2/HSDP |
| `PSGD` | General PSGD facade | Shape-dependent preconditioner factors | Auto-routes large, matrix-mergeable, and remaining leaves | Shape-dependent preconditioner path | DDP; FSDP2/HSDP |
| `SFAdamW` | Specialized train/eval lifecycle | Schedule-free train and evaluation iterates | Any floating-point parameter | Elementwise graph plus compiled train/eval swap | DDP; FSDP2/HSDP |

Use the registry instead of relying on a stale catalog:

- `heavyball.list_optimizers()` returns the canonical public facades.
- `heavyball.describe("SOAP")` reports the algorithm, signature defaults, routing, aliases, and
  distributed modes.
- `heavyball.estimate_state_bytes(params, heavyball.AdamW)` estimates the actual persistent state
  slabs without allocating parameter-sized state or running a step.

`PSGD` is the general auto-routing PSGD facade. `PSGDKron`, `PSGDLRA`, `PSGDNfactor`, `PSGDPro`, and
`QSGD` are explicit variants for advanced routing choices. Check an individual facade with
`describe()` before selecting FSDP2: distributed support is recipe-specific.

## HeavyBall is not a drop-in `torch.optim` replacement

The former “drop-in replacement” description is retracted.
HeavyBall subclasses `torch.optim.Optimizer`, supports parameter groups and scheduler updates to
`optimizer.param_groups`, and follows the familiar backward/step/zero-grad loop. Its storage and step
contracts deliberately differ from PyTorch's optimizers.

### Defaults

These are constructor defaults in HeavyBall 4.0 and supported PyTorch 2.13:

| Setting | `heavyball.AdamW` | `torch.optim.AdamW` |
| --- | ---: | ---: |
| `lr` | `0.0025` | `0.001` |
| first-moment decay | `beta1=0.9` | `betas[0]=0.9` |
| second-moment decay | `beta2=0.99` | `betas[1]=0.999` |
| `eps` | `1e-8` | `1e-8` |
| `weight_decay` | `0.0` | `0.01` |

`heavyball.SGD` defaults to `lr=0.0025` and `weight_decay=0.0`. It is stateless raw SGD and exposes
neither momentum nor Nesterov; it is not the full `torch.optim.SGD` algorithm.

### Behavioral differences

- Parameters and gradients are views into persistent slabs. Do not replace `p.data` or `p.grad`, and
  do not change parameter storage after construction. Copy values in place when necessary.
- `optimizer.zero_grad()` means `set_to_none=False`. Passing `set_to_none=True` raises because the
  gradient slab must remain bound.
- By default, every optimized parameter advances on every `step()`, including its weight decay,
  state, and age, even if backward did not touch that parameter.
- `observed=` controls HeavyBall's optimizer activity only. Supply one host `bool` per trainable
  parameter in construction order, or a mapping containing every trainable parameter. A `False`
  leaf does not advance its parameter, moments, decay, or clock, but this mask does not change the
  autograd graph or suppress DDP/FSDP collectives. Conditional or MoE graphs under DDP still require
  `find_unused_parameters=True`; pass `observed=` as well so HeavyBall does not advance inactive
  leaves. Under FSDP2, conditional graphs require `set_reduce_scatter_unused_params(True)` in
  addition to the HeavyBall mask.
- Hyperparameters are keyword-only. HeavyBall names separate `beta1` and `beta2` arguments rather
  than PyTorch's `betas` tuple, and it does not expose PyTorch implementation switches such as
  `foreach` or `fused`.
- `add_param_group()` rebuilds the Engine and therefore causes new compiled graphs on the next step.
  Declare all groups up front when compile latency matters.

## Compatibility

| Feature | 4.0 status and required usage |
| --- | --- |
| CPU | Supported through the same compile-first Engine. |
| CUDA | Supported; the shipped performance harness requires CUDA. |
| AMP | Autocast and the standard `torch.amp.GradScaler` path work with the dense persistent gradient buffers. CPU autocast plus `GradScaler` was exercised in this pass; CUDA AMP was not rerun here. State precision is configured separately with `storage_dtype` or `ecc`. |
| DDP | Supported with `gradient_as_bucket_view=False` (the DDP default). Construct HeavyBall after device placement and before `DistributedDataParallel(model)`. Conditional graphs still require `find_unused_parameters=True`; `observed=` does not replace DDP unused-parameter detection. |
| FSDP2 | Implemented for recipes reported as supported by `heavyball.describe(name)`. Call `fully_shard(model)` first, then `heavyball.AdamW.fsdp2(model, ...)`; do not pass the DTensor parameters to the plain constructor. |
| HSDP | The FSDP2 adapter accepts a 2-D device mesh named `("replicate", "shard")`; use the same `.fsdp2(model)` path. |
| Sparse gradients | Sparse gradient storage is not preserved: HeavyBall binds a dense persistent gradient slab. |
| Unused/conditional parameters | No automatic grad-`None` detection. Pass the complete `observed=` activity mask on every conditional step to stop HeavyBall state advancement. DDP also needs `find_unused_parameters=True`; FSDP2 needs `set_reduce_scatter_unused_params(True)`. |
| Tied weights | A single shared `nn.Parameter` is supported because it appears once in `model.parameters()`. Distinct `nn.Parameter` objects with overlapping storage are rejected. The `register_truegrad()` observation helper rejects tied/shared parameters. |

FSDP2 also rejects `clip_global_norm`, observation-bearing recipes such as TrueGrad, recipes whose
callable scopes are not supported, scalar parameters sharded on dimension 0, and parameters that
were not managed by `fully_shard`.

## Compile lifecycle and cache

Each Engine creates full-graph, static-shape compiled callables with max-autotune enabled. Compilation
is lazy: the first normal step pays the normal-graph compile cost. Preconditioner recipes can also
select a refresh graph, whose first use has its own compile cost. `SFAdamW` and `MSAM` additionally
compile their train/evaluation swap when first used.

The persistent cache is PyTorch Inductor's cache. HeavyBall controls the optimizer compile policy and
gives its custom scalar-CSE graph pass a versioned UUID, allowing Inductor's persistent FX-graph and
autotune caches to key and reuse compatible artifacts. On a cache hit, the max-autotune work can be
reused across matching runs. HeavyBall does not create a separate cache service or a
HeavyBall-specific cache directory.

Configure the cache before Python imports PyTorch:

```bash
# Put Inductor artifacts in an explicit persistent location.
TORCHINDUCTOR_CACHE_DIR=/path/to/inductor-cache python train.py

# Invalidate otherwise compatible cache entries with an explicit version tag.
TORCH_COMPILE_CACHE_KEY_TAG=my-training-stack-v1 python train.py

# Disable all Torch compile caches for a cold-cache diagnostic run.
TORCH_COMPILE_FORCE_DISABLE_CACHES=1 python train.py
```

Changing parameter shapes, dtypes, devices, routing, state precision, or parameter groups changes the
compiled plan and can require compilation. Construct all groups before the first step where possible.

## Precision and state memory

Optimizer state defaults to fp32 for fp16, bf16, and fp32 parameters; fp64 parameters retain fp64
state. Low-precision storage changes persistent state representation while optimizer math is promoted
to at least fp32.

For AdamW's two full-size moment tensors, the exact state storage is:

| State configuration | Representation per moment | Total bytes per parameter element | Ratio to fp32 |
| --- | --- | ---: | ---: |
| default | fp32 | `8 B` | `1.00x` |
| `storage_dtype=torch.bfloat16` | bf16 | `4 B` | `0.50x` |
| `ecc=8` | bf16 + int8 correction | `6 B` | `0.75x` |
| `ecc=16` | bf16 + int16 correction | `8 B` | `1.00x` |

These figures include the ECC correction slabs. They can be reproduced directly with
`heavyball.estimate_state_bytes`; use that helper as the authoritative answer for a real model and
optimizer because preconditioner state is shape- and route-dependent. Boolean flags, integer counters,
and deliberately fp64 stability scalars retain their natural or required dtype.

`storage_dtype` currently accepts only `None` or `torch.bfloat16`; `ecc=8` and `ecc=16` imply bf16
state plus their correction. For sustained extreme-magnitude gradients that exceed fp32's useful
range, use float64 parameters so the corresponding state is float64. Passing
`storage_dtype=torch.float64` is not supported by the current constructor.

## Checkpointing

Save the model and optimizer together and reconstruct the same facade and parameter-group layout
before loading:

```text
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    "checkpoint.pt",
)

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

The optimizer state dict includes Engine state, cadence, train/eval mode, hyperparameter cells, and
the optimizer's counter-based RNG identity. It does not replace the need to save model weights.

HeavyBall 3.x optimizer checkpoints do **not** migrate to 4.0 because the Engine state structure
changed. Restart optimization or warm-start from model weights only.

For an optimizer constructed with `.fsdp2(model)`, use the distributed-checkpoint path:
`optimizer.dcp_save(checkpoint_dir)` and `optimizer.dcp_load(checkpoint_dir)`. These methods save and
restore the FSDP2 parameters owned by the optimizer and its optimizer state with resharding metadata;
they reject plain, non-FSDP2 optimizers. Save model buffers separately if the model has any.

## Optimizer identity and 3.x migration

Several familiar names do not mean the same algorithm as they did in 3.x:

- `SGD` is now a stateless raw-gradient step with decoupled weight decay.
- `Muon` routes 2-D weights through Muon and all other parameters through AdamW.
- `SOAP` contains the eigenvalue-sorted QR and Hadamard-square moment transport previously identified
  by the `HeavySOAP` name. `HeavySOAP` now builds the same recipe as `SOAP`.
- `ScheduleFree` is a deprecated alias for `SFAdamW`.
- `WhitenAdamW` is a deprecated alias for `Whitening`.
- `PSGD` is the general auto-routing facade. Use `PSGDKron`, `PSGDLRA`, `PSGDNfactor`, `PSGDPro`, or
  `QSGD` only when the explicit routing and algorithm are intentional.

`SplitOpt` remains exported and delegates parameter groups to separate HeavyBall facades. Ordinary
per-group hyperparameters usually need only one facade with `torch.optim`-style group dictionaries;
`SplitOpt` itself does not support adding groups after construction.

The 3.x composable flag surface is not the 4.0 API. Use named facades for common variants and the
`Recipe` / `Route` / transform API for custom composition.

## Known limitations

- Multi-GPU FSDP2/HSDP was not revalidated in this CPU-only README pass. The implementation and
  multi-GPU tests are present, but release claims still require the supported GPU CI lanes.
- `add_param_group()` rebuilds the slab plan and compiled callables; the next step recompiles. FSDP2
  optimizers reject post-construction parameter additions.
- Sustained extreme-magnitude gradients can require float64 parameters and state.
- Sparse gradients are accumulated into dense persistent gradient slabs.
- Parameter storage must be contiguous, must not overlap another distinct parameter, and must not
  change after optimizer construction.

## Benchmark status and methodology

All GPU-dependent release tables are intentionally pending. The previous README's measurements used
PyTorch 2.12, outside this package's supported range, and its ECC state accounting omitted correction
slabs. Those results are not carried into the 4.0 prerelease documentation.

The shipped harness uses a transformer-shaped model with `D=512`, vocabulary size `4096`, batch size
`4`, and sequence length `8`, and covers fp32, bf16, ECC8, and ECC16 state. For each
optimizer/precision cell it records:

- the first normal full training step for that facade, including its first compiled execution;
- one explicitly forced refresh step and `12` total untimed warmup steps;
- the mean of a `20`-step steady full-training window;
- peak allocated GPU memory and all persistent optimizer-state slabs, including ECC corrections;
- whether Dynamo reported recompilation contamination.

The steady timing is a full training iteration—zeroing, forward, loss, backward, and optimizer
step—not an optimizer-only microbenchmark. The harness clears its cache once per precision and then
shares it across optimizer cells, so `compile_s` is not an independent cold-cache measurement for
every cell. The current JSON schema also records the steady-window mean but not per-step variance.
Release timing tables must therefore remain pending until cold and repeated steady runs, with their
variance, are captured alongside exact hardware, driver, CUDA, and supported PyTorch build metadata.

Benchmark provenance for this README audit:

| Field | Value |
| --- | --- |
| README audit base revision | `d9cd27652b7c4b8af8241d4e61fae2e9193e2102`; record a clean exact benchmark commit before publishing |
| Hardware / driver / CUDA | GPU regeneration pending |
| PyTorch build | GPU regeneration pending; must satisfy `>=2.13.0,<3.0.0` |
| Shapes and warmup | Defined above and in `benchmarks/feature_matrix.py` |
| Raw JSONL | `benchmarks/results/d9cd276-feature-matrix.jsonl` |
| Variance | GPU regeneration pending; not emitted by the current harness |

Run this exact command from the repository root on a CUDA machine with a supported PyTorch build and
a new output path:

```bash
python benchmarks/feature_matrix.py --measure benchmarks/results/d9cd276-feature-matrix.jsonl --out benchmarks/results/d9cd276-feature-matrix.md
```

For parallel execution, add `--gpus N`, where `N` is the number of available GPUs. Record
`git rev-parse HEAD`, the full `torch.__version__`, `torch.version.cuda`, GPU model/count, driver, and
the repeated-run variance next to the raw JSONL before publishing results.

<!-- heavyball:feature-matrix begin -->
### Cold-cache first-step time (s) — GPU regeneration pending

| Optimizer/state precision | Result |
| --- | --- |
| All measured cells | To be regenerated on GPU with the exact command above. |

### Steady full training step time (ms) — GPU regeneration pending

| Optimizer/state precision | Result |
| --- | --- |
| All measured cells | To be regenerated on GPU with the exact command above. |

### Peak allocated GPU memory (MB) — GPU regeneration pending

| Optimizer/state precision | Result |
| --- | --- |
| All measured cells | To be regenerated on GPU with the exact command above. |

<!-- heavyball:feature-matrix end -->

After measurement, the generated detailed tables can be rendered into the marked block with:

```bash
python benchmarks/feature_matrix.py --render benchmarks/results/d9cd276-feature-matrix.jsonl --update-readme README.md
```

The AdamW byte table above is not GPU-pending: its `8/4/6/8 B` totals come from the live state-slab
accounting used by `heavyball.estimate_state_bytes`.

## License

HeavyBall is distributed under the BSD 2-Clause license. See [LICENSE](LICENSE).
