# HeavyBall

[![PyPI version](https://img.shields.io/pypi/v/heavyball?color=blue)][pypi] [![Downloads](https://img.shields.io/pypi/dm/heavyball)][pypi] [![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)][license]

HeavyBall is a compile-first optimizer library for PyTorch. Every optimizer's entire step
compiles into a single `torch.compile(fullgraph=True)` graph, and each optimizer is assembled from
small, composable transforms. It ships drop-in `torch.optim` replacements (AdamW, SGD, RMSprop)
alongside Muon, SOAP, Shampoo, PSGD (Kronecker, PRO, QSGD), LATHER, Scion, Schedule-Free, MSAM, and
more.

## Install

```bash
pip install heavyball
```

Requires PyTorch >= 2.13.

## Quick start

Every optimizer is a `torch.optim.Optimizer` subclass and drives the usual
`loss.backward()` / `opt.step()` / `opt.zero_grad()` loop.

```python
from heavyball import AdamW, SOAP, Muon

opt = AdamW(model.parameters(), lr=1e-3)   # drop-in torch.optim.AdamW replacement
opt = SOAP(model.parameters(), lr=3e-3)    # Shampoo-style eigenbasis preconditioning
opt = Muon(model.parameters(), lr=0.02)    # Newton-Schulz orthogonalized updates
```

The preconditioning optimizers route by shape: a weight that merges to a matrix (including a
convolution kernel whose trailing dims collapse to 2-D) is preconditioned, a 1-D bias or norm weight
falls back to AdamW, and an oversized axis is handled with a reduced factor (diagonal in PSGD/LATHER,
dropped in SOAP/Shampoo) rather than a full one. So you pass `model.parameters()` directly, biases and
all.

For the compiled path's full speed, call `heavyball.set_torch()` once at startup:

```python
import heavyball

heavyball.set_torch()   # TF32 matmuls + opt_einsum optimal contraction paths
```

It sets `torch.set_float32_matmul_precision("high")` (TF32 on tensor-core GPUs, ~2x faster fp32
matmuls) and opt_einsum's path strategy. For the optimizers' preconditioner contractions the TF32
step is marginal for training (below seed-to-seed noise); pass `set_torch(matmul_precision="highest")`
for numerically sensitive work. It mutates process-wide torch state (your model's matmuls too), so it
is opt-in, not applied on import.

## Parameter groups

Pass `torch.optim`-style parameter groups for per-group hyperparameters, for example a lower learning
rate on the embedding and no weight decay on the norms. A learning-rate scheduler that mutates
`opt.param_groups[i]["lr"]` works as usual.

```python
opt = AdamW([
    {"params": body.parameters(), "lr": 1e-3, "weight_decay": 0.1},
    {"params": norms.parameters(), "lr": 1e-3, "weight_decay": 0.0},
])
```

## Optimizers

Every optimizer below is an importable class (`from heavyball import <Name>`).

- **First-order:** `AdamW`, `AdamC`, `NAdam`, `RMSprop`, `Lion`, `SGD`, `SignSGD`, `LaProp`,
  `SignLaProp`, `ADOPT`, `AdEMAMix`, `MARSAdamW`, `CautiousAdamW`, `OrthoGradAdamW`, `UnscaledAdamW`,
  `SUDSAdamW`
- **Orthogonal / norm-constrained:** `Muon`, `MuonLaProp`, `AdaMuon`, `NorMuon`, `HyperBallAdamW`,
  `OrthoLaProp`, `LaPropOrtho`, `Scion`, `PolarGrad`, `Aurora`, `Oblique`, `SpEL`
- **Shampoo / SOAP:** `SOAP`, `Shampoo`, `KLSOAP`, `KLShampoo`, `SOAPNAdam`, `SOAPAdEMAMix`, `SOLP`
- **PSGD:** `PSGDKron`, `PSGDPro`, `QSGD`, `LATHER`
- **Whitening:** `Whitening`, `WhitenAdamW`
- **Schedule-Free / SAM:** `ScheduleFree`, `MSAM`

`ScheduleFree` and `MSAM` keep separate training and evaluation parameter states. Call `opt.eval()`
before validation and `opt.train()` before resuming; the swap is reversible and training continues
seamlessly.

## Composition

Each optimizer is a `Recipe`: a `chain` of pure transforms plus a terminal `commit` that writes the
update. A transform maps `(update, observations, param, state, tempo)` to a new update and its next
state; the engine is the only writer. `Route` dispatches by a shape predicate, which is how the
shape-aware optimizers fall back to AdamW on parameters they cannot precondition.

```python
from heavyball import build, adamw, soap, Route
from heavyball.matrix import matrix_route

# Preconditioning 2-D leaves with soap and everything else with adamw is exactly what the
# heavyball.SOAP facade builds:
recipe = Route(matrix_route, soap, adamw)
opt = build(model.parameters(), recipe, lr=3e-3)
```

Hyperparameters are a strict contract: `tempo.hyper` carries only the declared hyperparameters, so a
transform that reads one the recipe never declared raises rather than silently substituting a default.

See [`examples/`](examples/) for full training scripts, including DDP, which works transparently.

For FSDP2 (`torch.distributed.fsdp.fully_shard`), wrap the model with `fully_shard(model)`, then build
the optimizer with the `.fsdp2` classmethod -- `heavyball.AdamW.fsdp2(model, lr=1e-3)` rather than
`heavyball.AdamW(model.parameters())` -- which binds each slab into FSDP2's sharded storage instead of a
plain one. Passing a `fully_shard`'d model's parameters to the plain constructor raises with a message
pointing to `.fsdp2()` (HeavyBall binds each parameter into a contiguous slab at construction, which
FSDP2's sharded DTensor storage does not permit). Shard-separable and owner-whole optimizers are covered;
the whole-model scoped ones (`clip_global_norm`, SAM) are rejected under FSDP2. DDP works transparently.

## Checkpointing

`state_dict()` / `load_state_dict()` round-trip the full optimizer state, including the
preconditioner refresh cadence and the per-optimizer random seed. Restore it alongside
`model.load_state_dict()` (the optimizer state does not carry the parameters), and resume matches an
uninterrupted run bit-for-bit -- including the optimizers that draw randomness in the step (Muon's
stochastic rounding, the PSGD family, Scion, LATHER, and any `storage_dtype`/`ecc` low-precision state
write). Their per-step randomness is a stateless counter-based stream keyed by the stored seed, leaf
index, and step count, so it resumes bit-identically from `state_dict()` alone, without restoring the
process-wide `torch.get_rng_state()`.

## Performance

Compiling the whole step is the point. Measured on an RTX 5060 Ti (PyTorch 2.12) for AdamW, the
compiled step is about **6x faster than the same optimizer run eagerly**, **2-3x faster than
`torch.optim.AdamW(foreach=True)`**, and within a few percent of `torch.optim.AdamW(fused=True)` at
scale (a compiled Python optimizer matching a hand-written fused CUDA kernel). For most optimizers the
compiled step matches the eager result to floating-point precision. The orthogonalizing and spectral
families are the exception (the Muon variants, Scion, PolarGrad, SpEL, KLSOAP, Aurora): their
Newton-Schulz and eigendecomposition iterations amplify the small difference between the compiled and
eager matmul kernels, so over a run compiled and eager drift apart like two seeds. That drift is
matmul reduction order, not a correctness bug. The preconditioning families trade a higher
per-step cost for sample efficiency; `benchmarks/conditioning.py` measures this across a controlled
input-conditioning sweep, where whitening with momentum (`LATHER`) increasingly outperforms `AdamW`
as conditioning worsens.

The step compiles with `fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"`. One graph
is compiled per step type (a normal step and a preconditioner-refresh step), and no per-step
recompilation occurs.

## Low-precision state

Optimizer state defaults to fp32. Two opt-ins shrink it, with compute still promoted to fp32 so only
storage changes: `storage_dtype=torch.bfloat16` stores state in bfloat16 (half the state memory, and
faster than `torch.optim.AdamW(fused=True)` on the tested models from the lower bandwidth), and `ecc=8`
adds a per-value int8 residual for near-fp16 precision at 0.75x fp32 (`ecc=16` uses int16 but costs as
much memory as fp32, so prefer plain fp32 there). See `examples/ecc_bf16.py` and
`benchmarks/precision_speed.py`.

## Compatibility matrix

`test/test_feature_matrix.py` is the compile-first compatibility matrix: for every optimizer, over each
distinct state-precision setting (`fp32`, `bf16`, `ecc=8`, `ecc=16`), it compiles the live max-autotune
step — the exact path production runs, not eager and not a cheaper mode — on real transformer-shaped
weights, and asserts the cell compiles, stays finite, updates every parameter, and actually allocates the
requested state precision. Each cell also records its compile time, steady step time, peak GPU memory, and
optimizer-state bytes (set `FEATURE_MATRIX_MEASURE=<path.jsonl>`), so a run characterizes the whole surface
rather than only passing or failing. Companion guards in `test/test_storage_dtype.py` assert that `bf16`
halves the state for every optimizer and that every optimizer updates `bf16` *parameters* without underflow.

The full grid -- **43 optimizers × 4 state precisions = 172 cells -- compiles and passes with no holes**
(each cell runs the live max-autotune step, stays finite, updates every parameter, and allocates the
requested precision; `bf16` halves the float state for every stateful optimizer):

| state precision | optimizer state stored as |
| --- | --- |
| `fp32` (default) | fp32 |
| `bf16` | bf16 -- half the float-state memory |
| `ecc=8` | bf16 + int8 (top 8 low-mantissa bits, rest stochastically rounded) -- near-fp16 at 0.75x fp32 |
| `ecc=16` | bf16 + int16 (all 16 low-mantissa bits) -- **bit-exact fp32** at the same 4 bytes |

It is a heavy GPU test — max-autotune compiles the whole grid — so CI runs it on the on-demand `gpu-test`
lane across 8 GPUs with `pytest-xdist` (`test/conftest.py` pins one worker per GPU). On a CUDA box:

```bash
pytest test/test_feature_matrix.py -n 8   # -n <#gpus>; drop -n to run serially
```

## Migrating from 3.x

HeavyBall 4 is a rewrite. The 3.x composable-flag surface is gone: `heavyball.chainable`,
`SplitOpt`, the `mars=` / `caution=` keyword flags, and `capture_param_shapes` no longer exist. The
`ecc=` low-precision-state option and `heavyball.set_torch` (also exposed at `heavyball.utils.set_torch`)
returned in 4.0; see Low-precision state above and Quick start. Use the optimizer classes above for the
common case, per-group
dictionaries instead of `SplitOpt`, and the `Recipe` / `Route` / transform API to compose custom
optimizers. `cautious`/`mars` variants that used to be flags are their own recipes (for example
`CautiousAdamW`, `MARSAdamW`).

## Contributing

Fork the repository, install with `pip install -e .[dev]`, and run `pytest`.

## License

BSD-2-Clause, see [LICENSE](LICENSE).

The name "HeavyBall" comes from [Polyak's heavy-ball method](https://doi.org/10.1016/0041-5553(64)90137-5), the momentum
technique underlying most modern optimizers.

[pypi]: https://pypi.org/project/heavyball/

[license]: LICENSE
