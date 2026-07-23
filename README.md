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

The three measured tables below are auto-attached output from HeavyBall's benchmark pipeline, not a
"run this yourself" recipe. These numbers are shown here and are produced by
`benchmarks/feature_matrix.py`, the same harness the 8-GPU CI lane runs. CI regenerates the marked block
in place from that lane's merged JSONL artifact.

The harness runs every optimizer at each distinct state-precision setting (`fp32`, `bf16`, `ecc=8`,
`ecc=16`) on a real ~7M-parameter transformer (`D=512`, vocab 4096). It compiles the live
`max-autotune-no-cudagraphs` step, warms both the normal and preconditioner-refresh graphs, times a
20-step steady window, and verifies that every cell stays finite, updates every parameter, and allocates
the requested state precision. Companion guards in `test/test_storage_dtype.py` verify that `bf16`
halves the state for every optimizer and that every optimizer updates `bf16` parameters without underflow.

These numbers were measured with `heavyball.set_torch()` on RTX 5060 Ti ×8 (PyTorch 2.12): **51
optimizers × 4 state precisions = 204 cells compile and pass with no holes.** The four `TrueGrad*`
facades skip because they need an external observation producer, so the shipped surface is 55
`HeavyBallOptimizer` classes.

<!-- heavyball:feature-matrix begin -->
Generated by `benchmarks/feature_matrix.py`; regenerate with `python benchmarks/feature_matrix.py --render feature-matrix.jsonl --update-readme README.md`.

### Steady optimizer step time (ms)

| optimizer | fp32 | bf16 | ecc8 | ecc16 |
| --- | ---: | ---: | ---: | ---: |
| ADOPT | 3.35 | 3.95 | 2.56 | 3.55 |
| AdEMAMix | 3.11 | 3.44 | 3.95 | 3.47 |
| AdaMuon | 6.58 | 6.45 | 7.51 | 7.06 |
| AdamC | 3.35 | 2.57 | 3.68 | 2.44 |
| AdamW | 4.04 | 3.82 | 3.32 | 3.58 |
| Aurora | 9.23 | 10.09 | 9.40 | 9.41 |
| CautiousAdamW | 4.03 | 3.31 | 3.99 | 2.74 |
| HeavyKLSOAP | 31.11 | 32.21 | 33.06 | 33.01 |
| HeavyKLShampoo | 27.82 | 28.85 | 28.49 | 28.77 |
| HeavySOAP | 28.99 | 29.68 | 30.02 | 29.59 |
| HeavySOAPAdEMAMix | 29.81 | 30.68 | 31.52 | 30.88 |
| HeavySOAPNAdam | 28.11 | 29.19 | 29.74 | 29.11 |
| HeavySOLP | 28.04 | 29.26 | 29.71 | 29.14 |
| HyperBallAdamW | 3.51 | 4.53 | 5.28 | 5.41 |
| KLSOAP | 30.10 | 31.72 | 32.16 | 31.50 |
| KLShampoo | 28.52 | 29.75 | 29.38 | 29.52 |
| LATHER | 64.86 | 64.56 | 66.11 | 65.34 |
| LaProp | 3.46 | 3.87 | 3.90 | 3.27 |
| LaPropOrtho | 4.04 | 4.40 | 4.64 | 4.05 |
| Lion | 3.24 | 3.31 | 2.79 | 3.43 |
| MARSAdamW | 3.70 | 5.34 | 4.59 | 4.37 |
| MSAM | 3.89 | 3.50 | 4.54 | 4.89 |
| Muon | 5.60 | 6.59 | 6.68 | 6.46 |
| MuonLaProp | 6.35 | 6.62 | 7.22 | 6.66 |
| NAdam | 2.42 | 3.69 | 4.88 | 3.73 |
| NorMuon | 5.34 | 5.83 | 6.75 | 6.43 |
| Oblique | 2.43 | 3.93 | 3.33 | 3.43 |
| OrthoGradAdamW | 3.96 | 3.07 | 4.24 | 3.89 |
| OrthoLaProp | 3.73 | 3.10 | 3.08 | 3.87 |
| PSGDKron | 16.91 | 16.38 | 16.80 | 16.89 |
| PSGDLRA | 122.36 | 126.70 | 125.28 | 119.85 |
| PSGDNfactor | 20.84 | 22.13 | 21.01 | 21.50 |
| PSGDPro | 20.87 | 20.81 | 20.37 | 21.10 |
| PolarGrad | 6.41 | 5.78 | 6.81 | 6.55 |
| QSGD | 18.90 | 19.24 | 19.77 | 18.95 |
| RMSprop | 2.25 | 3.53 | 3.10 | 3.60 |
| SGD | 3.29 | 3.26 | 3.32 | 3.47 |
| SOAP | 28.90 | 29.20 | 29.95 | 29.33 |
| SOAPAdEMAMix | 29.44 | 30.72 | 31.91 | 30.65 |
| SOAPNAdam | 29.66 | 30.46 | 31.07 | 30.33 |
| SOLP | 28.86 | 29.42 | 29.91 | 29.32 |
| SUDSAdamW | 8.00 | 7.44 | 7.41 | 8.36 |
| ScheduleFree | 2.60 | 4.94 | 3.12 | 2.68 |
| Scion | 6.17 | 6.64 | 6.43 | 6.06 |
| Shampoo | 66.31 | 74.94 | 68.06 | 66.80 |
| SignLaProp | 4.43 | 3.83 | 4.90 | 4.87 |
| SignSGD | 3.26 | 3.39 | 3.55 | 3.41 |
| SpEL | 22.81 | 23.05 | 22.91 | 22.87 |
| UnscaledAdamW | 3.57 | 2.48 | 3.76 | 3.60 |
| WhitenAdamW | 6.29 | 7.22 | 6.75 | 5.64 |
| Whitening | 5.51 | 6.24 | 6.66 | 6.76 |

### Peak GPU memory (MB)

| optimizer | fp32 | bf16 | ecc8 | ecc16 |
| --- | ---: | ---: | ---: | ---: |
| ADOPT | 321 | 259 | 291 | 321 |
| AdEMAMix | 385 | 290 | 339 | 385 |
| AdaMuon | 315 | 351 | 386 | 348 |
| AdamC | 321 | 259 | 291 | 321 |
| AdamW | 321 | 259 | 291 | 321 |
| Aurora | 249 | 345 | 347 | 254 |
| CautiousAdamW | 319 | 257 | 291 | 320 |
| HeavyKLSOAP | 480 | 497 | 702 | 613 |
| HeavyKLShampoo | 386 | 415 | 484 | 432 |
| HeavySOAP | 449 | 460 | 604 | 486 |
| HeavySOAPAdEMAMix | 491 | 489 | 642 | 535 |
| HeavySOAPNAdam | 447 | 460 | 604 | 486 |
| HeavySOLP | 449 | 458 | 603 | 480 |
| HyperBallAdamW | 240 | 227 | 250 | 259 |
| KLSOAP | 480 | 497 | 701 | 612 |
| KLShampoo | 387 | 431 | 485 | 431 |
| LATHER | 588 | 584 | 946 | 759 |
| LaProp | 321 | 259 | 291 | 321 |
| LaPropOrtho | 302 | 240 | 291 | 320 |
| Lion | 257 | 228 | 243 | 259 |
| MARSAdamW | 351 | 308 | 405 | 368 |
| MSAM | 399 | 441 | 509 | 382 |
| Muon | 238 | 227 | 347 | 254 |
| MuonLaProp | 302 | 375 | 467 | 320 |
| NAdam | 321 | 259 | 289 | 321 |
| NorMuon | 231 | 219 | 348 | 251 |
| Oblique | 319 | 257 | 291 | 320 |
| OrthoGradAdamW | 338 | 276 | 307 | 340 |
| OrthoLaProp | 338 | 276 | 307 | 340 |
| PSGDKron | 436 | 411 | 428 | 440 |
| PSGDLRA | 1880 | 1597 | 1620 | 1804 |
| PSGDNfactor | 398 | 373 | 388 | 401 |
| PSGDPro | 398 | 373 | 388 | 401 |
| PolarGrad | 238 | 256 | 347 | 254 |
| QSGD | 399 | 373 | 388 | 401 |
| RMSprop | 257 | 228 | 243 | 259 |
| SGD | 193 | 193 | 193 | 193 |
| SOAP | 449 | 460 | 604 | 486 |
| SOAPAdEMAMix | 491 | 489 | 642 | 535 |
| SOAPNAdam | 447 | 460 | 604 | 486 |
| SOLP | 449 | 458 | 603 | 480 |
| SUDSAdamW | 402 | 554 | 499 | 402 |
| ScheduleFree | 305 | 360 | 314 | 307 |
| Scion | 241 | 344 | 359 | 241 |
| Shampoo | 350 | 378 | 408 | 355 |
| SignLaProp | 302 | 240 | 333 | 320 |
| SignSGD | 193 | 193 | 193 | 193 |
| SpEL | 260 | 248 | 253 | 261 |
| UnscaledAdamW | 321 | 259 | 291 | 321 |
| WhitenAdamW | 321 | 259 | 291 | 321 |
| Whitening | 321 | 259 | 291 | 321 |

### Optimizer-state memory (MB)

| optimizer | fp32 | bf16 | ecc8 | ecc16 |
| --- | ---: | ---: | ---: | ---: |
| ADOPT | 58.80 | 29.40 | 29.40 | 29.40 |
| AdEMAMix | 88.20 | 44.10 | 44.10 | 44.10 |
| AdaMuon | 58.80 | 29.40 | 29.40 | 29.40 |
| AdamC | 58.80 | 29.40 | 29.40 | 29.40 |
| AdamW | 58.80 | 29.40 | 29.40 | 29.40 |
| Aurora | 29.44 | 14.72 | 14.72 | 14.72 |
| CautiousAdamW | 58.80 | 29.40 | 29.40 | 29.40 |
| HeavyKLSOAP | 159.50 | 79.75 | 79.75 | 79.75 |
| HeavyKLShampoo | 130.14 | 65.07 | 65.07 | 65.07 |
| HeavySOAP | 159.46 | 79.73 | 79.73 | 79.73 |
| HeavySOAPAdEMAMix | 188.82 | 94.41 | 94.41 | 94.41 |
| HeavySOAPNAdam | 159.46 | 79.73 | 79.73 | 79.73 |
| HeavySOLP | 159.46 | 79.73 | 79.73 | 79.73 |
| HyperBallAdamW | 29.44 | 14.72 | 14.72 | 14.72 |
| KLSOAP | 159.50 | 79.75 | 79.75 | 79.75 |
| KLShampoo | 130.14 | 65.07 | 65.07 | 65.07 |
| LATHER | 159.49 | 79.75 | 79.75 | 79.75 |
| LaProp | 58.80 | 29.40 | 29.40 | 29.40 |
| LaPropOrtho | 58.80 | 29.40 | 29.40 | 29.40 |
| Lion | 29.40 | 14.70 | 14.70 | 14.70 |
| MARSAdamW | 88.20 | 44.10 | 44.10 | 44.10 |
| MSAM | 117.60 | 58.80 | 58.80 | 58.80 |
| Muon | 29.44 | 14.72 | 14.72 | 14.72 |
| MuonLaProp | 58.80 | 29.40 | 29.40 | 29.40 |
| NAdam | 58.80 | 29.40 | 29.40 | 29.40 |
| NorMuon | 29.49 | 14.74 | 14.74 | 14.74 |
| Oblique | 58.80 | 29.40 | 29.40 | 29.40 |
| OrthoGradAdamW | 58.80 | 29.40 | 29.40 | 29.40 |
| OrthoLaProp | 58.80 | 29.40 | 29.40 | 29.40 |
| PSGDKron | 50.44 | 25.22 | 25.22 | 25.22 |
| PSGDLRA | 617.38 | 308.69 | 308.69 | 308.69 |
| PSGDNfactor | 50.44 | 25.22 | 25.22 | 25.22 |
| PSGDPro | 50.44 | 25.22 | 25.22 | 25.22 |
| PolarGrad | 29.44 | 14.72 | 14.72 | 14.72 |
| QSGD | 50.44 | 25.22 | 25.22 | 25.22 |
| RMSprop | 29.40 | 14.70 | 14.70 | 14.70 |
| SGD | 0.00 | 0.00 | 0.00 | 0.00 |
| SOAP | 159.46 | 79.73 | 79.73 | 79.73 |
| SOAPAdEMAMix | 188.82 | 94.41 | 94.41 | 94.41 |
| SOAPNAdam | 159.46 | 79.73 | 79.73 | 79.73 |
| SOLP | 159.46 | 79.73 | 79.73 | 79.73 |
| SUDSAdamW | 88.20 | 44.10 | 44.10 | 44.10 |
| ScheduleFree | 58.80 | 29.40 | 29.40 | 29.40 |
| Scion | 29.40 | 14.70 | 14.70 | 14.70 |
| Shampoo | 100.74 | 50.37 | 50.37 | 50.37 |
| SignLaProp | 58.80 | 29.40 | 29.40 | 29.40 |
| SignSGD | 0.00 | 0.00 | 0.00 | 0.00 |
| SpEL | 29.44 | 14.72 | 14.72 | 14.72 |
| UnscaledAdamW | 58.80 | 29.40 | 29.40 | 29.40 |
| WhitenAdamW | 58.80 | 29.40 | 29.40 | 29.40 |
| Whitening | 58.80 | 29.40 | 29.40 | 29.40 |

<!-- heavyball:feature-matrix end -->

State precision maps to bytes per state element (the `ecc` int residual is why `ecc=8` is 0.75× and `ecc=16`
is bit-exact fp32 at 1.0×, even though both narrow the float slab to bf16):

| state precision | optimizer state stored as |
| --- | --- |
| `fp32` (default) | fp32 |
| `bf16` | bf16, half the float-state memory |
| `ecc=8` | bf16 + int8 (top 8 low-mantissa bits, rest stochastically rounded), near-fp16 at 0.75× fp32 |
| `ecc=16` | bf16 + int16 (all 16 low-mantissa bits), bit-exact fp32 at the same 4 bytes |

The 8-GPU CI lane maintains this section with these pipeline commands:

```bash
python benchmarks/feature_matrix.py --gpus 8 --measure feature-matrix.jsonl
python benchmarks/feature_matrix.py --render feature-matrix.jsonl --update-readme README.md
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
