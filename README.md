# HeavyBall

[![PyPI version](https://img.shields.io/pypi/v/heavyball?color=blue)][pypi] [![Downloads](https://img.shields.io/pypi/dm/heavyball)][pypi] [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)][license]

**A compile-first optimizer library for PyTorch.**

Drop-in replacements for `torch.optim` — AdamW, SGD, RMSprop — plus second-order and spectral methods: Muon, SOAP (Shampoo), PSGD (Kronecker), ADOPT, Schedule-Free, LaProp, and 30+ more. Every optimizer is assembled from 100+ compiled building blocks that fuse into minimal Triton kernels via `torch.compile`. Features like MARS gradient correction, cautious updates, [ECC](#ecc) memory compression, and stochastic rounding compose as flags on any optimizer. DDP and FSDP work transparently, with automatic round-robin compute distribution for second-order methods. Build custom optimizers from the same primitives — they compile and fuse automatically.

```python
from heavyball import AdamW, Muon

opt = AdamW(model.parameters(), lr=1e-3)                                       # drop-in
opt = Muon(model.parameters(), lr=0.02, ecc="bf16+8", mars=True, caution=True)  # composed
```

- **Drop-in and fast.** Same API as `torch.optim`. Optimizer internals compile via `torch.compile(fullgraph=True)` — for fused first-order methods (AdamW, LaProp, ADOPT, NAdam, AdEMAMix), the entire step fuses into minimal Triton kernels. Second-order methods (Muon, PSGD) compile their preconditioning steps as separate regions while their elementwise portions still fuse.
- **Mix any feature, any optimizer.** `mars=True` for gradient variance reduction, `caution=True` for cautious masking, `ecc="bf16+8"` for 25% less optimizer state memory, `palm=True` for PaLM-style scheduling — flags compose freely. Or go deeper: build entirely custom optimizers from the same compiled primitives via the chainable transform API. Custom optimizers inherit compilation, ECC, MARS, caution, clipping, warmup, stochastic rounding, and `foreach` batching automatically.
- **30+ optimizers.** AdamW, Muon, SOAP (Shampoo), PSGD (Kronecker, related to K-FAC), ADOPT, Schedule-Free AdamW, LaProp, NAdam, AdEMAMix, Newton-PSGD hybrids, SAM, and more. Mix per layer with `SplitOpt`. DDP and FSDP supported with automatic compute distribution for second-order methods.

## Quick Start

```bash
pip install heavyball
```

Requires PyTorch >= 2.2.

```python
from heavyball import AdamW
opt = AdamW(model.parameters(), lr=1e-3)
```

```python
from heavyball import SOAP  # Shampoo-based second-order preconditioning
opt = SOAP(model.parameters(), lr=3e-3)
```

```python
from heavyball import SplitOpt, Muon, AdamW  # different optimizer per layer
opt = SplitOpt([
    {'params': matrices, 'optimizer': Muon, 'lr': 0.02},
    {'params': vectors, 'optimizer': AdamW, 'lr': 1e-3},
])
```

If you're coming from `torch.optim.AdamW`, start with `heavyball.AdamW` — same API. When ready, try `SOAP` for Shampoo-based preconditioning, `Muon` for orthogonal updates, or stack flags like `mars=True` and `ecc="bf16+8"`. See [`examples/`](examples/) for training examples.

## Supported Optimizers

AdamW, Muon, SOAP (Shampoo), PSGD (Kronecker/K-FAC family), Schedule-Free AdamW, ADOPT, LaProp, NAdam, AdEMAMix, SAM, and 20+ variants — all sharing the same compiled, composable infrastructure.

<details>
<summary>Full list</summary>

**First-order:**
AdamW, NAdam, RMSprop, ADOPT, AdEMAMix, LaProp, SignLaProp, SGD, Scion, UnscaledAdamW, ForeachAdamC, SUDSAdamW

**Schedule-Free:**
SFAdamW, PaLMSFAdamW

**Orthogonal / Muon:**
Muon, MuonLaProp, OrthoLaProp, LaPropOrtho

**Shampoo-based (SOAP):**
SOAP, PaLMSOAP, PrecondScheduleSOAP, PrecondSchedulePaLMSOAP, SOAPNAdam, SOAPAdEMAMix, SOLP

**PSGD (Kronecker):**
PSGDKron, CachedPSGDKron, DelayedPSGD, CachedDelayedPSGDKron, PurePSGD, NewtonPSGDKron, NewtonHybrid2PSGDKron

**PSGD (Low-Rank):**
PSGDLRA, DelayedPSGDLRA, NewtonPSGDLRA, NewtonHybrid2PSGDLRA

**SAM:**
SAMWrapper, MSAMLaProp

**Meta:**
SplitOpt

</details>

## Composable Features

Flags compose freely — `Muon(..., ecc="bf16+8", mars=True, caution=True, palm=True)` is valid. Available on all optimizers except the meta-optimizers SAMWrapper and SplitOpt (which delegate to inner optimizers).

| Flag | Effect |
|------|--------|
| `mars=True` | MARS gradient correction — variance reduction via previous gradients. |
| `caution=True` | Cautious updates — mask update elements that disagree with the gradient direction. |
| `ecc="bf16+8"` | Compress optimizer state to bf16 + int8 correction (3 bytes vs fp32's 4). See [ECC](#ecc). |
| `param_ecc="bf16+8"` | Same compression applied to parameters. |
| `palm=True` | PaLM-style beta2 scheduling (`1 - step^(-beta2_scale)`). Not available on SGD, AdEMAMix, Scion, or PSGD variants. |
| `gradient_clipping=...` | Clip incoming gradients. Multiple modes: L2, RMS, trust region, compressive — or pass a custom function. |
| `update_clipping=...` | Clip outgoing updates after all transforms. Same options. |
| `promote=True` | Promote gradients to fp32 before the update. |
| `warmup_steps=N` | Linear learning rate warmup over N steps. |

These flags are the surface API. Underneath, each is a chainable transform built on compiled building blocks — the same primitives available to you via `heavyball.chainable` for building custom optimizers.

### ECC

ECC stores each optimizer state tensor as bf16 + int8 correction (3 bytes vs fp32's 4) — inspired by the approach in [FlashOptim](https://arxiv.org/abs/2602.23349). HeavyBall makes it composable: an attribute-based architecture attaches correction tensors at call time, so any compiled optimizer function handles ECC transparently. No per-optimizer kernel needed — add `ecc="bf16+8"` to AdamW, SOAP, Muon, PSGD, or any custom chain.

```python
opt = AdamW(model.parameters(), lr=1e-3, ecc="bf16+8")
opt = Muon(model.parameters(), lr=0.02, ecc="bf16+8", param_ecc="bf16+8")  # state + params
```

For first-order optimizers (where all state is momentum + variance), `bf16+8` gives 25% state memory savings. For second-order methods, preconditioner matrices are not compressed, so total savings are lower. Encode/decode are fully elementwise and fuse into the compiled kernel.

Available modes: `bf16+8`, `bf16+16`, `fp16+8`, `fp16+16`. Start with `bf16+8`.

## Distributed Training

Works with DDP and FSDP out of the box. For second-order methods (Muon, SOAP, PSGD), HeavyBall auto-detects FSDP-sharded parameters on the first step, gathers them to their original shapes, and distributes preconditioning compute across ranks in round-robin — each rank owns a subset of weight matrices and broadcasts the results. This saves `(N-1)/N` of the second-order compute per rank. First-order methods work without any special handling.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from heavyball import Muon

model = FSDP(model, use_orig_params=True)
opt = Muon(model.parameters(), lr=0.02)  # auto-detects sharding, distributes compute
```

For non-FSDP sharding backends, capture shapes before wrapping:

```python
from heavyball import SOAP, capture_param_shapes

shapes = capture_param_shapes(model)
model = your_sharding_wrapper(model)
opt = SOAP(model.parameters(), lr=3e-3, orig_shapes=shapes)
```

## Building Custom Optimizers

Every built-in optimizer is a chain of `FunctionTransform`s — the same API available to you. Use `Branch` to run parallel transform paths and merge their outputs for grafted optimizers, ensemble updates, or any topology.

```python
import heavyball.chainable as C

def graft(outputs, eps=1e-8):
    adam_update, sgd_update = outputs
    return [s * (a.norm() / s.norm().add(eps)) for a, s in zip(adam_update, sgd_update)]

class GraftedAdam(C.BaseOpt):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_steps=0, foreach=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        branch = C.Branch(branches=[[C.scale_by_adam], [C.identity]], merge_fn=graft)
        super().__init__(params, defaults, foreach, fns=(branch,))
```

Custom optimizers built this way automatically get: `torch.compile` fusion, ECC support, MARS, caution, gradient/update clipping, warmup, stochastic rounding, and `foreach` batching — all inherited from `BaseOpt`.

Key transforms: `scale_by_adam`, `scale_by_laprop`, `scale_by_soap`, `scale_by_psgd`, `scale_by_adopt`, `scale_by_ademamix`, `orthogonalize_update`, `exp_avg`, `nesterov_ema`, `heavyball_momentum`, `mars`, `palm_beta2`, `sign`, `identity`.

<details>
<summary>How it compiles</summary>

Every building block in `utils.py` — stochastic rounding, EMA updates, Adam scaling, Newton-Schulz iterations, Kronecker factor updates, gradient clipping, ECC encode/decode — is wrapped with `torch.compile(fullgraph=True)`. When one compiled function calls another, the inner function detects it's already compiling via `is_compiling()` and inlines, so nested calls fuse into the same compiled graph.

For fused first-order optimizers (AdamW, LaProp, ADOPT, NAdam, AdEMAMix), the entire update runs in a single compiled function and fuses into minimal Triton kernels. Features like stochastic rounding, ECC encode/decode, weight decay, and cautious masking fold into the same graph. MARS gradient correction runs as a separate compiled function before the update.

Second-order methods compile their preconditioning steps separately: Newton-Schulz iterations (Muon) and Kronecker factor updates (PSGD) each compile as individual regions, while their elementwise portions still fuse.

Custom optimizers built via the chainable API inherit this behavior — the building blocks they compose are the same compiled functions.

</details>

## Benchmarks

HeavyBall includes a diagnostic benchmark suite ([docs/benchmark.md](docs/benchmark.md)) via [LightBench](https://github.com/HomebrewML/LightBench). It tests for silent optimizer failures across difficulty levels, not leaderboard rankings.

## Migrating from 1.x

See the [2.0.0 migration notes](docs/heavyball2.md) for a full checklist and `scripts/migrate_optimizer_state.py` for checkpoint conversion.

## Contributing

Fork, `pip install -e .[dev]`, `pytest`.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

The name "HeavyBall" comes from [Polyak's heavy-ball method](https://doi.org/10.1016/0041-5553(64)90137-5), the momentum technique underlying most modern optimizers.

[pypi]: https://pypi.org/project/heavyball/
[license]: LICENSE
