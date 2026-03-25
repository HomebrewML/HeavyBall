# HeavyBall

[![PyPI version](https://img.shields.io/pypi/v/heavyball?color=blue)][pypi] [![Downloads](https://img.shields.io/pypi/dm/heavyball)][pypi] [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)][license]

HeavyBall is an optimizer library for PyTorch where every optimizer is assembled from composable, compiled building
blocks. It includes API-compatible replacements for `torch.optim.AdamW`, `SGD`, and `RMSprop`, alongside Muon, SOAP (
Shampoo), PSGD (Kronecker), ADOPT, Schedule-Free, LaProp, and others.

The building blocks, over 100 functions in [`utils.py`](heavyball/utils.py), are each compiled with
`torch.compile(fullgraph=True)` and fuse into Triton kernels. Features like MARS gradient correction,
cautious updates, and [ECC state compression](#ecc) are implemented as chainable transforms that work as flags on any
optimizer. DDP and FSDP are [supported](#distributed-training), with automatic repartitioning for second-order methods.

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
from heavyball import SOAP  # Shampoo-based preconditioning
opt = SOAP(model.parameters(), lr=3e-3)
```

```python
from heavyball import Muon
opt = Muon(model.parameters(), lr=0.02, ecc="bf16+8", mars=True, caution=True)
```

```python
from heavyball import SplitOpt, Muon, AdamW
opt = SplitOpt([
    {'params': matrices, 'optimizer': Muon, 'lr': 0.02},
    {'params': vectors, 'optimizer': AdamW, 'lr': 1e-3},
])
```

The API matches `torch.optim`, with the same parameter groups, same `step()`/`zero_grad()` interface. See [
`examples/`](examples/) for training scripts.

## Optimizers

The library covers first-order methods (AdamW, NAdam, RMSprop, ADOPT, LaProp, SGD), orthogonal methods (Muon),
Shampoo-based preconditioning (SOAP and variants), PSGD with Kronecker and low-rank factorization, Schedule-Free
training, and SAM.

<details>
<summary>Full list</summary>

**First-order:**
AdamW, NAdam, RMSprop, ADOPT, ForeachAdEMAMix, LaProp, SignLaProp, SGD, Scion, UnscaledAdamW, ForeachAdamC, SUDSAdamW

**Schedule-Free:**
SFAdamW, PaLMSFAdamW

Schedule-Free optimizers override `.eval()` and `.train()` to swap between training and evaluation parameter states.
Call `opt.eval()` before validation and `opt.train()` before resuming training.

**Orthogonal:**
Muon, MuonLaProp, OrthoLaProp, LaPropOrtho

**Shampoo-based (SOAP):**
SOAP, PaLMSOAP, PrecondScheduleSOAP, PrecondSchedulePaLMSOAP, SOAPNAdam, SOAPAdEMAMix, ForeachSOLP

**PSGD (Kronecker):**
PSGDKron, CachedPSGDKron, DelayedPSGD, CachedDelayedPSGDKron, PurePSGD, NewtonPSGDKron, NewtonHybrid2PSGDKron

`Newton`-PSGD requires a closure passed to `step()`.

**PSGD (Low-Rank):**
PSGDLRA, DelayedPSGDLRA, NewtonPSGDLRA, NewtonHybrid2PSGDLRA

`Newton`-PSGD requires a closure passed to `step()`.

**SAM:**
SAMWrapper, MSAMLaProp

SAMWrapper requires a closure passed to `step()`.

MSAMLaProp overrides `.eval()` and `.train()` to swap between training and evaluation parameter states.
Call `opt.eval()` before validation and `opt.train()` before resuming training.

**Meta:**
SplitOpt

</details>

## Composable Features

These flags compose freely. For example, `LaProp(..., ecc="bf16+8", mars=True, caution=True, palm=True)` is valid.
They are available on all optimizers except SAMWrapper and SplitOpt, which delegate to inner optimizers.

| Flag                    | Effect                                                                                                                                                                           |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mars=True`             | Applies [MARS variance reduction](https://arxiv.org/pdf/2411.10438) via previous gradients.                                                                                      |
| `caution=True`          | [Masks update elements](https://arxiv.org/abs/2411.16085) that disagree with the gradient direction.                                                                             |
| `ecc="bf16+8"`          | [Compresses optimizer state](https://arxiv.org/pdf/2602.23349) to bf16 + int8 correction (3 bytes vs fp32's 4). See [ECC](#ecc).                                                 |
| `param_ecc="bf16+8"`    | Applies the same compression to parameters.                                                                                                                                      |
| `palm=True`             | Enables [PaLM-style beta2 scheduling](https://arxiv.org/abs/2204.02311). Only available on optimizers with beta2                                                                 |
| `gradient_clipping=...` | Clips incoming gradients. Accepts `"l2_clip_"`, `"rmsnorm_clip_"`, `"trust_region_clip_"`, `"a_law_compress"`, `"mu_law_compress"`, `"softsign_compress"`, or a custom callable. |
| `update_clipping=...`   | Clips outgoing updates after all transforms. Same options as `gradient_clipping`.                                                                                                |
| `promote=True`          | Promotes gradients to fp32 before the update.                                                                                                                                    |
| `warmup_steps=N`        | Linear learning rate warmup over N steps.                                                                                                                                        |

### ECC

ECC stores each optimizer state tensor as a bf16 value plus an int8 correction term (3 bytes total vs fp32's 4 bytes),
based on the approach from [FlashOptim](https://arxiv.org/abs/2602.23349). HeavyBall integrates ECC as a composable
flag: correction tensors are attached as attributes at call time, so any built-in optimizer handles ECC without
per-optimizer changes.

```python
opt = AdamW(model.parameters(), lr=1e-3, ecc="bf16+8")
opt = Muon(model.parameters(), lr=0.02, ecc="bf16+8", param_ecc="bf16+8")  # state + params
```

For first-order optimizers (where all state is momentum and variance), `bf16+8` gives roughly 25% state memory savings
compared to fp32.
For second-order methods, preconditioner matrices are not compressed, so total savings are lower. The encode and decode
operations are fully elementwise and fuse into the compiled kernel.

Available modes: `bf16+8`, `bf16+16`, `fp16+8`, `fp16+16`.

## Distributed Training

HeavyBall works with both DDP and FSDP. First-order optimizers are elementwise and operate directly on FSDP shards with
no repartitioning. Second-order methods (Muon, SOAP, PSGD) need the full parameter to compute their update, so HeavyBall
auto-detects FSDP-sharded parameters on the first step and repartitions them: each weight matrix is assigned to one rank
in round-robin, which reconstructs the full parameter, computes the update, and broadcasts the result. This saves both
compute and memory compared to DDP-style redundant updates, at the cost of communication.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from heavyball import Muon

model = FSDP(model, use_orig_params=True)  # use_orig_params required for shape detection
opt = Muon(model.parameters(), lr=0.02)
```

For non-FSDP sharding backends, capture the original parameter shapes before wrapping:

```python
from heavyball import SOAP, capture_param_shapes

shapes = capture_param_shapes(model)
model = your_sharding_wrapper(model)
opt = SOAP(model.parameters(), lr=3e-3, orig_shapes=shapes)
```

## Building Custom Optimizers

Every built-in optimizer is a chain of `FunctionTransform`s, an API also available for building custom optimizers.
`Branch` runs parallel transform paths with a merge function, which is useful for grafted optimizers or ensemble
updates.

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

Custom optimizers that inherit from `BaseOpt` get ECC, MARS, caution, clipping, warmup, and stochastic rounding
automatically.

Key transforms: `scale_by_adam`, `scale_by_laprop`, `scale_by_soap`, `scale_by_psgd`, `scale_by_adopt`,
`scale_by_ademamix`, `orthogonalize_update`, `exp_avg`, `nesterov_ema`, `heavyball_momentum`, `mars`, `palm_beta2`,
`sign`, `identity`.

<details>
<summary>How it compiles</summary>

Every building block in `utils.py` is wrapped with `torch.compile(fullgraph=True)`. When one
compiled function calls another, the inner function inlines and nested calls fuse into the same compiled graph.

For fused first-order optimizers (AdamW, LaProp, ADOPT, NAdam, AdEMAMix), the entire update runs in a single compiled
function and fuses into minimal kernels. Stochastic rounding, ECC encode/decode, weight decay, and cautious
masking all fold into the same graph, reducing the memory traffic to a minimum. Adam without add-ons gets reduced from
14 reads + 9 writes in O(N) kernels to 4 reads + 3 writes in one kernel, a 3x speedup.

Second-order methods compile their preconditioning steps separately: Newton-Schulz iterations (Muon) and Kronecker
factor updates (PSGD, SOAP) each compile as individual regions, while their elementwise portions still fuse. This avoids
suboptimal code paths, at the cost of one graph break.

Custom optimizers built via the chainable API inherit this behavior.

</details>

## Benchmarks

HeavyBall includes a diagnostic benchmark suite via [LightBench](https://github.com/HomebrewML/LightBench) that tests
for silent optimizer failures across difficulty levels. Results and methodology are documented
in [docs/benchmark.md](docs/benchmark.md).

## Migrating from 1.x

See the [2.0.0 migration notes](docs/heavyball2.md) for a full checklist, and `scripts/migrate_optimizer_state.py` for
checkpoint conversion.

## Contributing

To contribute, fork the repository, install with `pip install -e .[dev]`, and run `pytest`.

## License

BSD-3-Clause, see [LICENSE](LICENSE).

The name "HeavyBall" comes from [Polyak's heavy-ball method](https://doi.org/10.1016/0041-5553(64)90137-5), the momentum
technique underlying most modern optimizers.

[pypi]: https://pypi.org/project/heavyball/

[license]: LICENSE
