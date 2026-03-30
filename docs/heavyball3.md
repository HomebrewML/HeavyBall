# HeavyBall 3.0.0

## Highlights

* Simplified public API: `Foreach*` prefixes removed, short names are now the canonical classes
* New optimizers: `HyperBallAdamW`, `MuonAdamW`, `PSGDPRO`
* `Route`-based param dispatch replaces manual `SplitOpt` for mixed-architecture optimizers
* `ScheduleFree` and `MSAM` mode switches are now idempotent (`eval()` twice is safe)
* Higher-precision PSGD preconditioner updates
* `torch.compile`-friendly step with automatic eager fallback for init/preconditioning

---

## Breaking changes

### Class renames

Every `Foreach*` class is renamed to its short form. The old short-form aliases (which existed
in 2.x) keep working — only the `Foreach*` imports break.

| 2.x name | 3.x name |
|---|---|
| `ForeachAdamW` | `AdamW` |
| `ForeachNAdam` | `NAdam` |
| `ForeachAdEMAMix` | `AdEMAMix` |
| `ForeachAdamC` | `AdamC` |
| `ForeachRMSprop` | `RMSprop` |
| `ForeachSFAdamW` | `SFAdamW` |
| `ForeachADOPT` | `ADOPT` |
| `ForeachMuon` | `Muon` |
| `ForeachLaProp` | `LaProp` |
| `ForeachSignLaProp` | `SignLaProp` |
| `ForeachSOAP` | `SOAP` |
| `ForeachSOAPNAdam` | `SOAPNAdam` |
| `ForeachSOAPAdEMAMix` | `SOAPAdEMAMix` |
| `ForeachSOLP` | `SOLP` |
| `ForeachPSGDKron` | `PSGDKron` |
| `ForeachPSGDLRA` | `PSGDLRA` |

### Removed optimizer classes

These were thin subclasses that only set a class-level default. Use the parent class with the
corresponding constructor argument instead.

| 2.x class | 3.x equivalent |
|---|---|
| `PaLMForeachSFAdamW` / `PaLMSFAdamW` | `SFAdamW(..., palm=True)` |
| `PaLMForeachSOAP` / `PaLMSOAP` / `PalmForEachSoap` | `SOAP(..., palm=True)` |
| `PrecondScheduleForeachSOAP` / `PrecondScheduleSOAP` | `SOAP(..., use_precond_schedule=True)` |
| `PrecondSchedulePaLMForeachSOAP` / `PrecondSchedulePaLMSOAP` | `SOAP(..., palm=True, use_precond_schedule=True)` |
| `ForeachPurePSGD` / `PurePSGD` | `PSGDKron(..., exp_avg_input=False)` |
| `ForeachCachedPSGDKron` / `CachedPSGDKron` | `PSGDKron(...)` (caching is now the default) |
| `ForeachDelayedPSGD` / `DelayedPSGD` | `PSGDKron(..., delayed=True)` |
| `ForeachCachedDelayedPSGDKron` / `CachedDelayedPSGDKron` | `PSGDKron(..., delayed=True)` |
| `ForeachCachedNewtonPSGD` / `NewtonPSGDKron` | `PSGDKron(..., hessian_approx=True)` |
| `NewtonHybrid2PSGDKron` | `PSGDKron(..., hessian_approx=True, hvp_interval=2)` |
| `ForeachDelayedPSGDLRA` / `DelayedPSGDLRA` | `PSGDLRA(..., delayed=True)` |
| `ForeachNewtonPSGDLRA` / `NewtonPSGDLRA` | `PSGDLRA(..., hessian_approx=True)` |
| `NewtonHybrid2PSGDLRA` | `PSGDLRA(..., hessian_approx=True, hvp_interval=2)` |

### Renamed parameters

| 2.x parameter | 3.x parameter | Notes |
|---|---|---|
| `foreach` | `multi_tensor` | Passing `foreach` emits a `FutureWarning` and remaps automatically |

### Removed parameters

These raise `TypeError` if passed. They were either unused or replaced by better defaults.

| Parameter | Previously on | Notes                                                    |
|---|---|----------------------------------------------------------|
| `stochastic_schedule` | SOAP, PSGDKron, PSGDLRA | Deterministic accumulation schedule is now the only mode |
| `normalize_grads` | SOAP variants | Was unused in the transform pipeline                     |
| `correct_bias` | SOAP variants | Was unused in the transform pipeline                     |
| `inverse_free` | PSGDKron | Use `quad_torch` or PSGDPRO for inverse-free PSGD        |
| `adaptive` | PSGDKron | Removed                                                  |

### Chainable API renames

| 2.x name | 3.x name |
|---|---|
| `Branch` | `Parallel` |

### Behavioral changes

* **ScheduleFree / MSAM `eval()` / `train()`**: Now idempotent. Calling `eval()` twice no
  longer flips back to train mode. Both methods accept a `mode` argument matching
  `nn.Module.train(mode)` and return `self`.
* **PSGD dampening**: `dampen_grad` default changed from `2**-13` to `1e-9`, and dampening
  epsilon uses `torch.finfo(float32).eps` regardless of input dtype. This improves
  preconditioner accuracy but may change convergence behavior.

---

## Checkpoint migration

Use the migration CLI to convert 1.x or 2.x checkpoints:

```bash
python scripts/migrate_optimizer_state.py <checkpoint.pt> <OptimizerClass>
```

Old class names (including all aliases listed above) are resolved automatically.
The `foreach` → `multi_tensor` key rename in param groups is handled automatically.

---

## Upgrade checklist

1. Replace `from heavyball import Foreach*` with the short name (e.g., `ForeachAdamW` → `AdamW`)
2. Replace `foreach=` with `multi_tensor=` in constructor calls
3. Replace removed subclass instantiations with parent + kwargs (see table above)
4. Remove any `stochastic_schedule`, `normalize_grads`, `correct_bias`, `inverse_free`, or `adaptive` kwargs
5. Replace `Branch(...)` with `Parallel(...)` in custom chainable code
6. Migrate checkpoints: `python scripts/migrate_optimizer_state.py <ckpt> heavyball.<Optimizer>`
7. If you relied on `eval(); eval()` toggling back to train mode, update your code
