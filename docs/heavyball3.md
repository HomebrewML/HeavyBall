# HeavyBall 3.0.0

## Highlights

* Simplified public API: `Foreach*` prefixes removed, short names are now the canonical classes
* New optimizers: `HyperBallAdamW`, `MuonAdamW`, `LATHER`, `PSGDPRO`
* `LATHER`, "Lie-group Adam Through Harmonic Eigenbasis Rotations", performs AdamW in the PSGD eigenbasis
* `Route`-based parameter dispatch handles mixed paths within one optimizer; `SplitOpt` remains available when the paths
  need different optimizer classes
* `ScheduleFree` and `MSAM` mode switches are now idempotent (`eval()` twice is safe)
* Higher-precision PSGD preconditioner updates
* New `consume_grad` option: `step()` clears `p.grad` after consuming it by default; set `consume_grad=False` to keep gradients attached after the step
* `orig_shapes` is now an explicit documented optimizer argument; use `capture_param_shapes(...)` before wrapping models with sharding backends that do not preserve original parameter shapes
* `torch.compile`-friendly step with automatic eager fallback for init/preconditioning

---

## Release benchmarks

HeavyBall 3.0.0 was benchmarked against HeavyBall 2.0.0 and `torch.optim` with
[`benchmarks/bench_optimizer_step.py`](../benchmarks/bench_optimizer_step.py), with compiled AdamW step latency
dropping from 10.63 ms in HeavyBall 2.0.0 to 4.15 ms in HeavyBall 3.0.0, a 2.56x speedup.

## Breaking changes

### Class renames

Every `Foreach*` class is renamed to its short form. The old short-form aliases (which existed
in 2.x) keep working, only the `Foreach*` imports break.

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
| `ForeachCachedPSGDKron` / `CachedPSGDKron` | `PSGDKron(..., cached=True)` |
| `ForeachDelayedPSGD` / `DelayedPSGD` | `PSGDKron(..., delayed=True)` |
| `ForeachCachedDelayedPSGDKron` / `CachedDelayedPSGDKron` | `PSGDKron(..., delayed=True, cached=True)` |
| `ForeachCachedNewtonPSGD` / `NewtonPSGDKron` | `PSGDKron(..., hessian_approx=True, cached=True)` |
| `NewtonHybrid2PSGDKron` | `PSGDKron(..., hessian_approx=True, hvp_interval=2, cached=True)` |
| `ForeachDelayedPSGDLRA` / `DelayedPSGDLRA` | `PSGDLRA(..., delayed=True)` |
| `ForeachNewtonPSGDLRA` / `NewtonPSGDLRA` | `PSGDLRA(..., hessian_approx=True)` |
| `NewtonHybrid2PSGDLRA` | `PSGDLRA(..., hessian_approx=True, hvp_interval=2)` |

### Renamed parameters

| 2.x parameter | 3.x parameter | Notes |
|---|---|---|
| `foreach` | `multi_tensor` | The old name is removed |

### Removed parameters

These raise `TypeError` if passed. They were either unused or replaced by better defaults.

| Parameter | Previously on | Notes                                                    |
|---|---|----------------------------------------------------------|
| `stochastic_schedule` | SOAP, PSGDKron, PSGDLRA | Deterministic accumulation schedule is now the only mode |
| `normalize_grads` | SOAP variants | Was unused in the transform pipeline                     |
| `correct_bias` | SOAP variants | Was unused in the transform pipeline                     |
| `inverse_free` | PSGDKron | Use `quad_torch` or PSGDPRO for inverse-free PSGD        |
| `adaptive` | PSGDKron | Removed                                                  |
| `palm`, `beta2_scale` | Muon, KLShampoo | These optimizers have no second-moment coefficient |

### Helper sampler kwargs

These compatibility kwargs were removed from `heavyball.helpers` samplers and now raise
`TypeError`.

| Class | Removed kwargs |
|---|---|
| `BoTorchSampler` | `constraints_func`, `consider_running_trials` |
| `HEBOSampler` | `constant_liar` |
| `ImplicitNaturalGradientSampler` | `lr`, `warn_independent_sampling` |
| `AutoSampler` | `constraints_func` |

### Chainable API renames

| 2.x name | 3.x name |
|---|---|
| `Branch` | `Parallel` |

### Behavioral changes

* **ScheduleFree / MSAM `eval()` / `train()`**: Now idempotent. Calling `eval()` twice no
  longer flips back to train mode. Both methods accept a `mode` argument matching
  `nn.Module.train(mode)` and return `self`.
* **Gradient lifetime**: `consume_grad=True` is available on all optimizers and clears `p.grad`
  during `step()` once the gradient has been consumed. Set `consume_grad=False` if your code
  reads gradients after stepping or relies on them remaining attached.
* **Sharded parameter shapes**: Built-in optimizers now expose `orig_shapes` explicitly. Use
  `capture_param_shapes()` before wrapping parameters if your sharding backend hides original
  shapes.
* **HyperBallAdamW routing**: Matrix parameters use HyperBall; 1D parameters now use full AdamW rather than only its
  second-moment scaling.
* **SplitOpt construction**: Each spec now requires a HeavyBall optimizer class rather than an instance, with constructor
  options in the spec. Hessian-approximation children and `add_param_group()` are unsupported.
* **SAMWrapper construction**: Class form accepts per-group hyperparameters from parameter-group dictionaries. Use a
  preconfigured instance for optimizer-wide flags.
* **PSGD dampening**: `dampening` default changed from `2**-13` to `1e-9`, and the dampening
  epsilon follows the promoted working dtype. This improves
  preconditioner accuracy but may change convergence behavior.

---

## Upgrade checklist

1. Replace `from heavyball import Foreach*` with the short name (e.g., `ForeachAdamW` → `AdamW`)
2. Replace `foreach=` with `multi_tensor=` in constructor calls
3. Replace removed subclass instantiations with parent + kwargs (see table above)
4. Remove any `stochastic_schedule`, `normalize_grads`, `correct_bias`, `inverse_free`, or `adaptive` kwargs
5. Replace `Branch(...)` with `Parallel(...)` in custom chainable code
6. If you relied on `eval(); eval()` toggling back to train mode, update your code
7. If your training loop reads `p.grad` after `step()`, pass `consume_grad=False`
8. Remove obsolete compatibility kwargs from `heavyball.helpers` samplers
