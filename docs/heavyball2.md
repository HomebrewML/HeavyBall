# HeavyBall 2.0.0

## Highlights

* First‑class SAM via `SAMWrapper` (closure‑based)
* More robust checkpoint/restore with HeavyBall‑internal state
* New optimizers: `SGD`, `ForeachAdamC`, `MSAMLaProp`
* Overhauled chainable pipeline: indexed transforms, branching, internal gradient‑accumulation, and `SqueezeGrad`
* Faster, more accurate code paths
* New `heavyball.helpers` with Optuna‑compatible samplers and utilities

---

## Detailed changes

### New & improved

* `SAMWrapper` applies sharpness‑aware minimization to any HeavyBall optimizer while preserving the wrapped step logic;
  requires a closure
* `SGD` built on the chainable internals
* `ForeachAdamC`, a ["corrected version of Adam"](https://arxiv.org/abs/2506.02285) with weight decay normalized by the
  maximum LR
* `MSAMLaProp` built on top of [Momentum‑SAM](https://arxiv.org/abs/2401.12033)
* Chainable pipeline:
    * Every transform carries a `transform_idx`; state keys include this index
    * Branching supported
    * `PrecondGradAccumGuard` enables gradient accumulation for preconditioner fitting
    * `SqueezeGrad` removes size‑1 dims before functional transforms, improving PSGD's speed and preconditioner fitting
* PSGD and SOAP speedups through new, more accurate SVD calculation and better compilation
* `heavyball.helpers` module with Optuna‑compatible samplers and sweep utilities

### Breaking changes

* Default orthogonalization switches to Newton-Schulz, impacting Muon; SOAP relies on `precise_zeroth_power_mode="qr"`
  and remains unchanged
* Optimizer state keys include the per‑transform index (e.g., `exp_avg_3`), breaking old checkpoints

## Upgrade checklist

1. Re‑test optimizers sensitive to orthogonalization; set `utils.zeroth_power_mode="qr"` to restore 1.x behavior
2. Migrate checkpoints using `python scripts/migrate_optimizer_state.py <checkpoint_path> heavyball.<OptimizerClass>`
3. Update any custom state‑dict tooling to handle transform‑indexed keys