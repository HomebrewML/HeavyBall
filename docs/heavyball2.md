# Migrating from 1.x

!!! warning "Breaking changes"
    - Default orthogonalization switches to Newton-Schulz, impacting Muon. SOAP is unaffected (it uses QR decomposition through a separate code path).
    - Optimizer state keys now include a per-transform index (e.g., `exp_avg_3`), breaking old checkpoints.

## Upgrade checklist

1. Re-test optimizers sensitive to orthogonalization. To restore 1.x behavior:
   ```python
   heavyball.utils.zeroth_power_mode = "qr"
   ```
2. Migrate checkpoints:
   ```bash
   python scripts/migrate_optimizer_state.py <checkpoint_path> heavyball.<OptimizerClass>
   ```
3. Update any custom state-dict tooling to handle transform-indexed keys.

## What's new

* First-class SAM via `SAMWrapper` (closure-based)
* Robust checkpoint/restore with HeavyBall-internal state
* New optimizers: `SGD`, `ForeachAdamC`, `MSAMLaProp`
* Overhauled chainable pipeline: indexed transforms, [branching](https://github.com/HomebrewML/HeavyBall/blob/2f7e095fb8217a58600d86ea6b19682c10e7eb33/examples/branched_optimizer.py#L15-L28) for native [grafting](https://openreview.net/forum?id=FpKgG31Z_i9) support, `PrecondGradAccumGuard`, `SqueezeGrad`
* Faster, more accurate PSGD and SOAP via improved SVD and better compilation
* `heavyball.helpers` module with Optuna-compatible samplers and sweep utilities
