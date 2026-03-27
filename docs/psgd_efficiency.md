# PSGD Efficiency

## `store_triu_as_line`

Stores only the upper triangle of the triangular preconditioner `Q` as a 1D array, halving Q memory. Enabled by default.

The cost is remapping the 1D array to 2D on every preconditioner application, which is memory-bandwidth bound. On a
high-overhead test case (`python3 -m lightbench.xor_digit --batch 16 --size 1024 --length 4 --depth 1`), total step time
increased by ~58%. Larger batch sizes amortize this.

![psgd_efficiency_triu_as_line.png](assets/psgd_efficiency_triu_as_line.png)

```python
from heavyball import PSGDKron
opt = PSGDKron(model.parameters(), lr=1e-3, store_triu_as_line=True)
```

## Cached Preconditioner

`CachedPSGDKron` precomputes and caches the full preconditioning matrix rather than reconstructing it from `Q` each
step. Faster steps, at the cost of storing an additional full-size Q copy (2x Q memory without `store_triu_as_line`, or
1.5x Q with it).

![psgd_efficiency_cache.png](assets/psgd_efficiency_cache.png)

```python
from heavyball import CachedPSGDKron
opt = CachedPSGDKron(model.parameters(), lr=1e-3)
```

Combining both (`CachedPSGDKron` with `store_triu_as_line=True`) gives 1.5x Q: compressed triangular storage plus a
full-size cache.

![psgd_efficiency_cache_triu_as_line.png](assets/psgd_efficiency_cache_triu_as_line.png)

!!! tip "Which variant?"
    - Memory-constrained: `store_triu_as_line=True` (default)
    - Speed-constrained: `CachedPSGDKron`
    - Both: `CachedPSGDKron` with `store_triu_as_line=True`

    See [Choosing an Optimizer](optimizer_guide.md) for the full comparison.
