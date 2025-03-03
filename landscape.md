# Benchmark Results
Generated: 2025-02-22 10:51:19.457377
Last updated: 2025-02-22 10:51:19.457388

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| ForeachSOAP | No | No | 1/1 | 19.37s | 5.0 |
| LaProp | No | No | 1/1 | 20.04s | 5.0 |
| AdamW | No | No | 1/1 | 17.83s | 5.0 |
| Muon | No | No | 0/1 | 0.00s | 0.0 |
| ForeachCachedNewtonPSGD | No | No | 1/1 | 51.90s | 18.0 |
| RMSprop | No | No | 1/1 | 31.00s | 10.0 |
| OrthoLaProp | No | No | 0/1 | 0.00s | 0.0 |
| ForeachSFAdamW | No | No | 1/1 | 23.61s | 5.0 |
| ForeachADOPT | No | No | 0/1 | 0.00s | 0.0 |
| LaPropOrtho | No | No | 0/1 | 0.00s | 0.0 |
| CachedPSGDKron | No | No | 1/1 | 22.17s | 4.0 |
| SignLaProp | No | No | 1/1 | 68.20s | 36.0 |
| ForeachSOLP | No | No | 1/1 | 18.54s | 5.0 |
| AdamW | Yes | No | 1/1 | 20.71s | 5.0 |
| AdamW | Unscaled | No | 1/1 | 23.29s | 5.0 |
| AdamW | No | Yes | 1/1 | 19.08s | 5.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts |
|-----------|-----------|---------|---|---|----------|------|---|
| dynamic_landscape | AdamW | No | No | ✓ | 17.83s | 8.95e-03 | 5 |
| dynamic_landscape | CachedPSGDKron | No | No | ✓ | 22.17s | 9.35e-03 | 4 |
| dynamic_landscape | ForeachADOPT | No | No | ✗ | 645.10s | 4.90e-01 | 1000 |
| dynamic_landscape | ForeachCachedNewtonPSGD | No | No | ✓ | 51.90s | 9.94e-03 | 18 |
| dynamic_landscape | ForeachSFAdamW | No | No | ✓ | 23.61s | 9.37e-03 | 5 |
| dynamic_landscape | ForeachSOAP | No | No | ✓ | 19.37s | 9.74e-03 | 5 |
| dynamic_landscape | ForeachSOLP | No | No | ✓ | 18.54s | 8.25e-03 | 5 |
| dynamic_landscape | LaProp | No | No | ✓ | 20.04s | 9.22e-03 | 5 |
| dynamic_landscape | LaPropOrtho | No | No | ✗ | 290.69s | 9.62e-01 | 453 |
| dynamic_landscape | Muon | No | No | ✗ | 287.13s | 3.79e-01 | 245 |
| dynamic_landscape | OrthoLaProp | No | No | ✗ | 238.36s | 9.30e-01 | 349 |
| dynamic_landscape | RMSprop | No | No | ✓ | 31.00s | 9.80e-03 | 10 |
| dynamic_landscape | SignLaProp | No | No | ✓ | 68.20s | 9.78e-03 | 36 |
| dynamic_landscape | AdamW | Yes | No | ✓ | 20.71s | 8.92e-03 | 5 |
| dynamic_landscape | AdamW | No | Yes | ✓ | 19.08s | 8.86e-03 | 5 |
| dynamic_landscape | AdamW | Unscaled | No | ✓ | 23.29s | 9.75e-03 | 5 |

## Errors
