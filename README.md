# heavyball

[![PyPI version](https://img.shields.io/pypi/v/heavyball?color=blue)][pypi]
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)][license]

_High-performance, extensible, chainable optimizers for PyTorch._

## Why heavyball

- **Lightning-Fast Training**: Batched `foreach` operations deliver significant speedups on large models.
- **Adaptive & Extensible**: Built-in AdamW, RMSprop, Schedule-Free algorithms, and PaLM-inspired schedules.
- **Plug-and-Play**: Drop-in replacements for `torch.optim` with seamless integration.
- **Customizable**: Chainable API lets you compose optimizers and transforms (MARS correction, cautious updates, orthogonal updates).
- **Battle-Tested**: Extensive benchmarks and real-world examples included.

## Key Features

- Foreach-based optimizers: `ForeachAdamW`, `ForeachRMSprop`, `ForeachSFAdamW`, `Muon`, `ADOPT`, `MSAM` (Momentum SAM), …
- Schedule-Free optimizers with dynamic learning rate adaptation.
- Advanced update rules: MARS correction, cautious updates, PaLM beta2 scheduling.
- Chainable transforms for custom optimization recipes.
- Comprehensive benchmark suite packaged separately as LightBench (`../LightBench`).
- Detailed documentation and example-driven tutorials.

## Quickstart

**Install:**
```bash
pip install heavyball
```

**Basic usage:**
```python
import torch
from torch import nn
from heavyball import ForeachAdamW

model = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)
optimizer = ForeachAdamW(model.parameters(), lr=1e-3)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
```

## Benchmarks

> Reproduce benchmarks with LightBench (install it via `pip install -e ../LightBench` from the repo root):
> ```bash
> python3 -m lightbench.run_all_benchmarks --opt ForeachSOAP --opt LaProp --opt AdamW --opt Muon --opt ForeachCachedNewtonPSGD  --opt RMSprop --opt OrthoLaProp --opt ForeachSFAdamW --opt ForeachADOPT --opt LaPropOrtho --opt CachedPSGDKron --opt SignLaProp --opt ForeachSOLP --opt PSGDLRA --opt NewtonPSGDLRA --opt NewtonHybrid2PSGDKron --opt NewtonHybrid2PSGDLRA --opt mars-NewtonHybrid2PSGDLRA --opt MSAMLaProp --opt mars-adaptive-NewtonHybrid2PSGDKron  --opt mars-ortho-NewtonHybrid2PSGDKron --opt MuonLaProp --opt mars-unscaled-NewtonHybrid2PSGDKron --opt mars-NewtonHybrid2PSGDKron --opt cautious-AdamW --opt unscaled_cautious-AdamW --opt mars-AdamW  --dtype float32 --steps 1000000 --trials 1000 --parallelism 256 --seeds 1 --difficulties trivial --difficulties easy --difficulties medium --difficulties hard --difficulties extreme --difficulties nightmare --timeout 2880
> ```

## Migrating from HeavyBall 1.x

- Read the detailed [2.0.0 migration notes](docs/heavyball2.md) for an end-to-end checklist, including guidance for restoring legacy behaviour when needed.
- Use `scripts/migrate_optimizer_state.py` to rewrite pre-2.0 optimizer checkpoints:
  ```bash
  python scripts/migrate_optimizer_state.py path/to/checkpoint.pt heavyball.ForeachAdamW --state-key optimizer
  ```
  The utility renames legacy state entries, fans them out per parameter view, and injects the HeavyBall metadata block expected by 2.0.0.


## Contributing

We welcome contributions! Please check the [issue tracker][tracker] and follow these steps:
1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e .[dev]`.
3. Run tests: `pytest`.
4. Submit a pull request.

## License

BSD 3-Clause — see the [LICENSE](LICENSE) file.

---
<p align="center">
  Made by the HeavyBall team.
</p>

[pypi]: https://pypi.org/project/heavyball/
[license]: LICENSE
[tracker]: https://github.com/HomebrewML/HeavyBall/issues
