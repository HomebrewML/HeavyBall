"""Every optimizer's state round-trips through a checkpoint bit-for-bit.

Restoring the model parameters, the optimizer state_dict, and the torch RNG, a resumed run matches an
uninterrupted one exactly. Before this only SOAP (cadence) and AdamW (multigroup) were covered; this
gates state serialization across all facades. The RNG restore isolates state completeness from the
per-step randomness Muon (stochastic rounding), the PSGD family, Scion, and LATHER draw -- without it
those resume correctly but not bit-for-bit (a documented checkpointing caveat, not a state bug).
"""

from unittest.mock import patch

import pytest
import torch

import heavyball.optim as optim

FACADES = tuple(sorted(
    name for name in optim.__all__
    if name[0].isupper()
    and name != "HeavyBallOptimizer"
    and isinstance(getattr(optim, name, None), type)
    and issubclass(getattr(optim, name), optim.HeavyBallOptimizer)
))


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(12, 12), torch.nn.GELU(), torch.nn.Linear(12, 6))


def _needs_observations(optimizer):
    return any(
        group.recipe.observations for engine in optimizer._engines for group in engine.groups
    )


def _train(model, optimizer, inputs, targets, steps):
    produce = _needs_observations(optimizer)
    for _ in range(steps):
        ((model(inputs) - targets) ** 2).mean().backward()
        if produce:
            for p in model.parameters():
                if p.grad is not None:
                    optimizer.produce(p, "sum_grad_squared", p.grad.square())
        optimizer.step()
        optimizer.zero_grad()


# storage_dtype/ecc add a bf16 state slab plus int8/int16 correction slabs (and stochastic rounding) to
# serialize; default resume was covered, these gate the low-precision slabs too. Measured bit-for-bit
# exact on CPU eager and GPU.
_PRECISIONS = [("default", {}), ("bf16", {"storage_dtype": torch.bfloat16}), ("ecc8", {"ecc": 8}), ("ecc16", {"ecc": 16})]


@pytest.mark.parametrize("precision", _PRECISIONS, ids=lambda item: item[0])
@pytest.mark.parametrize("name", FACADES)
def test_checkpoint_resume_is_bit_identical(name, precision):
    opt_kwargs = precision[1]
    torch.manual_seed(1)
    inputs, targets = torch.randn(24, 12), torch.randn(24, 6)

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        base = _model()
        base_opt = getattr(optim, name)(base.parameters(), lr=1e-2, **opt_kwargs)
        _train(base, base_opt, inputs, targets, 8)

        interrupted = _model()
        interrupted_opt = getattr(optim, name)(interrupted.parameters(), lr=1e-2, **opt_kwargs)
        _train(interrupted, interrupted_opt, inputs, targets, 4)
        model_checkpoint = {key: value.clone() for key, value in interrupted.state_dict().items()}
        optimizer_checkpoint = interrupted_opt.state_dict()
        rng_checkpoint = torch.get_rng_state()

        resumed = _model()
        resumed_opt = getattr(optim, name)(resumed.parameters(), lr=1e-2, **opt_kwargs)
        resumed.load_state_dict(model_checkpoint)
        resumed_opt.load_state_dict(optimizer_checkpoint)
        torch.set_rng_state(rng_checkpoint)
        _train(resumed, resumed_opt, inputs, targets, 4)

    for base_param, resumed_param in zip(base.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(base_param, resumed_param, rtol=0, atol=0)
