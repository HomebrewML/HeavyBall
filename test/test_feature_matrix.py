"""Feature-compatibility matrix -- compile-first, the LIVE system.

Every optimizer is one Engine running a different transform recipe; the state-precision features
(storage_dtype, ecc) are handled uniformly by the Engine, orthogonally to the recipe. HeavyBall is
compile-first -- the whole step is one fullgraph compiled with max-autotune (core.py). This compiles that
EXACT live step (no compile-mode override, no eager) for every optimizer x distinct state-precision feature
on real transformer-shaped weights, and MEASURES each cell: compile time, steady step time, peak GPU memory,
optimizer-state bytes (so bf16's halving shows), and the loss delta.

It asserts COMPATIBILITY, not convergence -- whether a default lr reduces the loss here is lr/task-dependent
(ADOPT, for one, diverges identically eager and compiled). Each cell must: compile+run the live step, stay
finite, and move every parameter. All-moved catches a fully dead update, but not a matrix route silently
masked by an AdamW fallback: routing is independent, and per-route correctness belongs to the parity tests.
Every floating state slab must also use the feature's requested state dtype; stateless optimizers pass this
check trivially. The measurements are printed as `MEASURE {json}` lines so a run characterizes the whole
surface, not just pass/fail. Deep bf16/ecc numerics live in test_storage_dtype.py / test_ecc.py; this is the
breadth sweep.
"""
import inspect
import json
import math
import os
import time

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import heavyball

D, VOCAB, STEPS = 512, 4096, 25
OPTIMIZERS = sorted(
    name
    for name in dir(heavyball)
    if isinstance(getattr(heavyball, name), type)
    and issubclass(getattr(heavyball, name), heavyball.HeavyBallOptimizer)
    and getattr(heavyball, name) is not heavyball.HeavyBallOptimizer
)
assert len(OPTIMIZERS) >= 55, f"optimizer feature matrix unexpectedly shrank: {len(OPTIMIZERS)} < 55"
# Distinct state-precision settings. ecc forces bf16 storage, so (storage=bf16, ecc) duplicates (None, ecc);
# only these four compile a distinct Engine path. Each carries the state dtype the feature must produce.
FEATURES = [
    ("fp32", dict(storage_dtype=None, ecc=None), torch.float32),
    ("bf16", dict(storage_dtype=torch.bfloat16, ecc=None), torch.bfloat16),
    ("ecc8", dict(storage_dtype=None, ecc=8), torch.bfloat16),
    ("ecc16", dict(storage_dtype=None, ecc=16), torch.bfloat16),
]


class Transformer(nn.Module):
    """Real transformer weight shapes: embedding, square, wide (in_proj), tall/wide MLP, norm, head."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.in_proj = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.up = nn.Linear(D, 4 * D)
        self.down = nn.Linear(4 * D, D)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, idx):
        x = self.embed(idx).mean(1)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        x = self.out(torch.tanh(q) * torch.sigmoid(k) + v)
        return self.head(self.norm(x + self.down(F.gelu(self.up(x)))))


def _state_slabs(opt):
    return [
        slab
        for engine in opt._engines
        for group in engine.groups
        for slots in (*group.states, group.commit_state)
        for slab in slots.values()
    ]


def _correction_slabs(opt):
    return [
        slab
        for engine in opt._engines
        for group in engine.groups
        for slots in (*group.state_corrections, group.commit_corrections)
        for slab in slots.values()
    ]


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f[0])
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_full_step_compiled(name, feature):
    """Compile the live max-autotune full step on real shapes; assert compatible and measure the cell."""
    fname, kwargs, state_dtype = feature
    if not torch.cuda.is_available():
        pytest.skip("compile-first GPU path requires CUDA")
    if name.startswith("TrueGrad"):
        pytest.skip("TrueGrad facades require an external observation producer")
    facade = getattr(heavyball, name)
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = Transformer().cuda()
    idx = torch.randint(VOCAB, (4, 8), device="cuda")
    target = torch.randint(VOCAB, (4,), device="cuda")
    lr = inspect.signature(facade).parameters["lr"].default
    opt = facade(model.parameters(), lr=lr, **kwargs)
    before = [p.detach().clone() for p in model.parameters()]
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = F.cross_entropy(model(idx), target)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    slabs = _state_slabs(opt)
    float_dtypes = {slab.dtype for slab in slabs if slab.is_floating_point()}
    tag = f"{name} {fname}"
    assert all(map(math.isfinite, losses)), f"{tag}: NONFINITE loss"
    assert all(p.isfinite().all() for p in model.parameters()), f"{tag}: NONFINITE params"
    assert all(
        not torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True)
    ), f"{tag}: a parameter route never updated"
    # Every narrowable state slab must take the feature's dtype (this is where bf16 halves the state);
    # deliberately higher-precision slabs (PSGD/LATHER keep fp64 stability scalars) are allowed. A slab at
    # the WRONG low precision means the feature did not engage, or a matrix route was masked by an fp32/bf16
    # fallback. Stateless optimizers have no float slabs and pass trivially.
    wrong = torch.float32 if state_dtype is torch.bfloat16 else torch.bfloat16
    if float_dtypes:
        assert state_dtype in float_dtypes, f"{tag}: feature did not engage ({state_dtype} state not allocated)"
        assert wrong not in float_dtypes, f"{tag}: a float state slab is {wrong} (unnarrowed state or masked route)"

    # The bf16 engage-check above cannot tell ecc8/ecc16 apart from plain bf16 -- both narrow float state to
    # bf16. ecc's whole point is the extra int residual, so require it: the correction slabs must exist, be the
    # right int width (int8 vs int16), and be written (nonzero after real steps). Without this, a silent
    # ecc->plain-bf16 regression passes every ecc cell. Non-ecc cells must carry no correction slabs.
    corrections = _correction_slabs(opt)
    ecc = kwargs.get("ecc")
    if ecc and float_dtypes:
        corr_dtype = torch.int8 if ecc == 8 else torch.int16
        assert corrections, f"{tag}: ecc={ecc} added no int correction slab (silently plain bf16)"
        assert all(c.dtype is corr_dtype for c in corrections), f"{tag}: ecc={ecc} correction not {corr_dtype}"
        assert any(c.any() for c in corrections), f"{tag}: ecc={ecc} corrections all zero (never written)"
    if not ecc:
        assert not corrections, f"{tag}: {fname} allocated correction slabs without ecc"

    record = {
        "opt": name,
        "feat": fname,
        "compile_s": round(times[0], 2),
        "step_ms": round(1000 * sum(times[1:]) / max(len(times) - 1, 1), 2),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
        "state_mb": round(sum(s.numel() * s.element_size() for s in slabs) / 1e6, 2),
        "n_slabs": len(slabs),
        "loss": [round(losses[0], 3), round(losses[-1], 3)],
    }
    path = os.environ.get("FEATURE_MATRIX_MEASURE")
    if path:  # append is atomic per short line, so xdist workers can share one file
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
