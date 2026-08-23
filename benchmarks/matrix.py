#!/usr/bin/env python3
"""The one test.

Cells are the outer product {shipped compositions, atom/pair chains} x precision
x scale x topology. Each cell trains one seeded problem for STEPS steps at
``max-autotune-no-cudagraphs`` and is judged against its upstream: the pinned
clones under ``references/`` and ``torch.optim`` are the accuracy oracle and the
speed baseline, run on the same recorded gradients. The gate is
``err_ours <= max(err_ref, err_semantic)`` -- our distance to the upstream fp64
run may not exceed the upstream's own at that precision, nor our own fp64
algorithmic floor. Twenty-three compositions carry an external upstream oracle (the pinned
official implementation of the same algorithm, run on the same gradients);
sixteen do not -- no same-algorithm upstream exists -- and their rows say
``upstream: null`` with every gate labeled ``self_*`` to make it
clear the comparison is ourselves-against-ourselves, not a baseline. Chain-law cells (``pairs``) are additionally
judged against themselves run eagerly at the same precision.

Usage:
  matrix.py list-cells | sample | full | pairs | cells | report | plots
  matrix.py ensure-references   # provision the pinned upstream clones
  matrix.py worker | batch      # internal
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import functools
import hashlib
import importlib
import importlib.util
import inspect
import io
import itertools
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.distributed as dist
import typer
from torch import nn
from torch.backends import opt_einsum
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
# Inserted before the heavyball imports on purpose: a stale editable install or
# PYTHONPATH pointing at a different checkout silently renders the matrix
# against foreign code.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REFERENCES = REPO_ROOT / "references"

import heavyball  # noqa: E402 -- must follow the sys.path guard above
from heavyball import chainable as C  # noqa: E402
from heavyball import utils  # noqa: E402
from heavyball.chainable import use_default  # noqa: E402

# ---------------------------------------------------------------------------
# The baselines: pinned upstream implementations, run as-is. The accuracy oracle
# and the speed baseline are the same object. ``kwargs`` maps our drawn
# hyperparameter names onto the baseline's constructor (the mapping is the only
# adapter there is); ``constants`` mirror our structural defaults onto knobs the
# draw does not jitter; ``subset`` names a parameter subset the baseline owns
# (official Muon trains >=2D parameters only -- both sides of the distance are
# restricted to that subset); ``promoted`` names the constructor knobs that give
# the baseline low-precision storage with fp32 compute, where it can express that
# at all. Pins and licenses live in references/PINS.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    name: str
    repo: str  # references/ subdirectory, or "torch" for torch.optim
    module: str
    cls: str
    kwargs: tuple[tuple[str, str], ...] = ()  # our hyper name -> baseline kwarg
    constants: dict = field(default_factory=dict)
    subset: str | None = None  # "ndim2": the baseline owns only >=2D parameters
    promoted: tuple[str, ...] = ()  # kwargs that set the compute dtype
    prepare: tuple[str, ...] = ()  # methods official usage calls after construction
    closure: bool = False  # upstream steps from a closure (its own forward)
    wrap_base: str = ""  # wrap this baseline's class as base_optimizer_cls (OrthoGrad style)
    groups: str = ""  # "muon_aux": upstream takes use_muon param groups (MuonWithAuxAdam)
    url: str = ""
    pin: str = ""
    license: str = ""
    note: str = ""


BASELINES = {
    "adamw": Baseline(
        "adamw",
        "torch",
        "torch.optim",
        "AdamW",
        kwargs=(("lr", "lr"), ("beta1", "betas"), ("beta2", "betas"), ("eps", "eps"), ("weight_decay", "weight_decay")),
        license="PyTorch",
    ),
    "sgd": Baseline(
        "sgd",
        "torch",
        "torch.optim",
        "SGD",
        kwargs=(("lr", "lr"), ("weight_decay", "weight_decay")),
        license="PyTorch",
    ),
    "muon": Baseline(
        "muon",
        "Muon",
        "muon",
        "SingleDeviceMuon",
        kwargs=(("lr", "lr"), ("beta1", "momentum"), ("weight_decay", "weight_decay")),
        subset="ndim2",
        url="https://github.com/KellerJordan/Muon",
        pin="f98f1ca",
        license="MIT",
        note="official Muon trains >=2D parameters; distance is over that subset",
    ),
    "soap": Baseline(
        "soap",
        "SOAP",
        "soap",
        "SOAP",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("shampoo_beta", "shampoo_beta"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("precondition_frequency", "precondition_frequency"),
        ),
        constants={"merge_dims": True, "max_precond_dim": 2048},  # our facade's structural defaults
        url="https://github.com/nikhilvyas/SOAP",
        pin="a1e5535",
        license="MIT",
    ),
    "psgd_kron": Baseline(
        "psgd_kron",
        "kron_torch",
        "kron_torch",
        "Kron",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "b1"),
            ("weight_decay", "weight_decay"),
            ("precond_lr", "precond_lr"),
            ("precond_init_scale", "precond_init_scale"),
        ),
        constants={
            "preconditioner_update_probability": 1.0,
            "merge_dims": False,
        },  # our precond schedule is 1.0 within STEPS
        promoted=("mu_dtype", "precond_dtype"),
        url="https://github.com/evanatyourservice/kron_torch",
        pin="884427c",
        license="CC-BY-4.0",
        note="the lineage PSGDKron's docstring cites; Procrustes variants have no torch.optim upstream",
    ),
    "schedule_free": Baseline(
        "schedule_free",
        "schedule_free",
        "schedulefree",
        "AdamWScheduleFree",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("r", "r"),
            ("weight_lr_power", "weight_lr_power"),
        ),
        prepare=("train",),
        url="https://github.com/facebookresearch/schedule_free",
        pin="70785b5",
        license="Apache-2.0",
    ),
    "nadam": Baseline(
        "nadam",
        "torch",
        "torch.optim",
        "NAdam",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("momentum_decay", "momentum_decay"),
        ),
        license="PyTorch",
    ),
    "rmsprop": Baseline(
        "rmsprop",
        "torch",
        "torch.optim",
        "RMSprop",
        kwargs=(("lr", "lr"), ("beta2", "alpha"), ("eps", "eps"), ("weight_decay", "weight_decay")),
        constants={"momentum": 0.0, "centered": False},
        license="PyTorch",
        note="torch RMSprop has no bias correction; the divergence lands in err_semantic",
    ),
    "adopt": Baseline(
        "adopt",
        "adopt",
        "src/adopt/adopt.py",
        "ADOPT",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
        ),
        url="https://github.com/iShohei220/adopt",
        pin="6468572",
        license="Apache-2.0",
    ),
    "ademamix": Baseline(
        "ademamix",
        "ademamix-optimizer-pytorch",
        "AdEMAMix.py",
        "AdEMAMix",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("beta3", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("alpha", "alpha"),
        ),
        url="https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch",
        pin="0f52410",
        license="MIT",
        note="the standard community port; AdEMAMix is Apple's algorithm",
    ),
    "mars": Baseline(
        "mars",
        "mars",
        "MARS/optimizers/mars.py",
        "MARS",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("mars_gamma", "gamma"),
        ),
        constants={"mars_type": "mars-adamw", "optimize_1d": True},
        url="https://github.com/AGI-Arena/MARS",
        pin="4831e28",
        license="Apache-2.0",
    ),
    "scion": Baseline(
        "scion",
        "scion",
        "scion.py",
        "Scion",
        kwargs=(("lr", "lr"), ("beta1", "momentum")),
        constants={"norm": "Auto"},
        url="https://github.com/LIONS-EPFL/scion",
        pin="f58a393",
        license="MIT",
        note="official momentum is one minus the traditional momentum",
    ),
    "laprop": Baseline(
        "laprop",
        "laprop-optimizer",
        "laprop.py",
        "LaProp",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
        ),
        url="https://github.com/Z-T-WANG/LaProp-Optimizer",
        pin="a419916",
        license="MIT",
    ),
    "cadamw": Baseline(
        "cadamw",
        "c-optim",
        "c_adamw.py",
        "AdamW",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
        ),
        url="https://github.com/kyleliang919/C-Optim",
        pin="a506ee3",
        license="MIT",
    ),
    "kl_shampoo": Baseline(
        "kl_shampoo",
        "kl-methods",
        "optim/kl_opt.py",
        "KLOpt",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("shampoo_beta", "shampoo_beta"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("precondition_frequency", "precondition_frequency"),
        ),
        url="https://github.com/yorkerlin/KL-Methods",
        pin="a02e622",
        license="none stated; dev-time oracle use only",
    ),
    "kl_soap_official": Baseline(
        "kl_soap_official",
        "kl-methods",
        "optim/kl_opt.py",
        "KLOpt",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("shampoo_beta", "shampoo_beta"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("precondition_frequency", "precondition_frequency"),
        ),
        constants={"using_klsoap": True},
        url="https://github.com/yorkerlin/KL-Methods",
        pin="a02e622",
        license="none stated; dev-time oracle use only",
    ),
    "psgd_lra": Baseline(
        "psgd_lra",
        "psgd_torch",
        "preconditioned_stochastic_gradient_descent.py",
        "LRA",
        kwargs=(
            ("lr", "lr_params"),
            ("beta1", "momentum"),
            ("rank", "rank_of_approximation"),
            ("precond_init_scale", "preconditioner_init_scale"),
            ("precond_lr", "lr_preconditioner"),
        ),
        constants={"preconditioner_type": "whitening", "exact_hessian_vector_product": False},
        closure=True,
        url="https://github.com/lixilinx/psgd_torch",
        pin="c86b1cb",
        license="none stated; dev-time oracle use only",
    ),
    "xmat": Baseline(
        "xmat",
        "psgd_torch",
        "preconditioned_stochastic_gradient_descent.py",
        "XMat",
        kwargs=(
            ("lr", "lr_params"),
            ("beta1", "momentum"),
            ("precond_init_scale", "preconditioner_init_scale"),
            ("precond_lr", "lr_preconditioner"),
        ),
        constants={"preconditioner_type": "whitening", "exact_hessian_vector_product": False},
        closure=True,
        url="https://github.com/lixilinx/psgd_torch",
        pin="c86b1cb",
        license="none stated; dev-time oracle use only",
        note="XMat applies Q = sqrt(P) directly: QSGD's lineage",
    ),
    "orthograd_laprop": Baseline(
        "orthograd_laprop",
        "orthograd",
        "orthograd.py",
        "OrthoGrad",
        wrap_base="laprop",  # hyperparameters flow through the wrapped baseline's mapping
        url="https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability",
        pin="720d244",
        license="MIT",
        note="official OrthoGrad wrapping official LaProp; hyper flow into the base",
    ),
    "adamc": Baseline(
        "adamc",
        "pytorch_optimizer",
        "pytorch_optimizer.optimizer.adamc",
        "AdamC",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
            ("max_lr", "max_lr"),
        ),
        url="https://github.com/kozistr/pytorch_optimizer",
        pin="b452a0f",
        license="Apache-2.0",
    ),
    "muon_aux": Baseline(
        "muon_aux",
        "Muon",
        "muon.py",
        "MuonWithAuxAdam",
        groups="muon_aux",
        url="https://github.com/KellerJordan/Muon",
        pin="f98f1ca",
        license="MIT",
        note="official MuonWithAuxAdam: muon group for >=2D, aux Adam for the rest; its step issues collectives, so the single-process replay records the PG error -- a distributed-cell-only upstream",
    ),
    "msam": Baseline(
        "msam",
        "msam",
        "optimizer/adamW_msam.py",
        "AdamW_MSAM",
        kwargs=(
            ("lr", "lr"),
            ("beta1", "betas"),
            ("beta2", "betas"),
            ("eps", "eps"),
            ("weight_decay", "weight_decay"),
        ),
        url="https://github.com/MarlonBecker/MSAM",
        pin="780fa4f",
        license="MIT",
    ),
}


def _load_baseline(baseline: Baseline):
    """Import the upstream class from its pinned clone; the tree is dev-time only.
    ``module`` is a dotted import inside the repo root, or a .py file path when
    the repo has no importable package layout."""
    if baseline.repo == "torch":
        return getattr(importlib.import_module(baseline.module), baseline.cls)
    if baseline.module.endswith(".py"):
        path = REFERENCES / baseline.repo / baseline.module
        package = f"matrix_baseline_{baseline.repo.replace('-', '_')}"
        if package not in sys.modules:  # a synthetic package resolves the file's relative imports
            holder = types.ModuleType(package)
            holder.__path__ = [str(path.parent)]
            sys.modules[package] = holder
        spec = importlib.util.spec_from_file_location(f"{package}.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return getattr(module, baseline.cls)
    root = str(REFERENCES / baseline.repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    return getattr(importlib.import_module(baseline.module), baseline.cls)


def _validate_baseline_source(baseline: Baseline) -> None:
    if baseline.repo == "torch":
        return
    target = REFERENCES / baseline.repo
    actual = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = subprocess.run(
        ["git", "-C", str(target), "rev-parse", f"{baseline.pin}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{baseline.repo} is not at pinned revision {baseline.pin}")
    if baseline.wrap_base:
        _validate_baseline_source(BASELINES[baseline.wrap_base])


def _baseline_kwargs(baseline: Baseline, hyper: dict) -> dict:
    if baseline.wrap_base:
        base = BASELINES[baseline.wrap_base]
        return {"base_optimizer_cls": functools.partial(_load_baseline(base), **_baseline_kwargs(base, hyper))}
    if baseline.groups == "muon_aux":
        return {}  # MuonWithAuxAdam reads everything from its param groups
    kw = dict(baseline.constants)
    betas = tuple(hyper[f"beta{i}"] for i in (1, 2, 3) if f"beta{i}" in hyper)
    for source, target in baseline.kwargs:
        if target == "betas":
            kw["betas"] = betas
        elif source in hyper and hyper[source] is not None:
            # None is our "automatic" marker; the upstream default is its own
            # resolution of that knob, so it stays untouched.
            kw[target] = hyper[source]
    return kw


def _owned_indices(baseline: Baseline, init: list) -> list[int]:
    if baseline.subset == "ndim2":
        return [index for index, leaf in enumerate(init) if leaf.ndim >= 2]
    return list(range(len(init)))


def _apply_grads(make_optimizer, init: list, grads: list, dtype, *, prepare=(), trace=False, loss_fn=None):
    """The one primitive under every gradient replay (upstream and our fp64
    self): apply recorded gradients with an eagerly-built optimizer. Reference
    math, not a timed leg, so it runs on the GPU when one is present. Returns
    (finals, trace, error) -- the error names the exception when the run cannot
    happen; that is a datum, not a dropped cell."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        params = [nn.Parameter(value.detach().clone().to(device=device, dtype=dtype)) for value in init]
        trajectory, losses = [], []
        with contextlib.redirect_stdout(io.StringIO()):  # kron_torch prints buffer sizes
            optimizer = make_optimizer(params)
            for method in prepare:
                getattr(optimizer, method)()
            for step, step_grads in enumerate(grads):
                optimizer.zero_grad()
                for param, grad in zip(params, step_grads):
                    g = grad.detach().to(device=param.device, dtype=param.dtype)
                    if param.grad is None:
                        param.grad = g.clone()
                    else:
                        param.grad.copy_(g)  # slab-bound: never rebind after the first step
                optimizer.step()
                if trace:
                    trajectory.append(torch.cat([p.detach().reshape(-1).double().cpu() for p in params]))
                if loss_fn is not None:
                    losses.append(loss_fn(step, params))
        return [p.detach().double().cpu() for p in params], (trajectory, losses) if trace or loss_fn else None, ""
    except Exception as error:  # noqa: BLE001 -- the cause is the datum
        return None, None, f"{type(error).__name__}: {str(error)[:200]}"


def _closure_replay(baseline: Baseline, hyper: dict, shapes: dict, init: list, batches, dtype):
    """Closure-driven upstreams (the original psgd_torch classes) evaluate the
    loss themselves; the oracle runs its own forward over the same model and
    batches -- the same standing the fp64 self-oracle has."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        model = _build_model(shapes).to(device=device, dtype=dtype)
        with torch.no_grad():
            for parameter, value in zip(model.parameters(), init, strict=True):
                parameter.copy_(value.to(device=device, dtype=dtype))
        owned = _owned_indices(baseline, init)
        params = [parameter for index, parameter in enumerate(model.parameters()) if index in owned]
        optimizer = _load_baseline(baseline)(params, **_baseline_kwargs(baseline, hyper))
        state = {"step": 0}

        def closure():
            for parameter in params:  # psgd_torch classes are not torch.optim: no zero_grad
                parameter.grad = None
            idx, target = batches[state["step"]]
            return _forward_loss(model, idx.to(device), target.to(device), 1 / idx.shape[0])

        with contextlib.redirect_stdout(io.StringIO()):
            for _ in batches:
                optimizer.step(closure)
                state["step"] += 1
        return [value.detach().double().cpu() for value in model.parameters()]
    except Exception:  # noqa: BLE001 -- the cause is the datum
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None


def _muon_aux_groups(params: list, hyper: dict, owned: list[int], init: list) -> list:
    """MuonWithAuxAdam's exact group contract: the muon group carries only
    (params, lr, momentum, weight_decay, use_muon); the aux group is plain Adam."""
    muon_params = [p for p, index in zip(params, owned) if init[index].ndim >= 2]
    aux_params = [p for p, index in zip(params, owned) if init[index].ndim < 2]
    return [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": hyper.get("lr", 0.02),
            "momentum": hyper.get("beta1", 0.95),
            "weight_decay": hyper.get("weight_decay", 0.0),
        },
        {
            "params": aux_params,
            "use_muon": False,
            "lr": hyper.get("lr", 0.02),
            "betas": (hyper.get("beta1", 0.9), hyper.get("beta2", 0.999)),
            "eps": hyper.get("eps", 1e-8),
            "weight_decay": hyper.get("weight_decay", 0.0),
        },
    ]


def _replay(baseline: Baseline, hyper: dict, init: list, grads: list, storage, compute=None, trace=False, loss_fn=None):
    """The upstream optimizer over the recorded gradients, as-is: fp64/fp64 is
    the oracle, ``storage`` alone the naive same-math baseline, ``compute`` the
    baseline's own low-precision-storage/fp32-compute knobs where it has them.
    Returns (owned finals, owned indices, trace, error)."""
    owned = _owned_indices(baseline, init)
    kw = _baseline_kwargs(baseline, hyper)
    if compute is not None:
        for knob in baseline.promoted:
            kw[knob] = compute
    if baseline.groups == "muon_aux":
        make = lambda params: _load_baseline(baseline)(_muon_aux_groups(params, hyper, owned, init))  # noqa: E731
    else:
        make = lambda params: _load_baseline(baseline)(params, **kw)  # noqa: E731
    sliced = [[step[index] for index in owned] for step in grads]
    finals, trace_data, error = _apply_grads(
        make, [init[index] for index in owned], sliced, storage, prepare=baseline.prepare, trace=trace, loss_fn=loss_fn
    )
    return finals, owned, trace_data, error


@dataclass(frozen=True)
class Distances:
    """Distances to the upstream fp64 oracle: ours, the upstream at the cell
    precision, our fp64 self (the floor set by deliberate divergence), plus
    per-step traces. ``err`` names the run that could not happen."""

    ours: float | None = None
    ref: float | None = None
    semantic: float | None = None
    ref_promoted: float | None = None
    oracle_dtype: str = ""
    traj_ours: list | None = None
    traj_ref: list | None = None
    loss_ref: list | None = None
    loss_naive: list | None = None
    err: str = ""


@dataclass(frozen=True)
class AccuracyLeg:
    """Everything the accuracy comparison consumes, bound once: the upstream
    baseline, the drawn hyperparameters, the shared problem (init, recorded
    gradients, batches, storage dtype) and the cell's own finals."""

    baseline: Baseline
    hyper: dict
    init: list
    grads: list
    fulls: list
    storage: object
    shapes: dict = None
    batches: list = None

    def run(self, ours_fp64, ours_traj, loss_fn) -> Distances:
        if self.baseline.closure:
            return self._run_closure(ours_fp64)
        # fp64 where the upstream is fp64-clean, fp32 otherwise (recorded):
        # both sides are measured against the same run.
        oracle, owned, oracle_trace, replay_error = None, None, None, ""
        for dtype in (torch.float64, torch.float32):
            oracle, owned, oracle_trace, replay_error = _replay(
                self.baseline,
                self.hyper,
                self.init,
                self.grads,
                dtype,
                trace=ours_traj is not None,
                loss_fn=loss_fn,
            )
            if oracle is not None:
                oracle_dtype = str(dtype).removeprefix("torch.")
                break
        if oracle is None:
            return Distances(err=f"upstream cannot run at fp64 or fp32: {replay_error}")
        oracle_flat = torch.cat([value.reshape(-1) for value in oracle])
        scale = oracle_flat.norm().clamp_min(1e-12)

        def distance(values):
            return ((values - oracle_flat).norm() / scale).item()

        def flat_owned(step_params):
            return torch.cat([step_params[index].reshape(-1) for index in owned]).double()

        def trajectory(steps):
            return [((step - oracle_step).norm() / scale).item() for step, oracle_step in zip(steps, oracle_trace[0])]

        out = Distances(ours=distance(flat_owned(self.fulls)), oracle_dtype=oracle_dtype)
        if ours_fp64 is not None:
            out = replace(out, semantic=distance(flat_owned(ours_fp64)))
        if ours_traj is not None:
            out = replace(out, traj_ours=trajectory([flat_owned(step) for step in ours_traj]), loss_ref=oracle_trace[1])
        naive, _, naive_trace, replay_error = _replay(
            self.baseline, self.hyper, self.init, self.grads, self.storage, trace=True, loss_fn=loss_fn
        )
        if naive is None:
            return replace(out, err=f"upstream cannot run at {self.storage}: {replay_error}")
        out = replace(out, ref=distance(torch.cat([value.reshape(-1) for value in naive])))
        if naive_trace is not None:
            out = replace(
                out,
                traj_ref=trajectory(naive_trace[0]) if oracle_trace is not None else None,
                loss_naive=naive_trace[1],
            )
        if self.storage not in (torch.float32, torch.float64) and self.baseline.promoted:
            promoted, _, _, _ = _replay(
                self.baseline, self.hyper, self.init, self.grads, self.storage, compute=torch.float32
            )
            if promoted is not None:
                out = replace(out, ref_promoted=distance(torch.cat([value.reshape(-1) for value in promoted])))
        return out

    def _run_closure(self, ours_fp64) -> Distances:
        """Oracle, naive and our fp64 leg all run their own forward: the
        closure upstreams would have it no other way."""
        oracle = None
        for dtype in (torch.float64, torch.float32):
            oracle = _closure_replay(self.baseline, self.hyper, self.shapes, self.init, self.batches, dtype)
            if oracle is not None:
                oracle_dtype = str(dtype).removeprefix("torch.")
                break
        if oracle is None:
            return Distances(err="closure upstream cannot run at fp64 or fp32")
        oracle_flat = torch.cat([value.reshape(-1) for value in oracle]).double()
        scale = oracle_flat.norm().clamp_min(1e-12)
        flat = lambda finals: torch.cat([value.reshape(-1) for value in finals]).double()  # noqa: E731
        out = Distances(ours=((flat(self.fulls) - oracle_flat).norm() / scale).item(), oracle_dtype=oracle_dtype)
        if ours_fp64 is not None:
            out = replace(out, semantic=((flat(ours_fp64) - oracle_flat).norm() / scale).item())
        naive = _closure_replay(self.baseline, self.hyper, self.shapes, self.init, self.batches, self.storage)
        if naive is None:
            return replace(out, err=f"closure upstream cannot run at {self.storage}")
        return replace(out, ref=((flat(naive) - oracle_flat).norm() / scale).item())


def _profile_step(step_fn) -> dict:
    """One warm step under the profiler: device-op count and device time.

    The op count is the mechanism behind step-time differences -- a fused graph
    wins by issuing fewer device ops, and loses when its fixed cost dominates on
    small problems -- so it belongs beside the timing it explains."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step_fn()
        torch.cuda.synchronize()
    device_ops, device_us = 0, 0.0
    for event in prof.key_averages():
        if str(getattr(event, "device_type", "")).endswith("CUDA"):
            device_ops += event.count
            device_us += getattr(event, "self_device_time_total", 0.0) or 0.0
    return {"device_ops": device_ops, "device_ms": round(device_us / 1000, 4)}


def _baseline_metrics(baseline: Baseline, shapes: dict, hyper: dict, init: list, batches, device, dtype) -> dict:
    """Time and profile the upstream optimizer on the same problem in this process."""
    cls = _load_baseline(baseline)
    model = _build_model(shapes).to(device, dtype=dtype)
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), init, strict=True):
            parameter.copy_(value.to(device, dtype=dtype))
    owned = _owned_indices(baseline, init)
    params = [parameter for index, parameter in enumerate(model.parameters()) if index in owned]
    optimizer = cls(params, **_baseline_kwargs(baseline, hyper))
    for method in baseline.prepare:
        getattr(optimizer, method)()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for idx, target in batches:
        optimizer.zero_grad()
        _forward_loss(model, idx.to(device), target.to(device), 1 / idx.shape[0]).backward()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    metrics = _profile_step(optimizer.step)
    metrics["peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
    timed = timings[TIMED_FROM - 1 :]
    metrics["step_ms"] = round(sum(timed) / max(len(timed), 1), 3)
    return metrics


# ---------------------------------------------------------------------------
# The tested compositions: chainable functions and their shipped compositions.
# ``chain`` is the tuple of ``heavyball.chainable`` functions (routes included);
# ``baseline`` names the upstream oracle in BASELINES, or None (self-oracle gate);
# ``facade`` names the class whose signature supplies the hyperparameter schema;
# ``fixed`` are hyperparameters the composition pins (they have no upstream knob);
# ``clip`` is the update clipping the shipped composition carries (PSGD family);
# ``precond`` is how the composition wires its preconditioner-update schedule.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Composition:
    name: str
    chain: tuple
    baseline: str | None = None
    facade: str | None = None
    fixed: dict = field(default_factory=dict)
    clip: str | None = None
    precond: str | None = None  # "frequency" (1/precondition_frequency) or "default" (annealed schedule)


def _route_hyperball():
    return C.route((lambda p: p.ndim >= 2, C.update_by_hyperball), default=C.apply_update)


def _route_muon():
    return C.route((lambda p: p.ndim >= 2, (C.nesterov_ema, C.orthogonalize_update)), default=C.scale_by_adam)


def _psgd(fn):
    return functools.partial(fn, cached=True)


COMPOSITIONS = [
    Composition("adam", (C.update_by_adam,), baseline="adamw", facade="AdamW"),
    Composition("sgd", (C.heavyball_momentum,), baseline="sgd", facade="SGD", fixed={"beta": 0.0}),
    Composition("laprop", (C.update_by_laprop,), baseline="laprop", facade="LaProp"),
    Composition("rmsprop", (C.scale_by_exp_avg_sq,), baseline="rmsprop", facade="RMSprop"),
    Composition("nadam", (C.update_by_nadam,), baseline="nadam", facade="NAdam"),
    Composition("ademamix", (C.update_by_ademamix,), baseline="ademamix", facade="AdEMAMix"),
    Composition("unscaled_adam", (C.scale_by_unscaled_adam,), facade="UnscaledAdamW"),
    Composition("adopt", (C.update_by_adopt,), baseline="adopt", facade="ADOPT"),
    Composition("mars_adam", (C.mars, C.update_by_adam), baseline="mars", facade="AdamW"),
    Composition(
        "cautious_adam",
        (C.update_by_adam,),
        baseline="cadamw",
        facade="AdamW",
        fixed={"caution": True},
    ),
    Composition(
        "schedule_free", (C.scale_by_exp_avg_sq, C.update_by_schedule_free), baseline="schedule_free", facade="SFAdamW"
    ),
    Composition("hyperball", (C.scale_by_exp_avg_sq, _route_hyperball()), facade="HyperBallAdamW"),
    Composition(
        "ortho_laprop",
        (C.orthogonalize_grad_to_param, C.update_by_laprop),
        baseline="orthograd_laprop",
        facade="OrthoLaProp",
    ),
    Composition("laprop_ortho", (C.update_by_laprop, C.orthogonalize_grad_to_param), facade="LaPropOrtho"),
    Composition("sign_laprop", (C.update_by_laprop, C.sign), facade="SignLaProp"),
    Composition("soap", (C.scale_by_soap,), baseline="soap", facade="SOAP", precond="frequency"),
    Composition(
        "kl_soap",
        (C.scale_by_kl_soap,),
        baseline="kl_soap_official",
        facade="KLSOAP",
        precond="frequency",
    ),
    Composition(
        "shampoo",
        (C.scale_by_kl_shampoo,),
        baseline="kl_shampoo",
        facade="KLShampoo",
        precond="frequency",
    ),
    Composition("soap_nadam", (C.scale_by_soap_nadam,), facade="SOAPNAdam", precond="frequency"),
    Composition("soap_ademamix", (C.scale_by_soap_ademamix,), facade="SOAPAdEMAMix", precond="frequency"),
    Composition("soap_laprop", (C.scale_by_soap_laprop,), facade="SOLP", precond="frequency"),
    Composition("heavy_soap", (C.scale_by_heavy_soap,), facade="HeavySOAP", precond="frequency"),
    Composition("heavy_kl_soap", (C.scale_by_heavy_kl_soap,), facade="HeavyKLSOAP", precond="frequency"),
    Composition("heavy_shampoo", (C.scale_by_heavy_kl_shampoo,), facade="HeavyKLShampoo", precond="frequency"),
    Composition("heavy_soap_nadam", (C.scale_by_heavy_soap_nadam,), facade="HeavySOAPNAdam", precond="frequency"),
    Composition(
        "heavy_soap_ademamix", (C.scale_by_heavy_soap_ademamix,), facade="HeavySOAPAdEMAMix", precond="frequency"
    ),
    Composition("heavy_soap_laprop", (C.scale_by_heavy_soap_laprop,), facade="HeavySOLP", precond="frequency"),
    Composition(
        "psgd",
        (_psgd(C.scale_by_psgd),),
        baseline="psgd_kron",
        facade="PSGDKron",
        clip="trust_region_clip_",
        precond="default",
    ),
    Composition(
        "psgd_lra",
        (C.scale_by_psgd_lra,),
        baseline="psgd_lra",
        facade="PSGDLRA",
        clip="trust_region_clip_",
        precond="default",
    ),
    Composition(
        "psgd_pro",
        (_psgd(C.scale_by_psgd_pro),),
        baseline="psgd_kron",
        facade="PSGDPRO",
        clip="trust_region_clip_",
        precond="default",
        fixed={"store_triu_as_line": False},
    ),
    Composition(
        "qsgd",
        (_psgd(C.scale_by_psgd_pro),),
        baseline="xmat",
        facade="QSGD",
        clip="trust_region_clip_",
        precond="default",
        fixed={"store_triu_as_line": False, "sqrt": True},
    ),
    Composition("lather", (C.scale_by_lather,), facade="LATHER", clip="trust_region_clip_", precond="default"),
    Composition("suds", (C.scale_by_suds,), facade="SUDSAdamW"),
    Composition("scion", (C.exp_avg, C.scion_auto_norm), baseline="scion", facade="Scion"),
    Composition("muon", (C.nesterov_ema, C.orthogonalize_update), baseline="muon", facade="Muon"),
    Composition("muon_laprop", (C.scale_by_laprop, C.orthogonalize_update), facade="MuonLaProp"),
    Composition("muon_adamw", (_route_muon(),), baseline="muon_aux", facade="MuonAdamW"),
    Composition(
        "msam",
        (C.scale_by_exp_avg_sq, C.update_by_msam),
        baseline="msam",
        facade="MSAMLaProp",
    ),
    Composition("adamc", (C.update_by_adamc,), baseline="adamc", facade="AdamC"),
]
COMPOSITIONS = {entry.name: entry for entry in COMPOSITIONS}

# ---------------------------------------------------------------------------
# The composition tier: individual chainable functions and their ordered pairs,
# not shipped optimizers. The tested unit is the chain itself; the product is
# that these compose, so the tier enumerates singles and pairs over the atom
# set (transforms that take the standard state/update/grad/param signature and
# construct with no arguments) and judges each against three laws:
#   fusion    -- the compiled chain at the cell's precision sits no farther
#                from the eager chain at that precision than the dtype itself
#                costs (the eager-vs-eager-fp64 gap): compilation may not
#                change the math beyond the precision's own rounding;
#   precision -- the compiled chain is no farther from the eager fp64 chain
#                than the eager cell-precision chain is;
#   fullgraph -- the whole chain compiles as one graph (the standing assert).
# Hyperparameters are the unjittered union of every facade's surface, so any
# atom can index any group key it reads; the problem draw is the seeded one.
# ---------------------------------------------------------------------------

_ATOMS: tuple[str, ...] = (
    "identity",
    "apply_update",
    "weight_decay_to_init",
    "weight_decay_to_ema",
    "l1_weight_decay_to_ema",
    "exp_avg",
    "nesterov_momentum",
    "nesterov_ema",
    "heavyball_momentum",
    "scale_by_exp_avg_sq",
    "scale_by_adam",
    "update_by_adam",
    "scale_by_nadam",
    "update_by_nadam",
    "update_by_adamc",
    "scale_by_ademamix",
    "update_by_ademamix",
    "scale_by_laprop",
    "update_by_laprop",
    "scale_by_unscaled_adam",
    "update_by_adopt",
    "scale_by_adopt",
    "update_by_schedule_free",
    "update_by_msam",
    "orthogonalize_grad_to_param",
    "orthogonalize_update",
    "sign",
    "mars",
    "palm_beta2",
    "mup_approx",
    "scale_by_d_adaptation",
    "scale_by_lr_adaptation",
    "scale_by_pointwise_lr_adaptation",
    "scion_auto_norm",
)


def _composition_for(name: str) -> Composition:
    """The shipped composition by name, or the synthesized single/pair chain
    ``chain:a`` / ``chain:a+b`` of the composition tier."""
    if not name.startswith("chain:"):
        return COMPOSITIONS[name]
    parts = tuple(getattr(C, atom) for atom in name.removeprefix("chain:").split("+"))
    return Composition(name, parts)


_SUPERSET: dict | None = None


def _hyper_superset() -> dict:
    """Every facade's surface in one group: pair cells run law mode -- fixed
    hyperparameters, no jitter -- so any atom can index any key it reads."""
    global _SUPERSET
    if _SUPERSET is None:
        merged: dict = {}
        for entry in COMPOSITIONS.values():
            for name, default in _schema(entry):
                merged.setdefault(name, default)
        if any(f"beta{i}" in merged for i in (1, 2, 3)):
            merged["betas"] = tuple(merged[f"beta{i}"] for i in (1, 2, 3) if f"beta{i}" in merged)
        _SUPERSET = merged
    return dict(_SUPERSET)


# ---------------------------------------------------------------------------
# The problem draw
# ---------------------------------------------------------------------------

PRECISIONS = {
    "fp64": {"model": torch.float64, "optim": {"storage_dtype": "float64"}},
    "fp32": {"model": torch.float32, "optim": {}},
    "bf16": {"model": torch.bfloat16, "optim": {"storage_dtype": "bfloat16"}},
    "fp16": {"model": torch.float16, "optim": {"storage_dtype": "float16"}},
    "ecc16": {"model": torch.bfloat16, "optim": {"ecc": "bf16+16"}},
    "ecc16f": {"model": torch.float16, "optim": {"ecc": "fp16+16"}},
}
STEPS = 64  # per-step logging makes shorter horizons prefixes of this run
BATCH = 16
TIMED_FROM = 3
PASS_MARGIN = 1
INIT_SEED = 7
PORT_BASE = 29500
_TRAIN_T0 = 0.0
_WORKER_T0 = 0.0
SCHEMA_VERSION = 2


def _device_count() -> int:
    """Visible GPUs; lanes may exceed it -- accuracy passes multiplex several
    lanes per device (compile is CPU-bound, the step is microseconds), while
    timing passes keep one lane per device for exclusivity."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


# Scale multiplies the drawn shapes, capped for a 10 GB card; seq and out set
# the data pattern, not the parameter count.
_SCALE_CAPS = {"vocab": 50000, "embed": 1024, "hidden": 2048}

_SCHEMA_SKIP = frozenset(
    {
        "params",
        "storage_dtype",
        "ecc",
        "param_ecc",
        "orig_shapes",
        "gradient_clipping",
        "update_clipping",
        "palm",
        "compile_step",
        "promote",
        "multi_tensor",
        "precond_scheduler",  # only meaningful with use_precond_schedule=True (default False)
    }
)


def _draw_seed(master: int, name: str, precision: str) -> int:
    digest = hashlib.blake2b(f"{master}|{name}|{precision}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _schema(entry: Composition) -> list[tuple[str, object]]:
    """The drawn hyperparameters: the facade's full surface. The group dict the
    chain reads must carry every key the chain indexes -- structural ints and
    bools ride along at their defaults, unjittered -- and the baseline mapping
    picks out by name what the upstream constructor consumes. ``betas`` tuples
    expand to beta1..N so the coefficients are jittered; the group conversion
    happens at build."""
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    out = []
    for name, parameter in inspect.signature(getattr(heavyball, entry.facade)).parameters.items():
        if name in _SCHEMA_SKIP or parameter.kind not in kinds or parameter.default is use_default:
            continue
        if name == "betas":
            out.extend((f"beta{i}", beta) for i, beta in enumerate(parameter.default, start=1))
        else:
            out.append((name, parameter.default))
    # AdamC's max_lr defaults to None, which the facade resolves to the learning
    # rate; draw it in the lr style so the decay scaling is exercised.
    if any(name == "max_lr" and default is None for name, default in out):
        lr = next(default for name, default in out if name == "lr")
        out = [(name, lr if name == "max_lr" else default) for name, default in out]
    return out


def _bounded_coefficient(name: str) -> bool:
    # beta1/beta2/beta3 start with "beta" but do not end with it; shampoo_beta and
    # lower_bound_beta end with it; probabilities end with "probability". All are
    # engine-domain-checked to stay within [0, 1].
    return name.startswith("beta") or name.endswith("beta") or name.endswith("probability")


def _jitter(generator, name: str, default):
    # Only continuous knobs are drawn: int and bool hyperparameters are structural,
    # except precondition_frequency, which must land inside the run so the
    # preconditioner actually fires (a frequency above STEPS never binds).
    if name == "precondition_frequency" and isinstance(default, int):
        return int(torch.randint(1, STEPS + 1, (1,), generator=generator))
    if name == "rank" and default is None:
        # PSGDLRA's facade auto-sets rank from the parameter count; the direct
        # construction must draw it so the low-rank approximation is exercised.
        return int(torch.randint(1, 9, (1,), generator=generator))
    if not isinstance(default, float):
        return default
    if name in ("lr", "max_lr") and default > 0:
        factor = float(torch.exp(torch.normal(0.0, 0.4, (1,), generator=generator)))
        return default * min(max(factor, 0.25), 4.0)
    if name == "weight_decay" and default == 0.0:
        # Decoupled decay is drawn in even when the default is zero so the decay
        # path is exercised across the matrix.
        drawn = float(torch.rand(1, generator=generator))
        return float(10.0 ** (drawn * 2.0 - 3.0))
    if _bounded_coefficient(name) and default > 0.0:
        drawn = default + float(torch.rand(1, generator=generator)) * 0.2 - 0.1
        return min(max(drawn, 0.01), 0.99)
    if default > 0:
        factor = float(torch.exp(torch.normal(0.0, 0.3, (1,), generator=generator)))
        return default * min(max(factor, 0.4), 2.5)
    return default


def _draw_problem(seed: int, entry: Composition, scale: int = 1):
    """Shapes and hyperparameters, drawn on CPU so every rank and the oracle
    agree. The same seed at every scale draws the same problem multiplied by the
    scale, so the size axis isolates size: hyperparameters, data pattern and the
    draw itself do not move between scales."""
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "vocab": int(torch.randint(50, 200, (1,), generator=generator)),
        "embed": int(torch.randint(8, 33, (1,), generator=generator)),
        "hidden": int(torch.randint(8, 49, (1,), generator=generator)),
        "seq": int(torch.randint(3, 9, (1,), generator=generator)),
        "out": int(torch.randint(2, 6, (1,), generator=generator)),
    }
    shapes = {name: min(value * scale, _SCALE_CAPS.get(name, value)) for name, value in shapes.items()}
    if entry.facade is None:  # composition tier: law mode, no jitter
        return shapes, _hyper_superset()
    hyper = {name: _jitter(generator, name, default) for name, default in _schema(entry)}
    return shapes, hyper


def _build_model(shapes: dict):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(shapes["vocab"], shapes["embed"])
            self.conv = nn.Conv1d(shapes["embed"], shapes["hidden"], 3)
            self.norm = nn.LayerNorm(shapes["hidden"])
            self.lin = nn.Linear(shapes["hidden"], shapes["hidden"])
            self.head = nn.Linear(shapes["hidden"], shapes["out"])

        def forward(self, idx):
            tokens = self.emb(idx).transpose(1, 2)
            pooled = self.conv(tokens).mean(dim=2)
            return self.head(torch.tanh(self.lin(self.norm(pooled))))

    return Net()


def _initial_parameters(shapes: dict) -> list:
    """CPU-deterministic init, identical on every rank and in the upstream oracle."""
    torch.manual_seed(INIT_SEED)
    model = _build_model(shapes)
    return [value.detach().clone() for value in model.parameters()]


def _batch(shapes: dict, seed: int, batch: int):
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, shapes["vocab"], (batch, shapes["seq"]), generator=generator)
    target = torch.randn(batch, shapes["out"], generator=generator)
    return idx, target


def _forward_loss(model, idx, target, scale: float):
    """Per-rank loss under the standard data-parallel convention.

    DDP/FSDP average gradients over the data-parallel ranks, so a rank that
    scales its local sample sum by dp_world/global_batch lands the exact
    single-process global-batch gradient.
    """
    return ((model(idx) - target) ** 2).sum() * scale


def _build(entry: Composition, params, hyper: dict, precision: str, *, compile_step: bool = True):
    """Construct the composition as an optimizer directly from the chain; the
    facade never appears past the hyperparameter schema."""
    defaults = dict(hyper)
    defaults.update(entry.fixed)
    defaults["compile_step"] = compile_step
    defaults.update(PRECISIONS[precision]["optim"])
    if any(f"beta{i}" in defaults for i in (1, 2, 3)):
        defaults["betas"] = tuple(defaults.pop(f"beta{i}") for i in (1, 2, 3) if f"beta{i}" in defaults)
    optimizer = C.BaseOpt(params, defaults, update_clipping=entry.clip, fns=entry.chain)
    if entry.precond == "frequency":
        optimizer.precond_schedule = 1.0 / defaults["precondition_frequency"]
    elif entry.precond == "default":
        optimizer.precond_schedule = utils.precond_update_prob_schedule()
    return optimizer


def _hsdp_mesh(world: int) -> tuple[int, int] | None:
    replicate = next((r for r in range(2, world + 1) if world % r == 0), None)
    if replicate is None or replicate >= world:
        return None
    return replicate, world // replicate


def enumerate_cells(
    max_world: int, names: list[str], precisions: list[str], min_world: int = 1, scales: list[int] = ()
) -> list[dict]:
    cells = []
    for scale in scales or [1]:
        for name in names:
            for precision in precisions:
                if min_world <= 1:
                    cells.append(
                        {"topology": "default", "world": 1, "name": name, "precision": precision, "scale": scale}
                    )
                for world in range(max(2, min_world), max_world + 1):
                    cells.append(
                        {"topology": "ddp", "world": world, "name": name, "precision": precision, "scale": scale}
                    )
                    cells.append(
                        {"topology": "fsdp2", "world": world, "name": name, "precision": precision, "scale": scale}
                    )
                    if _hsdp_mesh(world) is not None:
                        cells.append(
                            {"topology": "hsdp", "world": world, "name": name, "precision": precision, "scale": scale}
                        )
    return cells


def _cell_key(cell: dict) -> str:
    return f"{cell['name']}/{cell['precision']}/{cell['topology']}/{cell['world']}/{cell.get('scale', 1)}"


def _cell_id(cell: dict) -> str:
    normalized = dict(cell)
    normalized.setdefault("scale", 1)
    return json.dumps(normalized, sort_keys=True)


def _git_short(revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", revision],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _matrix_meta(seed: int) -> dict:
    source_state = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT)
    if source_state.returncode:
        raise RuntimeError("matrix measurements require a clean tracked worktree")
    return {
        "schema_version": SCHEMA_VERSION,
        "steps": STEPS,
        "seed": seed,
        "product_sha": _git_short("HEAD"),
        "references": {baseline.repo: baseline.pin for baseline in BASELINES.values() if baseline.repo != "torch"},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compile": f"torch.compile fullgraph compile_step=True mode={utils.compile_mode}",
        "opt_einsum": str(getattr(opt_einsum, "strategy", None)),
        "fx_graph_cache": bool(torch._inductor.config.fx_graph_cache),
    }


def _meta_path(measure: Path) -> Path:
    return measure.with_suffix(".meta.json")


def _validate_meta(measure: Path, seed: int) -> None:
    meta_path = _meta_path(measure)
    if not meta_path.exists():
        raise ValueError(f"cannot carry {measure}: missing {meta_path.name}")
    actual = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _matrix_meta(seed)
    keys = (
        "schema_version",
        "steps",
        "seed",
        "product_sha",
        "references",
        "device",
        "torch",
        "cuda",
        "compile",
        "opt_einsum",
        "fx_graph_cache",
    )
    mismatches = [name for name in keys if actual.get(name) != expected[name]]
    if mismatches:
        raise ValueError(f"cannot carry {measure}: metadata mismatch in {', '.join(mismatches)}")


def _prepare_measure(measure: Path, seed: int) -> None:
    meta_path = _meta_path(measure)
    if measure.exists() and measure.stat().st_size:
        _validate_meta(measure, seed)
        return
    meta_path.write_text(json.dumps(_matrix_meta(seed), sort_keys=True, indent=1) + "\n", encoding="utf-8")


def _read_rows(paths: list[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        meta_path = _meta_path(path)
        if not meta_path.exists():
            raise ValueError(f"cannot read {path}: missing {meta_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION or meta.get("steps") != STEPS:
            raise ValueError(f"cannot read {path}: incompatible metadata schema")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("schema_version") != SCHEMA_VERSION or row.get("steps") != STEPS:
                    raise ValueError(f"cannot carry {path}: incompatible row schema")
                merged[_cell_id(row["cell"])] = row
    return list(merged.values())


def _carried_rows(paths: list[Path], seed: int) -> list[dict]:
    """Read resumable rows after proving that every source matches this run."""
    for path in paths:
        if path.exists():
            _validate_meta(path, seed)
    return _read_rows(paths)


def _finite_json(value, path: str, nonfinite: list[str]):
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(path)
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item, f"{path}.{key}" if path else key, nonfinite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, f"{path}[{index}]", nonfinite) for index, item in enumerate(value)]
    return value


def _normalize_record(row: dict, seed: int | None = None) -> dict:
    normalized = dict(row)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["steps"] = STEPS
    if seed is not None:
        normalized["cell_seed"] = seed
    old_nonfinite = list(normalized.pop("nonfinite_fields", ()))
    found: list[str] = []
    normalized = _finite_json(normalized, "", found)
    if old_nonfinite or found:
        normalized["nonfinite_fields"] = sorted(set(old_nonfinite + found))
    return normalized


def _json_row(row: dict, seed: int | None = None) -> str:
    return json.dumps(_normalize_record(row, seed), sort_keys=True, default=str, allow_nan=False)


def _emit(prefix: str, payload: dict, seed: int | None = None) -> None:
    print(f"{prefix} {_json_row(payload, seed)}", flush=True)


# ---------------------------------------------------------------------------
# The cell run
# ---------------------------------------------------------------------------


def _oracle(entry: Composition, shapes: dict, hyper: dict, init, batches, device, precision: str = "fp64"):
    """Single-process run of the same composition on the same draw.

    fp64 is the truth where the chain supports it; KL-Shampoo, PSGD and LATHER
    fix their preconditioner matrices at q_dtype=float32, so their einsums reject
    an fp64 gradient. Those cells fall back to an fp32 oracle (recorded) -- the
    self-oracle gates compare cells against the same oracle, so the verdict stays
    relative. Baseline cells are unaffected: their oracle is the upstream run.
    The composition tier also asks for the eager run at the cell's own precision
    (no fallback: if the eager chain cannot run there, the cell says so).
    """
    last_error = ""
    for dtype in (torch.float64, torch.float32) if precision == "fp64" else (PRECISIONS[precision]["model"],):
        model = _build_model(shapes).to(device, dtype=dtype)
        optimizer = None
        with torch.no_grad():
            for parameter, value in zip(model.parameters(), init, strict=True):
                parameter.copy_(value.to(device, dtype=dtype))
        try:
            optimizer = _build(entry, model.parameters(), hyper, precision, compile_step=False)
            for idx, target in batches:
                optimizer.zero_grad()
                _forward_loss(model, idx.to(device), target.to(device), 1 / idx.shape[0]).backward()
                optimizer.step()
            final = [value.detach().double().cpu() for value in model.parameters()]
            return final, dtype, ""
        except Exception as error:  # noqa: BLE001 -- a failed oracle must not erase the product result
            last_error = f"{type(error).__name__}: {error}"
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if precision == "fp64" and dtype is torch.float64:
                continue
            return None, dtype, last_error
    return None, None, last_error


_DTYPE_NAMES = frozenset({"float16", "bfloat16", "float32", "float64", "fp16", "bf16", "fp32", "fp64"})


def _self_replay(entry: Composition, hyper: dict, init: list, grads: list, device):
    """Our composition replayed eagerly at fp64 on the captured gradients: the
    semantic floor. A knob whose value names a dtype (PSGD's q_dtype) pins the
    chain below fp64; the replay retries with every such knob promoted, while
    the cell keeps testing the shipped default."""
    promoted = {name: "float64" for name, value in hyper.items() if isinstance(value, str) and value in _DTYPE_NAMES}
    for hyper_run in (hyper, {**hyper, **promoted}):
        make = lambda params: _build(entry, params, hyper_run, "fp64", compile_step=False)  # noqa: E731
        finals, _, _ = _apply_grads(make, init, grads, torch.float64)
        if finals is not None:
            return finals
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return None


def _validate_runtime():
    """Fail closed on everything that would make a result meaningless: no GPU,
    no einsum path search, no persistent Inductor cache, or any compile mode but
    the one that ships. A cell run in a weaker configuration is not evidence."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: heavyball is GPU-only; matrix cells never run on CPU")
    if not opt_einsum.is_available():
        raise RuntimeError("opt_einsum is required: einsum path search is part of the tested configuration")
    if not torch._inductor.config.fx_graph_cache:
        raise RuntimeError("persistent Inductor FX graph cache is required (TORCHINDUCTOR_FX_GRAPH_CACHE)")
    if utils.compile_mode != "max-autotune-no-cudagraphs":
        raise RuntimeError(f"compile_mode must be max-autotune-no-cudagraphs, got {utils.compile_mode!r}")


@dataclass
class Loop:
    """One cell's training loop and everything it measured."""

    step_ms: list
    fwd_bwd_ms: list
    traj: list | None  # rank-0 per-step full parameters; None under sharding
    grads: list | None  # rank-0 recorded gradients for the upstream replay
    graphs: int
    peak_mb: float | None
    profiled: dict | None


def _train_and_measure(optimizer, model, trainable, batches, local, scale, device, rank, want_traj, want_grads) -> Loop:
    graphs_before = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
    step_ms, fwd_bwd_ms = [], []
    traj = [] if want_traj and rank == 0 else None
    grads = [] if want_grads else None
    for step, (idx, target) in enumerate(batches):
        optimizer.zero_grad()
        started = time.perf_counter()
        _forward_loss(trainable, idx[local].to(device), target[local].to(device), scale).backward()
        if grads is not None:
            # The upstream baseline replays these gradients; full_tensor() is
            # collective under fsdp2, so every rank enters the gather even though
            # only rank 0 keeps the copies. fp64 capture round-trips bf16 exactly.
            captured = []
            for value in model.parameters():
                grad = value.grad
                if hasattr(grad, "full_tensor"):
                    grad = grad.full_tensor()
                captured.append(grad.detach().double().cpu())
            if rank == 0:
                grads.append(captured)
        # The speed leg is optimizer step time only: the bracket starts after a
        # synchronized backward so model time cannot mask or fake a step win.
        torch.cuda.synchronize()
        fwd_bwd_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        if step + 1 >= TIMED_FROM:
            step_ms.append((time.perf_counter() - started) * 1000)
        if traj is not None:
            traj.append([value.detach().double().cpu() for value in model.parameters()])
    graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0) - graphs_before
    assert graphs >= 1, "nothing compiled"
    return Loop(
        step_ms=step_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        traj=traj,
        grads=grads,
        graphs=int(graphs),
        peak_mb=round(torch.cuda.max_memory_allocated() / 1e6, 2) if rank == 0 else None,
        profiled=_profile_step(optimizer.step) if rank == 0 else None,
    )


@torch.no_grad()
def _gather_fulls(model) -> list:
    """Full-parameter copies on every rank. Every rank must call this exactly
    once: it issues one all_gather per parameter, and a rank that skips it
    deadlocks the others. Padded fixed sizes in parameter order keep the pairing
    exact across ranks."""
    fulls = []
    for value in model.parameters():
        if not isinstance(value, DTensor):
            fulls.append(value.detach().double().cpu())
            continue
        mesh = value.device_mesh
        shard = mesh if mesh.ndim == 1 else mesh["shard"]
        local = value.to_local()
        padded = local.new_zeros((math.ceil(value.shape[0] / shard.size()), *local.shape[1:]))
        padded.narrow(0, 0, local.shape[0]).copy_(local)
        chunks = [torch.empty_like(padded) for _ in range(shard.size())]
        dist.all_gather(chunks, padded, group=shard.get_group())
        fulls.append(torch.cat(chunks, dim=0).narrow(0, 0, value.shape[0]).double().cpu())
    return fulls


def _run_cell(cell: dict, seed: int, batch: bool = False) -> dict:
    """One cell. In batch mode the cell is default-topology and single-rank by
    construction, so there is no process group at all: torchrun, NCCL and the
    interpreter are paid once for the whole batch, not once per cell."""
    if batch and (cell["topology"] != "default" or cell["world"] != 1):
        raise RuntimeError("batch mode runs default-topology world-1 cells only")
    rank = 0 if batch else int(os.environ["RANK"])
    if batch:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())

    utils.set_torch()
    _validate_runtime()

    entry = _composition_for(cell["name"])
    baseline = BASELINES[entry.baseline] if entry.baseline is not None else None
    baseline_source_error = ""
    if baseline is not None and rank == 0:
        try:
            _validate_baseline_source(baseline)
        except Exception as error:
            baseline_source_error = f"{type(error).__name__}: {error}"
    shapes, hyper = _draw_problem(seed, entry, cell.get("scale", 1))
    init = _initial_parameters(shapes)
    batches = [_batch(shapes, seed + step + 1, BATCH) for step in range(STEPS)]
    model_dtype = PRECISIONS[cell["precision"]]["model"]
    storage = PRECISIONS[cell["precision"]]["model"]
    law_cell = entry.facade is None  # composition tier: judged against its eager self
    # Fresh-process repeats carry no oracle or upstream work: they exist to spread
    # the speed measurement, and the accuracy fields are identical by determinism.
    skip_reference = os.environ.get("HEAVYBALL_MATRIX_SKIP_REFERENCE") == "1"
    baseline_grads = [] if baseline is not None and not skip_reference else None

    # The oracle must run before the process group exists: building an optimizer
    # inside a live group issues collectives only this rank enters.
    oracle_result = _oracle(entry, shapes, hyper, init, batches, device) if rank == 0 and not skip_reference else None
    final_oracle = oracle_result[0] if oracle_result is not None else None
    self_oracle_error = oracle_result[2] if oracle_result is not None else ""
    eager_cell = None
    eager_oracle_error = ""
    if law_cell and rank == 0 and not skip_reference:
        if cell["precision"] == "fp64" and final_oracle is not None:
            eager_cell = final_oracle  # the eager fp64 leg IS the oracle at fp64
        else:
            eager_cell, _, eager_oracle_error = _oracle(entry, shapes, hyper, init, batches, device, cell["precision"])
    if not batch:
        dist.init_process_group("nccl")
    # The shipped step narrows fp32 intermediates to bf16 with stochastic
    # rounding drawn from the global RNG. Rank 0's oracle run advanced that RNG,
    # so reseed identically on every rank or the bf16 cells diverge by the
    # rounding noise alone. The problem draws use their own Generators.
    torch.manual_seed(seed)
    if rank == 0:
        torch.cuda.reset_peak_memory_stats()

    model = _build_model(shapes).to(device, dtype=model_dtype)
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), init, strict=True):
            parameter.copy_(value.to(device, dtype=model_dtype))
    mesh = wrapper = None
    dp_rank, dp_world = rank, cell["world"]
    try:
        if cell["topology"] in ("fsdp2", "hsdp"):
            if cell["topology"] == "hsdp":
                replicate, shard = _hsdp_mesh(cell["world"])
                mesh = init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("replicate", "shard"))
                dp_rank, dp_world = mesh.get_local_rank("replicate"), replicate
            else:
                mesh = init_device_mesh("cuda", (cell["world"],))
            for child in model.children():
                fully_shard(child, mesh=mesh)
            fully_shard(model, mesh=mesh)
        optimizer = _build(entry, model.parameters(), hyper, cell["precision"])
        if cell["topology"] == "ddp":
            wrapper = DDP(model)
    except (ValueError, TypeError) as error:
        # Design-level rejections raise ValueError/TypeError; anything else -- OOM
        # is a RuntimeError -- must fail the gate instead of laundering itself.
        if not batch:
            dist.destroy_process_group()
        return {"cell": cell, "status": "unsupported", "reason": str(error)[:400], "shapes": shapes, "hyper": hyper}

    global_batch = batches[0][0].shape[0]
    base, extra = divmod(global_batch, dp_world)
    start = dp_rank * base + min(dp_rank, extra)
    local = slice(start, start + base + (dp_rank < extra))
    scale = dp_world / global_batch
    sharded = cell["topology"] in ("fsdp2", "hsdp")  # DTensor params have no per-step full copy
    _TRAIN_T0 = time.perf_counter()
    loop = _train_and_measure(
        optimizer,
        model,
        wrapper if wrapper is not None else model,
        batches,
        local,
        scale,
        device,
        rank,
        want_traj=baseline is not None and not sharded,
        want_grads=baseline_grads is not None and not baseline.closure,
    )
    baseline_grads = loop.grads if baseline_grads is not None else None
    ours_traj = loop.traj
    step_ms, fwd_bwd_ms = loop.step_ms, loop.fwd_bwd_ms
    profiled = loop.profiled

    phases = {"startup_oracle": None, "train": None, "reference": None}
    if batch:
        fulls = [value.detach().double().cpu() for value in model.parameters()]
        ranks_identical, rank_gap = True, 0.0
    else:
        fulls = _gather_fulls(model)
        flat = torch.cat([value.reshape(-1) for value in fulls])
        reference = flat.to(device)
        dist.broadcast(reference, src=0)
        gap = (flat.to(device) - reference).abs().max().reshape(1)
        dist.all_reduce(gap, op=dist.ReduceOp.MAX)
        # Ranks recompute the same step from bitwise-identical gradients, but
        # Inductor's max_autotune picks kernels by wall-clock timing, so ranks can
        # choose kernels that accumulate in different orders: ulp-level drift, not
        # divergence. Identity is judged within a few hundred ulps of the largest
        # parameter; a rank that truly computed different work lands far above.
        tolerance = 256 * torch.finfo(torch.float32).eps * flat.abs().max().to(device)
        ranks_identical = bool(gap.item() <= tolerance.item())
        rank_gap = gap.item()

    started_phase = time.perf_counter()
    phases["startup_oracle"] = round(_TRAIN_T0 - _WORKER_T0, 1)
    phases["train"] = round(started_phase - _TRAIN_T0, 1)  # compile + 64 steps + gather
    record = {"cell": cell, "shapes": shapes, "hyper": hyper}
    if rank == 0:
        l2_rel = (
            [
                (value - oracle).norm().item() / max(oracle.norm().item(), 1e-12)
                for value, oracle in zip(fulls, final_oracle, strict=True)
            ]
            if final_oracle is not None
            else None
        )
        record.update(
            status="measured",
            l2_rel=max(l2_rel) if l2_rel else None,
            **(_law_leg(fulls, final_oracle, eager_cell) if law_cell and not skip_reference else {}),
            ranks_identical=ranks_identical,
            rank_gap=rank_gap,
            changed=max((value - before).abs().max().item() for value, before in zip(fulls, init, strict=True)),
            finite=bool(torch.isfinite(torch.cat([value.reshape(-1) for value in fulls])).all().item()),
            step_ms=round(sum(step_ms) / len(step_ms), 3),
            fwd_bwd_ms=round(sum(fwd_bwd_ms) / len(fwd_bwd_ms), 3),
            peak_mb=loop.peak_mb,
            graphs=loop.graphs,
            upstream=baseline.name if baseline is not None else None,
        )
        if self_oracle_error:
            record["self_oracle_error"] = self_oracle_error
        if eager_oracle_error:
            record["eager_oracle_error"] = eager_oracle_error
        if profiled is not None:
            record["device_ops"], record["device_ms"] = profiled["device_ops"], profiled["device_ms"]
        if baseline_source_error:
            record["baseline_source_error"] = baseline_source_error
        if baseline is not None and baseline_grads and not baseline_source_error:
            loss_model = _build_model(shapes).to(storage)

            def replay_loss(step, params):
                # Scored on the CPU model the replays also use: a GPU forward of
                # identical parameters differs by ~1e-2 in loss.
                with torch.no_grad():
                    for parameter, value in zip(loss_model.parameters(), params, strict=False):
                        parameter.copy_(value.to(dtype=parameter.dtype))
                idx, target = batches[step]
                return _forward_loss(loss_model, idx, target, 1 / idx.shape[0]).item()

            if ours_traj is not None:
                record["loss_ours"] = round(replay_loss(len(ours_traj) - 1, ours_traj[-1]), 6)
            ours64 = final_oracle if baseline.closure else _self_replay(entry, hyper, init, baseline_grads, device)
            leg = AccuracyLeg(baseline, hyper, init, baseline_grads, fulls, storage, shapes, batches).run(
                ours64, ours_traj, replay_loss
            )
            record.update(
                err_ours=leg.ours,
                err_ref=leg.ref,
                err_semantic=leg.semantic,
                err_ref_promoted=leg.ref_promoted,
                err_oracle_dtype=leg.oracle_dtype or None,
                onset=next((i + 1 for i, value in enumerate(leg.traj_ours or []) if value > 1e-6), None),
                loss_ref=round(leg.loss_ref[-1], 6) if leg.loss_ref else None,
                loss_naive=round(leg.loss_naive[-1], 6) if leg.loss_naive else None,
            )
            if leg.err:
                record["err_note"] = leg.err
        if baseline is not None and not baseline_source_error:
            runs = [("baseline", model_dtype, ("step_ms", "device_ops", "device_ms", "peak_mb"))]
            if storage is not torch.float32:
                runs.append(("baseline_fp32", torch.float32, ("step_ms", "device_ops")))
            for prefix, dtype, fields in runs:
                try:
                    metrics = _baseline_metrics(baseline, shapes, hyper, init, batches, device, dtype)
                except Exception as error:
                    record[f"{prefix}_error"] = f"{type(error).__name__}: {error}"
                    continue
                for field in fields:
                    record[f"{prefix}_{field}"] = metrics[field]
        phases = {k: v for k, v in phases.items() if v is not None}
        phases["reference"] = round(time.perf_counter() - started_phase, 1)
        record["phases"] = phases
    if not batch:
        dist.destroy_process_group()
    return record


def _worker_main() -> int:
    global _WORKER_T0
    _WORKER_T0 = time.perf_counter()
    cell = json.loads(os.environ["HEAVYBALL_MATRIX_CELL"])
    seed = int(os.environ["HEAVYBALL_MATRIX_SEED"])
    # Rank 0 alone reports the result: a non-zero rank's record carries no
    # measurements, and its failure surfaces when torchrun tears the cell down.
    reporting = int(os.environ["RANK"]) == 0
    started = time.perf_counter()
    try:
        record = _run_cell(cell, seed)
        if reporting:
            record["wall_s"] = round(time.perf_counter() - started, 1)
            _emit("MATRIX_RESULT", record, seed)
    except Exception as error:
        _emit(
            "MATRIX_ERROR",
            {
                "cell": cell,
                "error_type": type(error).__name__,
                "error_message": str(error)[:600],
                "traceback": traceback.format_exc()[-2000:],
                "wall_s": round(time.perf_counter() - started, 1),
            },
            seed,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _healthy(row: dict) -> bool:
    return bool(row["finite"] and row["changed"] > 0 and row["ranks_identical"])


def _law_leg(fulls: list, oracle: list | None, eager: list | None) -> dict:
    """The composition-tier legs, all flat distances relative to the fp64
    oracle: err_compiled (the tested cell), err_eager (the same chain run
    eagerly at the cell's precision), and err_fusion (compiled against that
    eager run)."""
    if oracle is None:
        return {"law_dead": True, "err_note": "self-oracle cannot run at fp64"}
    oracle_flat = torch.cat([value.reshape(-1) for value in oracle]).double()
    scale = oracle_flat.norm().clamp_min(1e-12)
    flat = lambda finals: torch.cat([value.reshape(-1) for value in finals]).double()  # noqa: E731

    if eager is None:
        return {"law_dead": True, "err_note": "eager chain cannot run at the cell precision"}
    return {
        "err_compiled": ((flat(fulls) - oracle_flat).norm() / scale).item(),
        "err_eager": ((flat(eager) - oracle_flat).norm() / scale).item(),
        "err_fusion": ((flat(fulls) - flat(eager)).norm() / scale).item(),
    }


def _row_gates(row: dict) -> dict[str, bool]:
    gates = {}
    if row.get("err_ours") is not None:
        # The global accuracy gate: the compiled composition at the cell's
        # precision must sit no farther from the upstream fp64 run than the
        # upstream implementation does at that same precision. Where our
        # algorithm deliberately diverges from the upstream (clamp-for-eps is
        # not add), err_semantic -- our own fp64 replay on the same gradients
        # against the same oracle -- is the floor no precision can beat, and
        # err_ref hits 0 exactly at cells where the upstream at cell precision
        # IS the oracle. The bound is the max of the two.
        bound = max(row["err_ref"], row.get("err_semantic") or 0.0)
        gates["accuracy"] = row["err_ours"] <= bound
    if row.get("err_fusion") is not None:
        # The composition-tier laws. Two valid computations of the same math at
        # one precision differ by rounding that accumulates as a random walk:
        # the floor is sqrt(steps) ulps of the storage dtype, not zero. fusion:
        # the compiled chain may not sit farther from the eager chain at the
        # same precision than that floor. precision: the compiled chain may not
        # sit farther from the eager fp64 chain than the eager chain at the
        # cell precision does, beyond the same floor.
        eps = torch.finfo(PRECISIONS[row["cell"]["precision"]]["model"]).eps
        floor = math.sqrt(STEPS) * eps
        eager_gap = row.get("err_eager") or 0.0
        gates["fusion"] = row["err_fusion"] <= PASS_MARGIN * max(eager_gap, floor)
        gates["precision"] = row["err_compiled"] <= PASS_MARGIN * max(eager_gap, floor)
    if row.get("law_dead") is not None:
        gates["eager_leg"] = not row["law_dead"]
    if row.get("step_ms") is not None and row.get("baseline_step_ms") is not None:
        gates["speed"] = row["step_ms"] < row["baseline_step_ms"]
    return gates


def _judge(rows: list[dict]) -> int:
    key = lambda cell: (cell["name"], cell["precision"], cell.get("scale", 1))  # noqa: E731
    # References first: their verdicts gate the selfmargin path of the cells
    # that judge against them.
    ordered = sorted(rows, key=lambda row: row["cell"]["topology"] != "default")
    references = {key(row["cell"]): row for row in ordered if row["cell"]["topology"] == "default"}
    for row in ordered:
        cell = row["cell"]
        if row["status"] != "measured":
            continue
        if row.get("nonfinite_fields"):
            row["gates"] = {"finite_record": False}
            row["status"] = "fail"
            continue
        if row.get("upstream") is not None:
            required = ["baseline_step_ms"]
            if cell["topology"] in ("default", "ddp"):
                required.extend(("err_ours", "err_ref"))
            missing = [
                field
                for field in required
                if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field])
            ]
            if missing:
                row["status"] = "invalid_reference"
                row["reason"] = "upstream reference missing " + ", ".join(missing)
                continue
        gates = _row_gates(row)
        if "accuracy" not in gates and cell["topology"] != "default":
            # No upstream baseline: the distributed cell must stay within
            # PASS_MARGIN of its own single-process fp64 self.
            reference = references.get(key(cell))
            if reference is None or reference.get("l2_rel") is None or reference.get("status") != "pass":
                row["status"] = "invalid_reference"
                continue
            row["reference_l2_rel"] = reference["l2_rel"]
            gates["self_eager"] = row["l2_rel"] <= reference["l2_rel"] * PASS_MARGIN
        row["gates"] = gates
        row["status"] = "pass" if _healthy(row) and all(gates.values()) else "fail"
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    for row in rows:
        if row["status"] in ("pass", "fail"):
            detail = ""
            if row.get("l2_rel") is not None:
                detail = f"l2_rel={row['l2_rel']:.3e}"
                if "reference_l2_rel" in row:
                    detail += f" ref={row['reference_l2_rel']:.3e}"
            if row.get("err_ours") is not None:
                detail += f" err={row['err_ours']:.3e}/{row['err_ref']:.3e}"
                if row.get("err_ref_promoted") is not None:
                    detail += f"/{row['err_ref_promoted']:.3e}"
            if row.get("baseline_step_ms") is not None:
                detail += f" ms={_median(row, 'step_ms', 'step_ms_spread')} base={_median(row, 'baseline_step_ms', 'baseline_step_ms_spread')}"
                if row.get("baseline_fp32_step_ms") is not None:
                    detail += f" base32={_median(row, 'baseline_fp32_step_ms', 'baseline_fp32_step_ms_spread')}"
            detail += " gates=" + (
                ",".join(f"{name}:{'ok' if ok else 'BAD'}" for name, ok in row.get("gates", {}).items()) or "none"
            )
        else:
            detail = (row.get("reason") or row.get("traceback", ""))[:110].replace("\n", " ")
        print(f"{row['status'].upper():18} {_cell_key(row['cell']):48} {detail}")
    print(
        f"TOTAL {len(rows)} cells: " + " ".join(f"{name}={count}" for name, count in sorted(counts.items())), flush=True
    )
    return 0 if rows and all(row.get("status") == "pass" for row in rows) else 1


def _median(row: dict, field: str, spread_field: str):
    spread = row.get(spread_field)
    if spread:
        return spread[1]
    return row.get(field)


# ---------------------------------------------------------------------------
# The two lists: where we are not more accurate, and where we are not faster
# ---------------------------------------------------------------------------


def _less_accurate(row: dict) -> bool:
    ours, ref = row.get("err_ours"), row.get("err_ref")
    if ours is None or ref is None:
        return False
    bound = max(ref, row.get("err_semantic") or 0.0)
    return not math.isfinite(bound) or ours > bound


def _slower(row: dict) -> bool:
    """Median-loses to the baseline, and -- where fresh-process spreads exist --
    the spread cannot explain the loss away."""
    ours, base = row.get("step_ms"), row.get("baseline_step_ms")
    if ours is None or base is None:
        return False
    ours_spread = row.get("step_ms_spread") or [ours]
    base_spread = row.get("baseline_step_ms_spread") or [base]
    if len(ours_spread) > 1 and min(ours_spread) < min(base_spread):
        return False
    return statistics.median(ours_spread) >= statistics.median(base_spread)


def _config(row: dict, **fields) -> dict:
    config = {
        "key": _cell_key(row["cell"]),
        "cell": row["cell"],
        "shapes": row.get("shapes"),
        "hyper": row.get("hyper"),
    }
    config.update({name: value for name, value in fields.items() if value is not None})
    return config


def _err_ratio(row: dict) -> float:
    bound = max(row.get("err_ref") or 0.0, row.get("err_semantic") or 0.0)
    ours = row.get("err_ours") or 0.0
    if not math.isfinite(bound) or bound <= 0:
        return math.inf if ours > 0 else 1.0
    return ours / bound


def _classify_error(row: dict) -> str:
    text = row.get("traceback") or row.get("reason") or ""
    if "lerp_" in text:
        return "soap-chain lerp_ dtype (non-fp32 storage)"
    if "_grid_" in text or "'stream'" in text:
        return "inductor triton launcher"
    if "nothing compiled" in text:
        return "nothing compiled"
    if "timeout" in text:
        return "timeout"
    if "out of memory" in text.lower():
        return "oom"
    return "other inductor/compiler"


def _fmt(value, spec=".2e"):
    if value is None:
        return "-"
    if isinstance(value, list):  # legacy rows carry curves; the final is the datum
        return _fmt(value[-1] if value else None, spec)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return format(value, spec)


def _summary_md(rows: list[dict]) -> str:
    """The one-stop digest: verdict grid, ranked defeats, mechanism tables."""
    measured = [row for row in rows if row.get("status") in ("pass", "fail", "measured")]
    lines = ["# Matrix summary", ""]

    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    lines.append(f"{len(rows)} cells: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())) + "\n")

    # Verdict grid: composition x precision, one grid per scale.
    compositions = sorted({row["cell"]["name"] for row in rows})
    precisions = [p for p in PRECISIONS if any(row["cell"]["precision"] == p for row in rows)]
    scales = sorted({row["cell"].get("scale", 1) for row in rows})
    bycell = {_cell_key(row["cell"]): row for row in rows}
    for scale in scales:
        lines += [
            f"## Verdicts (scale x{scale})",
            "",
            "| composition | " + " | ".join(precisions) + " |",
            "|" + "---|" * (len(precisions) + 1),
        ]
        for name in compositions:
            cells = []
            for precision in precisions:
                row = bycell.get(
                    _cell_key({"name": name, "precision": precision, "topology": "default", "world": 1, "scale": scale})
                )
                cells.append(
                    " . "
                    if row is None
                    else "ERR"
                    if row["status"] == "error"
                    else "uns"
                    if row["status"] == "unsupported"
                    else "acc!"
                    if row.get("err_ours") is not None and _less_accurate(row)
                    else "spd!"
                    if row.get("step_ms") is not None and row.get("baseline_step_ms") is not None and _slower(row)
                    else "ok"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        "`ok` passes every gate; `acc!` loses the accuracy bound; `spd!` loses the speed gate; `ERR` crashed.\n"
    )
    upstream = sum(1 for row in rows if row.get("upstream"))
    self_only = len(rows) - upstream
    lines += [
        f"**{upstream} rows carry an upstream oracle (official same-algorithm implementation); "
        f"{self_only} rows are self-validated only (no external reference exists).**\n"
    ]

    # Ranked accuracy defeats with the semantic decomposition.
    defeats = sorted((row for row in measured if _less_accurate(row)), key=_err_ratio, reverse=True)
    if defeats:
        lines += [
            "## Accuracy defeats (worst first)",
            "",
            "| cell | ours | ref | semantic | ours/bound | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for row in defeats:
            semantic = row.get("err_semantic")
            ref = row.get("err_ref")
            at_floor = semantic is not None and semantic >= (
                ref if isinstance(ref, float) and math.isfinite(ref) else 0.0
            )
            verdict = "at algorithmic floor" if at_floor else "precision deficit"
            lines.append(
                f"| {_cell_key(row['cell'])} | {_fmt(row['err_ours'])} | {_fmt(ref)} | {_fmt(semantic)} "
                f"| {_fmt(_err_ratio(row), '.2f')} | {verdict} |"
            )
        lines.append("")

    # Speed with the op-count mechanism.
    timed = [row for row in measured if row.get("baseline_step_ms") is not None]
    if timed:
        lines += [
            "## Speed vs upstream (with mechanism)",
            "",
            "| cell | ours ms | base ms | ratio | ours ops | base ops | ours dev-ms | base dev-ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(timed, key=lambda r: r["step_ms"] / r["baseline_step_ms"], reverse=True):
            lines.append(
                f"| {_cell_key(row['cell'])} | {_fmt(row['step_ms'], '.3f')} | {_fmt(row['baseline_step_ms'], '.3f')} "
                f"| {row['step_ms'] / row['baseline_step_ms']:.2f}x | {_fmt(row.get('device_ops'), 'd')} "
                f"| {_fmt(row.get('baseline_device_ops'), 'd')} | {_fmt(row.get('device_ms'), '.3f')} "
                f"| {_fmt(row.get('baseline_device_ms'), '.3f')} |"
            )
        lines.append("")

    # Optimization quality: final loss on the same gradient stream.
    curves = [row for row in measured if row.get("loss_ours") is not None and row.get("loss_naive") is not None]
    if curves:
        lines += [
            "## Final loss over the same gradient stream",
            "",
            "| cell | ours | upstream@cell | upstream fp64 |",
            "|---|---|---|---|",
        ]
        for row in curves:
            ours_final = row["loss_ours"]
            naive_final = row["loss_naive"]
            ref_final = row.get("loss_ref")
            lines.append(
                f"| {_cell_key(row['cell'])} | {_fmt(ours_final, '.6g')} | {_fmt(naive_final, '.6g')} | {_fmt(ref_final, '.6g')} |"
            )
        lines.append("")

    # Divergence onset: first step where ours separates from the oracle trajectory.
    onset = [row for row in measured if row.get("onset") is not None]
    if onset:
        lines += [
            "## Divergence onset (first step with err > 1e-6)",
            "",
            "| cell | first hot step |",
            "|---|---|",
        ]
        lines += [f"| {_cell_key(row['cell'])} | {row['onset']} |" for row in onset]
        lines.append("")

    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        counts = {}
        for row in errors:
            counts[_classify_error(row)] = counts.get(_classify_error(row), 0) + 1
        lines += ["## Cell errors", ""]
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            names = sorted({row["cell"]["name"] for row in errors if _classify_error(row) == reason})
            lines.append(f"- {reason} x{count}: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
        lines.append("")
    return "\n".join(lines)


def _report(measure: Path) -> int:
    rows = _read_rows([measure])
    verdict = _judge(rows)
    accuracy = [
        _config(
            row,
            err_ours=row.get("err_ours"),
            err_ref=row.get("err_ref"),
            err_semantic=row.get("err_semantic"),
            err_ref_promoted=row.get("err_ref_promoted"),
            onset=row.get("onset"),
            loss_ours=row.get("loss_ours"),
            loss_naive=row.get("loss_naive"),
        )
        for row in rows
        if _less_accurate(row)
    ]
    speed = [
        _config(
            row,
            step_ms=_median(row, "step_ms", "step_ms_spread"),
            step_ms_spread=row.get("step_ms_spread"),
            baseline_step_ms=_median(row, "baseline_step_ms", "baseline_step_ms_spread"),
            baseline_step_ms_spread=row.get("baseline_step_ms_spread"),
            device_ops=row.get("device_ops"),
            baseline_device_ops=row.get("baseline_device_ops"),
            device_ms=row.get("device_ms"),
            baseline_device_ms=row.get("baseline_device_ms"),
        )
        for row in rows
        if _slower(row)
    ]
    baselineless = [
        _config(
            row,
            note="no same-algorithm upstream exists anywhere; self-eager gate only -- this is NOT a baseline comparison",
        )
        for row in rows
        if row.get("upstream") is None
    ]
    for name, entries in (("accuracy_defeats", accuracy), ("speed_defeats", speed), ("baselineless", baselineless)):
        path = measure.parent / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(_json_row(entry) + "\n")
        print(f"{name}: {len(entries)} -> {path}")
    summary = measure.parent / "summary.md"
    summary.write_text(_summary_md(rows), encoding="utf-8")
    compared = sum(1 for row in rows if row.get("err_ours") is not None)
    print(
        f"{len(rows)} rows: {compared} upstream-compared, {len(rows) - compared} self-only (NOT baseline comparisons)"
    )
    print(f"summary -> {summary}")
    return 1 if verdict or accuracy or speed else 0


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _inductor_cache() -> Path:
    """The persistent Inductor artifact cache: pinned, not left to the TMPDIR default."""
    return Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or Path.home() / ".cache" / "heavyball" / "inductor")


def _prune_inductor_cache(cache: Path, target_free: int = 32 * 2**30) -> None:
    """Bound the cache: below the target, delete oldest entries until it is met."""
    if not cache.is_dir() or shutil.disk_usage(cache).free >= target_free:
        return
    entries = sorted(cache.iterdir(), key=lambda entry: entry.stat().st_mtime)
    for entry in entries:
        if shutil.disk_usage(cache).free >= target_free:
            break
        shutil.rmtree(entry, ignore_errors=True)


def _run_one(
    cell: dict, seed: int, timeout: int, *, skip_reference: bool = False, device: int | None = None, port: int
) -> dict:
    environment = dict(os.environ)
    environment["HEAVYBALL_MATRIX_CELL"] = json.dumps(cell, sort_keys=True)
    environment["HEAVYBALL_MATRIX_SEED"] = str(seed)
    if skip_reference:
        environment["HEAVYBALL_MATRIX_SKIP_REFERENCE"] = "1"
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(_inductor_cache())
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(cell["world"]),
        f"--master_port={port + cell['world']}",
        str(Path(__file__).resolve()),
        "worker",
    ]
    # The worker log lands in a file so a hung cell still yields its tail; a
    # timeout kills the whole process group because torchrun does not reap workers.
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "cell.log"
        with log.open("w") as handle:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {
                    "cell": cell,
                    "status": "error",
                    "reason": f"timeout after {timeout}s",
                    "output": log.read_text(errors="replace")[-1500:],
                }
        output = log.read_text(errors="replace")
        returncode = process.returncode
    record = None
    for line in output.splitlines():
        for prefix in ("MATRIX_RESULT", "MATRIX_ERROR"):
            if not line.startswith(prefix + " "):
                continue
            _, _, payload = line.partition(" ")
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue  # a line truncated mid-write is not the cell's verdict
            if prefix == "MATRIX_ERROR":
                record["status"] = "error"
    if record is None:
        return {
            "cell": cell,
            "status": "error",
            "reason": f"exit {returncode}, no result line",
            "output": output[-1500:],
        }
    return record


_MEASURE_LOCK = threading.Lock()


def _cell_lane(cell, seed, timeout, fresh_processes, device, port, measure) -> dict:
    """Every fresh-process repeat of one cell, sequentially on one device: the
    repeats exist to spread the speed measurement, so they never share the GPU
    with another cell. Returns the representative row (first measured repeat,
    carrying the timing spreads of the rest)."""
    base = last = None
    step_timings, baseline_timings, baseline32_timings = [], [], []
    for repeat in range(max(1, fresh_processes)):
        row = _run_one(
            cell,
            seed,
            timeout,
            skip_reference=repeat > 0,
            device=device,
            port=port,
        )
        last = row
        if measure:
            with _MEASURE_LOCK, open(measure, "a", encoding="utf-8") as handle:
                handle.write(_json_row(row, seed) + "\n")
        if row.get("status") != "measured":
            continue
        if base is None:
            base = row
        for key, timings in (
            ("step_ms", step_timings),
            ("baseline_step_ms", baseline_timings),
            ("baseline_fp32_step_ms", baseline32_timings),
        ):
            if row.get(key) is not None:
                timings.append(row[key])
    chosen = base if base is not None else last
    for key, timings, spread in (
        ("step_ms", step_timings, "step_ms_spread"),
        ("baseline_step_ms", baseline_timings, "baseline_step_ms_spread"),
        ("baseline_fp32_step_ms", baseline32_timings, "baseline_fp32_step_ms_spread"),
    ):
        if base is not None and len(timings) > 1:
            chosen[spread] = [round(min(timings), 3), round(statistics.median(timings), 3), round(max(timings), 3)]
    if base is not None and len(step_timings) > 1:
        chosen["fresh_processes"] = len(step_timings)
    if measure and max(1, fresh_processes) > 1:
        with _MEASURE_LOCK, open(measure, "a", encoding="utf-8") as handle:
            handle.write(_json_row(chosen, seed) + "\n")
    return chosen


def _append_locked(path: Path, row: dict, seed: int | None = None) -> None:
    """Append one row from one of several concurrent batch processes."""
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(_json_row(row, seed) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def _batch_main(manifest: Path, measure: Path, seed: int) -> int:
    """One process, one GPU (pinned by the spawner), many world-1 cells: the
    interpreter, torchrun and NCCL are paid once per batch, not once per cell.
    Timing cells still deserve fresh processes, so the driver only routes
    fresh-process-1 passes here."""
    global _WORKER_T0
    _WORKER_T0 = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    utils.set_torch()
    _validate_runtime()
    wanted = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for cell in wanted:
        seed_cell = _draw_seed(seed, cell["name"], cell["precision"])
        started = time.perf_counter()
        try:
            record = _run_cell(cell, seed_cell, batch=True)
            record["wall_s"] = round(time.perf_counter() - started, 1)
        except Exception as error:
            record = {
                "cell": cell,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:600],
                "traceback": traceback.format_exc()[-2000:],
                "wall_s": round(time.perf_counter() - started, 1),
            }
        _append_locked(measure, record, seed_cell)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


def _finish(rows: list[dict], note: str) -> int:
    """Judge the raw measurement rows."""
    print(note, file=sys.stderr)
    return _judge(rows)


def _drive(
    cells: list[dict],
    seed: int,
    measure: Path | None,
    timeout: int,
    fresh_processes: int = 1,
    gpus: int = 1,
    batch: bool = False,
    carried: list[dict] | None = None,
) -> int:
    started = time.perf_counter()
    _prune_inductor_cache(_inductor_cache())
    if measure:
        _prepare_measure(measure, seed)
    # Batch mode: every world-1 cell of a fresh-process-1 pass in one process
    # per GPU. Timing passes keep the per-cell path -- fresh processes are the
    # point of their spreads.
    if batch and fresh_processes == 1 and cells and all(cell["world"] == 1 for cell in cells):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            processes = []
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            environment["TORCHINDUCTOR_CACHE_DIR"] = str(_inductor_cache())
            for lane in range(gpus):
                manifest = Path(directory) / f"lane{lane}.jsonl"
                manifest.write_text("".join(json.dumps(cell) + "\n" for cell in cells[lane::gpus]), encoding="utf-8")
                lane_env = dict(environment)
                if gpus > 1:
                    lane_env["CUDA_VISIBLE_DEVICES"] = str(lane % _device_count())
                log = open(Path(directory) / f"lane{lane}.log", "w")  # noqa: SIM115 -- closed with the process
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "batch",
                            "--manifest",
                            str(manifest),
                            "--measure",
                            str(results),
                            "--seed",
                            str(seed),
                        ],
                        env=lane_env,
                        cwd=str(REPO_ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            for process, log in processes:
                process.wait()
                log.close()
            fresh = (
                [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
                if results.exists()
                else []
            )
        by_key = {_cell_id(row["cell"]): row for row in fresh}
        fresh = [
            by_key.get(
                _cell_id(cell),
                {"cell": cell, "status": "error", "error_type": "BatchLaneError"},
            )
            for cell in cells
        ]
        if measure:
            for row in fresh:
                _append_locked(measure, row)
        rows = list(carried or []) + fresh
        return _finish(rows, f"driver wall {time.perf_counter() - started:.0f}s across {gpus} gpu(s), batched")
    # Single-GPU cells run gpus-at-a-time, one pinned device each. Distributed
    # cells run alone with every GPU visible: torchrun claims one device per
    # rank itself, and NCCL rejects two ranks on one device, so a distributed
    # cell can never share the node with another lane.
    singles = [cell for cell in cells if cell["world"] == 1]
    distributed = [cell for cell in cells if cell["world"] > 1]
    lanes = [
        (
            _cell_lane,
            cell,
            _draw_seed(seed, cell["name"], cell["precision"]),
            timeout,
            fresh_processes,
            position % _device_count(),
            PORT_BASE + (position % 1400) * 16,
            measure,
        )
        for position, cell in enumerate(singles)
    ]
    if gpus > 1 and lanes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpus) as pool:
            rows = list(pool.map(lambda lane: lane[0](*lane[1:]), lanes))
    else:
        rows = [lane[0](*lane[1:]) for lane in lanes]
    rows = list(carried or []) + rows
    for position, cell in enumerate(distributed):
        rows.append(
            _cell_lane(
                cell,
                _draw_seed(seed, cell["name"], cell["precision"]),
                timeout,
                fresh_processes,
                None,
                PORT_BASE + ((len(lanes) + position) % 1400) * 16,
                measure,
            )
        )
    return _finish(rows, f"driver wall {time.perf_counter() - started:.0f}s across {gpus} gpu(s)")


app = typer.Typer(add_completion=False)


@app.command("list-cells")
def list_cells(
    seed: int = 0,
    max_world: int = 4,
    optimizers: list[str] = typer.Option(None, "--optimizer"),
    precisions: list[str] = typer.Option(None, "--precision"),
    scales: list[int] = typer.Option(None, "--scale"),
):
    """List every cell with its drawn shapes and hyperparameters."""
    names = optimizers or list(COMPOSITIONS)
    cells = enumerate_cells(max_world, names, precisions or list(PRECISIONS), scales=scales)
    for cell in cells:
        shapes, hyper = _draw_problem(
            _draw_seed(seed, cell["name"], cell["precision"]), COMPOSITIONS[cell["name"]], cell.get("scale", 1)
        )
        print(json.dumps({"key": _cell_key(cell), "shapes": shapes, "hyper": hyper}, sort_keys=True))
    print(f"{len(cells)} cells (max_world={max_world})", file=sys.stderr)


def _with_references(cells: list[dict]) -> list[dict]:
    """Every distributed cell judges against its default-topology reference, so
    a selection holding only distributed cells must carry the references too or
    every row reads invalid_reference."""
    needed = {(cell["name"], cell["precision"], cell.get("scale", 1)) for cell in cells}
    have = {
        (cell["name"], cell["precision"], cell.get("scale", 1))
        for cell in cells
        if cell["topology"] == "default" and cell["world"] == 1
    }
    merged, seen = list(cells), {_cell_key(cell) for cell in cells}
    for name, precision, scale in sorted(needed - have):
        reference = {"topology": "default", "world": 1, "name": name, "precision": precision, "scale": scale}
        if _cell_key(reference) not in seen:
            merged.append(reference)
    return merged


@app.command()
def sample(
    count: int = 64,
    seed: int = 0,
    max_world: int = 4,
    measure: Path = typer.Option(None),
    timeout: int = 2400,
    fresh_processes: int = 3,
    gpus: int = 1,
    optimizers: list[str] = typer.Option(None, "--optimizer"),
    precisions: list[str] = typer.Option(None, "--precision"),
    scales: list[int] = typer.Option(None, "--scale"),
):
    """Seeded subsample plus the reference cells it is judged against."""
    names = optimizers or list(COMPOSITIONS)
    cells = enumerate_cells(max_world, names, precisions or list(PRECISIONS), scales=scales)
    chosen = random.Random(seed).sample(cells, min(count, len(cells)))
    merged = sorted(_with_references(chosen), key=_cell_key)
    raise SystemExit(_drive(merged, seed, measure, timeout, fresh_processes, gpus, batch))


def _resume_cells(cells: list[dict], measure: Path | None, also_done: list[Path], seed: int):
    """The pending cells and the carried rows: everything already measured is
    truth, not work -- the tool resumes itself instead of being resumed."""
    sources = [measure, *(also_done or [])]
    carried = _carried_rows([s for s in sources if s], seed)
    done = {_cell_id(row["cell"]) for row in carried}
    pending = [cell for cell in cells if _cell_id(cell) not in done]
    print(f"{len(cells)} enumerated, {len(cells) - len(pending)} measured, {len(pending)} pending", file=sys.stderr)
    return pending, carried


@app.command()
def full(
    seed: int = 0,
    max_world: int = 4,
    min_world: int = 1,
    measure: Path = typer.Option(None),
    shard: str = typer.Option(None),
    timeout: int = 2400,
    fresh_processes: int = 1,
    gpus: int = 1,
    optimizers: list[str] = typer.Option(None, "--optimizer"),
    precisions: list[str] = typer.Option(None, "--precision"),
    scales: list[int] = typer.Option(None, "--scale"),
    worlds: list[int] = typer.Option(None, "--world"),
    batch: bool = False,
    resume: bool = False,
    also_done: list[Path] = typer.Option(None, "--also-done"),
):
    """Run every cell in the outer product, reference cells included.

    Worlds above the visible GPU count cannot run under NCCL (one device per rank
    per communicator), so the w5-8 tier must be driven with all GPUs visible and
    cannot share the node with another lane; min_world splits the tiers. --gpus
    runs that many single-GPU cells in flight, one device each; distributed cells
    always run alone with every GPU visible. --scale multiplies the drawn problem
    (the parameter-count axis); --world keeps only those world sizes.
    """
    names = optimizers or list(COMPOSITIONS)
    cells = enumerate_cells(max_world, names, precisions or list(PRECISIONS), min_world, scales or [1])
    if worlds:
        cells = [cell for cell in cells if cell["world"] in set(worlds) or cell["topology"] == "default"]
    if shard:
        index, total = (int(value) for value in shard.split("/"))
        if total < 1 or not 0 <= index < total:
            raise SystemExit(f"shard must be i/N with 0 <= i < N, got {shard}")
        cells = [cell for position, cell in enumerate(cells) if position % total == index]
        if not cells:
            raise SystemExit(f"shard {shard} selected no cells")
    carried = []
    if resume:
        cells, carried = _resume_cells(cells, measure, also_done, seed)
    if min_world > 1 or worlds or shard:
        cells = _with_references(cells)
    cells.sort(key=_cell_key)
    raise SystemExit(_drive(cells, seed, measure, timeout, fresh_processes, gpus, batch, carried))


@app.command()
def pairs(
    seed: int = 0,
    measure: Path = typer.Option(None),
    timeout: int = 1200,
    gpus: int = 1,
    precisions: list[str] = typer.Option(["fp32"], "--precision"),
    atoms: list[str] = typer.Option(None, "--atom"),
    order: int = 2,
    batch: bool = False,
    resume: bool = False,
    also_done: list[Path] = typer.Option(None, "--also-done"),
):
    """The composition tier: every chainable atom alone and every ordered pair
    (or --order N tuple), judged by the fusion and precision laws against the
    same chain run eagerly. This tests the composable primitive set -- the
    product -- not the shipped facade recipes."""
    chosen = atoms or list(_ATOMS)
    names = []
    for depth in range(1, order + 1):
        names.extend("chain:" + "+".join(combo) for combo in itertools.product(chosen, repeat=depth))
    cells = [
        {"topology": "default", "world": 1, "name": name, "precision": precision, "scale": 1}
        for name in names
        for precision in (precisions or ["fp32"])
    ]
    cells.sort(key=_cell_key)
    print(f"{len(cells)} composition cells ({len(chosen)} atoms, order {order})", file=sys.stderr)
    raise SystemExit(_drive(cells, seed, measure, timeout, 1, gpus))


@app.command()
def report(measure: Path = typer.Option(...)):
    """The two lists: every config where we are not more accurate or not faster."""
    raise SystemExit(_report(measure))


def _verdict(row: dict | None) -> str:
    """One cell of the grid: the worst gate it loses, or that it crashed."""
    if row is None:
        return "missing"
    if row["status"] in ("error", "unsupported"):
        return "crashed"
    loses_accuracy = row.get("err_ours") is not None and _less_accurate(row)
    loses_speed = row.get("step_ms") is not None and row.get("baseline_step_ms") is not None and _slower(row)
    if loses_accuracy and loses_speed:
        return "both"
    if loses_accuracy:
        return "accuracy"
    if loses_speed:
        return "speed"
    return "ok"


@app.command()
def plots(measure: Path = typer.Option(...)):
    """Render the default-topology tier as heatmaps: the verdict grid, the
    accuracy ratio against its bound, and the speed ratio against the baseline.

    Values above 1.0 (log scale) lose; below win. Cells without an upstream
    comparison are hatched gray; crashes are black."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    all_rows = [row for row in _read_rows([measure]) if row["cell"]["topology"] == "default"]
    names = sorted({row["cell"]["name"] for row in all_rows})
    precisions = [p for p in PRECISIONS if any(row["cell"]["precision"] == p for row in all_rows)]
    scales = sorted({row["cell"].get("scale", 1) for row in all_rows})

    verdict_cmap = ListedColormap(["#2e7d32", "#ef6c00", "#3949ab", "#b71c1c", "#111111", "#cfd8dc"])
    verdict_codes = {"ok": 0, "accuracy": 1, "speed": 2, "both": 3, "crashed": 4, "missing": 5}
    for scale in scales:
        rows = {_cell_key(r["cell"]): r for r in all_rows if r["cell"].get("scale", 1) == scale}
        verdict_grid = np.zeros((len(names), len(precisions)), dtype=int)
        accuracy_ratio = np.full((len(names), len(precisions)), np.nan)
        speed_ratio = np.full((len(names), len(precisions)), np.nan)
        for i, name in enumerate(names):
            for j, precision in enumerate(precisions):
                row = rows.get(
                    _cell_key({"name": name, "precision": precision, "topology": "default", "world": 1, "scale": scale})
                )
                verdict_grid[i, j] = verdict_codes[_verdict(row)]
                if row is not None and row.get("err_ours") is not None:
                    accuracy_ratio[i, j] = _err_ratio(row)
                if row is not None and row.get("baseline_step_ms"):
                    speed_ratio[i, j] = row["step_ms"] / row["baseline_step_ms"]

        figure, axes = plt.subplots(
            1, 3, figsize=(13, 0.32 * len(names) + 2.4), gridspec_kw={"width_ratios": [1, 1.4, 1.4]}, sharey=True
        )
        figure.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.06, wspace=0.12)

        axes[0].imshow(verdict_grid, cmap=verdict_cmap, aspect="auto", vmin=0, vmax=5)
        axes[0].set_title(f"verdict (scale x{scale})", fontsize=10, loc="left")
        axes[0].set_xticks(range(len(precisions)), precisions)
        axes[0].set_yticks(range(len(names)), names, fontsize=7)
        axes[0].legend(
            handles=[Patch(color=verdict_cmap.colors[code], label=label) for label, code in verdict_codes.items()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            fontsize=7,
            frameon=False,
        )

        for ax, grid, title in (
            (axes[1], accuracy_ratio, "accuracy:  err_ours / max(err_ref, err_semantic)"),
            (axes[2], speed_ratio, "speed:  step_ms / baseline_step_ms"),
        ):
            finite = grid[np.isfinite(grid)]
            norm = TwoSlopeNorm(
                vcenter=1.0,
                vmin=min(0.25, finite.min() if finite.size else 0.25),
                vmax=max(4.0, finite.max() if finite.size else 4.0),
            )
            ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_xticks(range(len(precisions)), precisions)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.5)

        path = measure.parent / (f"matrix_x{scale}.png" if len(scales) > 1 else "matrix.png")
        figure.savefig(path, dpi=220)
        plt.close(figure)
        print(f"{len(rows)} default-topology cells at scale x{scale} -> {path}")


@app.command("ensure-references")
def ensure_references():
    """Clone every pinned upstream into references/ at its recorded commit.

    references/ is untracked by design, so a fresh checkout (a CI lane, a GPU
    box) provisions its own oracles with this before running cells."""
    seen = {}
    for baseline in BASELINES.values():
        if baseline.repo != "torch":
            seen[baseline.repo] = (baseline.url, baseline.pin)
    for repo, (url, pin) in sorted(seen.items()):
        target = REFERENCES / repo
        if not (target / ".git").is_dir():
            subprocess.run(["git", "clone", url, str(target)], check=True)
        checked = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if checked != pin:
            subprocess.run(["git", "-C", str(target), "checkout", pin], check=True)
            print(f"{repo}: {checked} -> {pin}")
        else:
            print(f"{repo}: {pin} (pinned)")


@app.command()
def batch(
    manifest: Path = typer.Option(...),
    measure: Path = typer.Option(...),
    seed: int = 0,
):
    """Internal: run one manifest of world-1 cells on one GPU, in-process."""
    raise SystemExit(_batch_main(manifest, measure, seed))


@app.command()
def worker():
    """Execute the cell named by HEAVYBALL_MATRIX_CELL under torchrun."""
    raise SystemExit(_worker_main())


if __name__ == "__main__":
    app()
