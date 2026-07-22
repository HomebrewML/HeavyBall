"""Host-side programs that sequence compiled HeavyBall object-steps."""

from types import SimpleNamespace
from typing import Callable

import torch
from torch import Tensor

from .core import Engine
from .numerics import _wide


# Kept local: Program requires 0-d tensors, unlike Engine's one-element coercion.
def _scalar(value: float | Tensor, reference: Tensor) -> Tensor:
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError("program hyperparameters must be 0-d tensors or Python scalars")
        return value.detach().to(device=reference.device, dtype=dtype)
    return torch.tensor(value, device=reference.device, dtype=dtype)


def _sum_squares(value: Tensor) -> Tensor:
    return value.double().square().sum()


class Program:
    """A host-side controller around one slab-backed :class:`Engine`."""

    def __init__(self, base: Engine) -> None:
        if not isinstance(base, Engine):
            raise TypeError("Program requires an Engine base")
        self.base = base
        self.params = base.params

    @torch.no_grad()
    def zero_grad(self, *, set_to_none: bool = False) -> None:
        self.base.zero_grad(set_to_none=set_to_none)


class SAM(Program):
    """Sharpness-aware minimization as two compiled phases around two closures."""

    def __init__(self, base: Engine, *, rho: float | Tensor = 0.05, eps: float | Tensor = 1e-12) -> None:
        super().__init__(base)
        if len({group.param_slab.device for group in base.groups}) != 1:
            raise ValueError("SAM requires parameters on one device for its global perturbation norm")
        if hasattr(base, "_fsdp2_manifest"):
            raise ValueError(
                "SAM perturbation norm reduces only the local FSDP2 shard, producing "
                "an incorrect global perturbation scale; SAM is not supported with FSDP2 engines"
            )

        reference = base.groups[0].param_slab
        self.hyper = SimpleNamespace(rho=_scalar(rho, reference), eps=_scalar(eps, reference))
        self.rho, self.eps = self.hyper.rho, self.hyper.eps
        # Low-precision addition and subtraction do not necessarily recover the original slab.
        self._original = tuple(torch.empty_like(group.param_slab) for group in base.groups)
        self.compiled_phases = {
            "perturb": self._compile_perturb(),
            "restore": self._compile_restore(),
        }
        self.compiled_perturb = self.compiled_phases["perturb"]
        self.compiled_restore = self.compiled_phases["restore"]

    def _compile_perturb(self):
        plans = tuple(
            (group.param_slab, group.grad_slab, tuple(vars(group.observations).values()), original)
            for group, original in zip(self.base.groups, self._original, strict=True)
        )
        rho, eps = self.hyper.rho, self.hyper.eps

        def whole_step():
            squared = [_sum_squares(grad_slab) for _, grad_slab, _, _ in plans]
            total = squared[0]
            for value in squared[1:]:
                total = total + value
            scale = rho / (total.sqrt() + eps)
            for param_slab, grad_slab, observation_slabs, original in plans:
                original.copy_(param_slab)
                e = (_wide(grad_slab) * scale).to(param_slab.dtype)
                param_slab.add_(e)
                for slab in observation_slabs:
                    slab.zero_()

        return torch.compile(whole_step, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")

    def _compile_restore(self):
        plans = tuple(
            (group.param_slab, original)
            for group, original in zip(self.base.groups, self._original, strict=True)
        )

        def whole_step():
            for param_slab, original in plans:
                param_slab.copy_(original)

        return torch.compile(whole_step, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """Evaluate at the current and perturbed parameters, then commit the base step."""

        if closure is None:
            raise ValueError("SAM requires closure")
        for group in self.base.groups:
            group.observed.fill_(True)
        with torch.enable_grad():
            loss = closure()
        self.compiled_perturb()
        self.base.zero_grad()
        try:
            with torch.enable_grad():
                closure()
        finally:
            self.compiled_restore()
        self.base.step()
        return loss
