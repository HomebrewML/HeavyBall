"""Batched gradient-whitening PSGD-Kron for slab-native HeavyBall.

Axes up to ``max_size_triangular`` use full triangular factors; larger axes
use diagonal factors.  Triangular-line packing, factor caching, and true-HVP
modes remain deliberately out of scope.
"""

import torch
from torch import Tensor

from .core import Recipe, RefreshCadence
from .matrix import merge_matrix_transform, merged_matrix_shape
from .numerics import _wide, balance_factors, broadcast_leaf, stable_l2_normalize
from .transforms import WHOLE, Tempo, sgd_commit


def _max_singular_value_power_iter(value: Tensor, power_iterations: int) -> Tensor:
    """Legacy power iteration, batched over the slab axis."""

    scale = value.abs().amax(dim=(-2, -1))
    scaled = value / torch.where(scale != 0, scale, torch.ones_like(scale)).unsqueeze(-1).unsqueeze(-1)
    count = min(2, value.shape[-2])
    indices = scaled.norm(dim=-1).topk(count, dim=-1).indices
    vectors = scaled.gather(-2, indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
    vectors = stable_l2_normalize(vectors, dim=-1, eps=None)
    # The compiled loop requires its carried vector layout to remain stable.
    vectors = vectors.mT.contiguous().mT
    transpose = scaled.mH.contiguous()

    def multiply(vector: Tensor) -> Tensor:
        return (transpose @ (scaled @ vector.mT)).mT

    for _ in range(power_iterations):
        vectors = stable_l2_normalize(multiply(vectors), dim=-1, eps=None)
    return (vectors.conj() * multiply(vectors)).sum(dim=-1).real.clamp_min(0).sqrt().amax(dim=-1) * scale


def _next_lower_bound(ell: Tensor, lower_bound: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """Running spectral lower bound, weighting HISTORY by beta so the safety floor decays slowly.

    Li's psgd_torch is ``max(ell, beta*L + (1-beta)*ell)`` (default betaL=0.9 -> 90% history). PyTorch
    ``L.lerp(ell, w) = (1-w)*L + w*ell``, so the history weight beta means a sample weight ``1 - beta``.
    """

    dtype = ell.dtype
    compute_dtype = torch.promote_types(torch.promote_types(dtype, lower_bound.dtype), torch.float32)
    ell = ell.to(compute_dtype)
    lower_bound = lower_bound.to(compute_dtype)
    smoothed = lower_bound.lerp(ell, 1 - beta)
    smoothed = torch.where(torch.isfinite(smoothed), smoothed, 0)
    next_bound = ell.maximum(smoothed)
    return next_bound.to(dtype), next_bound


def _balance_q(q0: Tensor, q1: Tensor) -> tuple[Tensor, Tensor]:
    """Balance the two factor maxima exactly as PSGD's full-Q path does."""

    q0, q1 = balance_factors([q0, q1])
    return q0, q1


def _balance_mixed_q(q0: Tensor, q1: Tensor) -> tuple[Tensor, Tensor]:
    """Balance factors whose state shapes encode diagonal/triangular topology."""

    q0, q1 = balance_factors([q0, q1])
    return q0, q1


def _calc_a_and_conjb(hessian_vector: Tensor, q0: Tensor, q1: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
    """Form legacy PSGD's ``A`` and inverse-triangular whitened probe."""

    # Pairwise contraction (a, then b): the 3-operand einsum makes inductor materialize the
    # [n,i,a,j,b] outer product (O(n^4) -> OOM at >=256x256); pairwise keeps every step O(n^2).
    projected = torch.einsum("nia,nab->nib", q0, hessian_vector)
    a = torch.einsum("nib,njb->nij", projected, q1).contiguous()
    probe = vector
    members = probe.shape[0]

    moved = probe.movedim(1, -1).contiguous()
    flat = moved.reshape(members, -1, moved.shape[-1]).to(torch.promote_types(q0.dtype, moved.dtype))
    flat = torch.linalg.solve_triangular(q0, flat, upper=True, left=False)
    probe = flat.reshape(moved.shape).movedim(-1, 1).contiguous()

    moved = probe.movedim(2, -1).contiguous()
    flat = moved.reshape(members, -1, moved.shape[-1]).to(torch.promote_types(q1.dtype, moved.dtype))
    flat = torch.linalg.solve_triangular(q1, flat, upper=True, left=False)
    probe = flat.reshape(moved.shape).movedim(-1, 2).contiguous()
    return a, probe


def _calc_mixed_a_and_conjb(
    hessian_vector: Tensor, q0: Tensor, q1: Tensor, vector: Tensor
) -> tuple[Tensor, Tensor]:
    """Form legacy PSGD's ``A`` and inverse-Q probe for mixed factor topology."""

    if q0.ndim == 2 and q1.ndim == 3:
        scaled = q0.unsqueeze(-1) * hessian_vector
        a = torch.einsum("...ab,...Bb->...aB", scaled, q1).contiguous()
    elif q0.ndim == 3 and q1.ndim == 2:
        scaled = q1.unsqueeze(-2) * hessian_vector
        a = torch.einsum("...Aa,...ab->...Ab", q0, scaled).contiguous()
    else:
        a = torch.einsum("...a,...b,...ab->...ab", q0, q1, hessian_vector).contiguous()

    probe = vector
    if q0.ndim == 2:
        probe = probe / q0.unsqueeze(-1)
    if q1.ndim == 2:
        probe = probe / q1.unsqueeze(-2)

    members = probe.shape[0]
    if q0.ndim == 3:
        moved = probe.movedim(1, -1).contiguous()
        flat = moved.reshape(members, -1, moved.shape[-1]).to(torch.promote_types(q0.dtype, moved.dtype))
        flat = torch.linalg.solve_triangular(q0, flat, upper=True, left=False)
        probe = flat.reshape(moved.shape).movedim(-1, 1).contiguous()
    if q1.ndim == 3:
        moved = probe.movedim(2, -1).contiguous()
        flat = moved.reshape(members, -1, moved.shape[-1]).to(torch.promote_types(q1.dtype, moved.dtype))
        flat = torch.linalg.solve_triangular(q1, flat, upper=True, left=False)
        probe = flat.reshape(moved.shape).movedim(-1, 2).contiguous()
    return a, probe


def _refresh_q(
    update: Tensor,
    q0: Tensor,
    q1: Tensor,
    lower0: Tensor,
    lower1: Tensor,
    tempo: Tempo,
    power_iterations: int,
    vector: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One gradient-whitening PSGD Q update for a full two-factor slab."""

    damping = tempo.hyper.dampening + torch.finfo(update.dtype).eps * update.abs()
    hessian_vector = update + damping * vector

    q0, q1 = _balance_q(q0, q1)
    a, conjb = _calc_a_and_conjb(hessian_vector, q0, q1, vector)
    term1_0 = torch.einsum("nab,ncb->nac", a, a)
    term2_0 = torch.einsum("nab,ncb->nac", conjb, conjb)
    ell0, lower0 = _next_lower_bound(
        _max_singular_value_power_iter(term1_0 + term2_0, power_iterations), lower0, tempo.hyper.lower_bound_beta
    )
    q0_update = (term1_0 - term2_0).triu() @ q0
    q0 = q0 - q0_update / broadcast_leaf(ell0, q0_update) * tempo.hyper.precond_lr

    term1_1 = torch.einsum("nab,nac->nbc", a, a)
    term2_1 = torch.einsum("nab,nac->nbc", conjb, conjb)
    ell1, lower1 = _next_lower_bound(
        _max_singular_value_power_iter(term1_1 + term2_1, power_iterations), lower1, tempo.hyper.lower_bound_beta
    )
    q1_update = (term1_1 - term2_1).triu() @ q1
    q1 = q1 - q1_update / broadcast_leaf(ell1, q1_update) * tempo.hyper.precond_lr
    return q0, q1, lower0, lower1


def _refresh_mixed_q(
    update: Tensor,
    q0: Tensor,
    q1: Tensor,
    lower0: Tensor,
    lower1: Tensor,
    tempo: Tempo,
    power_iterations: int,
    vector: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One legacy-equivalent Q update with at least one diagonal factor."""

    damping = tempo.hyper.dampening + torch.finfo(update.dtype).eps * update.abs()
    hessian_vector = update + damping * vector

    q0, q1 = _balance_mixed_q(q0, q1)
    a, conjb = _calc_mixed_a_and_conjb(hessian_vector, q0, q1, vector)
    if q0.ndim == 2:
        term1_0 = torch.einsum("...ab,...ab->...a", a, a)
        term2_0 = torch.einsum("...ab,...ab->...a", conjb, conjb)
        ell0, lower0 = _next_lower_bound((term1_0 + term2_0).amax(dim=-1), lower0, tempo.hyper.lower_bound_beta)
        q0_update = q0 * (term1_0 - term2_0)
    else:
        term1_0 = torch.einsum("nab,ncb->nac", a, a)
        term2_0 = torch.einsum("nab,ncb->nac", conjb, conjb)
        ell0, lower0 = _next_lower_bound(
            _max_singular_value_power_iter(term1_0 + term2_0, power_iterations),
            lower0,
            tempo.hyper.lower_bound_beta,
        )
        q0_update = (term1_0 - term2_0).triu() @ q0
    q0 = q0 - q0_update / broadcast_leaf(ell0, q0_update) * tempo.hyper.precond_lr

    if q1.ndim == 2:
        term1_1 = torch.einsum("...ab,...ab->...b", a, a)
        term2_1 = torch.einsum("...ab,...ab->...b", conjb, conjb)
        ell1, lower1 = _next_lower_bound((term1_1 + term2_1).amax(dim=-1), lower1, tempo.hyper.lower_bound_beta)
        q1_update = q1 * (term1_1 - term2_1)
    else:
        term1_1 = torch.einsum("nab,nac->nbc", a, a)
        term2_1 = torch.einsum("nab,nac->nbc", conjb, conjb)
        ell1, lower1 = _next_lower_bound(
            _max_singular_value_power_iter(term1_1 + term2_1, power_iterations),
            lower1,
            tempo.hyper.lower_bound_beta,
        )
        q1_update = (term1_1 - term2_1).triu() @ q1
    q1 = q1 - q1_update / broadcast_leaf(ell1, q1_update) * tempo.hyper.precond_lr
    return q0, q1, lower0, lower1


def _precondition(update: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Apply every Kronecker factor twice, i.e. ``P = QᵀQ``."""

    # Pairwise: the 5-operand einsum makes torch/inductor materialize an O(n^4) intermediate
    # (OOM at >=256x256). Contract P0 = Q0^T Q0 and P1 = Q1^T Q1, then P0 @ update @ P1 -- all O(n^2).
    p0 = torch.einsum("nri,nra->nia", q0, q0)
    p1 = torch.einsum("nsj,nsb->njb", q1, q1)
    projected = torch.einsum("nia,nab->nib", p0, update)
    return torch.einsum("nib,njb->nij", projected, p1)


def _precondition_mixed(update: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Apply mixed diagonal/triangular factors twice, exactly as legacy does."""

    if q0.ndim == 2 and q1.ndim == 3:
        p1 = torch.einsum("...oB,...ob->...Bb", q1, q1)
        scaled = q0.square().unsqueeze(-1) * update
        return torch.einsum("...ab,...Bb->...aB", scaled, p1)
    if q0.ndim == 3 and q1.ndim == 2:
        p0 = torch.einsum("...nA,...na->...Aa", q0, q0)
        scaled = q1.square().unsqueeze(-2) * update
        return torch.einsum("...Aa,...ab->...Ab", p0, scaled)
    return torch.einsum("...a,...a,...b,...b,...ab->...ab", q0, q0, q1, q1, update)


def psgd_kron_init(ref_leaf: Tensor, *, max_size_triangular: Tensor | int = 2048) -> dict[str, Tensor]:
    """Allocate diagonal or triangular Q state in merged matrix coordinates."""

    limit = int(max_size_triangular)
    merged = merged_matrix_shape(tuple(ref_leaf.shape), limit)
    if len(merged) != 2:
        raise ValueError(f"kron requires a leaf whose dimensions merge to 2D at max_size_triangular={limit}")
    rows, columns = merged
    if rows == 0 or columns == 0:
        raise ValueError("kron requires nonempty merged 2D parameter leaves")
    ref = _wide(ref_leaf).reshape(rows, columns)

    def factor(size: int) -> Tensor:
        if size > limit:
            return torch.ones(size, dtype=ref.dtype, device=ref.device)
        return torch.eye(size, dtype=ref.dtype, device=ref.device)

    return {
        "Q_0": factor(rows),
        "Q_1": factor(columns),
        "running_lower_bound_0": torch.zeros((), dtype=torch.float64, device=ref.device),
        "running_lower_bound_1": torch.zeros((), dtype=torch.float64, device=ref.device),
    }


def make_psgd_kron(power_iterations: int = 2):
    """Build a PSGD-Kron transform with a trace-time power-iteration count."""

    if type(power_iterations) is not int:
        raise TypeError("power_iterations must be a Python int")

    def psgd_kron(update: Tensor, obs, param: Tensor, state: dict[str, Tensor], tempo: Tempo):
        """Apply PSGD-Kron; only the host-selected refresh artifact updates Q."""

        del obs, param
        q0 = _wide(state["Q_0"])
        q1 = _wide(state["Q_1"])
        lower0 = _wide(state["running_lower_bound_0"])
        lower1 = _wide(state["running_lower_bound_1"])
        all_triangular = q0.ndim == 3 and q1.ndim == 3
        if tempo.refresh:
            vector = tempo.randn_like(update)
            if all_triangular:
                q0, q1, lower0, lower1 = _refresh_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations, vector
                )
            else:
                q0, q1, lower0, lower1 = _refresh_mixed_q(
                    update, q0, q1, lower0, lower1, tempo, power_iterations, vector
                )
        preconditioned = _precondition(update, q0, q1) if all_triangular else _precondition_mixed(update, q0, q1)
        return preconditioned, {
            "Q_0": q0,
            "Q_1": q1,
            "running_lower_bound_0": lower0,
            "running_lower_bound_1": lower1,
        }, tempo.live

    psgd_kron.distributed_scope = WHOLE
    psgd_kron = merge_matrix_transform(psgd_kron)
    psgd_kron.init = psgd_kron_init
    psgd_kron.state_init_hyper = ("max_size_triangular",)
    psgd_kron.config = {"power_iterations": power_iterations}
    return psgd_kron


psgd_kron = make_psgd_kron()


KronCadence = RefreshCadence


kron = Recipe(
    chain=(make_psgd_kron(2),),
    commit=sgd_commit,
    defaults=dict(
        lr=1e-3,
        preconditioner_update_probability=1.0,
        precond_lr=0.1,
        lower_bound_beta=0.9,
        dampening=1e-9,
        max_size_triangular=2048,
        weight_decay=0.0,
    ),
)

__all__ = ["KronCadence", "RefreshCadence", "kron", "make_psgd_kron", "psgd_kron", "psgd_kron_init"]
