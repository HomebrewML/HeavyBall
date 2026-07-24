"""Pure slab transforms for the HeavyBall 4.0 optimizer core."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard

from .numerics import (
    _caution,
    _wide,
    broadcast_leaf,
    stable_l2_normalize,
    stochastic_round_bfloat16,
)


@dataclass(frozen=True)
class Whole:
    inputs: tuple[str, ...] = ("update",)

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if not inputs or inputs[0] != "update" or len(set(inputs)) != len(inputs):
            raise ValueError("whole distributed scope inputs must start with one unique 'update'")
        if any(not isinstance(value, str) for value in inputs):
            raise TypeError("whole distributed scope inputs must be strings")
        object.__setattr__(self, "inputs", inputs)


WHOLE = Whole()
SHARD = "shard"
_STATE_PHILOX_ROUNDS = 3


class Tempo(NamedTuple):
    step: Tensor
    age: Tensor
    live: Tensor
    hyper: object
    refresh: bool
    raw_grad: Tensor | None = None
    logical_shape: tuple[int, ...] | None = None
    base_seed: Tensor | None = None
    leaf_indices: Tensor | None = None
    element_offset: int = 0
    rounds: int = 7

    def random_like(self, value: Tensor, *, _stream: int = 0) -> Tensor:
        """Return stateless Philox uniform noise keyed per logical leaf and age."""

        if isinstance(value, DTensor) and self.base_seed is not None:
            if len(value.placements) != 1:
                raise ValueError("Tempo.random_like supports one-dimensional DTensor placements")
            placement = value.placements[0]
            local = value.to_local()
            if isinstance(placement, Shard):
                if placement.dim != 0:
                    raise ValueError("Tempo.random_like supports DTensor sharding over the leaf axis")
                mesh = value.device_mesh
                _, leaf_offset = placement.local_shard_size_and_offset(
                    value.shape[0], mesh.size(0), mesh.get_local_rank(0)
                )
                local_tempo = self._replace(
                    age=self.age.narrow(0, leaf_offset, local.shape[0]),
                    leaf_indices=self.leaf_indices.narrow(0, leaf_offset, local.shape[0]),
                    element_offset=0,
                )
            elif isinstance(placement, Replicate):
                local_tempo = self._replace(element_offset=0)
            else:
                raise ValueError("Tempo.random_like does not support partial DTensor placements")
            local_random = local_tempo.random_like(local, _stream=_stream)
            return DTensor.from_local(
                local_random,
                value.device_mesh,
                value.placements,
                run_check=False,
                shape=value.shape,
                stride=value.stride(),
            )

        mask = 0xFFFFFFFF
        round0, round1 = 0xD2511F53, 0xCD9E8D57
        bump0, bump1 = 0x9E3779B9, 0xBB67AE85

        def mulhilo32(word: Tensor, multiplier: int) -> tuple[Tensor, Tensor]:
            word_low = word.bitwise_and(0xFFFF)
            word_high = word.bitwise_right_shift(16)
            multiplier_low = multiplier & 0xFFFF
            multiplier_high = multiplier >> 16
            low_product = word_low * multiplier_low
            middle0 = word_low * multiplier_high
            middle1 = word_high * multiplier_low
            carry = (
                low_product.bitwise_right_shift(16)
                + middle0.bitwise_and(0xFFFF)
                + middle1.bitwise_and(0xFFFF)
            )
            low = low_product.bitwise_and(0xFFFF).bitwise_or(
                carry.bitwise_and(0xFFFF).bitwise_left_shift(16)
            )
            high = (
                word_high * multiplier_high
                + middle0.bitwise_right_shift(16)
                + middle1.bitwise_right_shift(16)
                + carry.bitwise_right_shift(16)
            ).bitwise_and(mask)
            return high, low

        base_seed = self.base_seed
        if base_seed is None:
            return torch.randint_like(value, low=0, high=1 << 16, dtype=torch.int32).float() / (1 << 16)
        leaf_indices = self.leaf_indices
        if leaf_indices is None:
            leaf_indices = torch.arange(value.shape[0], dtype=torch.int64, device=value.device)

        elements_per_leaf = math.prod(value.shape[1:])
        offsets = (
            torch.arange(elements_per_leaf, dtype=torch.int64, device=value.device)
            + self.element_offset
        ).reshape(1, -1)
        counter0 = offsets.bitwise_and(mask)
        counter1 = offsets.bitwise_right_shift(32).bitwise_and(mask)
        counter2 = self.age.reshape(-1, 1).bitwise_and(mask)
        counter3 = self.age.reshape(-1, 1).bitwise_right_shift(32).bitwise_and(mask)

        leaf = leaf_indices.reshape(-1, 1)
        leaf_low = leaf.bitwise_and(mask)
        leaf_high = leaf.bitwise_right_shift(32).bitwise_and(mask)
        _, folded0 = mulhilo32(leaf_low, bump0)
        _, folded1_low = mulhilo32(leaf_low, bump1)
        _, folded1_high = mulhilo32(leaf_high, bump0)
        key0 = (base_seed[0] + folded0).bitwise_and(mask)
        key1 = (base_seed[1] + folded1_low + folded1_high).bitwise_and(mask)

        # The default parameter path uses Philox-4x32-7 (BigCrush-passing). 10 unrolled rounds spill
        # (~1ms/round of spill traffic: 10->11.6, 9->10.4, 8->9.1, 7->8.6 ms/step on 151M bf16 Adam);
        # 7 rounds fit, so stochastic rounding is nearly free (8.6 vs 8.58 no-SR) and precision vs the
        # fp64 ideal is unchanged (err 0.259 -> 0.268; torch bf16 is 4.17).
        for _ in range(self.rounds):
            high0, low0 = mulhilo32(counter0, round0)
            high1, low1 = mulhilo32(counter2, round1)
            counter0, counter1, counter2, counter3 = (
                high1.bitwise_xor(counter1).bitwise_xor(key0),
                low1,
                high0.bitwise_xor(counter3).bitwise_xor(key1),
                low0,
            )
            key0 = (key0 + bump0).bitwise_and(mask)
            key1 = (key1 + bump1).bitwise_and(mask)

        # The upper 24 bits convert exactly to float32 values in [0, 1).
        output = counter0 if _stream == 0 else counter1
        return (output.bitwise_right_shift(8).float() * (2.0**-24)).reshape(value.shape)

    def randn_like(self, value: Tensor) -> Tensor:
        """Return stateless Gaussian noise keyed per logical leaf and age."""

        uniform0 = self.random_like(value)
        uniform1 = self.random_like(value, _stream=1)
        radius = (-2.0 * torch.log1p(-uniform0)).sqrt()
        gaussian = radius * torch.cos((2.0 * math.pi) * uniform1)
        return gaussian.to(value.dtype)


def _eigh_regularized_decomposition(
    gram: Tensor, eps: Tensor, exponent: float
) -> tuple[Tensor, Tensor]:
    """Return eigenvectors and a regularized root with relative null-space classification."""

    regularized = gram * 0.5 + gram.mH * 0.5 + eps * torch.eye(  # halve before adding: no overflow near fp32 max
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    values, vectors = torch.linalg.eigh(regularized)
    signal = (values - eps).clamp_min(0)
    maximum = signal.amax(dim=-1, keepdim=True)
    roundoff = gram.shape[-1] * torch.finfo(gram.dtype).eps * maximum
    rooted = values.clamp_min(eps).pow(exponent)
    null_root = torch.as_tensor(eps, dtype=values.dtype, device=values.device).pow(exponent)
    rooted = torch.where(signal <= roundoff, null_root, rooted)
    return vectors, rooted


def _eigh_scaled_gram_decomposition(
    gram: Tensor, scale: Tensor, eps: Tensor, exponent: float
) -> tuple[Tensor, Tensor]:
    """Decompose ``gram * scale**2`` without materializing its large entries.

    ``scale == 1`` follows the original fp32 refresh exactly.  At large scales the
    absolute regularizer may underflow in normalized coordinates, so null modes are
    classified relatively and receive the physical ``eps**exponent`` root directly.
    """

    scale = scale.abs().to(gram.dtype)
    broadcast_scale = broadcast_leaf(scale, gram)
    epsilon = torch.as_tensor(eps, dtype=gram.dtype, device=gram.device)
    normalized_epsilon = (
        epsilon.double() / broadcast_scale.double().square()
    ).to(gram.dtype)
    regularized = gram * 0.5 + gram.mH * 0.5 + normalized_epsilon * torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    values, vectors = torch.linalg.eigh(regularized)
    normalized_epsilon = normalized_epsilon[..., 0, 0].unsqueeze(-1)
    signal = (values - normalized_epsilon).clamp_min(0)
    maximum = signal.amax(dim=-1, keepdim=True)
    roundoff = gram.shape[-1] * torch.finfo(gram.dtype).eps * maximum
    physical_scale = scale.unsqueeze(-1).pow(2 * exponent)
    rooted = values.clamp_min(torch.finfo(values.dtype).tiny).pow(exponent) * physical_scale
    null_root = epsilon.pow(exponent)
    rooted = torch.where(signal <= roundoff, null_root, rooted)
    return vectors, rooted


def _eigh_regularized_root(gram: Tensor, eps: Tensor, exponent: float) -> Tensor:
    """Materialize a symmetric regularized matrix root."""

    vectors, rooted = _eigh_regularized_decomposition(gram, eps, exponent)
    return (vectors * rooted.unsqueeze(-2)) @ vectors.mT


def _spectral_root_application(
    value: Tensor,
    vectors: Tensor,
    rooted: Tensor,
    *,
    left: bool,
) -> Tensor:
    """Apply a matrix root in spectral coordinates without materializing ill-conditioned sums."""

    value = value.to(vectors.dtype)
    if left:
        return vectors @ (rooted.unsqueeze(-1) * (vectors.mT @ value))
    return ((value @ vectors) * rooted.unsqueeze(-2)) @ vectors.mT


def _root_requires_spectral_application(rooted: Tensor, dtype: torch.dtype) -> Tensor:
    """Identify roots whose mode ratio cannot survive materialization in ``dtype``."""

    ratio = rooted.amax(dim=-1) / rooted.amin(dim=-1).clamp_min(torch.finfo(rooted.dtype).tiny)
    return ratio * torch.finfo(dtype).eps > 0.25


def _matrix_inv_sqrt(gram: Tensor, eps: Tensor) -> Tensor:
    """Symmetric inverse square root via eigendecomposition (batched over leading dims)."""

    return _eigh_regularized_root(gram, eps, -0.5)


def beta_debias(beta: Tensor, age: Tensor) -> Tensor:
    age = age.to(beta.dtype)
    log_beta = beta.log()
    value = beta * (-torch.expm1((age - 1) * log_beta)) / (-torch.expm1(age * log_beta))
    value = torch.where(beta == 0, torch.zeros_like(value), value)
    value = torch.where(beta == 1, (age - 1) / age.clamp_min(1), value)
    return torch.where(age <= 1, torch.zeros_like(value), value)


def _second_moment(previous: Tensor, observation: Tensor, beta: Tensor) -> Tensor:
    """Update a stored RMS without forming the observation's square.

    ``previous`` stores ``sqrt(v)`` even though the public state slot retains its
    historical ``exp_avg_sq`` name.  ``hypot`` evaluates the exact real recurrence
    ``sqrt(beta * v + (1 - beta) * observation**2)`` without overflowing for a
    finite observation whose square is not representable in the storage dtype.
    """

    previous = _wide(previous)
    observation = _wide(observation)
    beta = beta.to(previous.dtype).clamp(0, 1)
    return torch.hypot(previous * beta.sqrt(), observation * (1 - beta).sqrt())


def _second_moment_from_squared(
    previous: Tensor, squared_observation: Tensor, beta: Tensor
) -> Tensor:
    """Update a stored RMS when a producer already supplies a squared observation."""

    squared_observation = _wide(squared_observation).clamp_min(0)
    return _second_moment(previous, squared_observation.sqrt(), beta)


def _second_moment_denom(moment: Tensor, eps: Tensor, dtype: torch.dtype) -> Tensor:
    """Return ``sqrt(clamp(v, eps))`` from the stored ``sqrt(v)`` state.

    The square root on ``eps`` is intentional: the corresponding inverse form is
    ``v.clamp_min(eps).rsqrt()``.  Applying ``rsqrt`` to the RMS itself would instead
    compute a fourth root and change the optimizer.
    """

    moment = _wide(moment)
    epsilon = torch.as_tensor(eps, dtype=moment.dtype, device=moment.device).sqrt()
    return moment.clamp_min(epsilon).to(dtype)


def first_moment_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return {"exp_avg": torch.zeros_like(_wide(ref_leaf))}


def first_moment(update, obs, param, state, tempo):
    """Debiased first-moment EMA over the incoming update stream."""

    del obs, param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    exp_avg = _wide(state["exp_avg"]) * beta1 + update * (1 - beta1)
    return exp_avg, {"exp_avg": exp_avg}, tempo.live


first_moment.init = first_moment_init


def adam_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    ref = _wide(ref_leaf)
    return {
        "exp_avg": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
    }


def adam(update, obs, param, state, tempo):
    """Pure bias-corrected Adam direction over a full ``[N, *shape]`` slab."""

    del obs, param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg = _wide(state["exp_avg"]) * beta1 + update * (1 - beta1)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    update = exp_avg / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    return update, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, tempo.live


adam.init = adam_init


def rmsprop_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return {"exp_avg_sq": torch.zeros_like(_wide(ref_leaf))}


def rmsprop(update, obs, param, state, tempo):
    """Debiased RMSprop's second-moment normalization."""

    del obs, param
    update = _wide(update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    update = update / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    return update, {"exp_avg_sq": exp_avg_sq}, tempo.live


rmsprop.init = rmsprop_init


def lion_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return {"exp_avg": torch.zeros_like(_wide(ref_leaf))}


def lion(update, obs, param, state, tempo):
    """Lion's two-rate EMA and sign direction."""

    del obs, param
    update = _wide(update)
    exp_avg = _wide(state["exp_avg"])
    direction = (exp_avg * tempo.hyper.beta1 + update * (1 - tempo.hyper.beta1)).sign()
    exp_avg = exp_avg * tempo.hyper.beta2 + update * (1 - tempo.hyper.beta2)
    return direction, {"exp_avg": exp_avg}, tempo.live


lion.init = lion_init


def laprop_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return adam_init(ref_leaf)


def laprop(update, obs, param, state, tempo):
    """LaProp: normalize before accumulating first-moment momentum."""

    del obs, param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    normalized = update / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    exp_avg = _wide(state["exp_avg"]) * beta1 + normalized * (1 - beta1)
    return exp_avg, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, tempo.live


laprop.init = laprop_init


def nadam_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    ref = _wide(ref_leaf)
    return {
        "exp_avg": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
        "mu_product": torch.ones((), dtype=ref.dtype, device=ref.device),
    }


def nadam(update, obs, param, state, tempo):
    """NAdam with a per-leaf product of scheduled Nesterov momentum."""

    del obs, param
    update = _wide(update)
    exp_avg = _wide(state["exp_avg"]) * tempo.hyper.beta1 + update * (1 - tempo.hyper.beta1)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    denom = _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)

    age = tempo.age.to(tempo.hyper.beta1.dtype)
    base = torch.ones_like(tempo.hyper.beta1) * 0.96
    mu = tempo.hyper.beta1 * (1 - 0.5 * torch.pow(base, age * tempo.hyper.momentum_decay))
    mu_next = tempo.hyper.beta1 * (1 - 0.5 * torch.pow(base, (age + 1) * tempo.hyper.momentum_decay))
    mu_product = _wide(state["mu_product"]) * mu
    one = torch.ones_like(tempo.hyper.beta1)
    grad_weight = broadcast_leaf((one - mu) / (one - mu_product), update)
    avg_weight = broadcast_leaf(mu_next / (one - mu_product * mu_next), update)
    update = update / denom * grad_weight + exp_avg / denom * avg_weight
    return update, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "mu_product": mu_product}, tempo.live


nadam.init = nadam_init


def _ademamix_schedules(tempo) -> tuple[Tensor, Tensor]:
    """Legacy AdEMAMix beta3 half-life and alpha warmups, per leaf age."""

    age = tempo.age.to(tempo.hyper.beta1.dtype)
    alpha_warmup = tempo.hyper.alpha_warmup
    alpha_progress = (age / alpha_warmup.clamp_min(1)).clamp(min=0, max=1)
    alpha = torch.where(
        alpha_warmup > 0,
        tempo.hyper.alpha * alpha_progress,
        torch.ones_like(age) * tempo.hyper.alpha,
    )

    beta3_warmup = tempo.hyper.beta3_warmup
    beta3_progress = (age / beta3_warmup.clamp_min(1)).clamp(min=0, max=1)
    half_life_start = math.log(0.5) / (tempo.hyper.beta1 + 1e-8).log() - 1
    half_life_end = math.log(0.5) / (tempo.hyper.beta3 + 1e-8).log() - 1
    half_life = half_life_start + beta3_progress * (half_life_end - half_life_start)
    warmed_beta3 = torch.exp2(-1 / (half_life + 1)).clamp(min=0, max=1 - 1e-8)
    beta3 = torch.where(beta3_warmup > 0, warmed_beta3, torch.ones_like(age) * tempo.hyper.beta3)
    return beta3, alpha


def ademamix_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    ref = _wide(ref_leaf)
    return {
        "exp_avg_fast": torch.zeros_like(ref),
        "exp_avg_slow": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
    }


def ademamix(update, obs, param, state, tempo):
    """AdEMAMix's fast and slow EMAs with per-leaf warmup schedules."""

    del obs, param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    beta3, alpha = _ademamix_schedules(tempo)
    beta3 = broadcast_leaf(beta3, update)
    alpha = broadcast_leaf(alpha, update)
    exp_avg_fast = _wide(state["exp_avg_fast"]) * beta1 + update * (1 - beta1)
    exp_avg_slow = _wide(state["exp_avg_slow"]) * beta3 + update * (1 - beta3)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    update = (exp_avg_fast + exp_avg_slow * alpha) / _second_moment_denom(
        exp_avg_sq, tempo.hyper.eps, update.dtype
    )
    return update, {
        "exp_avg_fast": exp_avg_fast,
        "exp_avg_slow": exp_avg_slow,
        "exp_avg_sq": exp_avg_sq,
    }, tempo.live


ademamix.init = ademamix_init


def unscaled_adam_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return adam_init(ref_leaf)


def unscaled_adam(update, obs, param, state, tempo):
    """Adam with first-moment accumulation in variance-normalized coordinates."""

    del obs, param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment(state["exp_avg_sq"], update, beta2)
    denom = _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    normalized = update / denom
    exp_avg = _wide(state["exp_avg"]) * beta1 + normalized * (1 - beta1)
    return exp_avg * denom, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, tempo.live


unscaled_adam.init = unscaled_adam_init


def sign_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def sign(update, obs, param, state, tempo):
    """Replace the inbound update with its elementwise sign."""

    del obs, param, state
    return _wide(update).sign(), {}, tempo.live


sign.init = sign_init


def mars_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return {"mars_old_grad": torch.zeros_like(_wide(ref_leaf))}


def mars(update, obs, param, state, tempo):
    """Apply MARS's variance-reduction correction, clipped to unit L2 norm, before the Adam moments."""

    del obs, param
    update = _wide(update)
    old_grad = _wide(state["mars_old_grad"])
    a = -tempo.hyper.mars_gamma * tempo.hyper.beta1 / (1 - tempo.hyper.beta1)
    corrected = update + a * (old_grad - update)
    if corrected.numel():
        # MARS (Algorithm 2) clips the corrected gradient to unit L2 norm per leaf before both moments.
        flat, scale, norm = _slab_l2_components(corrected)
        corrected = (flat / (scale * norm).clamp_min(1.0)).reshape_as(corrected)
    return corrected, {"mars_old_grad": update}, tempo.live


mars.init = mars_init
mars.distributed_shard_separable = False


def _slab_l2_components(value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return flattened per-leaf stable L2 components for a full slab."""

    flat = value.reshape(value.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True)
    safe = torch.where(scale != 0, scale, torch.ones_like(scale))
    norm = torch.linalg.vector_norm(flat / safe, dim=1, keepdim=True)
    return flat, scale, norm


def sign_graft(update, obs, param, state, tempo):
    """Replace updates with signs grafted to their per-leaf stable-L2 norms."""

    del obs, param, state
    update = _wide(update)
    if update.numel() == 0:
        return update, {}, tempo.live
    update_flat, update_scale, update_norm = _slab_l2_components(update)
    direction = update_flat.sign()
    direction_scale = direction.abs().amax(dim=1, keepdim=True)
    safe_direction_scale = torch.where(
        direction_scale != 0, direction_scale, torch.ones_like(direction_scale)
    )
    direction_norm = torch.linalg.vector_norm(direction / safe_direction_scale, dim=1, keepdim=True)
    graft_eps = torch.as_tensor(1e-6, dtype=update.dtype, device=update.device)
    normalized = direction / safe_direction_scale / torch.where(
        direction_norm != 0, direction_norm, torch.ones_like(direction_norm)
    )
    direct = direction / graft_eps
    normalized = torch.where(direction_norm * direction_scale >= graft_eps, normalized, direct)
    return (normalized * update_norm * update_scale).reshape_as(update), {}, tempo.live


sign_graft.init = sign_init
sign_graft.distributed_shard_separable = False


def orthograd_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def orthograd(update, obs, param, state, tempo):
    """Project each update orthogonally to its parameter, then graft its norm back."""

    del obs, state
    update = _wide(update)
    param = _wide(param)
    if update.numel() == 0:
        return update, {}, tempo.live
    update_flat, update_scale, update_norm = _slab_l2_components(update)
    param_flat, param_scale, _ = _slab_l2_components(param)
    safe_update_scale = torch.where(update_scale != 0, update_scale, torch.ones_like(update_scale))
    safe_param_scale = torch.where(param_scale != 0, param_scale, torch.ones_like(param_scale))
    scaled_update = update_flat / safe_update_scale
    scaled_param = param_flat / safe_param_scale
    # OrthoGrad's projection (Grokking at the Edge of Numerical Stability, eq. 11) is exact: the amax
    # scaling keeps this sum in {0} u [1, numel], so the zero-param case is the only guard needed. It
    # is decoupled from the adaptive eps, which is 22 orders larger and would leak a parallel gradient.
    denominator = scaled_param.square().sum(dim=1, keepdim=True)
    numerator = (scaled_param * scaled_update).sum(dim=1, keepdim=True)
    projection = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(denominator))
    projected = (scaled_update - projection * scaled_param) * safe_update_scale

    projected_scale = projected.abs().amax(dim=1, keepdim=True)
    safe_projected_scale = torch.where(
        projected_scale != 0, projected_scale, torch.ones_like(projected_scale)
    )
    projected_norm = torch.linalg.vector_norm(projected / safe_projected_scale, dim=1, keepdim=True)
    graft_eps = torch.as_tensor(1e-6, dtype=update.dtype, device=update.device)
    normalized = projected / safe_projected_scale / torch.where(
        projected_norm != 0, projected_norm, torch.ones_like(projected_norm)
    )
    direct = projected / graft_eps
    normalized = torch.where(projected_norm * projected_scale >= graft_eps, normalized, direct)
    return (normalized * update_norm * update_scale).reshape_as(update), {}, tempo.live


orthograd.init = orthograd_init
orthograd.distributed_shard_separable = False


def caution_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def caution(update, obs, param, state, tempo):
    """Keep only update entries with the raw gradient's strict sign, preserving scale."""

    del param, state
    update = _wide(update)
    return _caution(_wide(obs.grad), update), {}, tempo.live


caution.init = caution_init
caution.distributed_shard_separable = False


def truegrad_adam_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return adam_init(ref_leaf)


def truegrad_adam(update, obs, param, state, tempo):
    """Adam whose second moment consumes the TrueGrad observation port.

    ``sum_grad_squared`` must be produced in place before each step, for example
    by registering HeavyBall's linear-layer TrueGrad hooks.
    """

    del param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg = _wide(state["exp_avg"]) * beta1 + update * (1 - beta1)
    exp_avg_sq = _second_moment_from_squared(
        state["exp_avg_sq"], obs.sum_grad_squared, beta2
    )
    update = exp_avg / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    return update, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, tempo.live


truegrad_adam.init = truegrad_adam_init


def truegrad_rmsprop(update, obs, param, state, tempo):
    del param
    update = _wide(update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment_from_squared(
        state["exp_avg_sq"], obs.sum_grad_squared, beta2
    )
    update = update / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    return update, {"exp_avg_sq": exp_avg_sq}, tempo.live


truegrad_rmsprop.init = rmsprop_init


def truegrad_laprop(update, obs, param, state, tempo):
    del param
    update = _wide(update)
    beta1 = broadcast_leaf(beta_debias(tempo.hyper.beta1, tempo.age), update)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment_from_squared(
        state["exp_avg_sq"], obs.sum_grad_squared, beta2
    )
    normalized = update / _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)
    exp_avg = _wide(state["exp_avg"]) * beta1 + normalized * (1 - beta1)
    return exp_avg, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, tempo.live


truegrad_laprop.init = laprop_init


def truegrad_nadam(update, obs, param, state, tempo):
    del param
    update = _wide(update)
    exp_avg = _wide(state["exp_avg"]) * tempo.hyper.beta1 + update * (1 - tempo.hyper.beta1)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    exp_avg_sq = _second_moment_from_squared(
        state["exp_avg_sq"], obs.sum_grad_squared, beta2
    )
    denom = _second_moment_denom(exp_avg_sq, tempo.hyper.eps, update.dtype)

    age = tempo.age.to(tempo.hyper.beta1.dtype)
    base = torch.ones_like(tempo.hyper.beta1) * 0.96
    mu = tempo.hyper.beta1 * (1 - 0.5 * torch.pow(base, age * tempo.hyper.momentum_decay))
    mu_next = tempo.hyper.beta1 * (1 - 0.5 * torch.pow(base, (age + 1) * tempo.hyper.momentum_decay))
    mu_product = _wide(state["mu_product"]) * mu
    one = torch.ones_like(tempo.hyper.beta1)
    grad_weight = broadcast_leaf((one - mu) / (one - mu_product), update)
    avg_weight = broadcast_leaf(mu_next / (one - mu_product * mu_next), update)
    update = update / denom * grad_weight + exp_avg / denom * avg_weight
    return update, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "mu_product": mu_product}, tempo.live


truegrad_nadam.init = nadam_init


def adopt_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    ref = _wide(ref_leaf)
    return {
        "exp_avg": torch.zeros_like(ref),
        "exp_avg_sq": torch.zeros_like(ref),
        "seen": torch.zeros((), dtype=torch.bool, device=ref.device),
    }


def adopt(update, obs, param, state, tempo):
    """ADOPT: seed the second moment from the first gradient, then take raw-EMA (un-debiased) moment
    steps that normalize by the previous second moment (Algorithm 2 of the ADOPT paper, without clipping)."""

    del obs, param
    update = _wide(update)
    seen = state["seen"]
    first = seen.logical_not()
    # ADOPT's EMAs are raw, not bias-corrected; seeding v from g_0^2 is the only correction.
    beta1 = tempo.hyper.beta1
    beta2 = tempo.hyper.beta2
    exp_avg = _wide(state["exp_avg"])
    exp_avg_sq = _wide(state["exp_avg_sq"])
    next_avg = exp_avg * beta1 + update / _second_moment_denom(
        exp_avg_sq, tempo.hyper.eps, update.dtype
    ) * (1 - beta1)
    next_avg_sq = _second_moment(exp_avg_sq, update, beta2)
    seeded_avg_sq = update.abs()
    candidate = {
        "exp_avg": torch.where(broadcast_leaf(first, next_avg), exp_avg, next_avg),
        "exp_avg_sq": torch.where(
            broadcast_leaf(first, next_avg_sq), seeded_avg_sq, next_avg_sq
        ),
        "seen": torch.ones_like(seen),
    }
    return next_avg, candidate, tempo.live & seen


adopt.init = adopt_init


def momentum_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    return {"exp_avg": torch.zeros_like(_wide(ref_leaf))}


def momentum(update, obs, param, state, tempo):
    """Muon's Nesterov heavy-ball direction over the incoming update stream."""

    del obs, param
    update = _wide(update)
    exp_avg = _wide(state["exp_avg"]) * tempo.hyper.beta1 + update
    return update + exp_avg * tempo.hyper.beta1, {"exp_avg": exp_avg}, tempo.live


momentum.init = momentum_init


def _stable_matrix_normalize(value: Tensor, eps: float) -> Tensor:
    value = _wide(value)
    if value.numel() == 0:
        return value
    eps_tensor = torch.as_tensor(eps, dtype=value.dtype, device=value.device).reshape(())
    scale = value.abs().amax(dim=(-2, -1), keepdim=True)
    safe = torch.where(scale != 0, scale, torch.ones_like(scale))
    scaled = value / safe
    norm = torch.linalg.vector_norm(scaled, dim=(-2, -1), keepdim=True)
    unit = scaled / torch.where(norm != 0, norm, torch.ones_like(norm))
    direct = value / torch.where(eps_tensor != 0, eps_tensor, torch.ones_like(eps_tensor))
    return torch.where(norm * scale >= eps_tensor, unit, direct)


def orthogonalize_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def orthogonalize(update, obs, param, state, tempo):
    """Apply Muon's five-step batched Newton-Schulz orthogonalization."""

    del obs, param, state
    dtype = update.dtype
    normalized = _stable_matrix_normalize(update, eps=1e-7)
    # Newton-Schulz runs in bf16 for non-fp64 inputs by design (Muon-standard: matches legacy and Keller
    # Jordan's reference, for tensor-core throughput). This is an intended ~2%-vs-fp64 tradeoff on the fp32
    # path, not an fp64 baseline; pass fp64 params for a full-precision orthogonalization.
    x = (
        normalized
        if normalized.dtype == torch.float64
        else stochastic_round_bfloat16(normalized, tempo.random_like(normalized))
    )
    transposed = normalized.shape[-2] > normalized.shape[-1]
    if transposed:
        x = x.mT
    for a, b, c in (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ):
        s = x @ x.mT
        # Evaluate b*S + c*S^2 directly so baddbmm can keep the coefficients in the GEMM epilogue.
        y = torch.baddbmm(s, s, s, beta=b, alpha=c)
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x
    if transposed:
        x = x.mT
    return x.to(dtype), {}, tempo.live


orthogonalize.init = orthogonalize_init
orthogonalize.distributed_scope = WHOLE


def oblique_tangent_projection(update, obs, param, state, tempo):
    """Project an update onto the tangent space of the unit-norm-rows manifold."""

    del obs, state
    wide = _wide(update)
    w = _wide(param)
    if wide.numel() == 0:
        return wide, {}, tempo.live
    scale = wide.abs().amax(dim=-1, keepdim=True)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    scaled = wide / safe_scale
    inner = (w * scaled).sum(dim=-1, keepdim=True)
    return (scaled - w * inner) * safe_scale, {}, tempo.live


oblique_tangent_projection.init = orthogonalize_init


def polargrad_direction(update, obs, param, state, tempo):
    """Scale the polar direction by the momentum's nuclear norm."""

    if update.numel() == 0:
        return _wide(update), {}, tempo.live
    orth = orthogonalize(update, obs, param, state, tempo)[0]
    update_scale = update.abs().amax(dim=(-2, -1), keepdim=True)
    safe_scale = torch.where(update_scale != 0, update_scale, torch.ones_like(update_scale))
    scale = (orth * (update / safe_scale)).sum(dim=(-2, -1), keepdim=True)
    return orth * scale * safe_scale, {}, tempo.live


polargrad_direction.init = orthogonalize_init
polargrad_direction.distributed_scope = WHOLE


def normuon_normalize_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    wide = _wide(ref_leaf)
    shape = list(wide.shape)
    shape[-1] = 1  # per output-neuron (row) second moment; NorMuon (arXiv:2510.05491) Alg. 1
    return {"moment2": torch.zeros(shape, dtype=wide.dtype, device=wide.device)}


def normuon_normalize(update, obs, param, state, tempo):
    """Normalize Muon's direction by a reduced second moment while preserving its Frobenius norm."""

    del obs, param
    update = _wide(update)
    if update.numel() == 0:
        # Zero-element matrices (e.g. shape (0, n)) are routed here as 2-D leaves; the reductions
        # below would reduce over an empty axis. Pass them through unchanged.
        return update, {"moment2": _wide(state["moment2"])}, tempo.live
    # Reduce over the input axis -> a per-output-neuron (row) second moment, applied to tall and wide
    # matrices alike, per NorMuon (arXiv:2510.05491) Alg. 1. (Was shorter-axis, wrong for wide leaves.)
    row_max = update.abs().amax(dim=-1, keepdim=True)
    safe_row_max = torch.where(row_max != 0, row_max, torch.ones_like(row_max))
    observation_rms = (
        (update / safe_row_max).square().mean(dim=-1, keepdim=True).sqrt()
        * safe_row_max
    )
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), update)
    moment2 = _second_moment(state["moment2"], observation_rms, beta2)
    normalized = update / _second_moment_denom(
        moment2, tempo.hyper.eps, update.dtype
    )

    matrix_max = update.abs().amax(dim=(-2, -1), keepdim=True)
    matrix_limit = math.sqrt(
        torch.finfo(update.dtype).max / (update.shape[-2] * update.shape[-1])
    )
    safe_matrix_max = torch.where(matrix_max != 0, matrix_max, torch.ones_like(matrix_max))
    stable_scale = (update / safe_matrix_max).norm(
        dim=(-2, -1), keepdim=True
    ) / normalized.norm(
        dim=(-2, -1), keepdim=True
    ).clamp_min(tempo.hyper.eps)
    stable_output = normalized * stable_scale * safe_matrix_max
    direct_scale = update.norm(dim=(-2, -1), keepdim=True) / normalized.norm(
        dim=(-2, -1), keepdim=True
    ).clamp_min(tempo.hyper.eps)
    direct_output = normalized * direct_scale
    output = torch.where(matrix_max <= matrix_limit, direct_output, stable_output)
    return output, {"moment2": moment2}, tempo.live


normuon_normalize.init = normuon_normalize_init
normuon_normalize.distributed_scope = WHOLE


def rms_align(update, obs, param, state, tempo):
    """Rescale each leaf's update to RMS 0.2, AdaMuon's RMS-aligned rescaling (arXiv:2507.11005): the
    second-moment-normalized orthogonal direction is not itself unit-scaled, so this restores an
    Adam-matched magnitude, shape-invariantly, letting Adam's LR schedules transfer."""

    del obs, param, state
    update = _wide(update)
    if update.numel() == 0:
        return update, {}, tempo.live
    count = update.shape[-2] * update.shape[-1]
    flat, scale, norm = _slab_l2_components(update)
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    safe_norm = torch.where(norm != 0, norm, torch.ones_like(norm))
    target_norm = 0.2 * math.sqrt(count)
    aligned = flat / safe_scale / safe_norm * target_norm
    epsilon = torch.as_tensor(tempo.hyper.eps, dtype=update.dtype, device=update.device)
    direct = flat * (0.2 / epsilon)
    rms = scale * (norm / math.sqrt(count))
    return torch.where(rms >= epsilon, aligned, direct).reshape_as(update), {}, tempo.live


rms_align.init = orthogonalize_init
rms_align.distributed_scope = WHOLE


def balanced_orthogonalize(update, obs, param, state, tempo):
    """Apply Aurora's two-round leverage-balanced polar direction."""

    if update.shape[-2] == update.shape[-1]:
        return orthogonalize(update, obs, param, state, tempo)

    transposed = update.shape[-2] < update.shape[-1]
    tall = update.mT if transposed else update
    target_row_sq = tall.shape[-1] / tall.shape[-2]
    balanced = stable_l2_normalize(tall, dim=-1, eps=1e-7)
    for round_index in range(2):
        direction = orthogonalize(balanced, obs, param, state, tempo)[0]
        if round_index < 1:
            row_sum_sq = direction.square().sum(dim=-1, keepdim=True).clamp_min(1e-7 * 1e-7)
            balanced = balanced * (target_row_sq / row_sum_sq).sqrt()
    if transposed:
        direction = direction.mT
    return direction, {}, tempo.live


balanced_orthogonalize.init = orthogonalize_init
balanced_orthogonalize.distributed_scope = WHOLE


def whiten_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    """Initialize a matrix whitening state for square 2D parameter leaves only."""

    if ref_leaf.ndim != 2 or ref_leaf.shape[0] != ref_leaf.shape[1]:
        raise ValueError("whiten requires square 2D parameter leaves")
    ref = _wide(ref_leaf)
    return {
        "Q": torch.eye(ref.shape[0], dtype=ref.dtype, device=ref.device),
        "GG": torch.zeros_like(ref),
        "GG_scale": torch.ones((), dtype=ref.dtype, device=ref.device),
    }


def whiten(update, obs, param, state, tempo):
    """Left-whiten square 2D slabs; ``Q`` is rebuilt only in a refresh variant.

    As in Shampoo, a negative Gram scale marks ``Q`` as a factored spectral
    root; a positive scale marks the ordinary materialized-root representation.
    """

    del param
    update = _wide(update)
    grad = _wide(obs.grad)
    dimensions = tuple(range(1, grad.ndim))
    maximum = grad.abs().amax(dim=dimensions)
    limit = math.sqrt(torch.finfo(grad.dtype).max / max(1, grad.shape[-1]))
    needs_scale = maximum > limit
    outer_scale = torch.where(needs_scale, maximum, torch.ones_like(maximum))
    scaled_grad = grad / broadcast_leaf(outer_scale, grad)
    direct_outer = grad @ grad.mT
    normalized_outer = scaled_grad @ scaled_grad.mT
    outer = torch.where(
        broadcast_leaf(needs_scale, direct_outer), normalized_outer, direct_outer
    )
    stored_scale = _wide(state["GG_scale"])
    spectral = stored_scale < 0
    old_scale = stored_scale.abs()
    gg_scale = torch.maximum(old_scale, outer_scale)
    safe_scale = torch.where(gg_scale != 0, gg_scale, torch.ones_like(gg_scale))
    old_ratio = broadcast_leaf(old_scale / safe_scale, state["GG"])
    new_ratio = broadcast_leaf(outer_scale / safe_scale, state["GG"])
    gg = (
        _wide(state["GG"]) * old_ratio.square()
        + outer.to(update.dtype) * new_ratio.square()
    )
    q = _wide(state["Q"])
    if tempo.refresh:
        vectors, roots = _eigh_scaled_gram_decomposition(
            gg, gg_scale, tempo.hyper.eps, -0.5
        )
        spectral = _root_requires_spectral_application(roots, q.dtype)
        direct_root = ((vectors * roots.unsqueeze(-2)) @ vectors.mT).to(q.dtype)
        factored_root = (vectors * roots.sqrt().unsqueeze(-2)).to(q.dtype)
        q = torch.where(
            broadcast_leaf(spectral, direct_root), factored_root, direct_root
        )
    direct = q @ update
    factored = q @ (q.mT @ update)
    preconditioned = torch.where(
        broadcast_leaf(spectral, direct), factored, direct
    )
    return preconditioned, {
        "Q": q,
        "GG": gg,
        "GG_scale": torch.where(spectral, -gg_scale, gg_scale),
    }, tempo.live


whiten.init = whiten_init
whiten.distributed_shard_separable = False


def _empty_commit_init(ref_leaf: Tensor) -> dict[str, Tensor]:
    del ref_leaf
    return {}


def sgd_commit(param, update, state, tempo):
    """The sole terminal commit rule; it is pure and never writes storage."""

    del state
    wide = _wide(param)
    wd = tempo.hyper.weight_decay
    return wide * (1 - tempo.hyper.lr * wd) - tempo.hyper.lr * update, {}


sgd_commit.init = _empty_commit_init


def adamc_commit(param, update, state, tempo):
    del state
    wide = _wide(param)
    max_lr = tempo.hyper.lr if tempo.hyper.max_lr is None else tempo.hyper.max_lr
    # When construction ``lr`` (and thus inherited ``max_lr``) is 0, ``lr / max_lr`` is 0/0 and would
    # NaN-poison the update even though the step is a no-op. Substitute 1.0 there (the ratio is unused
    # because ``lr`` zeroes the whole update); nonzero ``max_lr`` keeps the exact ``lr / max_lr`` ratio.
    safe_max_lr = torch.where(max_lr != 0, max_lr, torch.ones_like(max_lr))
    decay = tempo.hyper.weight_decay * tempo.hyper.lr / safe_max_lr
    return wide * (1 - tempo.hyper.lr * decay) - tempo.hyper.lr * update, {}


adamc_commit.init = _empty_commit_init


def muon_commit(param, update, state, tempo):
    """Muon's decoupled-weight-decay terminal parameter candidate."""

    del state
    shape = tempo.logical_shape or update.shape[1:]
    scale = max(1, shape[-2] / shape[-1]) ** 0.5
    update = update * scale
    return _wide(param) * (1 - tempo.hyper.lr * tempo.hyper.weight_decay) + update * -tempo.hyper.lr, {}


muon_commit.init = _empty_commit_init


def stiefel_projection(param, tempo):
    del tempo
    transposed = param.shape[-2] < param.shape[-1]
    x = param.mT if transposed else param
    q, r = torch.linalg.qr(x)
    signs = r.diagonal(dim1=-2, dim2=-1).sign()
    signs = torch.where(signs != 0, signs, torch.ones_like(signs))
    q = q * signs.unsqueeze(-2)
    return q.mT if transposed else q


stiefel_projection.distributed_scope = WHOLE


def oblique_normalization(param, tempo):
    del tempo
    wide = _wide(param)
    if wide.numel() == 0:
        return wide
    normalized = stable_l2_normalize(wide, dim=-1, eps=1e-8)
    zero = wide.abs().amax(dim=-1, keepdim=True) == 0
    fallback = torch.zeros_like(wide)
    if wide.shape[-1]:
        fallback[..., 0] = 1
    return torch.where(zero, fallback, normalized)


def make_retraction_commit(base_commit, projection, *, name):
    def retraction_commit(param, update, state, tempo):
        new_param, state_out = base_commit(param, update, state, tempo)
        return projection(new_param, tempo), state_out

    retraction_commit.init = base_commit.init
    retraction_commit.config = {"base": base_commit.__name__, "projection": name, "projection_fn": projection.__name__}
    if hasattr(projection, "distributed_scope"):
        retraction_commit.distributed_scope = projection.distributed_scope
    retraction_commit.__name__ = "retraction_commit"
    return retraction_commit
