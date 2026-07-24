from types import SimpleNamespace
from unittest.mock import patch

import torch

from heavyball.codecs import decode, encode
from heavyball.kl import _apply_kl_preconditioner, kl_soap_init
from heavyball.kron import _precondition_mixed, _refresh_q
from heavyball.lra import _refresh_lra
from heavyball.matrix import _gram_value, _inverse_fourth_root, _outer, shampoo, shampoo_init
from heavyball.numerics import balance_factors, broadcast_leaf, stochastic_round_bfloat16
from heavyball.scion import _copy_initialized_
from heavyball.transforms import (
    Tempo,
    _matrix_inv_sqrt,
    _second_moment_denom,
    adam,
    adam_init,
    ademamix,
    ademamix_init,
    adopt,
    adopt_init,
    balanced_orthogonalize,
    beta_debias,
    laprop,
    mars,
    nadam,
    normuon_normalize,
    normuon_normalize_init,
    oblique_normalization,
    oblique_tangent_projection,
    orthogonalize,
    polargrad_direction,
    rms_align,
    rmsprop,
    sgd_commit,
    stiefel_projection,
    unscaled_adam,
    whiten,
    whiten_init,
)


def _tempo(count=1, age=1, refresh=False, **overrides):
    hyper = {
        "alpha": torch.tensor(2.0),
        "alpha_warmup": torch.tensor(10.0),
        "beta1": torch.tensor(0.9),
        "beta2": torch.tensor(0.95),
        "beta3": torch.tensor(0.999),
        "beta3_warmup": torch.tensor(10.0),
        "dampening": torch.tensor(1e-9),
        "eps": torch.tensor(1e-8),
        "lower_bound_beta": torch.tensor(0.9),
        "lr": torch.tensor(0.1),
        "mars_gamma": torch.tensor(0.0025),
        "momentum_decay": torch.tensor(0.004),
        "precond_lr": torch.tensor(0.1),
        "weight_decay": torch.tensor(4.0),
    }
    hyper.update(overrides)
    return Tempo(
        step=torch.ones((), dtype=torch.long),
        age=torch.full((count,), age, dtype=torch.long),
        live=torch.ones(count, dtype=torch.bool),
        hyper=SimpleNamespace(**hyper),
        refresh=refresh,
    )


def _slab_state(initializer, reference):
    return {name: value.unsqueeze(0) for name, value in initializer(reference).items()}


def test_quadratic_state_extremes_are_finite_and_normal_formula_is_preserved():
    normal = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
    extreme = torch.full((1, 3), 1e20, dtype=torch.float32)
    transforms = (
        (adam, adam_init),
        (rmsprop, lambda ref: {"exp_avg_sq": adam_init(ref)["exp_avg_sq"]}),
        (laprop, adam_init),
        (nadam, lambda ref: {**adam_init(ref), "mu_product": torch.ones(())}),
        (ademamix, ademamix_init),
        (unscaled_adam, adam_init),
    )
    for transform, initializer in transforms:
        normal_state = _slab_state(initializer, normal[0])
        _, next_normal, _ = transform(
            normal, None, None, normal_state, _tempo()
        )
        old_normal = normal.square()
        torch.testing.assert_close(
            next_normal["exp_avg_sq"].float().square(), old_normal, rtol=1e-6, atol=1e-7
        )

        extreme_state = _slab_state(initializer, extreme[0])
        output, next_extreme, _ = transform(
            extreme, None, None, extreme_state, _tempo()
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            next_extreme["exp_avg_sq"].double().square(),
            extreme.double().square(),
            rtol=1e-7,
            atol=0,
        )

    adopt_state = _slab_state(adopt_init, extreme[0])
    _, adopt_next, _ = adopt(extreme, None, None, adopt_state, _tempo())
    assert torch.isfinite(adopt_next["exp_avg_sq"]).all()
    torch.testing.assert_close(
        adopt_next["exp_avg_sq"].double().square(),
        extreme.double().square(),
        rtol=1e-7,
        atol=0,
    )

    rms = torch.tensor([[1e-6, 1e-3]])
    eps = torch.tensor(1e-8)
    torch.testing.assert_close(
        _second_moment_denom(rms, eps, rms.dtype),
        rms.square().clamp_min(eps).sqrt(),
        rtol=0,
        atol=0,
    )

    torch.manual_seed(4)
    matrix = torch.randn(1, 4, 3)
    torch.testing.assert_close(
        _outer(matrix, matrix.mT).float(),
        matrix @ matrix.mT,
        rtol=1e-6,
        atol=1e-6,
    )


def test_extreme_shampoo_whitening_and_kl_grams_follow_fp64():
    update = torch.full((1, 2, 2), 1e20)
    expected_gram = update.double() @ update.double().mT

    shampoo_state = _slab_state(
        lambda ref: shampoo_init(ref, max_precond_dim=torch.tensor(8)),
        torch.zeros(2, 2),
    )
    shampoo_output, shampoo_next, _ = shampoo(
        update, None, None, shampoo_state, _tempo(refresh=True)
    )
    torch.testing.assert_close(
        _gram_value(shampoo_next["GG_l"], shampoo_next["GG_l_scale"]),
        expected_gram,
        rtol=1e-7,
        atol=0,
    )
    torch.testing.assert_close(
        shampoo_output, torch.full_like(update, 0.5), rtol=2e-6, atol=1e-6
    )

    whitening_state = _slab_state(whiten_init, torch.zeros(2, 2))
    whitening_output, whitening_next, _ = whiten(
        update,
        SimpleNamespace(grad=update),
        None,
        whitening_state,
        _tempo(refresh=True),
    )
    torch.testing.assert_close(
        whitening_next["GG"].double()
        * whitening_next["GG_scale"].double().reshape(-1, 1, 1).square(),
        expected_gram,
        rtol=1e-7,
        atol=0,
    )
    torch.testing.assert_close(
        whitening_output, torch.full_like(update, 0.5), rtol=2e-6, atol=1e-6
    )

    kl_state = _slab_state(
        lambda ref: kl_soap_init(ref, max_precond_dim=torch.tensor(8)),
        torch.zeros(2, 2),
    )
    kl_next = _apply_kl_preconditioner(
        update, kl_state, _tempo(refresh=True), torch.tensor(0.95)
    )
    for name in ("GG_l", "GG_r", "eigenvalues_l", "eigenvalues_r", "Q_l", "Q_r"):
        assert torch.isfinite(kl_next[name]).all(), name
    torch.testing.assert_close(
        kl_next["eigenvalues_l"].double().square(),
        torch.full_like(kl_next["eigenvalues_l"].double(), 5.0000015e39),
        rtol=1e-6,
        atol=0,
    )


def test_psgd_extreme_refreshes_are_finite_and_normal_ratio_matches_raw_formula():
    identity = torch.eye(2).unsqueeze(0)
    lower = torch.zeros(1, dtype=torch.float64)
    probe = torch.ones(1, 2, 2)

    normal = torch.full((1, 2, 2), 0.25)
    normal_result = _refresh_q(
        normal,
        identity,
        identity.clone(),
        lower,
        lower.clone(),
        _tempo(refresh=True),
        2,
        probe,
    )
    hessian = normal + (
        _tempo().hyper.dampening + torch.finfo(normal.dtype).eps * normal.abs()
    ) * probe
    a = hessian
    conjb = probe
    term1 = a @ a.mT
    term2 = conjb @ conjb.mT
    ell = torch.linalg.matrix_norm(term1 + term2, ord=2)
    old_q = identity - (term1 - term2).triu() @ identity / ell * 0.1
    torch.testing.assert_close(normal_result[0], old_q, rtol=1e-6, atol=1e-7)

    extreme = torch.full((1, 2, 2), 1e20)
    q0, q1, lower0, lower1 = _refresh_q(
        extreme,
        identity,
        identity.clone(),
        lower,
        lower.clone(),
        _tempo(refresh=True),
        2,
        probe,
    )
    for value in (q0, q1, lower0, lower1):
        assert torch.isfinite(value).all()
    expected_q = torch.tensor([[[0.95, -0.05], [0.0, 0.95]]])
    torch.testing.assert_close(q0, expected_q, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(q1, expected_q, rtol=1e-6, atol=1e-7)

    U = torch.zeros(1, 2, 1)
    V = torch.zeros_like(U)
    d = torch.ones(1, 2)
    with patch.object(Tempo, "randn_like", lambda self, value: torch.ones_like(value)):
        normal_lra = _refresh_lra(
            torch.full((1, 2), 0.25), U, V, d, _tempo(refresh=True)
        )
        extreme_lra = _refresh_lra(
            torch.full((1, 2), 1e20), U, V, d, _tempo(refresh=True)
        )
    old_d = 1 - 0.1 * 0.25 / (0.25**2 + 1) ** 0.5
    torch.testing.assert_close(
        normal_lra[2], torch.full_like(d, old_d), rtol=1e-6, atol=1e-7
    )
    for value in extreme_lra:
        assert torch.isfinite(value).all()
    torch.testing.assert_close(
        extreme_lra[2], torch.full_like(d, 0.9), rtol=1e-6, atol=1e-7
    )


def test_scale_first_norms_and_dots_match_normal_formulas_and_extremes():
    torch.manual_seed(2)
    normal = torch.randn(1, 3, 2, dtype=torch.float64)
    old_rms = normal * (
        0.2
        / (
            normal.norm(dim=(-2, -1), keepdim=True)
            / (normal.shape[-2] * normal.shape[-1]) ** 0.5
        ).clamp_min(1e-8)
    )
    new_rms = rms_align(
        normal, None, None, {}, _tempo(eps=torch.tensor(1e-8, dtype=torch.float64))
    )[0]
    torch.testing.assert_close(new_rms, old_rms, rtol=1e-15, atol=1e-15)

    extreme = torch.full((1, 2, 2), 1e20)
    aligned = rms_align(extreme, None, None, {}, _tempo())[0]
    torch.testing.assert_close(
        aligned, torch.full_like(extreme, 0.2), rtol=1e-6, atol=1e-7
    )

    oblique = torch.randn(2, 5, dtype=torch.float64)
    torch.testing.assert_close(
        oblique_normalization(oblique, None),
        oblique / oblique.norm(dim=-1, keepdim=True).clamp_min(1e-8),
        rtol=1e-15,
        atol=1e-15,
    )
    huge_oblique = oblique_normalization(torch.full((1, 16), 1e20), None)
    torch.testing.assert_close(
        huge_oblique.norm(dim=-1), torch.ones(1), rtol=1e-6, atol=1e-6
    )

    w = torch.full((1, 16), 0.25)
    parallel = torch.full((1, 16), 1e38)
    tangent = oblique_tangent_projection(
        parallel, None, w, {}, _tempo()
    )[0]
    assert torch.equal(tangent, torch.zeros_like(tangent))

    old_tangent = normal - normal * (normal * normal).sum(
        dim=-1, keepdim=True
    )
    new_tangent = oblique_tangent_projection(
        normal, None, normal, {}, _tempo()
    )[0]
    torch.testing.assert_close(new_tangent, old_tangent, rtol=1e-15, atol=1e-15)

    diagonal = torch.zeros(1, 3, 2)
    diagonal[0, 0, 0] = 1e20
    diagonal[0, 1, 1] = 1e20
    balanced = balanced_orthogonalize(
        diagonal, None, None, {}, _tempo()
    )[0]
    assert torch.linalg.vector_norm(balanced) > 1
    assert torch.isfinite(balanced).all()

    polar = polargrad_direction(normal, None, None, {}, _tempo())[0]
    orth = orthogonalize(normal, None, None, {}, _tempo())[0]
    old_polar = orth * (orth * normal).sum(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(polar, old_polar, rtol=1e-15, atol=1e-15)

    torch.manual_seed(0)
    extreme_polar = torch.randn(1, 4, 4) * 5e37
    assert torch.isfinite(extreme_polar).all()
    with patch.object(
        Tempo,
        "random_like",
        lambda self, value, _stream=0: torch.zeros_like(value),
    ):
        extreme_orth = orthogonalize(
            extreme_polar, None, None, {}, _tempo()
        )[0]
        old_extreme_scale = (extreme_orth * extreme_polar).sum(
            dim=(-2, -1), keepdim=True
        )
        assert not torch.isfinite(old_extreme_scale).all()
        extreme_scale = extreme_polar.abs().amax(dim=(-2, -1), keepdim=True)
        normalized_scale = (extreme_orth * (extreme_polar / extreme_scale)).sum(
            dim=(-2, -1), keepdim=True
        )
        expected_polar = extreme_orth * normalized_scale * extreme_scale
        actual_polar = polargrad_direction(
            extreme_polar, None, None, {}, _tempo()
        )[0]
    assert torch.isfinite(actual_polar).all()
    torch.testing.assert_close(actual_polar, expected_polar, rtol=1e-6, atol=0)


def test_scale_first_reductions_return_empty_inputs_unchanged():
    for shape in ((1, 0, 3), (1, 3, 0)):
        empty = torch.empty(shape)
        outputs = (
            rms_align(empty, None, None, {}, _tempo())[0],
            polargrad_direction(empty, None, None, {}, _tempo())[0],
            oblique_tangent_projection(empty, None, empty, {}, _tempo())[0],
            oblique_normalization(empty, None),
        )
        for output in outputs:
            assert output.shape == empty.shape
            assert output.numel() == 0


def test_rank_deficient_retractions_return_valid_manifold_points():
    torch.manual_seed(3)
    full_rank = torch.randn(5, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(full_rank)
    old = q * r.diagonal().sign().unsqueeze(-2)
    torch.testing.assert_close(
        stiefel_projection(full_rank, None), old, rtol=0, atol=0
    )

    stiefel = stiefel_projection(torch.zeros(3, 2), None)
    torch.testing.assert_close(
        torch.linalg.svdvals(stiefel), torch.ones(2), rtol=0, atol=0
    )
    oblique = oblique_normalization(torch.zeros(2, 3), None)
    torch.testing.assert_close(
        oblique.norm(dim=-1), torch.ones(2), rtol=0, atol=0
    )


def test_normuon_two_step_extreme_state_and_normal_equivalence():
    torch.manual_seed(8)
    normal = torch.randn(4, 4)
    state = normuon_normalize_init(normal)
    state["moment2"].copy_(torch.rand_like(state["moment2"]) + 0.1)
    tempo = _tempo(4, age=3)
    beta2 = broadcast_leaf(beta_debias(tempo.hyper.beta2, tempo.age), normal)
    old_moment = state["moment2"].float().square() * beta2 + normal.square().mean(
        dim=-1, keepdim=True
    ) * (1 - beta2)
    old_normalized = normal * old_moment.clamp_min(tempo.hyper.eps).rsqrt()
    old_output = old_normalized * (
        normal.norm(dim=(-2, -1), keepdim=True)
        / old_normalized.norm(dim=(-2, -1), keepdim=True).clamp_min(
            tempo.hyper.eps
        )
    )
    output, next_state, _ = normuon_normalize(
        normal, None, None, state, tempo
    )
    torch.testing.assert_close(output, old_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        next_state["moment2"].float().square(), old_moment, rtol=1e-6, atol=1e-7
    )

    extreme = torch.full((1, 2, 2), 1e20)
    extreme_state = {"moment2": torch.zeros(1, 2, 1)}
    outputs = []
    for age in (1, 2):
        output, extreme_state, _ = normuon_normalize(
            extreme, None, None, extreme_state, _tempo(age=age)
        )
        outputs.append(output)
    for output in outputs:
        torch.testing.assert_close(output, extreme, rtol=0, atol=0)
    torch.testing.assert_close(
        extreme_state["moment2"].double().square(),
        torch.full_like(extreme_state["moment2"].double(), 1e40),
        rtol=1e-7,
        atol=0,
    )


def _old_regularized_root(gram, eps, exponent):
    regularized = gram * 0.5 + gram.mT * 0.5 + eps * torch.eye(
        gram.shape[-1], dtype=gram.dtype
    )
    values, vectors = torch.linalg.eigh(regularized)
    return (vectors * values.clamp_min(eps).pow(exponent).unsqueeze(-2)) @ vectors.mT


def test_rank_deficient_roots_classify_roundoff_as_null():
    torch.manual_seed(9)
    full_rank = torch.randn(1, 8, 8)
    gram = full_rank @ full_rank.mT + torch.eye(8)
    eps = torch.tensor(1e-8)
    for function, exponent in (
        (_matrix_inv_sqrt, -0.5),
        (_inverse_fourth_root, -0.25),
    ):
        torch.testing.assert_close(
            function(gram, eps),
            _old_regularized_root(gram, eps, exponent),
            rtol=1e-6,
            atol=1e-6,
        )

    torch.manual_seed(2401)
    factor = torch.randn(1, 24, 1)
    gram32 = factor @ factor.mT
    gram64 = factor.double() @ factor.double().mT
    null = torch.linalg.qr(factor.double(), mode="complete").Q[:, :, 1]
    for function, expected_gain in (
        (_matrix_inv_sqrt, 1e4),
        (_inverse_fourth_root, 1e2),
    ):
        root = function(gram32, eps).double()
        gain = (root @ null.unsqueeze(-1)).norm()
        assert abs(float(gain) - expected_gain) / expected_gain < 2e-6
        reference = function(gram64, eps.double())
        assert float((root - reference).norm() / reference.norm()) < 2e-6


def test_sgd_and_mars_reordering_preserves_normal_and_avoids_cancellation():
    normal_param = torch.tensor([1.25, -0.5])
    normal_update = torch.tensor([0.2, -0.3])
    tempo = _tempo(lr=torch.tensor(0.01), weight_decay=torch.tensor(0.1))
    old = normal_param - tempo.hyper.lr * (
        normal_update + tempo.hyper.weight_decay * normal_param
    )
    new = sgd_commit(normal_param, normal_update, {}, tempo)[0]
    torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-7)

    extreme = sgd_commit(
        torch.tensor([1e38]), torch.zeros(1), {}, _tempo()
    )[0]
    torch.testing.assert_close(
        extreme, torch.tensor([6e37]), rtol=1e-6, atol=0
    )

    normal_history = torch.tensor([[0.5, -0.25]])
    a = (
        -tempo.hyper.mars_gamma
        * tempo.hyper.beta1
        / (1 - tempo.hyper.beta1)
    )
    old_corrected = normal_update.unsqueeze(0) * (1 - a) + normal_history * a
    old_corrected = old_corrected / old_corrected.norm().clamp_min(1)
    new_corrected = mars(
        normal_update.unsqueeze(0),
        None,
        None,
        {"mars_old_grad": normal_history},
        tempo,
    )[0]
    torch.testing.assert_close(
        new_corrected, old_corrected, rtol=1e-6, atol=1e-7
    )

    repeated = torch.full((1, 2), 1e38)
    corrected = mars(
        repeated,
        None,
        None,
        {"mars_old_grad": repeated.clone()},
        _tempo(beta1=torch.tensor(0.9999)),
    )[0]
    torch.testing.assert_close(
        corrected,
        torch.full_like(corrected, 2**-0.5),
        rtol=1e-6,
        atol=1e-7,
    )


def test_stochastic_rounding_saturates_only_finite_overflow():
    torch.manual_seed(11)
    normal = torch.randn(128)
    random = torch.rand_like(normal)
    bits = normal.view(torch.int32)
    old = (
        bits + (random * (1 << 16)).to(torch.int32)
    ).bitwise_and(-65536).view(torch.float32).bfloat16()
    assert torch.equal(stochastic_round_bfloat16(normal, random), old)

    maximum = torch.tensor(
        [torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]
    )
    half = torch.full_like(maximum, 0.5)
    rounded = stochastic_round_bfloat16(maximum, half)
    assert torch.isfinite(rounded).all()
    assert torch.equal(rounded.sign(), maximum.sign())

    for correction_dtype in (torch.int8, torch.int16):
        narrow, correction = encode(
            maximum, correction_dtype=correction_dtype, random=half
        )
        restored = decode(narrow, correction)
        assert torch.isfinite(restored).all()
        assert torch.equal(restored.sign(), maximum.sign())

    def fixed_noise(low, high, size, *, dtype, device, generator):
        del low, high, generator
        return torch.full(size, 1 << 15, dtype=dtype, device=device)

    scion_target = torch.empty_like(maximum, dtype=torch.bfloat16)
    with patch("heavyball.scion.torch.randint", fixed_noise):
        _copy_initialized_(scion_target, maximum, torch.Generator())
    assert torch.isfinite(scion_target).all()
    assert torch.equal(scion_target.sign(), maximum.sign())

    normal_narrow, normal_correction = encode(
        normal, correction_dtype=torch.int8, random=random
    )
    old_rounded = (
        bits + (random * (1 << 8)).to(torch.int32)
    ).bitwise_and(-256)
    old_decoded = old_rounded.view(torch.float32)
    assert torch.equal(decode(normal_narrow, normal_correction), old_decoded)

    nonfinite = torch.tensor([float("inf"), -float("inf"), float("nan")])
    nonfinite_random = torch.full_like(nonfinite, 0.5)
    assert not torch.isfinite(
        stochastic_round_bfloat16(nonfinite, nonfinite_random)
    ).any()
    for correction_dtype in (torch.int8, torch.int16):
        narrow, correction = encode(
            nonfinite,
            correction_dtype=correction_dtype,
            random=nonfinite_random,
        )
        assert not torch.isfinite(decode(narrow, correction)).any()


def test_extreme_factor_balancing_uses_log_reconstruction_only_when_needed():
    torch.manual_seed(12)
    normal = [torch.rand(3, 4) + 0.1, torch.rand(3, 5) + 0.1]
    old_logs = [factor.amax(dim=1).log() for factor in normal]
    mean = (old_logs[0] + old_logs[1]) * 0.5
    old = [
        factor * (mean - log).exp().unsqueeze(-1)
        for factor, log in zip(normal, old_logs, strict=True)
    ]
    new = balance_factors(normal)
    for actual, expected in zip(new, old, strict=True):
        assert torch.equal(actual, expected)

    minimum = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    q0 = torch.tensor([[1e38]])
    q1 = minimum.reshape(1, 1)
    balanced_q0, balanced_q1 = balance_factors([q0, q1])
    result = _precondition_mixed(
        torch.ones(1, 1, 1), balanced_q0, balanced_q1
    )
    reference = (q0.double().item() * q1.double().item()) ** 2
    torch.testing.assert_close(
        result.double(),
        torch.tensor([[[reference]]], dtype=torch.float64),
        rtol=1e-6,
        atol=0,
    )
