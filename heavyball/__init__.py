"""HeavyBall: a compile-first PyTorch optimizer library.

Each optimizer compiles to a single ``torch.compile(fullgraph=True)`` graph over slab-batched
state. Construct one through its ``torch.optim.Optimizer``-style facade::

    optimizer = heavyball.AdamW(model.parameters(), lr=1e-3)

Every facade's hyperparameters are explicit in its signature (IDE autocomplete / ``inspect.signature``)
and its docstring names the algorithm (``help(heavyball.SOAP)``). Matrix-preconditioned optimizers
(SOAP, Shampoo, PSGDKron, KLSOAP, ...) route non-matrix parameters through AdamW; ``PSGD`` instead
selects a PSGD-family preconditioner for every leaf by shape.

Optimizer state can be stored in low precision to save memory and bandwidth: pass
``storage_dtype=torch.bfloat16`` (half the state memory, often faster) or ``ecc=8`` (bfloat16 plus an
int8 residual, near-fp16 precision at 0.75x fp32). See ``heavyball.HeavyBallOptimizer`` and
``benchmarks/precision_speed.py``.

The bfloat16, ECC, and PSGD stochastic paths draw from a stateless counter-based stream keyed by the
per-optimizer seed, each parameter's leaf index, and its step count -- all carried in ``state_dict()`` --
so a run resumes bit-for-bit from the checkpoint alone, without restoring ``torch.get_rng_state()``.
"""

import math

from .core import Engine, Group, ParamInfo, Recipe, RefreshCadence, Route, build, produce
from .hyperball import hyperball_commit
from .kl import (
    heavy_kl_shampoo_init,
    heavy_kl_shampoo_recipe,
    heavy_kl_soap_init,
    heavy_kl_soap_recipe,
    kl_shampoo_init,
    kl_shampoo_recipe,
    kl_soap_init,
    kl_soap_recipe,
)
from .kron import kron, make_psgd_kron, psgd_kron, psgd_kron_init
from .lra import lra, make_psgd_lra, psgd_lra, psgd_lra_init
from .lather import lather, lather_init, lather_transform, make_lather
from .matrix import (
    matrix_route,
    nfactor_route,
    shampoo_init,
    shampoo_recipe,
    soap_ademamix_recipe,
    soap_init,
    soap_nadam_recipe,
    soap_recipe,
    solp_recipe,
)
from .msam import msam_commit
from .programs import SAM, Program
from .psgd_pro import (
    make_psgd_nfactor,
    make_psgd_pro,
    psgd_nfactor,
    psgd_nfactor_init,
    psgd_nfactor_transform,
    psgd_pro,
    psgd_pro_init,
    psgd_pro_transform,
    qsgd,
    qsgd_transform,
)
from .schedulefree import schedule_free_commit
from .scion import scion, scion_lmo, scion_lmo_init, scion_param_init, scion_route
from .suds import suds, suds_adamw
from .transforms import (
    adam,
    adamc_commit,
    ademamix as ademamix_transform,
    adopt as adopt_transform,
    balanced_orthogonalize,
    beta_debias,
    caution,
    laprop as laprop_transform,
    lion as lion_transform,
    make_retraction_commit,
    mars,
    momentum,
    momentum_init,
    muon_commit,
    nadam as nadam_transform,
    normuon_normalize,
    oblique_normalization,
    oblique_tangent_projection,
    orthogonalize,
    orthogonalize_init,
    orthograd,
    polargrad_direction,
    rms_align,
    rmsprop as rmsprop_transform,
    sgd_commit,
    sign,
    sign_graft,
    stiefel_projection,
    truegrad_adam as truegrad_adam_transform,
    truegrad_laprop as truegrad_laprop_transform,
    truegrad_nadam as truegrad_nadam_transform,
    truegrad_rmsprop as truegrad_rmsprop_transform,
    unscaled_adam,
    whiten,
    whiten_init,
)
from .truegrad import register_truegrad
from .utils import set_torch

KronCadence = RefreshCadence
whiten.__doc__ = (
    "Whiten each square-matrix parameter's gradient to unit covariance by left-multiplying it by "
    "the inverse spectral root of its running Gram (Shampoo-style factored spectral root)."
)

adamw = Recipe(
    chain=(adam,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
adamc = Recipe(
    chain=(adam,),
    commit=adamc_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0, max_lr=None),
)
mars_adamw = Recipe(
    chain=(mars, adam),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0, mars_gamma=0.0025),
)
orthograd_adamw = Recipe(
    chain=(orthograd, adam),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
cautious_adamw = Recipe(
    chain=(adam, caution),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
rmsprop = Recipe(
    chain=(rmsprop_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta2=0.99, eps=1e-6, weight_decay=0.0),
)
sf_adamw = Recipe(
    chain=(rmsprop_transform,),
    commit=schedule_free_commit,
    defaults=dict(
        lr=0.0025,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.0,
        weight_lr_power=2.0,
        r=0.0,
        caution=0.0,
        cautious_weight_decay=0.0,
    ),
)
msam_laprop = Recipe(
    chain=(rmsprop_transform,),
    commit=msam_commit,
    defaults=dict(
        lr=0.0025,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.0,
        sam_step_size=0.1,
        caution=0.0,
        cautious_weight_decay=0.0,
    ),
)
hyperball = Recipe(
    chain=(rmsprop_transform,),
    commit=hyperball_commit,
    defaults=dict(
        lr=0.0025,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.0,
        caution=0.0,
        cautious_weight_decay=0.0,
    ),
)
lion = Recipe(
    chain=(lion_transform,),
    commit=sgd_commit,
    defaults=dict(lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0),
)
laprop = Recipe(
    chain=(laprop_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
ortho_laprop = Recipe(
    chain=(orthograd, laprop_transform),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
laprop_ortho = Recipe(
    chain=(laprop_transform, orthograd),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
sign_laprop = Recipe(
    chain=(laprop_transform, sign_graft),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
nadam = Recipe(
    chain=(nadam_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, momentum_decay=4e-3, weight_decay=0.0),
)
ademamix = Recipe(
    chain=(ademamix_transform,),
    commit=sgd_commit,
    defaults=dict(
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        eps=1e-8,
        alpha=2.0,
        beta3_warmup=0.0,
        alpha_warmup=0.0,
        weight_decay=0.0,
    ),
)
unscaled_adamw = Recipe(
    chain=(unscaled_adam,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
signsgd = Recipe(
    chain=(sign,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, weight_decay=0.0),
)
adopt = Recipe(
    chain=(adopt_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
truegrad_adam = Recipe(
    chain=(truegrad_adam_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
    observations=("sum_grad_squared",),
)
truegrad_rmsprop = Recipe(
    chain=(truegrad_rmsprop_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta2=0.99, eps=1e-8, weight_decay=0.0),
    observations=("sum_grad_squared",),
)
truegrad_laprop = Recipe(
    chain=(truegrad_laprop_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
    observations=("sum_grad_squared",),
)
truegrad_nadam = Recipe(
    chain=(truegrad_nadam_transform,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, momentum_decay=0.004, weight_decay=0.0),
    observations=("sum_grad_squared",),
)
sgd = Recipe(
    chain=(),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, weight_decay=0.0),
)
whitening = Recipe(
    chain=(whiten,),
    commit=sgd_commit,
    defaults=dict(lr=0.0025, eps=1e-8, preconditioner_update_probability=1.0, weight_decay=0.0),
)
soap = soap_recipe
soap_ademamix = soap_ademamix_recipe
soap_nadam = soap_nadam_recipe
solp = solp_recipe
shampoo = shampoo_recipe
kl_soap = kl_soap_recipe
kl_shampoo = kl_shampoo_recipe
heavy_kl_soap = heavy_kl_soap_recipe
heavy_kl_shampoo = heavy_kl_shampoo_recipe
_muon_matrix = Recipe(
    chain=(momentum, orthogonalize),
    commit=muon_commit,
    # lr 0.02 (README's Muon value; SOAP likewise defaults to its 3e-3). The aspect-scaled orthogonal update
    # under-steps at 0.0025, so Muon converges slowly -- below AdamW for the first ~200 steps on a real MNIST
    # autoencoder, above it after -- while 0.02 converges ~2x faster and stays ahead at every horizon tested
    # (still ~1.15x at 600 steps). AdaMuon's RMS-align makes it lr-robust; plain orthogonalize+muon_commit not.
    defaults=dict(lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
_spel_matrix = Recipe(
    chain=(momentum, orthogonalize),
    commit=make_retraction_commit(sgd_commit, stiefel_projection, name="stiefel"),
    defaults=dict(lr=0.02, beta1=0.9, weight_decay=0.0),
)
_oblique_matrix = Recipe(
    chain=(adam, oblique_tangent_projection),
    commit=make_retraction_commit(sgd_commit, oblique_normalization, name="oblique"),
    defaults=dict(lr=0.0025, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
_polargrad_matrix = Recipe(
    chain=(momentum, polargrad_direction),
    commit=sgd_commit,
    defaults=dict(lr=0.02, beta1=0.95, weight_decay=0.0),
)
_normuon_matrix = Recipe(
    chain=(momentum, orthogonalize, normuon_normalize),
    commit=muon_commit,
    defaults=dict(lr=0.02, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
)
_adamuon_matrix = Recipe(
    # RMS-align the second-moment-normalized orthogonal direction to RMS 0.2 and commit without Muon's
    # aspect scale (sgd_commit) -- this is AdaMuon's (arXiv:2507.11005) core RMS-aligned rescaling; the
    # aspect scale is only correct for a truly-orthogonal update, which O/sqrt(v) is not.
    chain=(momentum, orthogonalize, rmsprop_transform, rms_align),
    commit=sgd_commit,
    # lr 0.02 (Muon-family), not 0.0025: the RMS-align fixes the update to RMS 0.2 (~10x smaller than the
    # old buggy RMS~2 that 0.0025 was tuned to), so a Muon-scale lr restores a well-trained effective step.
    defaults=dict(lr=0.02, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
)
_aurora_matrix = Recipe(
    chain=(momentum, balanced_orthogonalize),
    commit=muon_commit,
    defaults=dict(lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
_muon_laprop_matrix = Recipe(
    chain=(laprop_transform, orthogonalize),
    commit=muon_commit,
    defaults=dict(lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0),
)
whiten_adamw = Route(
    lambda info: info.ndim == 2 and info.shape[0] == info.shape[1],
    whitening,
    adamw,
)


def _psgd_lra_route(info):
    """Use LRA after a multidimensional leaf crosses PSGD's default full-factor axis limit."""

    return info.ndim >= 2 and any(size > 2048 for size in info.shape)


shampoo_adamw = Route(matrix_route, shampoo, adamw)
soap_adamw = Route(matrix_route, soap, adamw)
hyperball_adamw = Route(matrix_route, hyperball, adamw)
soap_ademamix_adamw = Route(matrix_route, soap_ademamix, adamw)
soap_nadam_adamw = Route(matrix_route, soap_nadam, adamw)
solp_adamw = Route(matrix_route, solp, adamw)
kl_soap_adamw = Route(matrix_route, kl_soap, adamw)
kl_shampoo_adamw = Route(matrix_route, kl_shampoo, adamw)
heavy_kl_soap_adamw = Route(matrix_route, heavy_kl_soap, adamw)
heavy_kl_shampoo_adamw = Route(matrix_route, heavy_kl_shampoo, adamw)
kron_adamw = Route(matrix_route, kron, adamw)
lather_adamw = Route(matrix_route, lather, adamw)
psgd = Route(
    _psgd_lra_route,
    lra,
    Route(matrix_route, kron, psgd_nfactor),
)
psgd_nfactor_adamw = Route(nfactor_route, psgd_nfactor, Route(matrix_route, psgd_pro, adamw))
psgd_pro_adamw = Route(matrix_route, psgd_pro, adamw)
qsgd_adamw = Route(matrix_route, qsgd, adamw)
psgd_lra_adamw = Route(lambda info: info.ndim >= 1 and math.prod(info.shape) > 1, lra, adamw)
aurora = Route(lambda info: info.ndim == 2, _aurora_matrix, adamw)
muon = Route(lambda info: info.ndim == 2, _muon_matrix, adamw)
spel = Route(lambda info: info.ndim == 2, _spel_matrix, adamw)
oblique = Route(lambda info: info.ndim == 2, _oblique_matrix, adamw)
muon_laprop = Route(lambda info: info.ndim == 2, _muon_laprop_matrix, adamw)
polargrad = Route(lambda info: info.ndim == 2, _polargrad_matrix, adamw)
normuon = Route(lambda info: info.ndim == 2, _normuon_matrix, adamw)
adamuon = Route(lambda info: info.ndim == 2, _adamuon_matrix, adamw)

from .optim import (  # noqa: E402
    ADOPT,
    KLSOAP,
    LATHER,
    MSAM,
    QSGD,
    SGD,
    SOAP,
    SOAPAdEMAMix,
    SOAPNAdam,
    SOLP,
    SpEL,
    AdaMuon,
    AdamC,
    AdamW,
    AdEMAMix,
    Aurora,
    CautiousAdamW,
    HeavyBallOptimizer,
    HeavyKLSOAP,
    HeavyKLShampoo,
    HeavySOAP,
    HeavySOAPAdEMAMix,
    HeavySOAPNAdam,
    HeavySOLP,
    HyperBallAdamW,
    KLShampoo,
    LaProp,
    LaPropOrtho,
    Lion,
    MARSAdamW,
    Muon,
    MuonLaProp,
    NAdam,
    NorMuon,
    Oblique,
    OrthoGradAdamW,
    OrthoLaProp,
    PSGD,
    PSGDKron,
    PSGDLRA,
    PSGDNfactor,
    PSGDPro,
    PolarGrad,
    RMSprop,
    SFAdamW,
    ScheduleFree,
    Scion,
    Shampoo,
    SignLaProp,
    SignSGD,
    SplitOpt,
    SUDSAdamW,
    TrueGradAdam,
    TrueGradLaProp,
    TrueGradNAdam,
    TrueGradRMSprop,
    UnscaledAdamW,
    WhitenAdamW,
    Whitening,
)
from .registry import describe, estimate_state_bytes, list_optimizers  # noqa: E402

__all__ = [
    "ADOPT", "AdEMAMix", "AdaMuon", "AdamC", "AdamW", "Aurora", "CautiousAdamW", "HeavyBallOptimizer", "HeavyKLSOAP", "HeavyKLShampoo", "HeavySOAP", "HeavySOAPAdEMAMix", "HeavySOAPNAdam", "HeavySOLP", "HyperBallAdamW", "KLSOAP",
    "KLShampoo", "LATHER", "LaProp", "LaPropOrtho", "Lion", "MARSAdamW", "MSAM", "Muon", "MuonLaProp",
    "NAdam", "NorMuon", "Oblique", "OrthoGradAdamW", "OrthoLaProp", "PSGD", "PSGDKron", "PSGDLRA", "PSGDNfactor", "PSGDPro", "PolarGrad", "QSGD", "RMSprop", "SGD", "SOAP", "SOAPAdEMAMix", "SOAPNAdam", "SOLP", "SpEL",
    "SUDSAdamW", "SFAdamW", "ScheduleFree", "Scion", "Shampoo", "SignLaProp", "SignSGD", "SplitOpt", "TrueGradAdam", "TrueGradLaProp", "TrueGradNAdam", "TrueGradRMSprop", "UnscaledAdamW",
    "WhitenAdamW", "Whitening", "describe", "estimate_state_bytes", "list_optimizers",
    "Engine", "Group", "ParamInfo", "Program", "Recipe", "RefreshCadence", "Route", "SAM", "adam", "adamc", "adamc_commit", "adamw", "ademamix",
    "adamuon", "ademamix_transform", "adopt", "adopt_transform", "aurora", "balanced_orthogonalize", "beta_debias", "build", "caution", "cautious_adamw",
    "KronCadence", "kron", "kron_adamw", "lather", "lather_adamw", "lather_init", "lather_transform", "laprop", "laprop_ortho", "laprop_transform", "lion", "lion_transform", "make_lather", "make_psgd_kron", "make_retraction_commit",
    "heavy_kl_shampoo", "heavy_kl_shampoo_adamw", "heavy_kl_shampoo_init", "heavy_kl_shampoo_recipe", "heavy_kl_soap", "heavy_kl_soap_adamw", "heavy_kl_soap_init", "heavy_kl_soap_recipe",
    "kl_shampoo", "kl_shampoo_adamw", "kl_shampoo_init", "kl_shampoo_recipe", "kl_soap", "kl_soap_adamw",
    "kl_soap_init", "kl_soap_recipe",
    "hyperball", "hyperball_adamw", "hyperball_commit", "mars", "mars_adamw", "momentum", "momentum_init", "msam_commit", "msam_laprop",
    "muon", "muon_commit", "muon_laprop", "nadam", "nadam_transform", "normuon", "normuon_normalize", "oblique", "oblique_normalization", "oblique_tangent_projection", "orthogonalize", "orthogonalize_init", "orthograd", "polargrad", "polargrad_direction",
    "lra", "make_psgd_lra", "psgd", "psgd_lra", "psgd_lra_adamw", "psgd_lra_init",
    "psgd_kron", "psgd_kron_init", "make_psgd_nfactor", "make_psgd_pro", "nfactor_route", "psgd_nfactor", "psgd_nfactor_adamw", "psgd_nfactor_init", "psgd_nfactor_transform", "psgd_pro", "psgd_pro_adamw", "psgd_pro_init",
    "produce", "psgd_pro_transform", "qsgd", "qsgd_adamw", "qsgd_transform", "register_truegrad",
    "ortho_laprop", "orthograd_adamw", "rms_align", "rmsprop", "rmsprop_transform", "sgd", "sgd_commit", "shampoo",
    "schedule_free_commit", "sf_adamw",
    "scion", "scion_lmo", "scion_lmo_init", "scion_param_init", "scion_route", "shampoo_adamw", "sign_laprop", "spel", "stiefel_projection",
    "shampoo_init", "shampoo_recipe", "sign", "sign_graft", "signsgd", "soap", "soap_adamw",
    "soap_ademamix", "soap_ademamix_adamw", "soap_ademamix_recipe", "soap_init", "soap_nadam", "soap_nadam_adamw", "soap_nadam_recipe", "soap_recipe", "solp", "solp_adamw", "solp_recipe", "truegrad_adam",
    "set_torch", "suds", "suds_adamw",
    "truegrad_adam_transform", "truegrad_laprop", "truegrad_laprop_transform", "truegrad_nadam", "truegrad_nadam_transform", "truegrad_rmsprop", "truegrad_rmsprop_transform", "unscaled_adam", "unscaled_adamw", "whiten",
    "whiten_adamw", "whiten_init", "whitening",
]
