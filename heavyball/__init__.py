import functools
import math
from numbers import Real

import torch.optim

from . import chainable as C
from . import utils

ShapeMap = dict[int, tuple[int, ...]]


def _class_path(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


class SGD(C.BaseOpt):
    """
    SGD with heavy-ball momentum.
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        beta=0.9,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, fns=(C.heavyball_momentum,))


class AdamW(C.BaseOpt):
    """
    AdamW

    Sources:
        Decoupled Weight Decay Regularization
        Ilya Loshchilov, Frank Hutter
        https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.update_by_adam,))


class NAdam(C.BaseOpt):
    """
    NAdam

    Sources:
        Incorporating Nesterov Momentum into Adam
        Timothy Dozat
        https://cs229.stanford.edu/proj2015/054_report.pdf
    """

    def __init__(
        self,
        params,
        lr=0.002,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.update_by_nadam,))


class AdEMAMix(C.BaseOpt):
    """
    AdEMAMix

    Sources:
        The AdEMAMix Optimizer: Better, Faster, Older
        Matteo Pagliardini, Pierre Ablin, David Grangier
        https://arxiv.org/abs/2409.03137
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999, 0.9999),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        alpha: float = 2.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        if len(betas) != 3:
            raise ValueError("AdEMAMix expects betas with three coefficients.")
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, fns=(C.update_by_ademamix,))


class UnscaledAdamW(C.BaseOpt):
    """
    UnscaledAdamW

    An AdamW variant that accumulates first-moment momentum in variance-normalized
    coordinates, then restores the current second-moment scale before applying the update.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        AdamW (baseline):
            Decoupled Weight Decay Regularization
            Ilya Loshchilov, Frank Hutter
            https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.scale_by_unscaled_adam,))


class SUDSAdamW(C.BaseOpt):
    """
    SUDSAdamW

    AdamW augmented with SUDS, a rank-1 Fisher-direction preconditioner fit online via
    Oja's rule and applied before Adam. The rank-1 sketch captures the dominant Hessian
    direction at near-zero cost.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        precond_lr: float = 1e-2,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.scale_by_suds,))


class Scion(C.BaseOpt):
    """
    Scion

    Norm-constrained linear minimization oracle (LMO) optimizer with auto-norm selection:
    spectral norm for matrices, RMS for vectors, spectral norm of the unfolded mode for
    convolutions.

    Sources:
        Training Deep Learning Models with Norm-Constrained LMOs
        Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher
        https://arxiv.org/abs/2502.07529
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
        cautious_weight_decay: bool = False,
        warmup_steps: int = 0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        scale: float = 1.0,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) == 0:
            raise ValueError("Scion expects at least one beta.")
        beta1 = betas[0]
        if not 0 <= beta1 <= 1:
            raise ValueError(f"Invalid momentum value: {beta1}")
        beta2 = betas[1] if len(betas) > 1 else beta1
        betas = (beta1, beta2)
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, fns=(C.exp_avg, C.scion_auto_norm))


class AdamC(C.BaseOpt):
    """
    AdamC

    Adam with weight decay scaled by `lr / max_lr` to preserve the normalized-layer
    steady-state gradient-to-weight ratio under a decaying learning rate.

    Sources:
        AdamC: Confused Adam Optimizers
        Defazio, Mehta, Mishchenko
        https://arxiv.org/abs/2506.02285
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        max_lr: float | None = None,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        max_lr = lr if max_lr is None else max_lr

        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.update_by_adamc,))


class RMSprop(C.BaseOpt):
    """
    Debiased RMSprop (not torch.optim.RMSprop). The bias correction matches Adam's
    second-moment debiasing.

    Sources:
        Lecture 6.5 — RMSprop, COURSERA: Neural Networks for Machine Learning
        Tieleman & Hinton, 2012
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq,),
        )


class HyperBallAdamW(C.BaseOpt):
    """
    HyperBallAdamW

    Routes 2D+ parameters through HyperBall — updates are projected to keep each
    parameter on a hyperball whose radius is set at initialization — and 1D parameters
    through standard AdamW.

    Sources:
        HyperBall:
            Fantastic Pretraining Optimizers and Where to Find Them, Section 2.1: HyperBall Optimization
            https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd

        AdamW:
            Decoupled Weight Decay Regularization
            Ilya Loshchilov, Frank Hutter
            https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(
                C.route(
                    (lambda p: p.ndim >= 2, (C.scale_by_exp_avg_sq, C.update_by_hyperball)),
                    default=C.update_by_adam,
                ),
            ),
        )


class MuonAdamW(C.BaseOpt):
    """
    MuonAdamW

    Routes 2D+ parameters through Muon (orthogonalized momentum) and 1D parameters
    through AdamW.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        Muon:
            Muon: An optimizer for hidden layers in neural networks
            Keller Jordan
            https://kellerjordan.github.io/posts/muon/

        AdamW:
            Decoupled Weight Decay Regularization
            Ilya Loshchilov, Frank Hutter
            https://arxiv.org/abs/1711.05101
    """

    _TOPOLOGY_KEYS = C.BaseOpt._TOPOLOGY_KEYS + ("nesterov",)

    def _core_fns_from_group(self, group):
        _ema = C.nesterov_ema if group["nesterov"] else C.exp_avg
        return (
            C.route(
                (lambda p: p.ndim >= 2, (_ema, C.orthogonalize_update)),
                default=C.scale_by_adam,
            ),
        )

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        nesterov: bool = True,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=self._core_fns_from_group(defaults),
        )


class SFAdamW(C.ScheduleFree):
    """
    SFAdamW (Schedule-Free AdamW)

    Sources:
        The Road Less Scheduled
        Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
        https://arxiv.org/abs/2405.15682
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq, C.update_by_schedule_free),
        )


class MSAMLaProp(C.MSAM):
    """
    MSAMLaProp

    RMSprop-style adaptive scaling wrapped in Momentum-SAM (M-SAM). Despite the name,
    the inner update is RMSprop, not LaProp.

    Sources:
        Momentum-SAM:
            Momentum-SAM: Sharpness Aware Minimization without Computational Overhead
            Marlon Becker, Frederick Altrock, Benjamin Risse
            https://arxiv.org/abs/2401.12033
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        sam_step_size: float = 0.1,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq, C.update_by_msam),
        )


class ADOPT(C.BaseOpt):
    """
    ADOPT

    Sources:
        ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate
        Shohei Taniguchi, Keno Harada, Gouki Minegishi, Yuta Oshima, Seong Cheol Jeong,
        Go Nagahara, Tomoshi Iiyama, Masahiro Suzuki, Yusuke Iwasawa, Yutaka Matsuo
        https://arxiv.org/abs/2411.02853
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.update_by_adopt,))


class Muon(C.BaseOpt):
    """
    Muon

    Sources:
        Muon: An optimizer for hidden layers in neural networks
        Keller Jordan
        https://kellerjordan.github.io/posts/muon/
    """

    _TOPOLOGY_KEYS = C.BaseOpt._TOPOLOGY_KEYS + ("nesterov", "heavyball_momentum")

    def _core_fns_from_group(self, group):
        if group["heavyball_momentum"]:
            _ema = C.nesterov_momentum if group["nesterov"] else C.heavyball_momentum
        elif group["nesterov"]:
            _ema = C.nesterov_ema
        else:
            _ema = C.exp_avg
        return (_ema, C.orthogonalize_update)

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        nesterov: bool = True,
        heavyball_momentum: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            fns=self._core_fns_from_group(defaults),
        )


class LaProp(C.BaseOpt):
    """
    LaProp

    Sources:
        LaProp: Separating Momentum and Adaptivity in Adam
        Liu Ziyin, Zhikang T. Wang, Masahito Ueda
        https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm, fns=(C.update_by_laprop,))


class MuonLaProp(C.BaseOpt):
    """
    MuonLaProp

    LaProp's adaptivity feeding Muon's orthogonalization.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        Muon:
            Muon: An optimizer for hidden layers in neural networks
            Keller Jordan
            https://kellerjordan.github.io/posts/muon/

        LaProp:
            LaProp: Separating Momentum and Adaptivity in Adam
            Liu Ziyin, Zhikang T. Wang, Masahito Ueda
            https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.orthogonalize_update),
        )


class SOAPBase(C.BaseOpt):
    use_precond_schedule: bool = False
    init_factor: float = 0.0

    def _build_soap_defaults(self, locals_dict, fns):
        _use_precond_schedule = C.default(locals_dict["use_precond_schedule"], self.use_precond_schedule)
        locals_dict = {
            **locals_dict,
            "init_factor": C.default(locals_dict.get("init_factor", C.use_default), self.init_factor),
        }
        params, defaults = C._build_defaults(locals_dict)
        defaults.pop("use_precond_schedule")
        if _use_precond_schedule:
            del defaults["precondition_frequency"]
            _scheduler = defaults.pop("precond_scheduler")
            self.precond_schedule = utils.get_soap_precond_schedule(_scheduler)
            self._precond_schedule_spec = ("soap", _scheduler)
        else:
            del defaults["precond_scheduler"]
            _frequency = defaults.pop("precondition_frequency")
            self.precond_schedule = 1 / _frequency
            self._precond_schedule_spec = ("constant", self.precond_schedule)
        super().__init__(
            params,
            defaults,
            locals_dict["gradient_clipping"],
            locals_dict["update_clipping"],
            locals_dict.get("palm", False),
            fns=fns,
        )


class SOAP(SOAPBase):
    """
    SOAP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP
    """

    _chain_fns = (C.scale_by_soap,)

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        cautious_weight_decay: bool = False,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        warmup_steps: int = 0,
        split: bool = False,
        multi_tensor: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        init_factor: float = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        self._build_soap_defaults(locals(), fns=self._chain_fns)


class KLSOAP(SOAP):
    """
    KLSOAP

    SOAP with KL-Shampoo's corrected Kronecker factor accumulation: a two-sided fixed
    point from KL-divergence minimization replaces the one-sided outer products G@G.T,
    weighting each factor's update by the inverse of the other factor's eigenvalue EMA.

    Sources:
        KL-Shampoo:
            Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
            Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, Roger B. Grosse
            https://arxiv.org/abs/2509.03378

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
    """

    _chain_fns = (C.scale_by_kl_soap,)
    init_factor: float = 0.1


class KLShampoo(SOAPBase):
    """
    KL-Shampoo

    Shampoo with KL-corrected Kronecker factor accumulation, applied directly as
    ⊗_i Q[i] diag(λ_i^{-1/2}) Q[i].T to a momentum-EMA gradient. Unlike KL-SOAP,
    no Adam runs in the projected space. Each covariance factor starts at zero while its
    eigenvalue EMA starts at init_factor.

    Sources:
        KL-Shampoo:
            Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
            Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, Roger B. Grosse
            https://arxiv.org/abs/2509.03378
    """

    _chain_fns = (C.scale_by_kl_shampoo,)

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        cautious_weight_decay: bool = False,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
        precondition_1d: bool = False,
        warmup_steps: int = 0,
        split: bool = False,
        multi_tensor: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        precond_scheduler=(1 / 3, 9),
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        init_factor: float = 0.1,
        **kwargs,
    ):
        self._build_soap_defaults(locals(), fns=self._chain_fns)


class SOAPNAdam(SOAP):
    """
    SOAPNAdam

    SOAP with NAdam (Nesterov-Adam) running in the projected eigenbasis instead of
    vanilla Adam.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321

        NAdam:
            Incorporating Nesterov Momentum into Adam
            Timothy Dozat
            https://cs229.stanford.edu/proj2015/054_report.pdf
    """

    _chain_fns = (C.scale_by_soap_nadam,)

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        cautious_weight_decay: bool = False,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
        precondition_1d: bool = False,
        warmup_steps: int = 0,
        split: bool = False,
        multi_tensor: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        self._build_soap_defaults(locals(), fns=self._chain_fns)


class SOAPAdEMAMix(SOAP):
    """
    SOAPAdEMAMix

    SOAP with AdEMAMix's three-EMA scheme running in the projected eigenbasis instead
    of vanilla Adam.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321

        AdEMAMix:
            The AdEMAMix Optimizer: Better, Faster, Older
            Matteo Pagliardini, Pierre Ablin, David Grangier
            https://arxiv.org/abs/2409.03137
    """

    _chain_fns = (C.scale_by_soap_ademamix,)

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        cautious_weight_decay: bool = False,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
        precondition_1d: bool = False,
        warmup_steps: int = 0,
        split: bool = False,
        multi_tensor: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        alpha: float = 2.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        self._build_soap_defaults(locals(), fns=self._chain_fns)


class SignLaProp(C.BaseOpt):
    """
    SignLaProp

    LaProp followed by sign normalization of the resulting update.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        LaProp:
            LaProp: Separating Momentum and Adaptivity in Adam
            Liu Ziyin, Zhikang T. Wang, Masahito Ueda
            https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.sign),
        )


class SOLP(SOAP):
    """
    SOLP

    SOAP with LaProp running in the projected eigenbasis instead of vanilla Adam.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP

        LaProp:
            LaProp: Separating Momentum and Adaptivity in Adam
            Liu Ziyin, Zhikang T. Wang, Masahito Ueda
            https://arxiv.org/abs/2002.04839
    """

    _chain_fns = (C.scale_by_soap_laprop,)


_HEAVYBALL_SOURCE = """
    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360"""


class HeavySOAP(SOAP):
    __doc__ = "SOAP with post-orth Q sort and Hadamard-square second-moment transport.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_soap,)


class HeavyKLSOAP(KLSOAP):
    __doc__ = "KLSOAP with HeavySOAP's eigenbasis update and Moore-Penrose pinv KL inversion.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_kl_soap,)


class HeavyKLShampoo(KLShampoo):
    __doc__ = "KLShampoo with Moore-Penrose pinv KL inversion.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_kl_shampoo,)


class HeavySOAPNAdam(SOAPNAdam):
    __doc__ = "SOAPNAdam with HeavySOAP's eigenbasis update.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_soap_nadam,)


class HeavySOAPAdEMAMix(SOAPAdEMAMix):
    __doc__ = "SOAPAdEMAMix with HeavySOAP's eigenbasis update.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_soap_ademamix,)


class HeavySOLP(SOLP):
    __doc__ = "SOLP with HeavySOAP's eigenbasis update.\n" + _HEAVYBALL_SOURCE
    _chain_fns = (C.scale_by_heavy_soap_laprop,)


class OrthoLaProp(C.BaseOpt):
    """
    OrthoLaProp

    Applies OrthoGrad to the gradient (suppressing the radial component along the
    parameter direction) before running LaProp.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        OrthoGrad:
            Grokking at the Edge of Numerical Stability
            Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, Tolga Birdal
            https://arxiv.org/abs/2501.04697

        LaProp:
            LaProp: Separating Momentum and Adaptivity in Adam
            Liu Ziyin, Zhikang T. Wang, Masahito Ueda
            https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.orthogonalize_grad_to_param, C.scale_by_laprop),
        )


class LaPropOrtho(C.BaseOpt):
    """
    LaPropOrtho

    Runs LaProp first, then applies OrthoGrad to the resulting update.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        OrthoGrad:
            Grokking at the Edge of Numerical Stability
            Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, Tolga Birdal
            https://arxiv.org/abs/2501.04697

        LaProp:
            LaProp: Separating Momentum and Adaptivity in Adam
            Liu Ziyin, Zhikang T. Wang, Masahito Ueda
            https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        cautious_weight_decay: bool = False,
        warmup_steps=0,
        multi_tensor: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.orthogonalize_grad_to_param),
        )


class PSGDBase(C.BaseOpt):
    delayed: bool = False
    cached: bool = False
    exp_avg_input: bool = True
    sqrt: bool = False  # QSGD: apply Q (the matrix square root of P = QᵀQ) instead of P
    _TOPOLOGY_KEYS = C.BaseOpt._TOPOLOGY_KEYS + ("delayed", "cached", "exp_avg_input")

    def _psgd_core_fns_from_group(self, group):
        return self._psgd_core_fns

    def _core_fns_from_group(self, group):
        return (*(C.exp_avg,) * group["exp_avg_input"], *self._psgd_core_fns_from_group(group))

    def _build_psgd_defaults(
        self,
        locals_dict,
        fns,
        *,
        default_update_clipping=utils.trust_region_clip_,
        extra_defaults=None,
    ):
        _exp_avg_input = C.default(locals_dict.get("exp_avg_input", C.use_default), self.exp_avg_input)
        _update_clipping = C.default(locals_dict["update_clipping"], default_update_clipping)
        locals_dict = {**locals_dict, "exp_avg_input": _exp_avg_input, "update_clipping": _update_clipping}
        params, defaults = C._build_defaults(locals_dict)

        self.precond_schedule = C.default(
            defaults.pop("preconditioner_update_probability"), utils.precond_update_prob_schedule()
        )
        if isinstance(self.precond_schedule, Real):
            self._precond_schedule_spec = ("constant", float(self.precond_schedule))
        elif self.precond_schedule is None:
            self._precond_schedule_spec = ("none", None)
        elif (
            isinstance(self.precond_schedule, functools.partial)
            and self.precond_schedule.func is utils._inner_precond_update_prob_schedule
            and not self.precond_schedule.args
        ):
            self._precond_schedule_spec = ("psgd", dict(self.precond_schedule.keywords or {}))
        else:
            raise ValueError("preconditioner_update_probability must be a number or a HeavyBall schedule")

        if extra_defaults:
            defaults.update(extra_defaults)
        if self.sqrt:  # only QSGD carries the key; everything else is plain PSGD via the .get default
            defaults["sqrt"] = self.sqrt

        self._psgd_core_fns = tuple(fns)
        super().__init__(
            params,
            defaults,
            locals_dict["gradient_clipping"],
            _update_clipping,
            False,
            fns=(*(C.exp_avg,) * _exp_avg_input, *fns),
        )


class PSGDKron(PSGDBase):
    """
    PSGDKron

    Preconditioned Stochastic Gradient Descent with a Kronecker-factored preconditioner.

    Sources:
        PSGD:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li
            https://arxiv.org/abs/1512.04202
            https://github.com/lixilinx/psgd_torch

        Originally adapted from Evan Walters and Omead Pooladzandi, 2024,
        under Creative Commons Attribution 4.0 International:
            https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    def _psgd_core_fns_from_group(self, group):
        fn = C.scale_by_delayed_psgd if group["delayed"] else C.scale_by_psgd
        return (functools.partial(fn, cached=group["cached"]),)

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        cautious_weight_decay: bool = False,
        preconditioner_update_probability=C.use_default,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        store_triu_as_line: bool = True,
        multi_tensor: bool = True,
        q_dtype="float32",
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: bool | None = C.use_default,
        cached: bool | None = C.use_default,
        exp_avg_input: bool | None = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,  #
        precond_grad_accum: bool = False,
        lower_bound_beta: float = 0.9,
        dampening: float = 1e-9,
        precond_update_power_iterations: int = 2,
        # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: float | None = None,
        precond_lr: float = 0.1,
        finite_differences: bool = C.use_default,
        fallback_to_finite_differences: bool = C.use_default,
        hvp_interval: int = C.use_default,
        hessian_approx: bool = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        cached = C.default(cached, self.cached)
        self._build_psgd_defaults(
            locals(),
            fns=self._psgd_core_fns_from_group(locals()),
        )


class LATHER(PSGDBase):
    """
    LATHER (Lie-group Adam Through Harmonic Eigenbasis Rotations)

    Runs Adam in the approximate eigenspace induced by the PSGD-Kron preconditioner,
    then maps back to the original space.

    Sources:
        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360

        PSGD:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li
            https://arxiv.org/abs/1512.04202
            https://github.com/lixilinx/psgd_torch
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay=0.0,
        cautious_weight_decay: bool = False,
        preconditioner_update_probability=C.use_default,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        store_triu_as_line: bool = True,
        multi_tensor: bool = True,
        q_dtype="float32",
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        precond_grad_accum: bool = False,
        lower_bound_beta: float = 0.9,
        dampening: float = 1e-9,
        precond_update_power_iterations: int = 2,
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: float | None = None,
        precond_lr: float = 0.1,
        finite_differences: bool = C.use_default,
        fallback_to_finite_differences: bool = C.use_default,
        hvp_interval: int = C.use_default,
        hessian_approx: bool = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        self._build_psgd_defaults(
            {**locals(), "exp_avg_input": False},
            fns=(C.scale_by_lather,),
        )


class PSGDPRO(PSGDBase):
    """
    PSGDPRO

    PSGD-Kron with the Q0.5EQ1.5 (PRO / Procrustes) Q-update — Xi-Lin Li's default
    and recommended local coordinate for fitting Q, using an online orthogonal
    Procrustes solver to keep Q approximately SPD.

    Sources:
        Preconditioned Stochastic Gradient Descent
        Xi-Lin Li
        https://arxiv.org/abs/1512.04202

        Stochastic Hessian Fittings with Lie Groups
        Xi-Lin Li
        https://arxiv.org/abs/2402.11858

        https://github.com/lixilinx/psgd_torch
    """

    def _psgd_core_fns_from_group(self, group):
        return (functools.partial(C.scale_by_psgd_pro, cached=False if self.sqrt else group["cached"]),)

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        cautious_weight_decay: bool = False,
        preconditioner_update_probability=C.use_default,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        multi_tensor: bool = True,
        q_dtype="float32",
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        cached: bool | None = C.use_default,
        exp_avg_input: bool | None = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        precond_grad_accum: bool = False,
        lower_bound_beta: float = 0.9,
        dampening: float = 1e-9,
        precond_update_power_iterations: int = 2,
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: float | None = None,
        precond_lr: float = 0.1,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        cached = C.default(cached, self.cached)
        if self.sqrt and cached:
            cached = False
        self._build_psgd_defaults(
            locals(),
            fns=self._psgd_core_fns_from_group(locals()),
            default_update_clipping=None,
            extra_defaults={"store_triu_as_line": False},
        )


class QSGD(PSGDPRO):
    """
    QSGD

    PSGDPRO applying the Kronecker factor Q during live preconditioning instead of the full
    preconditioner P = QᵀQ. Q is learned identically to PSGDPRO — only the applied update changes
    from P·g = Qᵀ(Q·g) to Q·g. Since QᵀQ = P, ⟨g, P·g⟩ = ‖Q·g‖² exactly. PSGDPRO's PRO/Procrustes
    update additionally keeps each Q symmetric SPD, so Q ≈ P^½ and the step is ≈ P^½·g — a
    full-matrix generalization of Adam's √-scaling. Caching the full P (`cached=True`) is a no-op,
    as a single factor is applied directly. Note: unrelated to "Quantized SGD".

    Sources:
        PSGD:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li
            https://arxiv.org/abs/1512.04202
            https://github.com/lixilinx/psgd_torch

        HeavyBall:
            HeavyBall: a compile-first PyTorch optimizer library
            Lucas Nestler and HomebrewML contributors
            https://github.com/HomebrewML/HeavyBall
            https://zenodo.org/records/19824360
    """

    sqrt: bool = True


class PSGDLRA(PSGDBase):
    """
    PSGDLRA

    Preconditioned Stochastic Gradient Descent with a low-rank preconditioner.

    Note: `multi_tensor=True` (default) uses a single global low-rank approximation shared across all
    parameters, while `multi_tensor=False` fits an independent per-parameter LRA. These are different
    algorithms and will produce different results.

    Sources:
        PSGD:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li
            https://arxiv.org/abs/1512.04202
            https://github.com/lixilinx/psgd_torch

        Originally adapted from Evan Walters and Omead Pooladzandi, 2024,
        under Creative Commons Attribution 4.0 International:
            https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    def _psgd_core_fns_from_group(self, group):
        return (C.scale_by_delayed_psgd_lra if group["delayed"] else C.scale_by_psgd_lra,)

    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        weight_decay=0.0,
        cautious_weight_decay: bool = False,
        preconditioner_update_probability=C.use_default,
        momentum_into_precond_update=True,
        rank: int | None = None,
        warmup_steps: int = 0,
        multi_tensor: bool = True,  # True: global LRA across all params. False: independent per-param LRA.
        q_dtype="float32",
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: bool | None = C.use_default,
        exp_avg_input: bool | None = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        eps: float = 1e-8,
        precond_grad_accum: bool = False,
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: float | None = None,
        precond_lr: float = 0.1,
        finite_differences: bool = C.use_default,
        fallback_to_finite_differences: bool = C.use_default,
        hvp_interval: int = C.use_default,
        hessian_approx: bool = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        orig_shapes: ShapeMap | None = None,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        if rank is None:
            _normalized, _flat = [], []
            for _item in params:
                if isinstance(_item, dict):
                    _group_params = list(_item["params"])
                    _normalized.append({**_item, "params": _group_params})
                    _flat.extend(_group_params)
                else:
                    _normalized.append(_item)
                    _flat.append(_item)
            params = _normalized
            rank = max(1, round(math.log2(max(1, sum(p.numel() for p in _flat)))))

        self._build_psgd_defaults(
            locals(),
            fns=self._psgd_core_fns_from_group(locals()),
        )


class SplitOpt(utils.StatefulOptimizer):
    """
    Delegates different parameter groups to different underlying optimizers.

        opt = SplitOpt([
            {'params': matrices, 'optimizer': Muon, 'lr': 0.02},
            {'params': vectors, 'optimizer': AdamW, 'lr': 0.001},
        ])
    """

    def __init__(self, specs):
        normalized, all_params = [], []
        for spec in specs:
            spec = dict(spec)
            params = list(spec.pop("params"))
            if params:
                optimizer = spec.pop("optimizer")
                if not isinstance(optimizer, type) or not issubclass(optimizer, utils.StatefulOptimizer):
                    raise TypeError("SplitOpt only supports HeavyBall optimizer classes")
                hessian_approx = spec.get("hessian_approx", C.use_default)
                if C.default(hessian_approx, optimizer.hessian_approx):
                    raise ValueError("SplitOpt does not support Hessian-approximation optimizers")
                normalized.append((optimizer, params, spec))
                all_params.extend(params)
        if not normalized:
            raise ValueError("No optimizers created")
        if len({id(p) for p in all_params}) != len(all_params):
            raise ValueError("A parameter cannot belong to multiple SplitOpt optimizers")
        self.optimizers = [optimizer(params, **options) for optimizer, params, options in normalized]
        self._split_ready = False
        super().__init__(all_params, {"multi_tensor": True})
        self._split_ready = True

    def add_param_group(self, param_group):
        if getattr(self, "_split_ready", False):
            raise RuntimeError("SplitOpt does not support add_param_group; recreate it with all parameter groups")
        super().add_param_group(param_group)

    def _handle_closure(self, closure):
        if closure is None:
            return None
        with torch.enable_grad():
            return closure()

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._handle_closure(closure) if closure else None
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def ema_update(self):
        for opt in self.optimizers:
            opt.ema_update()

    def copy_emas_to_params(self):
        for opt in self.optimizers:
            opt.copy_emas_to_params()

    def copy_params_to_emas(self):
        for opt in self.optimizers:
            opt.copy_params_to_emas()

    def state_size(self):
        return sum(opt.state_size() for opt in self.optimizers)

    def train(self, mode: bool = True):
        for opt in self.optimizers:
            train = getattr(opt, "train", None)
            if train is not None:
                train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {
            "classes": [_class_path(opt) for opt in self.optimizers],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError(f"Expected {len(self.optimizers)} optimizer states, got {len(states)}")
        classes = [_class_path(opt) for opt in self.optimizers]
        if state_dict["classes"] != classes:
            raise ValueError(f"Expected optimizer classes {classes}, got {state_dict['classes']}")
        for opt, s in zip(self.optimizers, states):
            opt.load_state_dict(s)


class SAMWrapper(torch.optim.Optimizer):
    """
    SAMWrapper

    Adaptive Sharpness-Aware Minimization wrapper. The inner ascent step scales the
    gradient elementwise by p^2 — the ASAM parameterization — making the perturbation
    scale-invariant under per-parameter rescalings. Wraps any HeavyBall optimizer;
    requires a closure passed to step().

    Sources:
        ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks
        Jungmin Kwon, Jeongseop Kim, Hyunseo Park, In Kwon Choi
        https://arxiv.org/abs/2102.11600
    """

    def __init__(
        self,
        params,
        wrapped_optimizer: utils.StatefulOptimizer | type[utils.StatefulOptimizer] = AdamW,
        ball: float = 0.1,
    ):
        _items = list(params)
        if _items and isinstance(_items[0], dict):
            _groups = [{**item, "params": list(item["params"])} for item in _items]
            if any("ball" in group for group in _groups):
                raise TypeError("ball is optimizer-wide")
            _wrapper_params = [{"params": list(group["params"])} for group in _groups]
            _inner_params = _groups
        else:
            _wrapper_params = _inner_params = _items
        super().__init__(_wrapper_params, {})

        if isinstance(wrapped_optimizer, type):
            if not issubclass(wrapped_optimizer, utils.StatefulOptimizer):
                raise TypeError("SAMWrapper requires a HeavyBall StatefulOptimizer")
            wrapped_optimizer = wrapped_optimizer(_inner_params)
        elif _items and isinstance(_items[0], dict):
            unknown = set().union(*(group.keys() for group in _groups)) - {"params"}
            if unknown:
                raise TypeError(f"Configure instance-form optimizer options before wrapping: {sorted(unknown)}")
        if not isinstance(wrapped_optimizer, utils.StatefulOptimizer):
            raise TypeError("SAMWrapper requires a HeavyBall StatefulOptimizer")

        _wrapper_ids = sorted(id(p) for group in self.param_groups for p in group["params"])
        _inner_ids = sorted(id(p) for group in wrapped_optimizer.param_groups for p in group["params"])
        if _wrapper_ids != _inner_ids:
            raise ValueError("SAMWrapper and its optimizer must contain the same parameters")

        self.ball = ball
        self.wrapped_optimizer = wrapped_optimizer

    def add_param_group(self, param_group):
        if hasattr(self, "wrapped_optimizer"):
            raise RuntimeError("SAMWrapper does not support add_param_group; recreate the wrapper with all parameters")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAM requires closure")
        with torch.enable_grad():
            closure()
        params = [p for group in self.param_groups for p in group["params"]]
        old_params = utils.sam_step(params, self.ball)

        original_handle_closure = self.wrapped_optimizer._handle_closure
        restored = False

        def _handle_closure(closure):
            nonlocal restored
            try:
                _loss = original_handle_closure(closure)
            finally:
                utils.copy_stochastic_list_(params, old_params)
                restored = True
            return _loss

        try:
            self.wrapped_optimizer._handle_closure = _handle_closure
            loss = self.wrapped_optimizer.step(closure)
        finally:
            self.wrapped_optimizer._handle_closure = original_handle_closure
            if not restored:
                utils.copy_stochastic_list_(params, old_params)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.wrapped_optimizer.zero_grad(set_to_none=set_to_none)

    def ema_update(self):
        self.wrapped_optimizer.ema_update()

    def copy_emas_to_params(self):
        self.wrapped_optimizer.copy_emas_to_params()

    def copy_params_to_emas(self):
        self.wrapped_optimizer.copy_params_to_emas()

    def state_size(self):
        return self.wrapped_optimizer.state_size()

    def state_dict(self):
        return {
            "class": _class_path(self.wrapped_optimizer),
            "wrapped": self.wrapped_optimizer.state_dict(),
            "ball": self.ball,
        }

    def load_state_dict(self, state_dict):
        expected = _class_path(self.wrapped_optimizer)
        if state_dict["class"] != expected:
            raise ValueError(f"Expected wrapped optimizer {expected}, got {state_dict['class']}")
        ball = state_dict["ball"]
        self.wrapped_optimizer.load_state_dict(state_dict["wrapped"])
        self.ball = ball

    def train(self, mode: bool = True):
        train = getattr(self.wrapped_optimizer, "train", None)
        if train is not None:
            train(mode)
        return self

    def eval(self):
        return self.train(False)


capture_param_shapes = utils.capture_param_shapes
_BASE_CLASSES = {SOAPBase, PSGDBase}
__all__ = ["capture_param_shapes"] + [
    k
    for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, torch.optim.Optimizer) and v not in _BASE_CLASSES
]
