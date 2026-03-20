import functools
import math
from typing import Optional, Type, Union

import torch.optim

from . import chainable as C
from . import utils


class SGD(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        beta=0.9,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, fns=(C.heavyball_momentum,))


class ForeachAdamW(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.update_by_adam,))


class ForeachNAdam(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.002,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.update_by_nadam,))


class ForeachAdEMAMix(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999, 0.9999),
        eps=1e-8,
        weight_decay=0,
        alpha: float = 2.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        if len(betas) != 3:
            raise ValueError("AdEMAMix expects betas with three coefficients.")

        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, fns=(C.update_by_ademamix,))


class UnscaledAdamW(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.scale_by_unscaled_adam,)
        )


class SUDSAdamW(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.scale_by_suds,))


class Scion(C.BaseOpt):
    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        scale: float = 1.0,
        momentum: Optional[float] = None,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) == 0 and momentum is None:
            raise ValueError("Scion expects at least one beta or an explicit momentum.")

        beta1 = momentum if momentum is not None else betas[0]
        if not 0 <= beta1 <= 1:
            raise ValueError(f"Invalid momentum value: {beta1}")
        beta2 = betas[1] if len(betas) > 1 else beta1

        params, defaults = C._build_defaults(locals())
        defaults["betas"] = (beta1, beta2)
        defaults["scale"] = scale
        defaults.pop("momentum", None)

        super().__init__(
            params, defaults, foreach, gradient_clipping, update_clipping, fns=(C.exp_avg, C.scion_auto_norm)
        )


class ForeachAdamC(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        max_lr: float | None = None,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        if max_lr is None:
            utils.warn_once(
                "max_lr was not set. setting it to the current learning rate, under the assumption that it strictly decreases"
            )
            max_lr = lr

        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.update_by_adamc,))


class ForeachRMSprop(C.BaseOpt):
    """
    Debiased RMSprop (not torch.optim.RMSprop)
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq,),
        )


class ForeachSFAdamW(C.ScheduleFree):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq, C.update_by_schedule_free),
        )


class MSAMLaProp(C.MSAM):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_exp_avg_sq, C.update_by_msam),
        )


class PaLMForeachSFAdamW(ForeachSFAdamW):
    palm: bool = True


class ForeachADOPT(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.update_by_adopt,))


class ForeachMuon(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        nesterov: bool = True,
        heavyball_momentum: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())

        if heavyball_momentum:
            ema = C.nesterov_momentum if nesterov else C.heavyball_momentum
        elif nesterov:
            ema = C.nesterov_ema
        else:
            ema = C.exp_avg

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(ema, C.orthogonalize_update),
        )


class ForeachLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, fns=(C.update_by_laprop,))


class MuonLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.orthogonalize_update),
        )


class ForeachSOAP(C.BaseOpt):
    """
    ForeachSOAP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
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
        stochastic_schedule: bool = False,
        precond_grad_accum: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        params, defaults = C._build_defaults(locals())

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  #
            fns=(C.scale_by_soap,),
        )


class ForeachSOAPNAdam(C.BaseOpt):
    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
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
        stochastic_schedule: bool = False,
        precond_grad_accum: bool = False,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        params, defaults = C._build_defaults(locals())

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_soap_nadam,),
        )


class ForeachSOAPAdEMAMix(C.BaseOpt):
    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
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
        stochastic_schedule: bool = False,
        precond_grad_accum: bool = False,
        alpha: float = 2.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        params, defaults = C._build_defaults(locals())

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_soap_ademamix,),
        )


class ForeachSignLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.sign),
        )


class ForeachSOLP(C.BaseOpt):
    """
    ForeachSOLP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
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
        stochastic_schedule: bool = False,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        params, defaults = C._build_defaults(locals())

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  #
            fns=(C.scale_by_soap_laprop,),
        )


class PaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = False
    palm: bool = True


class PrecondScheduleForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True


class PrecondSchedulePaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True
    palm: bool = True


class OrthoLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.orthogonalize_grad_to_param, C.scale_by_laprop),
        )


class LaPropOrtho(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
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
        **kwargs,
    ):
        params, defaults = C._build_defaults(locals())
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            fns=(C.scale_by_laprop, C.orthogonalize_grad_to_param),
        )


class ForeachPSGDKron(C.BaseOpt):
    """
    Originally from Evan Walters and Omead Pooladzandi, 2024
    Modified under Creative Commons Attribution 4.0 International
    Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    delayed: bool = False
    cached: bool = False
    exp_avg_input: bool = True
    quad: bool = False

    def __init__(
        self,
        params,
        lr=0.001,
        beta=None,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        preconditioner_update_probability=C.use_default,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        store_triu_as_line: bool = True,
        foreach: bool = True,
        q_dtype="float32",
        stochastic_schedule: bool = False,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: Optional[bool] = C.use_default,
        cached: Optional[bool] = C.use_default,
        exp_avg_input: Optional[bool] = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,  #
        adaptive: bool = False,
        ortho_method: Optional[str] = None,  # If None, no orthogonalization
        precond_grad_accum: bool = False,
        lower_bound_beta: float = 0.9,  # 0.0 recovers pre-2.0.0 PSGD
        inverse_free: bool = C.use_default,
        dampening: float = 2**-13,
        precond_update_power_iterations: int = 2,
        # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: Optional[float] = None,
        precond_lr: float = 0.1,
        finite_differences: bool = C.use_default,
        fallback_to_finite_differences: bool = C.use_default,
        hvp_interval: int = C.use_default,
        hessian_approx: bool = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        cached = C.default(cached, self.cached)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)
        inverse_free = C.default(inverse_free, self.quad)
        if inverse_free:
            raise ValueError(
                "inverse_free (i.e., PSGD-QUAD) is not supported at the moment. Consider using https://github.com/evanatyourservice/quad_torch"
            )

        params, defaults = C._build_defaults(locals())

        self.precond_schedule = C.default(
            defaults.pop("preconditioner_update_probability"), utils.precond_update_prob_schedule()
        )

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            fns=(
                *(C.exp_avg,) * exp_avg_input,
                functools.partial(C.scale_by_delayed_psgd if delayed else C.scale_by_psgd, cached=cached),
            ),
        )


class ForeachPurePSGD(ForeachPSGDKron):
    exp_avg_input: bool = False


class ForeachCachedDelayedPSGDKron(ForeachPSGDKron):
    delayed: bool = True
    cached: bool = True


class ForeachCachedPSGDKron(ForeachPSGDKron):
    cached: bool = True


class ForeachDelayedPSGD(ForeachPSGDKron):
    delayed: bool = True


class ForeachCachedNewtonPSGD(ForeachCachedPSGDKron):
    hessian_approx = True


class NewtonHybrid2PSGDKron(ForeachCachedNewtonPSGD):
    hvp_interval = 2


class ForeachPSGDLRA(C.BaseOpt):
    """
    Originally from Evan Walters and Omead Pooladzandi, 2024
    Modified under Creative Commons Attribution 4.0 International
    Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    delayed: bool = False
    exp_avg_input: bool = True

    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=C.use_default,
        momentum_into_precond_update=True,
        rank: Optional[int] = None,
        warmup_steps: int = 0,
        foreach: bool = True,
        q_dtype="float32",
        stochastic_schedule: bool = False,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: Optional[bool] = C.use_default,
        exp_avg_input: Optional[bool] = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        eps: float = 1e-8,  #
        precond_grad_accum: bool = False,  # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: Optional[float] = None,
        precond_lr: float = 0.1,
        finite_differences: bool = C.use_default,
        fallback_to_finite_differences: bool = C.use_default,
        hvp_interval: int = C.use_default,
        hessian_approx: bool = C.use_default,
        compile_step: bool = C.use_default,
        promote: bool = C.use_default,
        ecc: str | None = None,
        param_ecc: str | None = None,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)

        params, defaults = C._build_defaults(locals())

        self.precond_schedule = C.default(
            defaults.pop("preconditioner_update_probability"), utils.precond_update_prob_schedule()
        )

        if rank is None:
            utils.warn_once(
                f"{rank=}. It will be set to log2(param_count). This requires `params` to be of type list. Currently, {type(params)=}"
            )
            params = list(params)
            defaults["rank"] = max(1, round(math.log2(sum(p.numel() for p in params))))
            utils.warn_once(f"rank was set to {defaults['rank']}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            fns=(*(C.exp_avg,) * exp_avg_input, C.scale_by_delayed_psgd_lra if delayed else C.scale_by_psgd_lra),
        )


class ForeachDelayedPSGDLRA(ForeachPSGDLRA):
    delayed: bool = True


class ForeachNewtonPSGDLRA(ForeachPSGDLRA):
    hessian_approx = True


class NewtonHybrid2PSGDLRA(ForeachNewtonPSGDLRA):
    hvp_interval = 2


class SplitOpt(utils.StatefulOptimizer):
    """
    Delegates different parameter groups to different underlying optimizers.

        opt = SplitOpt([
            {'params': matrices, 'optimizer': Muon, 'lr': 0.02},
            {'params': vectors, 'optimizer': AdamW, 'lr': 0.001},
        ])
    """

    def __init__(self, specs):
        self.optimizers, all_params = [], []
        for spec in specs:
            spec = dict(spec)
            params = list(spec.pop("params"))
            if params:
                self.optimizers.append(spec.pop("optimizer")(params, **spec))
                all_params.extend(params)
        if not self.optimizers:
            raise ValueError("No optimizers created")
        super().__init__(all_params, {}, foreach=True)

    def _step(self, group):
        pass

    def _handle_closure(self, closure):
        return self.optimizers[0]._handle_closure(closure)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._handle_closure(closure) if closure else None
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, s in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(s)


class SAMWrapper(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        wrapped_optimizer: Union[utils.StatefulOptimizer, Type[utils.StatefulOptimizer]] = ForeachAdamW,
        ball: float = 0.1,
    ):
        params = list(params)
        super().__init__(params, {"ball": ball})

        if isinstance(wrapped_optimizer, type):
            if not issubclass(wrapped_optimizer, utils.StatefulOptimizer):
                raise ValueError(f"{wrapped_optimizer.__name__} is not a HeavyBall optimizer")
            wrapped_optimizer = wrapped_optimizer(params)
        elif not isinstance(wrapped_optimizer, utils.StatefulOptimizer):
            raise ValueError(f"{wrapped_optimizer.__class__.__name__} is not a HeavyBall optimizer")

        self.wrapped_optimizer = wrapped_optimizer

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAM requires closure")
        with torch.enable_grad():
            closure()
        old_params = [utils.sam_step(group["params"], group["ball"]) for group in self.param_groups]

        original_handle_closure = self.wrapped_optimizer._handle_closure

        def _handle_closure(closure):
            try:
                _loss = original_handle_closure(closure)
            finally:
                for group, old in zip(self.param_groups, old_params):
                    utils.copy_stochastic_list_(group["params"], old)
            return _loss

        try:
            self.wrapped_optimizer._handle_closure = _handle_closure
            loss = self.wrapped_optimizer.step(closure)
        finally:
            self.wrapped_optimizer._handle_closure = original_handle_closure
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.wrapped_optimizer.zero_grad(set_to_none=set_to_none)


PalmForEachSoap = PaLMForeachSOAP
PaLMSOAP = PaLMForeachSOAP
PaLMSFAdamW = PaLMForeachSFAdamW
SOAP = ForeachSOAP
SOAPAdEMAMix = ForeachSOAPAdEMAMix
SOAPNAdam = ForeachSOAPNAdam
SFAdamW = ForeachSFAdamW
LaProp = ForeachLaProp
ADOPT = ForeachADOPT
RMSprop = ForeachRMSprop
PrecondScheduleSOAP = PrecondScheduleForeachSOAP
PrecondSchedulePaLMSOAP = PrecondSchedulePaLMForeachSOAP
PSGDKron = ForeachPSGDKron
AdamW = ForeachAdamW
NAdam = ForeachNAdam
PurePSGD = ForeachPurePSGD
DelayedPSGD = ForeachDelayedPSGD
CachedPSGDKron = ForeachCachedPSGDKron
CachedDelayedPSGDKron = ForeachCachedDelayedPSGDKron
Muon = ForeachMuon
SignLaProp = ForeachSignLaProp
DelayedPSGDLRA = ForeachDelayedPSGDLRA
PSGDLRA = ForeachPSGDLRA
NewtonPSGDLRA = ForeachNewtonPSGDLRA
NewtonPSGDKron = ForeachCachedNewtonPSGD

capture_param_shapes = utils.capture_param_shapes
__all__ = [k for k, v in globals().items() if isinstance(v, type) and issubclass(v, torch.optim.Optimizer)]
