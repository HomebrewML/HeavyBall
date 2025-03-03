import functools
from typing import Optional

from . import chainable as C
from . import utils


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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam
        )


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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
            C.update_by_schedule_free,
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adopt
        )


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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.nesterov_momentum if nesterov else C.heavyball_momentum,
            C.orthogonalize_update,
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_laprop
        )


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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.orthogonalize_update,
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
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(
                defaults.pop("precond_scheduler")
            )
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
            C.scale_by_soap,
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.sign,
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
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(
                defaults.pop("precond_scheduler")
            )
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
            functools.partial(C.scale_by_soap, inner="laprop"),
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.orthogonalize_grad_to_param,
            C.scale_by_laprop,
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
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.orthogonalize_grad_to_param,
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

    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
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
        # expert parameters
        precond_init_scale=1.0,
        precond_lr=0.1,
    ):
        defaults = locals()
        defaults.pop("self")
        self.precond_schedule = (
            defaults.pop("preconditioner_update_probability")
            or utils.precond_update_prob_schedule()
        )
        params = defaults.pop("params")

        delayed = C.default(delayed, self.delayed)
        cached = C.default(cached, self.cached)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            *(C.exp_avg,) * exp_avg_input,  #
            functools.partial(
                C.scale_by_delayed_psgd if delayed else C.scale_by_psgd, cached=cached
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


PalmForEachSoap = PaLMForeachSOAP
PaLMSOAP = PaLMForeachSOAP
PaLMSFAdamW = PaLMForeachSFAdamW
SOAP = ForeachSOAP
SFAdamW = ForeachSFAdamW
LaProp = ForeachLaProp
ADOPT = ForeachADOPT
RMSprop = ForeachRMSprop
PrecondScheduleSOAP = PrecondScheduleForeachSOAP
PrecondSchedulePaLMSOAP = PrecondSchedulePaLMForeachSOAP
PSGDKron = ForeachPSGDKron
AdamW = ForeachAdamW
PurePSGD = ForeachPurePSGD
DelayedPSGD = ForeachDelayedPSGD
CachedPSGDKron = ForeachCachedPSGDKron
CachedDelayedPSGDKron = ForeachCachedDelayedPSGDKron
Muon = ForeachMuon
SignLaProp = ForeachSignLaProp

__all__ = [
    "Muon",
    "RMSprop",
    "PrecondSchedulePaLMSOAP",
    "PSGDKron",
    "PurePSGD",
    "DelayedPSGD",
    "CachedPSGDKron",
    "CachedDelayedPSGDKron",
    "PalmForEachSoap",
    "PaLMSOAP",
    "PaLMSFAdamW",
    "LaProp",
    "ADOPT",
    "PrecondScheduleSOAP",
    "PrecondSchedulePaLMSOAP",
    "RMSprop",
    "MuonLaProp",
    "ForeachSignLaProp",  #
    "ForeachAdamW",
    "ForeachSFAdamW",
    "ForeachLaProp",
    "ForeachADOPT",
    "ForeachSOAP",
    "ForeachPSGDKron",
    "ForeachPurePSGD",
    "ForeachDelayedPSGD",
    "ForeachCachedPSGDKron",
    "ForeachCachedDelayedPSGDKron",
    "ForeachRMSprop",
    "ForeachMuon",
    "ForeachCachedNewtonPSGD",
    "OrthoLaProp",
    "LaPropOrtho",
    "SignLaProp",
]
