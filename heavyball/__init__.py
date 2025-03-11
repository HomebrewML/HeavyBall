import functools
from typing import Optional

import torch

from . import chainable as C
from . import utils


class ForeachAdamW(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)


class ForeachRMSprop(C.BaseOpt):
    """
    Debiased RMSprop (not torch.optim.RMSprop)
    """

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-6, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.scale_by_exp_avg_sq)


class ForeachSFAdamW(C.ScheduleFree):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-6, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.scale_by_exp_avg_sq,
                         C.update_by_schedule_free)


class PaLMForeachSFAdamW(ForeachSFAdamW):
    palm: bool = True


class ForeachADOPT(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adopt)


class ForeachMuon(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8,
                 nesterov: bool = True):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,
                         C.nesterov_momentum if nesterov else C.heavyball_momentum, C.orthogonalize_update)


class ForeachLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_laprop)


class MuonLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.scale_by_laprop,
                         C.orthogonalize_update)


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

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 correct_bias: bool = True, warmup_steps: int = 0, split: bool = False, foreach: bool = True,
                 mars: bool = False, caution: bool = False, mars_gamma: float = 0.0025, palm: bool = C.use_default,
                 precond_scheduler=(1 / 3, 9), beta2_scale: float = 0.8, use_precond_schedule: bool = C.use_default,
                 gradient_clipping: C.str_or_fn = C.use_default, update_clipping: C.str_or_fn = C.use_default,
                 storage_dtype: str = 'float32', stochastic_schedule: bool = False):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")

        if use_precond_schedule:
            del defaults['precondition_frequency']
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults['precond_scheduler']
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,  #
                         C.scale_by_soap)


class ForeachSignLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.scale_by_laprop, C.sign)


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

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 correct_bias: bool = True, warmup_steps: int = 0, split: bool = False, foreach: bool = True,
                 mars: bool = False, caution: bool = False, mars_gamma: float = 0.0025, palm: bool = C.use_default,
                 precond_scheduler=(1 / 3, 9), beta2_scale: float = 0.8, use_precond_schedule: bool = C.use_default,
                 gradient_clipping: C.str_or_fn = C.use_default, update_clipping: C.str_or_fn = C.use_default,
                 storage_dtype: str = 'float32', stochastic_schedule: bool = False):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")

        if use_precond_schedule:
            del defaults['precondition_frequency']
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults['precond_scheduler']
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,  #
                         functools.partial(C.scale_by_soap, inner='laprop'))


class PaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = False
    palm: bool = True


class PrecondScheduleForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True


class PrecondSchedulePaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True
    palm: bool = True


class OrthoLaProp(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,
                         C.orthogonalize_grad_to_param, C.scale_by_laprop)


class LaPropOrtho(C.BaseOpt):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default, beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.scale_by_laprop,
                         C.orthogonalize_grad_to_param)


class ForeachPSGDKron(C.BaseOpt):
    """
    Originally from Evan Walters and Omead Pooladzandi, 2024
    Modified under Creative Commons Attribution 4.0 International
    Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    delayed: bool = False
    cached: bool = False
    exp_avg_input: bool = True

    def __init__(self, params, lr=0.001, beta=0.9, weight_decay=0.0, preconditioner_update_probability=None,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None,
                 momentum_into_precond_update=True, warmup_steps: int = 0, merge_dims: bool = False,
                 split: bool = False, store_triu_as_line: bool = True, foreach: bool = True, q_dtype='float32',
                 stochastic_schedule: bool = False, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025, delayed: Optional[bool] = C.use_default,
                 cached: Optional[bool] = C.use_default, exp_avg_input: Optional[bool] = C.use_default,
                 gradient_clipping: C.str_or_fn = C.use_default, update_clipping: C.str_or_fn = C.use_default,  #
                 # expert parameters
                 precond_init_scale=1.0, precond_lr=0.1):
        defaults = locals()
        defaults.pop("self")
        self.precond_schedule = defaults.pop(
            "preconditioner_update_probability") or utils.precond_update_prob_schedule()
        params = defaults.pop("params")

        delayed = C.default(delayed, self.delayed)
        cached = C.default(cached, self.cached)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, False,  #
                         *(C.exp_avg,) * exp_avg_input,  #
                         functools.partial(C.scale_by_delayed_psgd if delayed else C.scale_by_psgd, cached=cached))


class ForeachSophiaH(C.BaseOpt):
    """
    Sophia optimizer with simplified diagonal Hessian estimation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.96, 0.99), eps=1e-12, weight_decay=0.1,
                 warmup_steps=0, gamma=0.01, update_freq=10, foreach: bool = True,
                 storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                 mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default,
                 beta2_scale: float = 0.8):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")

        # Add Sophia-specific parameters to defaults
        defaults['sophia_gamma'] = defaults.pop('gamma')
        defaults['sophia_update_freq'] = defaults.pop('update_freq')

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,
                         C.zero_guard("diag_hessian"),
                         C.scale_by_sophia)

    def _step(self, group):
        """Override _step to add Hessian estimation before standard optimization steps."""
        if 'base_lr' not in group:
            group['base_lr'] = group['lr']
        if 'prev_lr' in group and group['prev_lr'] != group['lr']:
            utils.warn_once(f'Learning rate changed between steps. This is an experimental feature and '
                            f'only supported with foreach=True (currently foreach={group["foreach"]}).')
            group['base_lr'] = group['lr']

        caution = group['caution']

        vals = list(self.split_p_and_g_in_group(group, should_promote=self.promote, beta1=utils.get_beta1(group)))

        if not vals:
            return
        p, g = zip(*vals)

        for param in p:
            state = self.state_(param)
            if 'step' in state:
                step = state['step']
            elif self.compile_step:
                step = utils.scalar_guard(0, param)
            else:
                step = 0
            break

        group['step'] = state['step'] = step = step + 1
        group['prev_lr'] = group['lr'] = group['base_lr'] * step / max(step, group['warmup_steps'] + 1)

        # Update Hessian estimates directly here
        beta2 = utils.get_beta2(group)
        k = group.get('sophia_update_freq', 10)

        for param, grad in zip(p, g):
            state = self.state_(param)

            # Initialize Hessian's status
            if 'hessian_step' not in state:
                state['hessian_step'] = 0
                state['next_hessian_update'] = 1

            # Move to the next step
            state['hessian_step'] += 1

            # Check if hessian is updated
            if state['hessian_step'] >= state['next_hessian_update']:
                state['next_hessian_update'] = state['hessian_step'] + k

                # Hessian approximation using simple GNB method
                h = grad.pow(2)

                # Update hessian EMA : diag_hessian is initialized to zero_guard
                if 'diag_hessian' in state:
                    state['diag_hessian'].mul_(beta2).add_(h, alpha=1-beta2)
                else:
                    state['diag_hessian'] = h.clone()

                # Set minimum value for stability
                state['diag_hessian'].clamp_(min=1e-6)

        # Process standard initialization
        if not group['foreach'] or len(p) == 1:
            for param, grad in zip(p, g):
                C.chain(self.state_, group, [grad], [param], *self.fns)
        else:
            C.chain(self.state_, group, g, p, *self.fns)

        group['caution'] = caution
        group['lr'] = group['prev_lr']
        group['step'] = None

class ForeachSophiaG(ForeachSophiaH):
    """
    Sophia optimizer with Gauss-Newton-Bartlett estimator (same implementation as SophiaH for simplicity).
    """
    def __init__(self, params, lr=1e-3, betas=(0.96, 0.99), eps=1e-12, weight_decay=0.1,
                warmup_steps=0, gamma=0.05, update_freq=10, foreach: bool = True,
                storage_dtype: str = 'float32', mars: bool = False, caution: bool = False,
                mars_gamma: float = 0.0025, gradient_clipping: C.str_or_fn = C.use_default,
                update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default,
                beta2_scale: float = 0.8):

        defaults = locals()
        defaults.pop("self")
        params_local = defaults.pop("params")

        # Add Sophia-specific parameters to defaults
        defaults['sophia_gamma'] = defaults.pop('gamma')
        defaults['sophia_update_freq'] = defaults.pop('update_freq')
        defaults['sophia_type'] = 'g'  # Gauss-Newton-Bartlett estimator

        C.BaseOpt.__init__(self, params_local, defaults, foreach, gradient_clipping, update_clipping, palm,
                         C.zero_guard("diag_hessian"),
                         C.scale_by_sophia)


    def estimate_hessian_g(self, state, group, update, grad, param):
        """
        Estimate diagonal Hessian using Gauss-Newton-Bartlett method.
        """
        k = group.get('sophia_update_freq', 10)
        beta2 = utils.get_beta2(group)
        state['hessian_step'] += 1

        # Check if it's time to update the Hessian estimate
        if state['hessian_step'] >= state['next_hessian_update']:
            state['next_hessian_update'] = state['hessian_step'] + k

            # For GNB we can use a simplified approximation based on squared gradients
            # In a real LM, this should be implemented with proper label resampling
            h = grad * grad  # Simplified GNB estimator

            # Update EMA of diagonal Hessian
            if 'diag_hessian' in state:
                state['diag_hessian'].mul_(beta2).add_(h.abs(), alpha=1-beta2)
            else:
                state['diag_hessian'] = h.abs().clone()

            # Ensure positive values for numerical stability
            state['diag_hessian'].clamp_(min=1e-6)

        return update

class ForeachAdaLomo(C.BaseOpt):
    """
    AdaLomo: Low-memory Optimization with Adaptive Learning Rate

    As described in the paper:
    "AdaLomo: Low-memory Optimization with Adaptive Learning Rate"
    by Kai Lv, Hang Yan, Qipeng Guo, Haijun Lv, Xipeng Qiu

    AdaLomo combines:
    1. The memory-efficient approach of LOMO (Low-Memory Optimization)
    2. Adaptive learning rates using factorized second moments
    3. Grouped update normalization for stability

    Like LOMO, it updates parameters during the backward pass to save memory.
    Unlike LOMO, it uses an adaptive learning rate for each parameter, improving
    convergence while maintaining memory efficiency.

    Memory usage is comparable to parameter-efficient tuning methods (PEFT),
    while allowing full parameter update like AdamW.
    """
    def __init__(self, params, lr=5e-4, beta=0.99, eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32', mars: bool = False,
                 caution: bool = False, mars_gamma: float = 0.0025,
                 gradient_clipping: C.str_or_fn = C.use_default,
                 update_clipping: C.str_or_fn = C.use_default, palm: bool = C.use_default):
        """
        Initialize AdaLomo optimizer.

        Args:
            params: iterable of parameters to optimize
            lr: learning rate (default: 5e-4)
            beta: coefficient for computing running averages (default: 0.99)
            eps: term added to denominator for numerical stability (default: 1e-8)
            weight_decay: weight decay coefficient (default: 0)
            warmup_steps: number of warmup steps (default: 0)
            foreach: use foreach implementation if True (default: True)
            storage_dtype: data type for storing optimizer states (default: 'float32')
            mars: apply MARS correction if True (default: False)
            caution: prevent parameter updates that oppose gradients (default: False)
            mars_gamma: MARS correction strength (default: 0.0025)
            gradient_clipping: function or method name for gradient clipping (default: None)
            update_clipping: function or method name for update clipping (default: None)
            palm: use PaLM beta2 scheduler if True (default: False)
        """
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm,
                         C.update_by_adalomo)


class SophiaH(torch.optim.Optimizer):
    """
    Sophia optimizer with Gauss-Newton-Bartlett approximation for Hessian.
    """
    def __init__(self, params, lr=1e-3, betas=(0.96, 0.99), eps=1e-12, weight_decay=0.1,
                gamma=0.01, update_freq=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       gamma=gamma, update_freq=update_freq)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['diag_hessian'] = grad.pow(2).clone()
                    state['hessian_step'] = 0
                    state['next_hessian_update'] = 1

                # Load parameters
                beta1, beta2 = group['betas']
                gamma = group['gamma']
                eps = group['eps']
                lr = group['lr']
                weight_decay = group['weight_decay']
                update_freq = group['update_freq']

                # Step by step
                state['step'] += 1
                state['hessian_step'] += 1

                # Update hessian
                if state['hessian_step'] >= state['next_hessian_update']:
                    state['next_hessian_update'] = state['hessian_step'] + update_freq
                    h = grad.pow(2)
                    state['diag_hessian'].mul_(beta2).add_(h, alpha=1-beta2)
                    state['diag_hessian'].clamp_(min=1e-6)

                # Weight descent
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update momentum
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1-beta1)

                # Update Sophia & Parameters
                denom = torch.maximum(gamma * state['diag_hessian'],
                                     torch.tensor(eps, device=grad.device, dtype=grad.dtype))
                update = state['exp_avg'] / denom
                update.clamp_(-1.0, 1.0)
                p.data.add_(update, alpha=-lr)

        return loss

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
SophiaH = SophiaH
SophiaG = SophiaH


__all__ = ["Muon", "RMSprop", "PrecondSchedulePaLMSOAP", "PSGDKron", "PurePSGD", "DelayedPSGD", "CachedPSGDKron",
           "CachedDelayedPSGDKron", "PalmForEachSoap", "PaLMSOAP", "PaLMSFAdamW", "LaProp", "ADOPT",
           "PrecondScheduleSOAP", "PrecondSchedulePaLMSOAP", 'RMSprop', 'MuonLaProp', "ForeachSignLaProp",
           "ForeachAdamW", "ForeachSFAdamW",
           "ForeachLaProp", "ForeachADOPT", "ForeachSOAP", "ForeachPSGDKron", "ForeachPurePSGD", "ForeachDelayedPSGD",
           "ForeachCachedPSGDKron", "ForeachCachedDelayedPSGDKron", "ForeachRMSprop", "ForeachMuon",
           'ForeachCachedNewtonPSGD', 'OrthoLaProp', 'LaPropOrtho', 'SignLaProp', "SophiaH", "SophiaG"]
