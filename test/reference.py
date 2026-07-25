"""Reference optimizers, implemented from standard formulas rather than optimizer code.

The parity tests in ``test_reference.py`` check the shipped optimizers against these, so they validate
the math instead of agreement between two implementations.

Each function computes at ``init``'s dtype: fp64 gives the truth; a low-precision ``init`` gives the
naive same-math baseline (optimizer state kept at that dtype) that the shipped fp32-state path must beat.
Each function runs the full trajectory over ``grads`` from ``init`` and returns the final parameter.
The bias correction is the textbook ``m_hat = m / (1 - beta**t)``. Two shipped numerical floors are
formulation choices these references do not independently re-derive: the denominator uses
``sqrt(max(v, eps))`` (matched here), and the norm grafts (OrthoGrad/sign) keep the paper's exact
norm-preservation while the shipped stable-L2 graft caps amplification for orthogonal/sign norms below
~1e-6 (so rounding noise is not blown up to the full update norm). The references are therefore exact
away from ``v ~ eps`` and orthogonal/sign norms ``~ 1e-6``.
"""

import torch


def _run(init, grads, step):
    # Computes at init's dtype: fp64 init gives the truth; a low-precision init gives the naive
    # same-math baseline (state kept at that dtype, no fp32 promotion) that the shipped optimizer must beat.
    p = init.clone()
    state = {}
    for t, grad in enumerate(grads, start=1):
        p = step(p, grad.to(p.dtype), t, state)
    return p


def adam(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * g
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        m_hat = s["m"] / (1 - beta1**t)
        v_hat = s["v"] / (1 - beta2**t)
        update = m_hat / v_hat.clamp_min(eps).sqrt()
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def rmsprop(init, grads, *, lr, beta2, eps, weight_decay):
    def step(p, g, t, s):
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        v_hat = s["v"] / (1 - beta2**t)
        update = g / v_hat.clamp_min(eps).sqrt()
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def laprop(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        v_hat = s["v"] / (1 - beta2**t)
        normalized = g / v_hat.clamp_min(eps).sqrt()
        s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * normalized
        m_hat = s["m"] / (1 - beta1**t)
        return p - lr * (m_hat + weight_decay * p)

    return _run(init, grads, step)


def sgd(init, grads, *, lr, weight_decay):
    def step(p, g, t, s):
        return p - lr * (g + weight_decay * p)

    return _run(init, grads, step)


def lion(init, grads, *, lr, beta1, beta2, weight_decay):
    def step(p, g, t, s):
        m = s.get("m", torch.zeros_like(p))
        direction = (m * beta1 + g * (1 - beta1)).sign()
        s["m"] = m * beta2 + g * (1 - beta2)
        return p - lr * (direction + weight_decay * p)

    return _run(init, grads, step)


def signsgd(init, grads, *, lr, weight_decay):
    def step(p, g, t, s):
        return p - lr * (g.sign() + weight_decay * p)

    return _run(init, grads, step)


def unscaled_adam(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    # Adam whose first moment accumulates the variance-normalized gradient, then rescales by denom.
    def step(p, g, t, s):
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        denom = (s["v"] / (1 - beta2**t)).clamp_min(eps).sqrt()
        s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * (g / denom)
        m_hat = s["m"] / (1 - beta1**t)
        return p - lr * (m_hat * denom + weight_decay * p)

    return _run(init, grads, step)


def cautious_adam(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    # Adam masked to entries whose sign matches the raw gradient, rescaled to preserve mean magnitude.
    def step(p, g, t, s):
        s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * g
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        m_hat = s["m"] / (1 - beta1**t)
        v_hat = s["v"] / (1 - beta2**t)
        update = m_hat / v_hat.clamp_min(eps).sqrt()
        aligned = ((g > 0) & (update > 0)) | ((g < 0) & (update < 0))
        scale = update.numel() / aligned.sum().clamp_min(1).to(p.dtype)
        update = torch.where(aligned, update, torch.zeros_like(update)) * scale
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def ademamix(init, grads, *, lr, beta1, beta2, beta3, alpha, eps, weight_decay):
    def step(p, g, t, s):
        s["fast"] = beta1 * s.get("fast", 0.0) + (1 - beta1) * g
        s["slow"] = beta3 * s.get("slow", 0.0) + (1 - beta3) * g
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        fast_hat = s["fast"] / (1 - beta1**t)
        v_hat = s["v"] / (1 - beta2**t)
        # AdEMAMix leaves the slow EMA un-debiased, so it must not be corrected here.
        update = (fast_hat + alpha * s["slow"]) / v_hat.clamp_min(eps).sqrt()
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def _orthograd_direction(update, param):
    # Exact orthogonal projection (Grokking at the Edge of Numerical Stability, eq. 11): strip the
    # component of update along param, then graft the original update's norm back. A zero param leaves
    # the update untouched; a fully parallel update projects to nothing.
    flat_update, flat_param = update.reshape(-1), param.reshape(-1)
    param_sq = flat_param @ flat_param
    projection = torch.where(param_sq != 0, (flat_param @ flat_update) / param_sq, torch.zeros_like(param_sq))
    orthogonal = flat_update - projection * flat_param
    orth_norm = orthogonal.norm()
    grafted = torch.where(orth_norm != 0, orthogonal * (flat_update.norm() / orth_norm), torch.zeros_like(orthogonal))
    return grafted.reshape_as(update)


def _sign_graft(update):
    # Sign direction rescaled to the update's L2 norm; the unit sign vector has norm sqrt(numel).
    direction = update.sign()
    return direction * (update.norm() / direction.norm())


def _laprop_direction(g, t, s, *, beta1, beta2, eps):
    s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
    normalized = g / (s["v"] / (1 - beta2**t)).clamp_min(eps).sqrt()
    s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * normalized
    return s["m"] / (1 - beta1**t)


def _adam_direction(g, t, s, *, beta1, beta2, eps):
    s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * g
    s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
    return (s["m"] / (1 - beta1**t)) / (s["v"] / (1 - beta2**t)).clamp_min(eps).sqrt()


def adopt(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    # ADOPT (Algorithm 2): seed the second moment with g_1^2, normalize by the PREVIOUS second moment,
    # and take RAW (un-debiased) EMA steps -- the seeding is the only correction, no bias correction.
    def step(p, g, t, s):
        if t == 1:
            s["m"], s["v"] = torch.zeros_like(g), g * g
            return p
        normalized = g / s["v"].clamp_min(eps).sqrt()
        s["m"] = s["m"] * beta1 + normalized * (1 - beta1)
        s["v"] = s["v"] * beta2 + g * g * (1 - beta2)
        return p - lr * (s["m"] + weight_decay * p)

    return _run(init, grads, step)


def mars_adam(init, grads, *, lr, beta1, beta2, eps, weight_decay, mars_gamma):
    coefficient = mars_gamma * beta1 / (1 - beta1)

    def step(p, g, t, s):
        corrected = g + coefficient * (g - s.get("old", torch.zeros_like(g)))
        s["old"] = g
        # MARS (Algorithm 2) clips the corrected gradient to unit L2 norm before both Adam moments.
        corrected = corrected / corrected.norm().clamp_min(1.0)
        update = _adam_direction(corrected, t, s, beta1=beta1, beta2=beta2, eps=eps)
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def orthograd_adam(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        update = _adam_direction(_orthograd_direction(g, p), t, s, beta1=beta1, beta2=beta2, eps=eps)
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def ortho_laprop(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        update = _laprop_direction(_orthograd_direction(g, p), t, s, beta1=beta1, beta2=beta2, eps=eps)
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)


def laprop_ortho(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        direction = _laprop_direction(g, t, s, beta1=beta1, beta2=beta2, eps=eps)
        return p - lr * (_orthograd_direction(direction, p) + weight_decay * p)

    return _run(init, grads, step)


def sign_laprop(init, grads, *, lr, beta1, beta2, eps, weight_decay):
    def step(p, g, t, s):
        direction = _laprop_direction(g, t, s, beta1=beta1, beta2=beta2, eps=eps)
        return p - lr * (_sign_graft(direction) + weight_decay * p)

    return _run(init, grads, step)


def schedule_free_adamw(init, grads, *, lr, beta1, beta2, eps, weight_decay, weight_lr_power=2.0, r=0.0):
    # Defazio schedule-free AdamW (facebookresearch/schedule_free): an rmsprop-normalized gradient fed to
    # a schedule-free averaging iterate z. The PARAMETER is the train point y; there is no first moment.
    # Weight uses |lr| == the official lr_max for the constant lr schedule-free requires. Returns (y, z).
    p = init.clone()
    z = init.clone()
    v = torch.zeros_like(init)
    weight_sum = torch.zeros((), dtype=init.dtype)
    for t, g in enumerate(grads, start=1):
        g = g.to(p.dtype)
        v = beta2 * v + (1 - beta2) * g * g
        normalized = g / (v / (1 - beta2**t)).clamp_min(eps).sqrt()
        if weight_decay != 0:
            normalized = normalized + p * weight_decay
        weight = abs(lr) ** weight_lr_power * t**r
        weight_sum = weight_sum + weight
        ckp1 = weight / weight_sum
        p = torch.lerp(p, z, ckp1) + normalized * (lr * (beta1 * (1 - ckp1)) - lr)
        z = z - normalized * lr
    return p, z


def schedule_free_eval(train_iterate, z, beta1):
    # The eval point x = lerp(y, z, 1 - 1/beta1); beta1 == 0 leaves the parameter unchanged (no averaging).
    if beta1 <= 0:
        return train_iterate.clone()
    return torch.lerp(train_iterate, z, 1 - 1 / beta1)


def hyperball(init, grads, *, lr, beta1, beta2, eps, weight_decay=0.0, caution=False, cautious_weight_decay=False):
    # Hyperball (arXiv 2606.16899, Algorithm 1): constrain the matrix to the sphere R = ||W_0||_F using an
    # rmsprop-normalized base update -- W_{t+1} = R*Normalize(W_t - lr*R*Normalize(u_t)). weight_decay is
    # UNUSED: the paper REPLACES weight decay with the constraint (so cautious_weight_decay is moot). caution
    # is a no-op for this update (u shares the gradient's sign, so nothing is masked); it is accepted only to
    # match the recipe's hyper signature.
    del beta1, weight_decay, caution, cautious_weight_decay
    radius = init.norm()
    p = init.clone()
    v = torch.zeros_like(init)
    for t, g in enumerate(grads, start=1):
        g = g.to(p.dtype)
        v = beta2 * v + (1 - beta2) * g * g
        u = g / (v / (1 - beta2**t)).clamp_min(eps).sqrt()
        trial = p - lr * radius * (u / u.norm())
        p = radius * trial / trial.norm()
    return p


def nadam(init, grads, *, lr, beta1, beta2, eps, weight_decay, momentum_decay):
    def step(p, g, t, s):
        s["m"] = beta1 * s.get("m", 0.0) + (1 - beta1) * g  # first moment uses the raw beta1
        s["v"] = beta2 * s.get("v", 0.0) + (1 - beta2) * g * g
        denom = (s["v"] / (1 - beta2**t)).clamp_min(eps).sqrt()
        mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentum_decay))
        mu_next = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentum_decay))
        s["mu_product"] = s.get("mu_product", 1.0) * mu
        # Nesterov look-ahead: the schedule product carries the momentum bias correction, not 1-beta1**t.
        grad_weight = (1 - mu) / (1 - s["mu_product"])
        avg_weight = mu_next / (1 - s["mu_product"] * mu_next)
        update = g / denom * grad_weight + s["m"] / denom * avg_weight
        return p - lr * (update + weight_decay * p)

    return _run(init, grads, step)
