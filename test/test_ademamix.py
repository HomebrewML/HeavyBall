import math
import os

import torch
import torch.testing

import heavyball

# Ensure Torch dynamo stays disabled on CI runners without GPU support.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
heavyball.utils.compile_mode = None


def _alpha_schedule(step: int, alpha: float, warmup: int | None) -> float:
    if not warmup or warmup <= 0:
        return alpha
    if step < warmup:
        return alpha * (step / float(warmup))
    return alpha


def _beta3_schedule(step: int, beta1: float, beta3: float, warmup: int | None, eps: float = 1e-8) -> float:
    if not warmup or warmup <= 0:
        return beta3
    if step >= warmup:
        return beta3

    half_life_start = math.log(0.5) / math.log(beta1 + eps) - 1.0
    half_life_end = math.log(0.5) / math.log(beta3 + eps) - 1.0
    a = step / float(warmup)
    target = (1.0 - a) * half_life_start + a * half_life_end
    beta = math.pow(0.5, 1.0 / (target + 1.0))
    return min(max(beta, 0.0), 1.0 - eps)


def _run_reference_ademamix(
    initial: torch.Tensor,
    grads: list[torch.Tensor],
    betas: tuple[float, float, float],
    lr: float,
    eps: float,
    alpha: float,
    weight_decay: float,
    beta3_warmup: int | None,
    alpha_warmup: int | None,
):
    def _beta_debias(beta: float, step: int) -> float:
        return 1.0 - (1.0 - beta) / (1.0 - beta**step)

    beta1, beta2, beta3 = betas
    param = initial.clone()
    m_fast = torch.zeros_like(initial)
    m_slow = torch.zeros_like(initial)
    v = torch.zeros_like(initial)

    for step, grad in enumerate(grads, start=1):
        grad_tensor = grad.to(param)
        alpha_eff = _alpha_schedule(step, alpha, alpha_warmup)
        beta3_eff = _beta3_schedule(step, beta1, beta3, beta3_warmup)

        beta1_eff = _beta_debias(beta1, step)
        beta2_eff = _beta_debias(beta2, step)

        m_fast = beta1_eff * m_fast + (1.0 - beta1_eff) * grad_tensor
        m_slow = beta3_eff * m_slow + (1.0 - beta3_eff) * grad_tensor
        v = beta2_eff * v + (1.0 - beta2_eff) * grad_tensor.square()

        denom = v.sqrt() + eps
        update = (m_fast + alpha_eff * m_slow) / denom

        param = param * (1.0 - weight_decay * lr) - lr * update

    return param, m_fast, m_slow, v


def test_ademamix_matches_reference_math():
    torch.manual_seed(2024)

    initial = torch.tensor([1.25, -0.75], dtype=torch.float32)
    grads = [
        torch.tensor([0.30, -0.10], dtype=torch.float32),
        torch.tensor([-0.50, 0.20], dtype=torch.float32),
        torch.tensor([0.15, 0.05], dtype=torch.float32),
    ]

    betas = (0.9, 0.999, 0.9999)
    lr = 0.0125
    eps = 1e-8
    alpha = 3.5
    weight_decay = 0.004
    beta3_warmup = 6
    alpha_warmup = 4

    param = torch.nn.Parameter(initial.clone())
    optimizer = heavyball.ForeachAdEMAMix(
        [param],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        alpha=alpha,
        beta3_warmup=beta3_warmup,
        alpha_warmup=alpha_warmup,
        foreach=False,
    )

    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()

    expected_param, expected_fast, expected_slow, expected_sq = _run_reference_ademamix(
        initial,
        grads,
        betas,
        lr,
        eps,
        alpha,
        weight_decay,
        beta3_warmup,
        alpha_warmup,
    )

    torch.testing.assert_close(param.detach(), expected_param, atol=1e-6, rtol=1e-5)

    state = optimizer.state[param][0]
    torch.testing.assert_close(state["update_by_ademamix_exp_avg_fast_0"], expected_fast, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state["update_by_ademamix_exp_avg_slow_0"], expected_slow, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state["update_by_ademamix_exp_avg_sq_0"], expected_sq, atol=1e-6, rtol=1e-5)


def test_soap_ademamix_projects_gradients_into_eigenbasis():
    torch.manual_seed(7)

    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = heavyball.ForeachSOAPAdEMAMix([param], lr=0.01, foreach=False)

    # First call initializes the SOAP preconditioner state without applying an update.
    param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param][0]
    Q = [q.clone() if q is not None else None for q in state["scale_by_soap_ademamix_Q_1"]]

    captured: dict[str, list[torch.Tensor] | tuple] = {}
    original = heavyball.utils.ademamix_

    def capture_ademamix(exp_avg_fast, exp_avg_slow, exp_avg_sq, grad, *rest):
        captured["grad"] = [g.clone() for g in grad]
        captured["betas"] = rest[0]
        captured["alpha"] = rest[3]
        captured["beta3_warmup"] = rest[4]
        captured["alpha_warmup"] = rest[5]
        return original(exp_avg_fast, exp_avg_slow, exp_avg_sq, grad, *rest)

    grad_step = torch.randn_like(param)

    heavyball.utils.ademamix_ = capture_ademamix
    try:
        param.grad = grad_step.clone()
        optimizer.step()
    finally:
        heavyball.utils.ademamix_ = original

    assert "grad" in captured, "AdEMAMix inner update was not invoked."

    expected_projected = heavyball.utils.project(grad_step.clone(), Q, False)
    torch.testing.assert_close(captured["grad"][0], expected_projected, atol=1e-6, rtol=1e-5)
    assert captured["betas"] == optimizer.param_groups[0]["betas"]
    assert captured["alpha"] == optimizer.param_groups[0]["alpha"]
    assert captured["beta3_warmup"] == optimizer.param_groups[0].get("beta3_warmup")
    assert captured["alpha_warmup"] == optimizer.param_groups[0].get("alpha_warmup")
