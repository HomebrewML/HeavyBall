"""The precision half of the global objective: heavyball's low-precision execution must be as close or
CLOSER to the full-fp64 result than a naive same-dtype baseline, not merely equal to a reference.

The mechanism is stochastic rounding, which is UNBIASED: it preserves small accumulated updates that a
deterministic round-to-nearest systematically discards once they fall below the dtype's ULP. Over a long
accumulation, round-to-nearest stalls (a large systematic bias) while stochastic rounding tracks fp64 in
the mean. This is the training-relevant metric (long-horizon bias), distinct from single-run L2 error,
where stochastic rounding trades a small variance for that unbiasedness.
"""
import torch

from heavyball.numerics import stochastic_round_bfloat16


def test_stochastic_rounding_tracks_fp64_where_round_to_nearest_stalls():
    torch.manual_seed(0)
    dim, steps, delta = 8192, 400, 3e-3
    ideal = 4.0 - steps * delta  # 2.8; each step subtracts a sub-ULP amount from a value near 4

    p = torch.full((dim,), 4.0, dtype=torch.bfloat16)  # naive round-to-nearest baseline
    for _ in range(steps):
        p = (p.double() - delta).to(torch.bfloat16)
    naive_bias = abs(p.double().mean().item() - ideal)

    q = torch.full((dim,), 4.0, dtype=torch.bfloat16)  # heavyball stochastic rounding
    for _ in range(steps):
        q = stochastic_round_bfloat16(q.double() - delta, torch.rand(dim, dtype=torch.float64))
    stoch_bias = abs(q.double().mean().item() - ideal)

    assert naive_bias > 1.0            # round-to-nearest lost the entire accumulation (stalled at 4.0)
    assert stoch_bias < 0.01           # stochastic rounding tracked the fp64 result
    assert stoch_bias < naive_bias     # the objective: closer to fp64 than the same-dtype baseline


def test_bf16_adam_lands_closer_to_fp64_than_torch_baseline():
    """End-to-end precision objective on a real optimizer: heavyball's bf16 AdamW must land closer to the
    full-fp64 ideal than the naive dtype-baseline (torch.optim.AdamW in bf16). Same Adam algorithm, but
    heavyball keeps optimizer state in fp32 and stochastically rounds the parameter, while torch keeps
    bf16 state and rounds to nearest and so drifts over a long run. This is the objective, not matching."""
    from unittest.mock import patch

    torch.manual_seed(0)
    dim, steps, lr = 4096, 600, 0.02
    target = torch.randn(dim, dtype=torch.float64)

    def set_grad(p, dtype):
        with torch.no_grad():
            p.grad.copy_((p.double() - target).to(dtype))

    with patch("heavyball.core.torch.compile", lambda f, **k: f):
        import heavyball
        p64 = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        o64 = heavyball.AdamW([p64], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            o64.zero_grad()
            set_grad(p64, torch.float64)
            o64.step()
        ideal = p64.detach().double()

        phb = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        ohb = heavyball.AdamW([phb], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            ohb.zero_grad()
            set_grad(phb, torch.bfloat16)
            ohb.step()
        err_hb = (phb.detach().double() - ideal).norm().item()

    pt = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
    ot = torch.optim.AdamW([pt], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        with torch.no_grad():
            if pt.grad is None:
                pt.grad = torch.zeros_like(pt)
            pt.grad.copy_((pt.double() - target).to(torch.bfloat16))
        ot.step()
    err_torch = (pt.detach().double() - ideal).norm().item()

    assert err_hb < err_torch * 0.5  # heavyball's bf16 lands at least 2x closer to fp64 than torch's bf16
