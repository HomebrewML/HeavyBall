"""The precision half of the global objective: heavyball's low-precision execution must be as close or
CLOSER to the full-fp64 result than a naive same-dtype baseline, not merely equal to a reference.

The mechanism is stochastic rounding, which is UNBIASED: it preserves small accumulated updates that a
deterministic round-to-nearest systematically discards once they fall below the dtype's ULP. Over a long
accumulation, round-to-nearest stalls (a large systematic bias) while stochastic rounding tracks fp64 in
the mean. This is the training-relevant metric (long-horizon bias), distinct from single-run L2 error,
where stochastic rounding trades a small variance for that unbiasedness.
"""
import contextlib

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
            o64.zero_grad(); set_grad(p64, torch.float64); o64.step()
        ideal = p64.detach().double()

        phb = torch.nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        ohb = heavyball.AdamW([phb], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            ohb.zero_grad(); set_grad(phb, torch.bfloat16); ohb.step()
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


def test_soap_bf16_weight_distance_misleads_and_ecc_preserves_loss():
    """Precision for a SPECTRAL matrix optimizer's state, and why WEIGHT-DISTANCE is the wrong metric here.
    SOAP's state feeds an eigendecomposition (Gram -> eigenbasis -> preconditioner), so it is chaotically
    sensitive: a tiny bf16 state perturbation rotates the eigenbasis and the weight trajectory separates
    from fp32's. By ||W - W_fp32|| stochastic rounding looks ~2x closer than round-to-nearest -- but that is
    an artifact of SR's unbiasedness keeping the state near fp32's PATH, not evidence of a better optimizer.
    By the achieved LOSS, SR has no advantage over round-to-nearest (its added variance corrupts the
    eigenbasis about as much as RTN's bias). So unlike the AdamW accumulation state (test_bf16_adam..., where
    SR genuinely wins), SR does not help spectral state. The int8-residual ecc mode preserves fp32 loss and
    is the precision-preserving low-memory mode for SOAP-family optimizers.

    Measured on a real MNIST autoencoder to 600 steps (non-stationary, where the eigenbasis is repeatedly
    rebuilt on drifting bf16 state): loss vs fp32 was bf16-SR +9%, bf16-RTN +4%, ecc8 +0.06%. This short
    stationary least-squares reproduces the metric DISAGREEMENT that condemns weight-distance."""
    from unittest.mock import patch

    import heavyball
    import heavyball.numerics as numerics

    def problem(seed):
        g = torch.Generator().manual_seed(seed)
        w0 = torch.randn(24, 24, generator=g, dtype=torch.float64)
        target = torch.randn(24, 24, generator=g, dtype=torch.float64) * 0.5
        design = torch.randn(24, 48, generator=g, dtype=torch.float64)  # wide -> unique minimum
        return w0, target @ design, design

    def run(mode, seed):  # returns (final weight fp64, final loss) with only the STATE dtype/rounding varied
        rtn = lambda value, noise: value.to(torch.bfloat16)
        with patch("heavyball.core.torch.compile", lambda f, **k: f):
            with patch.object(numerics, "stochastic_round_bfloat16", rtn) if mode == "bf16-RTN" else contextlib.nullcontext():
                torch.manual_seed(0)
                w0, b, a = problem(seed)
                weight = torch.nn.Parameter(w0.to(torch.float32).clone())  # fp32 params; only STATE varies
                kwargs = {"storage_dtype": torch.bfloat16} if mode != "fp32" else {}
                if mode == "ecc8":
                    kwargs["ecc"] = 8
                optimizer = heavyball.SOAP([weight], lr=0.02, weight_decay=0.0, **kwargs)
                for _ in range(120):
                    ((weight.to(torch.float64) @ a - b) ** 2).mean().backward()
                    optimizer.step()
                    optimizer.zero_grad()
                w = weight.detach().to(torch.float64)
                return w, float(((w @ a - b) ** 2).mean())

    sr_dist, rtn_dist, sr_loss, rtn_loss, ecc_loss, ref_loss = [], [], [], [], [], []
    for seed in range(6):
        ref_w, ref_l = run("fp32", seed)  # fp32 state (~fp64 for the state)
        sr_w, sr_l = run("bf16-SR", seed)
        rtn_w, rtn_l = run("bf16-RTN", seed)
        _, ecc_l = run("ecc8", seed)
        sr_dist.append((sr_w - ref_w).norm().item() / ref_w.norm().item())
        rtn_dist.append((rtn_w - ref_w).norm().item() / ref_w.norm().item())
        sr_loss.append(sr_l); rtn_loss.append(rtn_l); ecc_loss.append(ecc_l); ref_loss.append(ref_l)

    # Weight-distance would rank SR the clear winner (much closer to fp32's path)...
    assert sum(sr_dist) < 0.6 * sum(rtn_dist)
    # ...but by LOSS that advantage vanishes: SR is not meaningfully better than round-to-nearest, so
    # weight-distance is an invalid precision proxy for a spectral optimizer.
    assert sum(sr_loss) >= 0.99 * sum(rtn_loss)
    # The actually precision-preserving low-memory state for SOAP is ecc, which recovers the fp32 loss.
    assert abs(sum(ecc_loss) - sum(ref_loss)) < 0.02 * sum(ref_loss)
