"""SUDS moment transport: the stored moments must be consistent with the stored Fisher basis.

Four invariants:
1. Physical first-moment invariance: m_stored @ H(fisher_stored) is the same before and after transport.
2. Unchanged basis => unchanged moments (identity transport).
3. Second-moment trace preservation: sum(q) is conserved.
4. Rotating-basis momentum convergence: under a steadily rotating basis with constant physical gradient,
   transported momentum converges to the true gradient; untransported momentum lags with magnitude
   |1-beta| / |1 - beta*exp(i*delta)| < 1 and nonzero phase error.
"""

import torch
import pytest

from heavyball.suds import (
    _householder_vec_e1_to_v,
    _transport_rank1_first_moment,
    _transport_rank1_second_moment,
    eigvecs_product_rank1,
)


def _random_fisher(n_leaves, d, rng):
    v = torch.randn(n_leaves, d, generator=rng, dtype=torch.float64)
    return v / v.norm(dim=-1, keepdim=True)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


def test_physical_first_moment_invariance(rng):
    """m_old @ H_old == m_new @ H_new after transport."""
    n, d = 8, 64
    fisher_old = _random_fisher(n, d, rng)
    fisher_new = _random_fisher(n, d, rng)
    w_old = _householder_vec_e1_to_v(fisher_old)
    w_new = _householder_vec_e1_to_v(fisher_new)
    m_old = torch.randn(n, d, generator=rng, dtype=torch.float64)

    m_phys_before, _ = eigvecs_product_rank1(m_old, fisher_old, w_old)
    m_new = _transport_rank1_first_moment(m_old, w_old, w_new)
    m_phys_after, _ = eigvecs_product_rank1(m_new, fisher_new, w_new)

    torch.testing.assert_close(m_phys_before, m_phys_after, atol=1e-12, rtol=0)


def test_unchanged_basis_preserves_moments(rng):
    """Transport with old_w == new_w is identity."""
    n, d = 4, 32
    fisher = _random_fisher(n, d, rng)
    w = _householder_vec_e1_to_v(fisher)
    m = torch.randn(n, d, generator=rng, dtype=torch.float64)
    q = torch.rand(n, d, generator=rng, dtype=torch.float64).clamp_min(1e-8)

    m_t = _transport_rank1_first_moment(m, w, w)
    q_t = _transport_rank1_second_moment(q, w, w)

    torch.testing.assert_close(m_t, m, atol=1e-12, rtol=0)
    torch.testing.assert_close(q_t, q, atol=1e-12, rtol=0)


def test_second_moment_trace_preserved(rng):
    """sum(q_new) == sum(q_old) per leaf."""
    n, d = 8, 64
    fisher_old = _random_fisher(n, d, rng)
    fisher_new = _random_fisher(n, d, rng)
    w_old = _householder_vec_e1_to_v(fisher_old)
    w_new = _householder_vec_e1_to_v(fisher_new)
    q = torch.rand(n, d, generator=rng, dtype=torch.float64).clamp_min(1e-8)

    q_new = _transport_rank1_second_moment(q, w_old, w_new)
    torch.testing.assert_close(q.sum(dim=-1), q_new.sum(dim=-1), atol=1e-10, rtol=0)


def test_second_moment_matches_dense(rng):
    """O(d) formula matches the dense C^2 reference."""
    n, d = 4, 16
    fisher_old = _random_fisher(n, d, rng)
    fisher_new = _random_fisher(n, d, rng)
    w_old = _householder_vec_e1_to_v(fisher_old)
    w_new = _householder_vec_e1_to_v(fisher_new)
    q = torch.rand(n, d, generator=rng, dtype=torch.float64).clamp_min(1e-8)

    H_old = torch.eye(d, dtype=torch.float64) - 2.0 * w_old.unsqueeze(-1) * w_old.unsqueeze(-2)
    H_new = torch.eye(d, dtype=torch.float64) - 2.0 * w_new.unsqueeze(-1) * w_new.unsqueeze(-2)
    C = H_old @ H_new
    q_dense = torch.einsum("ni,nij->nj", q, C.square())

    q_fast = _transport_rank1_second_moment(q, w_old, w_new)
    torch.testing.assert_close(q_fast, q_dense, atol=1e-10, rtol=0)


def test_rotating_basis_momentum_converges(rng):
    """Transported momentum converges to the true gradient under steady basis rotation.

    d=2 so the Householder rotation spans the full space; in higher d only a 2D subspace rotates
    and untransported components outside that plane stay aligned, masking the bug.
    """
    d = 2
    beta1 = 0.9
    delta_deg = 5.0
    delta = delta_deg * torch.pi / 180.0
    steps = 200

    mu = torch.randn(1, d, generator=rng, dtype=torch.float64)
    mu = mu / mu.norm()

    theta = torch.tensor(0.3, dtype=torch.float64)
    m_transported = torch.zeros(1, d, dtype=torch.float64)
    m_raw = torch.zeros(1, d, dtype=torch.float64)

    for t in range(steps):
        angle = theta + t * delta
        fisher = torch.zeros(1, d, dtype=torch.float64)
        fisher[0, 0] = angle.cos()
        fisher[0, 1] = angle.sin()
        fisher = fisher / fisher.norm(dim=-1, keepdim=True)
        w = _householder_vec_e1_to_v(fisher)

        g_basis, _ = eigvecs_product_rank1(mu, fisher, w)
        m_transported = beta1 * m_transported + (1 - beta1) * g_basis
        m_raw = beta1 * m_raw + (1 - beta1) * g_basis

        if t < steps - 1:
            angle_next = theta + (t + 1) * delta
            fisher_next = torch.zeros(1, d, dtype=torch.float64)
            fisher_next[0, 0] = angle_next.cos()
            fisher_next[0, 1] = angle_next.sin()
            fisher_next = fisher_next / fisher_next.norm(dim=-1, keepdim=True)
            w_next = _householder_vec_e1_to_v(fisher_next)

            m_transported = _transport_rank1_first_moment(m_transported, w, w_next)
            # m_raw is NOT transported (simulates the bug)

    m_phys_transported, _ = eigvecs_product_rank1(m_transported, fisher, w)
    m_phys_raw, _ = eigvecs_product_rank1(m_raw, fisher, w)

    cos_transported = (m_phys_transported * mu).sum() / (m_phys_transported.norm() * mu.norm())
    cos_raw = (m_phys_raw * mu).sum() / (m_phys_raw.norm() * mu.norm())

    assert cos_transported.item() > 0.999, f"transported cos={cos_transported.item():.4f}, expected ~1.0"
    assert cos_raw.item() < 0.95, f"raw cos={cos_raw.item():.4f}, expected lag from basis drift"
