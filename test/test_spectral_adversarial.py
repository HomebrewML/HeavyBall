import torch

from heavyball.kron import _max_singular_value_power_iter, _next_lower_bound


def test_power_iter_adversarial_underestimate():
    """The estimate remains a valid nonnegative lower bound on an adversarial matrix."""

    n_dominant = 100
    n_orthogonal = 2
    matrix = torch.zeros(1, n_dominant + n_orthogonal, 2, dtype=torch.float64)
    matrix[0, :n_dominant, 0] = 1.0
    matrix[0, n_dominant:, 1] = 2.0

    true_sigma = torch.linalg.svdvals(matrix[0]).max()
    estimated = _max_singular_value_power_iter(matrix, power_iterations=16)

    assert estimated.item() >= 0.0
    assert estimated.item() <= true_sigma.item() * (1 + 1e-12)


def test_running_lower_bound_keeps_psgd_step_safe_under_an_underestimate():
    """A stale high-curvature history keeps the normalized factor step below one."""

    true_norm = torch.tensor(10.0, dtype=torch.float64)
    underestimate = torch.tensor(2.0, dtype=torch.float64)
    beta = torch.tensor(0.9, dtype=torch.float64)
    precond_lr = torch.tensor(0.1, dtype=torch.float64)

    divisor, next_bound = _next_lower_bound(underestimate, true_norm, beta)
    effective_step = precond_lr * true_norm / divisor

    torch.testing.assert_close(divisor, next_bound, rtol=0, atol=0)
    assert effective_step < 1


def test_power_iter_random_matrices_accurate():
    """For random matrices (the typical PSGD case), power iteration is accurate."""
    torch.manual_seed(42)
    for _ in range(10):
        matrix = torch.randn(1, 32, 32, dtype=torch.float64)
        true_sigma = torch.linalg.svdvals(matrix[0]).max()
        estimated = _max_singular_value_power_iter(matrix, power_iterations=16)
        ratio = estimated.item() / true_sigma.item()
        # For random matrices, the estimate should be within 5% of true
        assert 0.95 < ratio < 1.05, f"Random matrix: ratio={ratio}"


def test_power_iter_psd_matrices_accurate():
    """For PSD matrices (what PSGD actually passes), power iteration is accurate."""
    torch.manual_seed(0)
    for _ in range(10):
        # PSGD passes covariance matrices: A^T A or A A^T
        a = torch.randn(1, 16, 32, dtype=torch.float64)
        psd = torch.bmm(a.mT, a)  # (1, 32, 32) PSD
        true_sigma = torch.linalg.eigvalsh(psd[0]).max()
        estimated = _max_singular_value_power_iter(psd, power_iterations=16)
        ratio = estimated.item() / true_sigma.item()
        assert 0.95 < ratio < 1.05, f"PSD matrix: ratio={ratio}"


def test_power_iter_batched():
    """Power iteration works correctly with batched input."""
    torch.manual_seed(7)
    batch = 4
    matrices = torch.randn(batch, 16, 16, dtype=torch.float64)
    estimated = _max_singular_value_power_iter(matrices, power_iterations=16)
    assert estimated.shape == (batch,)
    for i in range(batch):
        true_sigma = torch.linalg.svdvals(matrices[i]).max()
        ratio = estimated[i].item() / true_sigma.item()
        assert 0.90 < ratio < 1.10, f"Batch {i}: ratio={ratio}"


def test_running_lower_bound_catches_up():
    """The running_lower_bound EMA eventually recovers from an underestimate."""
    # Simulate: true ell=10, but power iter gives 2.83 for first few steps,
    # then corrects to 10 (as the matrix evolves).
    lower_bound = torch.tensor(0.0, dtype=torch.float64)
    beta = torch.tensor(0.9)

    # First 5 steps: underestimate
    for _ in range(5):
        ell = torch.tensor(2.83, dtype=torch.float64)
        _, lower_bound = _next_lower_bound(ell, lower_bound, beta)

    after_underestimate = lower_bound.item()
    assert after_underestimate < 3.0  # lower bound hasn't seen the true value yet

    # Next 20 steps: correct estimate
    for _ in range(20):
        ell = torch.tensor(10.0, dtype=torch.float64)
        _, lower_bound = _next_lower_bound(ell, lower_bound, beta)

    after_correction = lower_bound.item()
    # lower_bound should be close to 10 after EMA catches up
    assert after_correction > 9.0, f"Lower bound should have caught up: {after_correction}"
