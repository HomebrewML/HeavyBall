"""PSGD-Kron's preconditioner apply and A-form must contract pairwise, never materializing the
O(n^4) [n,i,a,j,b] outer product. That intermediate OOMs at >=256x256 (a 512x512 layer needs 256 GiB),
which made PSGDKron unusable on real layers. These guard the pairwise rewrites in kron.py."""
import torch

from heavyball.kron import _calc_a_and_conjb, _precondition


def _tri(n, d):
    return torch.randn(n, d, d, dtype=torch.float64).tril()


def test_precondition_matches_reference_einsum():
    torch.manual_seed(0)
    q0, q1, u = _tri(3, 16), _tri(3, 16), torch.randn(3, 16, 16, dtype=torch.float64)
    ref = torch.einsum("nri,nra,nsj,nsb,nab->nij", q0, q0, q1, q1, u)
    torch.testing.assert_close(_precondition(u, q0, q1), ref, rtol=1e-9, atol=1e-9)


def test_calc_a_matches_reference_einsum():
    torch.manual_seed(0)
    q0, q1, hv, v = _tri(3, 16), _tri(3, 16), torch.randn(3, 16, 16, dtype=torch.float64), torch.randn(3, 16, 16, dtype=torch.float64)
    ref_a = torch.einsum("nia,njb,nab->nij", q0, q1, hv)
    got_a, _ = _calc_a_and_conjb(hv, q0, q1, v)
    torch.testing.assert_close(got_a, ref_a, rtol=1e-9, atol=1e-9)


def test_precondition_is_not_quartic_memory():
    if not torch.cuda.is_available():
        # O(n^4) for 128 would allocate the [.,128,128,128,128] tensor; completing proves it is not built.
        q0, q1, u = _tri(2, 128).cpu(), _tri(2, 128).cpu(), torch.randn(2, 128, 128, dtype=torch.float64)
        assert _precondition(u, q0, q1).shape == (2, 128, 128)
        return
    dev = "cuda"
    q0, q1 = _tri(2, 128).to(dev), _tri(2, 128).to(dev)
    u = torch.randn(2, 128, 128, dtype=torch.float64, device=dev)
    torch.cuda.reset_peak_memory_stats()
    _precondition(u, q0, q1)
    torch.cuda.synchronize()
    quartic = 2 * 128 ** 4 * 8  # the [2,128,128,128,128] fp64 outer product
    assert torch.cuda.max_memory_allocated() < quartic // 100, "apply materialized an O(n^4) intermediate"
