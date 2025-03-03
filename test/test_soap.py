from itertools import chain

import pytest
import torch

from heavyball import utils
from heavyball.utils import dim_merger, promote


def init_preconditioner(
    grad,
    state,
    precondition_frequency=10,
    shampoo_beta=0.95,
    max_precond_dim=10000,
    precondition_1d=False,
    merge_dims=False,
):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state["GG"] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state["GG"].append([])
        else:
            state["GG"].append(
                torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype)
            )
    else:
        if merge_dims:
            grad = dim_merger(grad, max_precond_dim)

        for sh in grad.shape:
            if sh > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

    state["Q"] = None  # Will hold all the eigenbases of the preconditioner.
    state["precondition_frequency"] = precondition_frequency
    state["shampoo_beta"] = shampoo_beta


def project(grad, state, merge_dims=False, max_precond_dim=10000):
    """
    Projects the gradient to the eigenbases of the preconditioner.
    """
    original_shape = grad.shape
    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    for mat in state["Q"]:
        if len(mat) > 0:
            grad = torch.tensordot(
                grad,
                mat,
                dims=[[0], [0]],
            )
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = grad.permute(permute_order)

    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


def update_preconditioner(
    grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False
):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    if grad.dim() == 1:
        if precondition_1d and grad.shape[0] <= max_precond_dim:
            state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
    else:
        if merge_dims:
            new_grad = dim_merger(grad, max_precond_dim)
            for idx, sh in enumerate(new_grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(
                        new_grad,
                        new_grad,
                        dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                    )
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
        else:
            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(
                        grad,
                        grad,  # Contracts across all dimensions except for k.
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

    if state["Q"] is None:
        state["Q"] = get_orthogonal_matrix(state["GG"])
    if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
        state["Q"] = get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)


def project_back(grad, state, merge_dims=False, max_precond_dim=10000):
    """
    Projects the gradient back to the original space.
    """
    original_shape = grad.shape
    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)
    for mat in state["Q"]:
        if len(mat) > 0:
            grad = torch.tensordot(
                grad,
                mat,
                dims=[[0], [1]],
            )
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = grad.permute(permute_order)

    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype in (torch.float16, torch.float32):
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue
        try:
            _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
        except Exception:
            _, Q = torch.linalg.eigh(
                m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
            )
            Q = Q.to(m.dtype)
        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)
    return final


def get_orthogonal_matrix_QR(state, max_precond_dim=10000, merge_dims=False):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    precond_list = state["GG"]
    orth_list = state["Q"]

    matrix = []
    orth_matrix = []
    for m, o in zip(precond_list, orth_list):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype in (torch.float16, torch.float32):
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            float_data = True
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    orig_shape = state["exp_avg_sq"].shape
    if merge_dims:
        exp_avg_sq = dim_merger(state["exp_avg_sq"], max_precond_dim)
    else:
        exp_avg_sq = state["exp_avg_sq"]

    final = []
    for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
        if len(m) == 0:
            final.append([])
            continue
        est_eig = torch.diag(o.T @ m @ o)
        sort_idx = torch.argsort(est_eig, descending=True)
        exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
        o = o[:, sort_idx]
        power_iter = m @ o
        Q, _ = torch.linalg.qr(power_iter)

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    if merge_dims:
        exp_avg_sq = exp_avg_sq.reshape(orig_shape)

    state["exp_avg_sq"] = exp_avg_sq
    return final


def _init(size, max_precond, merge_dims, precondition_1d, beta, precondition_frequency=1):
    grad = torch.randn(size, dtype=torch.double)
    ref_state = {
        "step": 1,
        "exp_avg": torch.randn_like(grad),
        "exp_avg_sq": torch.randn_like(grad),
    }
    new_state = {
        "step": 1,
        **{k: v.clone() for k, v in ref_state.items() if isinstance(v, torch.Tensor)},
    }
    init_preconditioner(
        grad.clone(),
        ref_state,
        precondition_frequency=precondition_frequency,
        shampoo_beta=beta,
        max_precond_dim=max_precond,
        precondition_1d=precondition_1d,
        merge_dims=merge_dims,
    )
    utils.init_preconditioner(
        grad.clone(),
        new_state,
        max_precond_dim=max_precond,
        precondition_1d=precondition_1d,
        merge_dims=merge_dims,
    )
    return grad, ref_state, new_state


def _updated(
    size, max_precond, merge_dims, precondition_1d, beta, iterations, precondition_frequency=1
):
    grad, ref_state, new_state = _init(
        size, max_precond, merge_dims, precondition_1d, beta, precondition_frequency
    )
    for _ in range(iterations):
        ref_state["step"] += 1
        new_state["step"] += 1
        grad = torch.randn_like(grad)
        update_preconditioner(grad.clone(), ref_state, max_precond, merge_dims, precondition_1d)
        utils.update_preconditioner(
            grad.clone(),
            new_state,
            max_precond,
            merge_dims,
            precondition_1d,
            beta,
            precondition_frequency == 1,
        )
        yield grad, ref_state, new_state


def _check(ref, new):
    for k, rr in ref.items():
        if k not in new:
            continue
        nn = new[k]
        if isinstance(rr, list):
            for r, n in zip(rr, nn):
                if isinstance(r, torch.Tensor):
                    assert ref["step"] and k and torch.allclose(r, n)
        elif isinstance(rr, torch.Tensor):
            assert ref["step"] and k and torch.allclose(rr, nn)


_size = 16


@pytest.mark.parametrize("size", [(_size,), (_size,) * 2, (_size,) * 3])
@pytest.mark.parametrize("max_precond", [_size**2 * 2, _size * 2, _size // 2])
@pytest.mark.parametrize("merge_dims", [True, False])
@pytest.mark.parametrize("precondition_1d", [True, False])
@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
@torch.no_grad()
def test_init(size, max_precond, merge_dims, precondition_1d, beta):
    _grad, ref_state, new_state = _init(size, max_precond, merge_dims, precondition_1d, beta)
    _check(ref_state, new_state)


@pytest.mark.parametrize("size", [(_size,), (_size,) * 2, (_size,) * 3])
@pytest.mark.parametrize("max_precond", [_size**2 * 2, _size * 2, _size // 2])
@pytest.mark.parametrize("merge_dims", [True, False])
@pytest.mark.parametrize("precondition_1d", [True, False])
@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
@torch.no_grad()
def test_ggt(size, max_precond, merge_dims, precondition_1d, beta, iterations: int = 5):
    for grad, ref_state, new_state in _updated(
        size,
        max_precond,
        merge_dims,
        precondition_1d,
        beta,
        iterations,
        precondition_frequency=10**12,
    ):
        _check(ref_state, new_state)


@pytest.mark.parametrize("size", [(_size,), (_size,) * 2, (_size,) * 3])
@pytest.mark.parametrize("max_precond", [_size**2 * 2, _size * 2, _size // 2])
@pytest.mark.parametrize("merge_dims", [True, False])
@pytest.mark.parametrize("precondition_1d", [True, False])
@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
@torch.no_grad()
def test_update(size, max_precond, merge_dims, precondition_1d, beta, iterations: int = 5):
    for grad, ref_state, new_state in _updated(
        size, max_precond, merge_dims, precondition_1d, beta, iterations
    ):
        _check(ref_state, new_state)


@pytest.mark.parametrize("size", [(_size,), (_size,) * 2, (_size,) * 3])
@pytest.mark.parametrize("max_precond", [_size**2 * 2, _size * 2, _size // 2])
@pytest.mark.parametrize("merge_dims", [True, False])
@pytest.mark.parametrize("precondition_1d", [True, False])
@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("back", [True, False])
@torch.no_grad()
def test_project(size, max_precond, merge_dims, precondition_1d, beta, back, iterations: int = 5):
    for grad, ref_state, new_state in _updated(
        size, max_precond, merge_dims, precondition_1d, beta, iterations
    ):
        proj_ref = (project_back if back else project)(
            grad.clone(), ref_state, merge_dims, max_precond
        )
        proj_new = utils.project(grad.clone(), ref_state["Q"], merge_dims, max_precond, back)

        assert ref_state["step"] and torch.allclose(proj_ref.contiguous(), proj_new.contiguous())
