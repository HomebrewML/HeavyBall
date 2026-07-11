import pytest
import torch

import heavyball
from heavyball.utils import clean, set_torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "size,merge,split,expected",
    [
        ((4, 4, 4, 4), False, False, [(4, 4, 4, 4)]),
        ((4, 4, 4, 4), True, False, [(4, 4, 16)]),
        ((33, 17), True, True, [(16, 16), (16, 16), (1, 16), (16,), (16,), (1,)]),
    ],
)
def test_psgd_merge_layout_and_step(size, merge, split, expected):
    clean()
    set_torch()
    param = torch.nn.Parameter(torch.randn(size, device=DEVICE))
    initial = param.detach().clone()
    opt = heavyball.PSGDKron(
        [param],
        lr=1e-3,
        merge_dims=merge,
        split=split,
        max_size_triangular=16,
        preconditioner_update_probability=1.0,
        precond_init_scale=1.0,
        delayed=False,
        update_clipping=None,
    )
    assert [tuple(view.shape) for view in opt._set_views(param, opt.param_groups[0])] == expected

    for _ in range(2):
        param.grad = torch.randn_like(param)
        opt.step()
        opt.zero_grad()

    assert not torch.equal(param, initial)
    assert param.isfinite().all()
