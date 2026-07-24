"""Behavioral coverage for small edge paths shared across HeavyBall modules."""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import heavyball
from heavyball.hyperball import _stable_l2_components, _unpack_init_norm
from heavyball.lather import _next_psgd_eigenbasis, lather_init, make_lather
from heavyball.numerics import stable_l2_normalize
from heavyball.optim import _materialize_params
from heavyball.programs import _scalar
from heavyball.scion import _scion_bias_rms_direction, scion_param_init
from heavyball.transforms import Tempo, _stable_matrix_normalize, orthograd, sign_graft


def _tempo(count: int) -> Tempo:
    return Tempo(
        step=torch.ones((), dtype=torch.int64),
        age=torch.ones(count, dtype=torch.int64),
        live=torch.ones(count, dtype=torch.bool),
        hyper=SimpleNamespace(),
        refresh=False,
    )


def test_autocomplete_script_writes_the_rendered_stub():
    stub_generator = Path(heavyball.__file__).with_name("_autocomplete_stub.py")
    output = Path(heavyball.optim.__file__).with_suffix(".pyi")
    original = output.read_text(encoding="utf-8")

    runpy.run_path(str(stub_generator), run_name="__main__")

    from heavyball._autocomplete_stub import render

    generated = output.read_text(encoding="utf-8")
    assert generated == original
    assert generated == render()


def test_hyperball_empty_components_and_legacy_norm_state():
    empty = torch.empty((3, 0), dtype=torch.float64)
    scale, normalized_norm = _stable_l2_components(empty)
    assert torch.equal(scale, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(normalized_norm, torch.ones(3, dtype=torch.float64))

    legacy_init_norm = torch.tensor(((3.5,), (0.0,)), dtype=torch.float64)
    norm_scale, scaled_norm = _unpack_init_norm(legacy_init_norm)
    assert torch.equal(norm_scale, torch.tensor((3.5, 0.0), dtype=torch.float64))
    assert torch.equal(scaled_norm, torch.ones(2, dtype=torch.float64))


def test_lather_rejects_invalid_bases_shapes_and_configuration():
    with pytest.raises(
        ValueError,
        match="_next_psgd_eigenbasis requires a previous basis for matrix blocks; use _initial_psgd_eigenbasis",
    ):
        _next_psgd_eigenbasis(torch.eye(2).unsqueeze(0), None)

    with pytest.raises(
        ValueError,
        match="lather requires a leaf whose dimensions merge to 2D at max_precond_dim=2048",
    ):
        lather_init(torch.ones(3))
    with pytest.raises(ValueError, match="lather requires nonempty merged 2D parameter leaves"):
        lather_init(torch.empty(0, 3))
    with pytest.raises(TypeError, match="power_iterations must be a Python int"):
        make_lather(2.0)


def test_empty_normalizers_and_grafts_preserve_empty_slab_contracts():
    empty = torch.empty((2, 0), dtype=torch.float64)
    normalized = stable_l2_normalize(empty, dim=1, eps=1e-8)
    assert normalized.shape == (2, 0)
    assert normalized.dtype is torch.float64
    assert normalized.numel() == 0

    tempo = _tempo(2)
    grafted, graft_state, graft_live = sign_graft(empty, None, None, {}, tempo)
    assert grafted.shape == (2, 0)
    assert graft_state == {}
    assert torch.equal(graft_live, tempo.live)

    projected, projected_state, projected_live = orthograd(empty, None, empty, {}, tempo)
    assert projected.shape == (2, 0)
    assert projected_state == {}
    assert torch.equal(projected_live, tempo.live)

    matrix_empty = torch.empty((2, 0, 3), dtype=torch.float64)
    matrix_normalized = _stable_matrix_normalize(matrix_empty, eps=1e-7)
    assert matrix_normalized.shape == (2, 0, 3)
    assert matrix_normalized.dtype is torch.float64
    assert matrix_normalized.numel() == 0


def test_stable_l2_normalize_infinite_inputs():
    inf_vec = torch.tensor([[float("inf"), 1.0]], dtype=torch.float64)
    result = stable_l2_normalize(inf_vec, dim=-1, eps=None)
    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0]], dtype=torch.float64), rtol=0, atol=1e-15)

    mixed_inf = torch.tensor([[-float("inf"), float("inf")]], dtype=torch.float64)
    result_mixed = stable_l2_normalize(mixed_inf, dim=-1, eps=None)
    expected = torch.tensor([[-1.0, 1.0]], dtype=torch.float64) / (2**0.5)
    torch.testing.assert_close(result_mixed, expected, rtol=0, atol=1e-15)

    batched = torch.tensor(
        [[float("inf"), 1.0], [3.0, 4.0], [-float("inf"), float("inf")]],
        dtype=torch.float64,
    )
    result_batched = stable_l2_normalize(batched, dim=-1, eps=None)
    expected_batched = torch.tensor(
        [[1.0, 0.0], [0.6, 0.8], [-1 / 2**0.5, 1 / 2**0.5]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result_batched, expected_batched, rtol=0, atol=1e-15)


def test_program_scalar_validation_and_zero_grad_delegation():
    reference = torch.zeros((), dtype=torch.float64)
    supplied = torch.tensor(0.25, requires_grad=True)
    scalar = _scalar(supplied, reference)
    assert scalar.dtype is torch.float64
    assert scalar.device == reference.device
    assert scalar.item() == 0.25
    assert not scalar.requires_grad

    with pytest.raises(
        ValueError,
        match="program hyperparameters must be 0-d tensors or Python scalars",
    ):
        _scalar(torch.tensor((0.25,)), reference)
    with pytest.raises(TypeError, match="Program requires an Engine base"):
        heavyball.Program(object())

    parameter = torch.nn.Parameter(torch.ones(2))
    program = heavyball.Program(heavyball.Engine([parameter], heavyball.sgd, lr=0.1))
    parameter.grad.fill_(3.0)
    program.zero_grad()
    assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))


def test_sam_rejects_fsdp2_engines():
    parameter = torch.nn.Parameter(torch.ones(2))
    engine = heavyball.Engine([parameter], heavyball.sgd, lr=0.1)
    engine._fsdp2_manifest = True
    with pytest.raises(ValueError, match="SAM perturbation norm reduces only the local FSDP2 shard"):
        heavyball.SAM(engine)


def test_sam_rejects_missing_closure_and_parameters_on_multiple_devices():
    parameter = torch.nn.Parameter(torch.ones(1))
    sam = heavyball.SAM(heavyball.Engine([parameter], heavyball.sgd, lr=0.1))
    with pytest.raises(ValueError, match="SAM requires closure"):
        sam.step(None)

    cpu_parameter = torch.nn.Parameter(torch.ones(1))
    meta_parameter = torch.nn.Parameter(torch.ones(1, device="meta"))
    multi_device = heavyball.Engine([cpu_parameter, meta_parameter], heavyball.sgd, lr=0.1)
    with pytest.raises(
        ValueError,
        match="SAM requires parameters on one device for its global perturbation norm",
    ):
        heavyball.SAM(multi_device)


def test_sam_perturbation_uses_global_norm_across_slab_groups():
    torch._dynamo.reset()
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))

    try:
        with torch.compiler.set_stance("force_eager"):
            base = heavyball.Engine([first, second], heavyball.sgd, lr=0.0)
            assert len(base.groups) == 2
            sam = heavyball.SAM(base, rho=0.13, eps=0.0)
            first.grad.copy_(torch.tensor((3.0, 4.0)))
            second.grad.copy_(torch.tensor((12.0,)))

            sam.compiled_perturb()
            torch.testing.assert_close(first, torch.tensor((0.03, 0.04)), rtol=0, atol=1e-7)
            torch.testing.assert_close(second, torch.tensor((0.12,)), rtol=0, atol=1e-7)
            sam.compiled_restore()
            assert torch.equal(first, torch.zeros_like(first))
            assert torch.equal(second, torch.zeros_like(second))
    finally:
        torch._dynamo.reset()


def test_optimizer_parameter_group_validation():
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(
        TypeError,
        match="params must be an iterable of parameters or parameter groups",
    ):
        _materialize_params([parameter, {"params": [parameter]}])
    with pytest.raises(ValueError, match="parameter group must contain 'params'"):
        _materialize_params([{"lr": 0.1}])


def test_optimizer_rejects_foreign_parameter_and_malformed_engine_checkpoints():
    owned = torch.nn.Parameter(torch.ones(2))
    foreign = torch.nn.Parameter(torch.ones(2))
    optimizer = heavyball.AdamW([owned])
    with pytest.raises(ValueError, match="parameter is not owned by this Engine"):
        optimizer.produce(foreign, "sum_grad_squared", torch.ones_like(foreign))

    checkpoint = optimizer.state_dict()
    missing = copy.deepcopy(checkpoint)
    del missing["engines"]
    with pytest.raises(ValueError, match="expected a HeavyBallOptimizer 4.0 state dict with Engine state"):
        optimizer.load_state_dict(missing)

    wrong_count = copy.deepcopy(checkpoint)
    wrong_count["engines"] = []
    with pytest.raises(ValueError, match="checkpoint Engine state does not match this optimizer"):
        optimizer.load_state_dict(wrong_count)

    wrong_type = copy.deepcopy(checkpoint)
    wrong_type["engines"] = [42]
    with pytest.raises(ValueError, match="checkpoint Engine state does not match this optimizer"):
        optimizer.load_state_dict(wrong_type)


def test_scion_scalar_direction_is_sign_and_zero_preserving():
    direction = _scion_bias_rms_direction(
        torch.tensor((-2.0, 0.0, 4.0), dtype=torch.float64),
        torch.tensor(1e-8, dtype=torch.float64),
    )
    assert torch.equal(direction, torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float64))


def test_scion_bfloat16_seeded_initialization_is_deterministic_and_scaled_orthogonal():
    """The current bf16 path is seed-stable and preserves Scion's scaled-orthogonal contract."""

    seed = 23
    scale = torch.tensor(1.75)
    initial = torch.full((8, 8), 0.125, dtype=torch.bfloat16)
    actual = initial.clone()
    repeated = initial.clone()
    other_seed = initial.clone()

    scion_param_init(actual, seed=seed, scale=scale)
    scion_param_init(repeated, seed=seed, scale=scale)
    scion_param_init(other_seed, seed=seed + 1, scale=scale)

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, repeated)
    assert not torch.equal(actual, initial)
    assert not torch.equal(actual, other_seed)
    gram = actual.float().mT @ actual.float()
    torch.testing.assert_close(gram, torch.eye(8) * scale.square(), rtol=0, atol=0.025)
