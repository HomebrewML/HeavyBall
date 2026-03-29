import importlib.util
import pathlib
import pickle
import random
import sys

import pytest
import torch
from typer.testing import CliRunner

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "migrate_optimizer_state.py"
MODULE_NAME = "scripts.migrate_optimizer_state"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC and SPEC.loader
migrate_script = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = migrate_script
SPEC.loader.exec_module(migrate_script)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Rename tables (exhaustive, mirrors _CLASS_RENAMES in the script)
# ---------------------------------------------------------------------------

_DIRECT_RENAMES = [
    ("ForeachAdamW", "AdamW"),
    ("ForeachNAdam", "NAdam"),
    ("ForeachAdEMAMix", "AdEMAMix"),
    ("ForeachAdamC", "AdamC"),
    ("ForeachRMSprop", "RMSprop"),
    ("ForeachSFAdamW", "SFAdamW"),
    ("ForeachADOPT", "ADOPT"),
    ("ForeachMuon", "Muon"),
    ("ForeachLaProp", "LaProp"),
    ("ForeachSOAP", "SOAP"),
    ("ForeachSOAPNAdam", "SOAPNAdam"),
    ("ForeachSOAPAdEMAMix", "SOAPAdEMAMix"),
    ("ForeachSignLaProp", "SignLaProp"),
    ("ForeachSOLP", "SOLP"),
    ("ForeachPSGDKron", "PSGDKron"),
    ("ForeachPSGDLRA", "PSGDLRA"),
]

_DELETED_RENAMES = [
    ("PaLMForeachSFAdamW", "SFAdamW"),
    ("PaLMForeachSOAP", "SOAP"),
    ("PrecondScheduleForeachSOAP", "SOAP"),
    ("PrecondSchedulePaLMForeachSOAP", "SOAP"),
    ("ForeachPurePSGD", "PSGDKron"),
    ("ForeachCachedPSGDKron", "PSGDKron"),
    ("ForeachCachedDelayedPSGDKron", "PSGDKron"),
    ("ForeachDelayedPSGD", "PSGDKron"),
    ("ForeachCachedNewtonPSGD", "PSGDKron"),
    ("NewtonHybrid2PSGDKron", "PSGDKron"),
    ("ForeachDelayedPSGDLRA", "PSGDLRA"),
    ("ForeachNewtonPSGDLRA", "PSGDLRA"),
    ("NewtonHybrid2PSGDLRA", "PSGDLRA"),
]

_ALL_RENAMES = _DIRECT_RENAMES + _DELETED_RENAMES

# PSGDLRA uses a different constructor signature (beta not betas, rank param, etc.)
# so e2e tests using AdamW-style param_groups must exclude it
_LRA_TARGETS = {"PSGDLRA"}
_DIRECT_RENAMES_NO_LRA = [(o, n) for o, n in _DIRECT_RENAMES if n not in _LRA_TARGETS]
_DELETED_RENAMES_NO_LRA = [(o, n) for o, n in _DELETED_RENAMES if n not in _LRA_TARGETS]
_DIRECT_RENAMES_LRA = [(o, n) for o, n in _DIRECT_RENAMES if n in _LRA_TARGETS]
_DELETED_RENAMES_LRA = [(o, n) for o, n in _DELETED_RENAMES if n in _LRA_TARGETS]

_PASSTHROUGH = ["AdamW", "NAdam", "PSGDKron", "SGD", "SOAP", "Muon", "ADOPT", "LaProp", "PSGDLRA", "SFAdamW"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_state(*shapes):
    return {
        i: {
            "update_by_adam_exp_avg": torch.ones(s),
            "update_by_adam_exp_avg_sq": torch.full(s, 2.0),
            "is_initialized": [0],
        }
        for i, s in enumerate(shapes)
    }


def _nested_state(*shapes):
    return {i: {0: {"key_0": torch.zeros(s), "is_initialized": [0]}} for i, s in enumerate(shapes)}


def _v1_group(pids):
    return {
        "params": pids,
        "lr": 0.0025,
        "betas": [0.9, 0.99],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "foreach": True,
        "stochastic_schedule": False,
        "storage_dtype": "float32",
        "mars": False,
        "caution": False,
        "mars_gamma": 0.0025,
        "gradient_clipping": "use_default",
        "update_clipping": "use_default",
        "palm": "use_default",
        "beta2_scale": 0.8,
    }


def _v2_group(pids):
    return {**_v1_group(pids), "__class__": "heavyball.ForeachAdamW"}


def _v3_group(pids):
    g = _v1_group(pids)
    g.pop("foreach")
    g.pop("stochastic_schedule")
    g["multi_tensor"] = True
    return g


def _v2_meta():
    return {
        "inner_group": {"stochastic_schedule": None},
        "stochastic_schedule": None,
        "precond_rng": pickle.dumps(random.Random(0x12312)),
        "use_ema": False,
    }


def _v3_meta():
    return {"inner_group": {}, "use_ema": False}


def _load_heavyball_fresh():
    package_root = pathlib.Path(__file__).resolve().parents[1]
    heavyball_pkg = package_root / "heavyball"
    saved = {n: sys.modules[n] for n in list(sys.modules) if n == "heavyball" or n.startswith("heavyball.")}
    for n in list(sys.modules):
        if n == "heavyball" or n.startswith("heavyball."):
            sys.modules.pop(n)
    spec = importlib.util.spec_from_file_location(
        "heavyball", heavyball_pkg / "__init__.py", submodule_search_locations=[str(heavyball_pkg)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["heavyball"] = mod
    spec.loader.exec_module(mod)
    return saved


def _restore_heavyball(saved):
    for n in list(sys.modules):
        if n == "heavyball" or n.startswith("heavyball."):
            sys.modules.pop(n)
    sys.modules.update(saved)


@pytest.fixture()
def runner():
    return CliRunner()


# ====================================================================
# _resolve_class_name
# ====================================================================


@pytest.mark.parametrize("old,new", _ALL_RENAMES)
def test_resolve_class_name_renames(old, new):
    assert migrate_script._resolve_class_name(old) == new


@pytest.mark.parametrize("name", _PASSTHROUGH)
def test_resolve_class_name_passthrough(name):
    assert migrate_script._resolve_class_name(name) == name


# ====================================================================
# _load_optimizer_class
# ====================================================================


@pytest.mark.parametrize("old,new", _ALL_RENAMES)
def test_load_optimizer_class_old_names(old, new):
    import heavyball

    assert migrate_script._load_optimizer_class(f"heavyball.{old}") is getattr(heavyball, new)


@pytest.mark.parametrize("name", _PASSTHROUGH)
def test_load_optimizer_class_current_names(name):
    import heavyball

    assert migrate_script._load_optimizer_class(name) is getattr(heavyball, name)


def test_load_optimizer_class_bare_name():
    import heavyball

    assert migrate_script._load_optimizer_class("AdamW") is heavyball.AdamW


def test_load_optimizer_class_invalid_raises():
    with pytest.raises(ValueError, match="not found"):
        migrate_script._load_optimizer_class("heavyball.TotallyFakeOptimizer")


# ====================================================================
# _detect_version
# ====================================================================

_DETECT_CASES = [
    ("flat_foreach", {"state": _flat_state((2,)), "param_groups": [{"foreach": True}]}, 1),
    ("flat_multi_tensor", {"state": _flat_state((2,)), "param_groups": [{"multi_tensor": True}]}, 1),
    ("flat_neither", {"state": _flat_state((2,)), "param_groups": [{}]}, 1),
    ("flat_multi_group", {"state": _flat_state((2,), (3,)), "param_groups": [{"foreach": True}, {}]}, 1),
    ("nested_foreach", {"state": _nested_state((2,)), "param_groups": [{"foreach": True}]}, 2),
    ("nested_foreach_multi_group", {"state": _nested_state((2,), (3,)), "param_groups": [{"foreach": True}, {"foreach": True}]}, 2),
    ("nested_multi_tensor", {"state": _nested_state((2,)), "param_groups": [{"multi_tensor": True}]}, 3),
    ("nested_multi_tensor_multi_group", {"state": _nested_state((2,), (3,)), "param_groups": [{"multi_tensor": True}, {}]}, 3),
    ("empty_state_multi_tensor", {"state": {}, "param_groups": [{"multi_tensor": True}]}, 3),
    ("empty_state_foreach", {"state": {}, "param_groups": [{"foreach": True}]}, 1),
    ("empty_state_bare", {"state": {}, "param_groups": [{}]}, 1),
    ("empty_state_empty_groups", {"state": {}, "param_groups": []}, 1),
]


@pytest.mark.parametrize("name,sd,expected", _DETECT_CASES, ids=[c[0] for c in _DETECT_CASES])
def test_detect_version(name, sd, expected):
    assert migrate_script._detect_version(sd) == expected


# ====================================================================
# _normalise_group_options
# ====================================================================

_NORM_CASES = [
    ("foreach_renamed", {"foreach": True, "lr": 0.01}, {"multi_tensor": True, "lr": 0.01}),
    ("foreach_false", {"foreach": False, "lr": 0.01}, {"multi_tensor": False, "lr": 0.01}),
    ("multi_tensor_passthrough", {"multi_tensor": False, "lr": 0.01}, {"multi_tensor": False, "lr": 0.01}),
    ("stochastic_stripped", {"foreach": True, "stochastic_schedule": False}, {"multi_tensor": True}),
    ("betas_tuple", {"betas": [0.9, 0.999]}, {"betas": (0.9, 0.999)}),
    ("weight_decay_steps_tuple", {"weight_decay_steps": [100, 200]}, {"weight_decay_steps": (100, 200)}),
    ("params_stripped", {"params": [0, 1], "lr": 0.01}, {"lr": 0.01}),
    ("empty_group", {}, {}),
    ("params_only", {"params": [0]}, {}),
    ("all_removed", {"params": [0], "stochastic_schedule": None}, {}),
    ("kitchen_sink", {"params": [0, 1], "foreach": True, "stochastic_schedule": True, "lr": 0.01, "betas": [0.9, 0.99]}, {"multi_tensor": True, "lr": 0.01, "betas": (0.9, 0.99)}),
]


@pytest.mark.parametrize("name,group,expected", _NORM_CASES, ids=[c[0] for c in _NORM_CASES])
def test_normalise_group_options(name, group, expected):
    assert migrate_script._normalise_group_options(group) == expected


# ====================================================================
# _ensure_set
# ====================================================================


@pytest.mark.parametrize(
    "inp,expected",
    [
        ({1, 2}, {1, 2}),
        ([3, 4], {3, 4}),
        ((5,), {5}),
        (None, set()),
        (7, {7}),
        ([], set()),
    ],
    ids=["set", "list", "tuple", "none", "scalar", "empty_list"],
)
def test_ensure_set(inp, expected):
    assert migrate_script._ensure_set(inp) == expected


# ====================================================================
# _guess_tensor_meta
# ====================================================================


@pytest.mark.parametrize(
    "entry,shape,dtype",
    [
        ({"a": torch.zeros(3, 4, dtype=torch.float16)}, (3, 4), torch.float16),
        ({"a": [torch.zeros(2, dtype=torch.bfloat16)]}, (2,), torch.bfloat16),
        ({"a": "not_a_tensor", "b": torch.ones(5)}, (5,), torch.float32),
        ({}, (1,), torch.float32),
        ({"a": 42}, (1,), torch.float32),
    ],
    ids=["tensor", "tensor_list", "mixed_finds_tensor", "empty", "no_tensors"],
)
def test_guess_tensor_meta(entry, shape, dtype):
    s, d = migrate_script._guess_tensor_meta(entry)
    assert s == shape
    assert d == dtype


# ====================================================================
# _resolve_state_container
# ====================================================================


def test_resolve_state_container_single_key():
    root = {"optimizer": {"state": {}, "param_groups": []}}
    assert migrate_script._resolve_state_container(root, ["optimizer"]) is root["optimizer"]


def test_resolve_state_container_nested_key():
    inner = {"state": {}, "param_groups": []}
    root = {"model": {"training": {"opt": inner}}}
    assert migrate_script._resolve_state_container(root, ["model", "training", "opt"]) is inner


def test_resolve_state_container_missing_key():
    with pytest.raises(KeyError, match="not found"):
        migrate_script._resolve_state_container({"a": {}}, ["a", "b"])


def test_resolve_state_container_not_optimizer():
    with pytest.raises(ValueError, match="not an optimizer state dict"):
        migrate_script._resolve_state_container({"opt": {"just_data": 1}}, ["opt"])


# ====================================================================
# _migrate_v2_to_v3
# ====================================================================


@pytest.mark.parametrize(
    "meta_keys",
    [
        {"stochastic_schedule": None, "precond_rng": pickle.dumps(random.Random(0))},
        {"stochastic_schedule": None},
        {"precond_rng": b"rng"},
        {},
    ],
    ids=["both", "stochastic_only", "precond_rng_only", "clean"],
)
def test_migrate_v2_to_v3_strips_stale_meta(meta_keys):
    sd = {
        "state": _nested_state((3,)),
        "param_groups": [{"params": [0], "foreach": True, "stochastic_schedule": False}],
        "heavyball": {"inner_group": {"stochastic_schedule": None}, "use_ema": False, **meta_keys},
    }
    migrate_script._migrate_v2_to_v3(sd)
    group = sd["param_groups"][0]
    assert group["multi_tensor"] is True
    assert "foreach" not in group
    assert "stochastic_schedule" not in group
    hb = sd["heavyball"]
    for key in ("stochastic_schedule", "precond_rng"):
        assert key not in hb
        assert key not in hb["inner_group"]
    assert hb["use_ema"] is False


def test_migrate_v2_to_v3_multi_group():
    sd = {
        "state": _nested_state((2,), (3,)),
        "param_groups": [
            {"params": [0], "foreach": True, "stochastic_schedule": False},
            {"params": [1], "foreach": False, "stochastic_schedule": True, "lr": 0.1},
        ],
        "heavyball": _v2_meta(),
    }
    migrate_script._migrate_v2_to_v3(sd)
    assert sd["param_groups"][0]["multi_tensor"] is True
    assert sd["param_groups"][1]["multi_tensor"] is False
    assert sd["param_groups"][1]["lr"] == 0.1
    for g in sd["param_groups"]:
        assert "foreach" not in g
        assert "stochastic_schedule" not in g


def test_migrate_v2_to_v3_preserves_non_stale_meta():
    sd = {
        "state": _nested_state((2,)),
        "param_groups": [{"params": [0], "foreach": True}],
        "heavyball": {**_v2_meta(), "use_ema": True, "ema_decay": 0.005, "compile_step": True},
    }
    migrate_script._migrate_v2_to_v3(sd)
    assert sd["heavyball"]["use_ema"] is True
    assert sd["heavyball"]["ema_decay"] == 0.005
    assert sd["heavyball"]["compile_step"] is True


def test_migrate_v2_to_v3_no_heavyball_key():
    sd = {
        "state": _nested_state((2,)),
        "param_groups": [{"params": [0], "foreach": True}],
    }
    migrate_script._migrate_v2_to_v3(sd)
    assert sd["param_groups"][0]["multi_tensor"] is True


# ====================================================================
# _migrate_single_state
# ====================================================================


@pytest.mark.parametrize("n_views", [1, 2, 5])
def test_migrate_single_state_rewrites_keys(n_views):
    mappings = [
        migrate_script.TransformMapping("exp_avg", "exp_avg_0", 0),
        migrate_script.TransformMapping("exp_avg_sq", "exp_avg_sq_0", 0),
    ]
    entry = {
        "exp_avg": [torch.ones(2)] * n_views if n_views > 1 else torch.ones(2),
        "exp_avg_sq": [torch.full((2,), 2.0)] * n_views if n_views > 1 else torch.full((2,), 2.0),
        "is_initialized": [0],
    }
    migrated = migrate_script._migrate_single_state(entry, mappings)
    for view_idx in range(n_views):
        bucket = migrated[view_idx]
        assert "exp_avg_0" in bucket
        assert "exp_avg_sq_0" in bucket
        assert "exp_avg" not in bucket
        assert "exp_avg_sq" not in bucket
        assert 0 in bucket["is_initialized"]


def test_migrate_single_state_empty_mappings():
    entry = {"some_key": torch.ones(3), "is_initialized": {0, 1}}
    migrated = migrate_script._migrate_single_state(entry, [])
    assert "some_key" in migrated[0]
    assert set(migrated[0]["is_initialized"]) == {0, 1}


def test_migrate_single_state_multiple_transforms():
    mappings = [
        migrate_script.TransformMapping("exp_avg", "update_by_adam_exp_avg_0", 0),
        migrate_script.TransformMapping("exp_avg_sq", "update_by_adam_exp_avg_sq_0", 0),
        migrate_script.TransformMapping("momentum", "heavyball_momentum_momentum_1", 1),
    ]
    entry = {
        "exp_avg": torch.ones(4),
        "exp_avg_sq": torch.full((4,), 2.0),
        "momentum": torch.full((4,), 0.5),
        "is_initialized": [0, 1],
    }
    migrated = migrate_script._migrate_single_state(entry, mappings)
    bucket = migrated[0]
    assert "update_by_adam_exp_avg_0" in bucket
    assert "update_by_adam_exp_avg_sq_0" in bucket
    assert "heavyball_momentum_momentum_1" in bucket
    assert set(bucket["is_initialized"]) == {0, 1}


def test_migrate_single_state_preserves_values():
    t = torch.arange(6, dtype=torch.float32)
    mappings = [migrate_script.TransformMapping("old", "new_0", 0)]
    migrated = migrate_script._migrate_single_state({"old": t, "is_initialized": [0]}, mappings)
    assert torch.equal(migrated[0]["new_0"], t)


# ====================================================================
# Full migrate_state_dict
# ====================================================================


def test_migrate_v2_state_dict():
    sd = {
        "state": _nested_state((4,)),
        "param_groups": [_v2_group([0])],
        "heavyball": _v2_meta(),
    }
    migrated = migrate_script.migrate_state_dict(sd, "heavyball.ForeachAdamW")
    group = migrated["param_groups"][0]
    assert group["multi_tensor"] is True
    assert "foreach" not in group
    assert "stochastic_schedule" not in group
    hb = migrated.get("heavyball", {})
    assert "stochastic_schedule" not in hb
    assert "precond_rng" not in hb


def test_migrate_v3_is_noop():
    sd = {
        "state": _nested_state((4,)),
        "param_groups": [_v3_group([0])],
        "heavyball": _v3_meta(),
    }
    original_key0 = sd["state"][0][0]["key_0"].clone()
    migrated = migrate_script.migrate_state_dict(sd, "heavyball.AdamW")
    assert migrated is not sd
    assert torch.equal(migrated["state"][0][0]["key_0"], original_key0)
    assert migrated["param_groups"][0]["multi_tensor"] is True


def test_migrate_v2_with_old_class_name():
    sd = {
        "state": _nested_state((4,)),
        "param_groups": [_v2_group([0])],
        "heavyball": _v2_meta(),
    }
    migrated = migrate_script.migrate_state_dict(sd, "ForeachAdamW")
    assert migrated["param_groups"][0]["multi_tensor"] is True


def test_migrate_v2_multi_param():
    sd = {
        "state": _nested_state((4, 4), (8,), (2, 2, 2)),
        "param_groups": [_v2_group([0, 1, 2])],
        "heavyball": _v2_meta(),
    }
    migrated = migrate_script.migrate_state_dict(sd, "AdamW")
    for pid in (0, 1, 2):
        assert pid in migrated["state"]
        assert 0 in migrated["state"][pid]


# ====================================================================
# CLI (typer) | mocked
# ====================================================================


def test_cli_dry_run(monkeypatch, runner, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.touch()
    container = {"state": {}, "param_groups": []}
    checkpoint = {"optimizer": container}

    monkeypatch.setattr(migrate_script.torch, "load", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(migrate_script, "migrate_state_dict", lambda s, _: {"state": {"ok": True}, "param_groups": []})
    monkeypatch.setattr(migrate_script.torch, "save", lambda *a, **kw: pytest.fail("save during dry-run"))

    result = runner.invoke(migrate_script.app, [str(ckpt), "heavyball.Mock", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert container["state"] == {"ok": True}


def test_cli_writes_output(monkeypatch, runner, tmp_path):
    ckpt = tmp_path / "source.pt"
    ckpt.touch()
    out = tmp_path / "out.pt"
    migrated = {"state": {"done": True}, "param_groups": []}
    checkpoint = {"optimizer": {"state": {}, "param_groups": []}}
    saved = {}

    monkeypatch.setattr(migrate_script.torch, "load", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(migrate_script, "migrate_state_dict", lambda s, _: migrated)
    monkeypatch.setattr(migrate_script.torch, "save", lambda obj, path: saved.update(obj=obj, path=pathlib.Path(path)))

    result = runner.invoke(migrate_script.app, [str(ckpt), "heavyball.Mock", "--output", str(out)])
    assert result.exit_code == 0
    assert saved["path"] == out
    assert saved["obj"]["optimizer"] == migrated


def test_cli_overwrites_input_by_default(monkeypatch, runner, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.touch()
    saved = {}

    monkeypatch.setattr(migrate_script.torch, "load", lambda *a, **kw: {"optimizer": {"state": {}, "param_groups": []}})
    monkeypatch.setattr(migrate_script, "migrate_state_dict", lambda s, _: {"state": {}, "param_groups": []})
    monkeypatch.setattr(migrate_script.torch, "save", lambda obj, path: saved.update(path=pathlib.Path(path)))

    result = runner.invoke(migrate_script.app, [str(ckpt), "heavyball.Mock"])
    assert result.exit_code == 0
    assert saved["path"] == ckpt


@pytest.mark.parametrize("key", ["optimizer", "model.optimizer", "a.b.c"])
def test_cli_state_key_parsing(monkeypatch, runner, tmp_path, key):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.touch()
    parts = key.split(".")
    inner = {"state": {}, "param_groups": []}
    root = inner
    for p in reversed(parts):
        root = {p: root}

    monkeypatch.setattr(migrate_script.torch, "load", lambda *a, **kw: root)
    monkeypatch.setattr(migrate_script, "migrate_state_dict", lambda s, _: {"state": {}, "param_groups": []})
    monkeypatch.setattr(migrate_script.torch, "save", lambda *a, **kw: None)

    result = runner.invoke(migrate_script.app, [str(ckpt), "heavyball.Mock", "--state-key", key])
    assert result.exit_code == 0


# ====================================================================
# CLI - real end-to-end (fresh heavyball import)
# ====================================================================


def _make_v1_checkpoint(path, shapes, *, group_overrides=None):
    g = _v1_group(list(range(len(shapes))))
    g["multi_tensor"] = True
    g.pop("foreach")
    g.pop("stochastic_schedule")
    if group_overrides:
        g.update(group_overrides)
    torch.save({"optimizer": {"state": _flat_state(*shapes), "param_groups": [g]}}, path)


def _make_v2_checkpoint(path, shapes, *, group_overrides=None):
    g = _v2_group(list(range(len(shapes))))
    if group_overrides:
        g.update(group_overrides)
    torch.save({"optimizer": {"state": _nested_state(*shapes), "param_groups": [g], "heavyball": _v2_meta()}}, path)


def _make_v3_checkpoint(path, shapes, *, group_overrides=None):
    g = _v3_group(list(range(len(shapes))))
    if group_overrides:
        g.update(group_overrides)
    torch.save({"optimizer": {"state": _nested_state(*shapes), "param_groups": [g], "heavyball": _v3_meta()}}, path)


def _run_cli_e2e(runner, ckpt, out, class_name):
    saved = _load_heavyball_fresh()
    try:
        result = runner.invoke(migrate_script.app, [str(ckpt), class_name, "--output", str(out)])
        assert result.exit_code == 0, result.stderr or result.stdout
        return torch.load(out)["optimizer"]
    finally:
        _restore_heavyball(saved)


# --- v1 end-to-end ---


def test_e2e_v1_adamw(runner, tmp_path):
    ckpt, out = tmp_path / "v1.pt", tmp_path / "out.pt"
    _make_v1_checkpoint(ckpt, [(2, 2), (2,)])
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    for pid, shape in [(0, (2, 2)), (1, (2,))]:
        view = migrated["state"][pid][0]
        assert view["update_by_adam_exp_avg_0"].shape == shape
        assert view["update_by_adam_exp_avg_sq_0"].shape == shape
        assert torch.allclose(view["update_by_adam_exp_avg_0"], torch.ones(shape))
        assert view["is_initialized"] == [0]
    assert "heavyball" in migrated


@pytest.mark.parametrize("old_name,new_name", _DIRECT_RENAMES_NO_LRA)
def test_e2e_v1_all_direct_renames(runner, tmp_path, old_name, new_name):
    ckpt, out = tmp_path / "v1.pt", tmp_path / "out.pt"
    _make_v1_checkpoint(ckpt, [(4,)])
    migrated = _run_cli_e2e(runner, ckpt, out, f"heavyball.{old_name}")
    assert 0 in migrated["state"]
    assert "heavyball" in migrated


@pytest.mark.parametrize("old_name,new_name", _DELETED_RENAMES_NO_LRA)
def test_e2e_v1_all_deleted_renames(runner, tmp_path, old_name, new_name):
    ckpt, out = tmp_path / "v1.pt", tmp_path / "out.pt"
    _make_v1_checkpoint(ckpt, [(4,)])
    migrated = _run_cli_e2e(runner, ckpt, out, f"heavyball.{old_name}")
    assert 0 in migrated["state"]
    assert "heavyball" in migrated


# PSGDLRA v1 e2e: PSGDLRA computes rank from param numel during __init__,
# which the migration script's _instantiate_optimizer can't replicate from
# a state dict alone (rank isn't stored in param_groups). The rename
# resolution and migration logic for LRA is covered by the unit tests above.


@pytest.mark.parametrize("n_params", [1, 3, 5])
def test_e2e_v1_varying_param_count(runner, tmp_path, n_params):
    ckpt, out = tmp_path / "v1.pt", tmp_path / "out.pt"
    shapes = [(i + 2,) for i in range(n_params)]
    _make_v1_checkpoint(ckpt, shapes)
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    for pid in range(n_params):
        assert pid in migrated["state"]


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
def test_e2e_v1_varying_shapes(runner, tmp_path, shape):
    ckpt, out = tmp_path / "v1.pt", tmp_path / "out.pt"
    _make_v1_checkpoint(ckpt, [shape])
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    assert migrated["state"][0][0]["update_by_adam_exp_avg_0"].shape == shape


# --- v2 end-to-end ---


def test_e2e_v2_basic(runner, tmp_path):
    ckpt, out = tmp_path / "v2.pt", tmp_path / "out.pt"
    _make_v2_checkpoint(ckpt, [(4,)])
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    group = migrated["param_groups"][0]
    assert group["multi_tensor"] is True
    assert "foreach" not in group
    assert "stochastic_schedule" not in group
    hb = migrated["heavyball"]
    assert "precond_rng" not in hb
    assert "stochastic_schedule" not in hb


@pytest.mark.parametrize("old_name,new_name", _DIRECT_RENAMES_NO_LRA)
def test_e2e_v2_all_direct_renames(runner, tmp_path, old_name, new_name):
    ckpt, out = tmp_path / "v2.pt", tmp_path / "out.pt"
    _make_v2_checkpoint(ckpt, [(4,)])
    migrated = _run_cli_e2e(runner, ckpt, out, f"heavyball.{old_name}")
    assert migrated["param_groups"][0]["multi_tensor"] is True
    assert "foreach" not in migrated["param_groups"][0]


@pytest.mark.parametrize("n_params", [1, 3, 5])
def test_e2e_v2_varying_param_count(runner, tmp_path, n_params):
    ckpt, out = tmp_path / "v2.pt", tmp_path / "out.pt"
    shapes = [(i + 2,) for i in range(n_params)]
    _make_v2_checkpoint(ckpt, shapes)
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    for pid in range(n_params):
        assert pid in migrated["state"]


# --- v3 end-to-end (noop) ---


def test_e2e_v3_noop(runner, tmp_path):
    ckpt, out = tmp_path / "v3.pt", tmp_path / "out.pt"
    _make_v3_checkpoint(ckpt, [(4,)])
    migrated = _run_cli_e2e(runner, ckpt, out, "heavyball.AdamW")
    assert migrated["param_groups"][0]["multi_tensor"] is True
    assert torch.equal(migrated["state"][0][0]["key_0"], torch.zeros(4))
