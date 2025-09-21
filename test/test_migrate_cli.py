import importlib.util
import pathlib
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


@pytest.fixture()
def runner():
    return CliRunner()


def test_cli_dry_run_updates_state(monkeypatch, runner, tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.touch()

    state_container = {"state": {"initial": True}, "param_groups": ["group"]}
    checkpoint = {"optimizer": state_container}

    def fake_load(path, map_location=None):
        return checkpoint

    def fake_migrate(state, _):
        return {"state": {"migrated": True}, "param_groups": []}

    def fail_save(*args, **kwargs):
        pytest.fail("torch.save should not run during dry-run")

    monkeypatch.setattr(migrate_script.torch, "load", fake_load)
    monkeypatch.setattr(migrate_script, "migrate_state_dict", fake_migrate)
    monkeypatch.setattr(migrate_script.torch, "save", fail_save)

    result = runner.invoke(
        migrate_script.app,
        [str(checkpoint_path), "heavyball.Mock", "--state-key", "optimizer", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert state_container == {"state": {"migrated": True}, "param_groups": []}


def test_cli_writes_output(monkeypatch, runner, tmp_path):
    checkpoint_path = tmp_path / "source.pt"
    checkpoint_path.touch()
    output_path = tmp_path / "out.pt"

    state_container = {"state": {"initial": True}, "param_groups": ["group"]}
    checkpoint = {"optimizer": state_container}
    migrated = {"state": {"migrated": True}, "param_groups": []}

    def fake_load(path, map_location=None):
        return checkpoint

    def fake_migrate(state, _):
        return migrated

    monkeypatch.setattr(migrate_script.torch, "load", fake_load)
    monkeypatch.setattr(migrate_script, "migrate_state_dict", fake_migrate)

    saved = {}

    def fake_save(obj, path):
        saved["obj"] = obj
        saved["path"] = pathlib.Path(path)

    monkeypatch.setattr(migrate_script.torch, "save", fake_save)

    result = runner.invoke(
        migrate_script.app,
        [str(checkpoint_path), "heavyball.Mock", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert saved["path"] == output_path
    assert saved["obj"]["optimizer"] == migrated


def test_cli_migrates_legacy_checkpoint(runner, tmp_path):
    package_root = pathlib.Path(__file__).resolve().parents[1]
    heavyball_pkg = package_root / "heavyball"
    saved_heavyball_modules = {
        name: sys.modules[name] for name in list(sys.modules) if name == "heavyball" or name.startswith("heavyball.")
    }
    for name in list(sys.modules):
        if name == "heavyball" or name.startswith("heavyball."):
            sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(
        "heavyball",
        heavyball_pkg / "__init__.py",
        submodule_search_locations=[str(heavyball_pkg)],
    )
    heavyball_module = importlib.util.module_from_spec(spec)
    sys.modules["heavyball"] = heavyball_module
    spec.loader.exec_module(heavyball_module)

    try:
        checkpoint_path = tmp_path / "legacy.pt"
        output_path = tmp_path / "migrated.pt"

        legacy_state = {
            "state": {
                0: {
                    "update_by_adam_exp_avg": torch.ones((2, 2), dtype=torch.float32),
                    "update_by_adam_exp_avg_sq": torch.full((2, 2), 2.0, dtype=torch.float32),
                    "is_initialized": [0],
                },
                1: {
                    "update_by_adam_exp_avg": torch.ones((2,), dtype=torch.float32),
                    "update_by_adam_exp_avg_sq": torch.full((2,), 2.0, dtype=torch.float32),
                    "is_initialized": [0],
                },
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 0.0025,
                    "betas": [0.9, 0.99],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "warmup_steps": 0,
                    "foreach": True,
                    "storage_dtype": "float32",
                    "mars": False,
                    "caution": False,
                    "mars_gamma": 0.0025,
                    "gradient_clipping": "use_default",
                    "update_clipping": "use_default",
                    "palm": "use_default",
                    "beta2_scale": 0.8,
                    "__class__": "heavyball.ForeachAdamW",
                }
            ],
        }

        torch.save({"optimizer": legacy_state}, checkpoint_path)

        result = runner.invoke(
            migrate_script.app,
            [str(checkpoint_path), "heavyball.ForeachAdamW", "--output", str(output_path)],
        )

        assert result.exit_code == 0, result.stderr or result.stdout
        assert output_path.exists()

        migrated = torch.load(output_path)
        migrated_state = migrated["optimizer"]

        for pid, shape in [(0, (2, 2)), (1, (2,))]:
            assert pid in migrated_state["state"], f"missing state for parameter {pid}"
            migrated_bucket = migrated_state["state"][pid]
            assert 0 in migrated_bucket, f"missing transformed view for parameter {pid}"
            view_state = migrated_bucket[0]
            assert view_state["update_by_adam_exp_avg_0"].shape == shape
            assert view_state["update_by_adam_exp_avg_sq_0"].shape == shape
            assert torch.allclose(view_state["update_by_adam_exp_avg_0"], torch.ones(shape))
            assert torch.allclose(view_state["update_by_adam_exp_avg_sq_0"], torch.full(shape, 2.0))
            assert view_state["is_initialized"] == [0]

        assert "heavyball" in migrated_state
        assert migrated_state["heavyball"]["inner_group"]["stochastic_schedule"] is None
        assert "Migrated checkpoint written to" in result.stdout
    finally:
        for name in list(sys.modules):
            if name == "heavyball" or name.startswith("heavyball."):
                sys.modules.pop(name)
        sys.modules.update(saved_heavyball_modules)
