"""Structural lock for HeavyBall's shared caution primitive."""

import ast
import importlib
from pathlib import Path


def test_one_shared_caution_definition():
    package_root = Path(__file__).parents[1] / "heavyball"
    names = ("_caution", "_strictly_aligned")
    definitions = {name: [] for name in names}

    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in definitions:
                definitions[node.name].append((source_path.relative_to(package_root), node.lineno))

    assert {name: len(locations) for name, locations in definitions.items()} == {
        "_caution": 1,
        "_strictly_aligned": 1,
    }, definitions

    numerics = importlib.import_module("heavyball.numerics")
    hyperball = importlib.import_module("heavyball.hyperball")
    msam = importlib.import_module("heavyball.msam")
    schedulefree = importlib.import_module("heavyball.schedulefree")
    transforms = importlib.import_module("heavyball.transforms")

    assert all(module._caution is numerics._caution for module in (hyperball, msam, schedulefree, transforms))
    # hyperball dropped its direct _strictly_aligned use with cautious_weight_decay (paper Hyperball has no
    # decay); it still shares _caution, which uses _strictly_aligned internally.
    assert all(module._strictly_aligned is numerics._strictly_aligned for module in (msam, schedulefree))
