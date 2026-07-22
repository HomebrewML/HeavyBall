"""Fail-loud contracts for hyperparameter reads."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import heavyball
from heavyball import Recipe, Route

PUBLIC_RECIPES = tuple(
    (name, value)
    for name in heavyball.__all__
    if isinstance((value := getattr(heavyball, name)), (Recipe, Route))
)


def test_hyper_reads_do_not_use_getattr_fallbacks():
    package = Path(heavyball.__file__).parent
    fallback_reads = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        fallback_reads.extend(
            f"{path.relative_to(package)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "hyper"
        )

    assert not fallback_reads, "getattr hyper read(s): " + ", ".join(sorted(fallback_reads))


@pytest.mark.parametrize(
    ("name", "recipe"),
    [
        pytest.param(
            name,
            recipe,
            id=name,
            marks=pytest.mark.skip(reason="truegrad recipes require a sum_grad_squared producer")
            if name.startswith("truegrad_")
            else (),
        )
        for name, recipe in PUBLIC_RECIPES
    ],
)
def test_every_public_recipe_declares_hypers_read_by_both_step_types(name, recipe):
    del name
    params = [torch.nn.Parameter(torch.ones(2, 2))]
    if isinstance(recipe, Route):
        params.append(torch.nn.Parameter(torch.ones(3)))

    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        optimizer = heavyball.build(params, recipe)

    for step_type in ("refresh", "normal"):
        for param in params:
            param.grad.fill_(0.25)
        optimizer.step(step_type=step_type)
