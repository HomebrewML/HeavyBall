"""README python snippets import and run against the current API, so the entry-point doc cannot drift."""

import re
from pathlib import Path
from unittest.mock import patch

from torch import nn

README = Path(__file__).parents[1] / "README.md"


def _python_blocks():
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


def test_readme_has_python_snippets():
    assert _python_blocks()


def test_readme_snippets_run():
    # One namespace across blocks: imports shown once carry to later snippets, as a reader sees them.
    # A fresh model per block keeps each snippet's parameters independent.
    namespace = {}
    with patch("heavyball.core.torch.compile", lambda function, **kwargs: function):
        for block in _python_blocks():
            model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 4))
            namespace.update(model=model, body=model[0], norms=model[1])
            exec(compile(block, "<readme>", "exec"), namespace)
