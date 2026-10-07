"""Colab setup installs into its runtime while ordinary notebook execution stays local."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import andrey

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def setup_source(path):
    notebook = json.loads(path.read_text())
    return "".join(
        next(c for c in notebook["cells"] if "colab-setup" in c["metadata"].get("tags", []))[
            "source"
        ]
    )


def run_setup(source, commands):
    """Run a setup cell as IPython would; each ``!`` line becomes a command in ``commands``."""
    python = re.sub(
        r"^(\s*)!(.*)$", lambda m: f"{m[1]}get_ipython().system({m[2]!r})", source, flags=re.M
    )
    shell = SimpleNamespace(system=commands.append)
    namespace = {"get_ipython": lambda: shell}
    exec(compile(python, "setup", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_colab_link_opens_the_notebook_from_the_public_repository(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    links = re.findall(
        r"https://colab\.research\.google\.com/github/\S+?(?=\))", "".join(cells[0]["source"])
    )
    public = "https://colab.research.google.com/github/Abel-ai-lab/andrey/blob/main/examples/"
    assert links == [public + notebook.name]
    # The badge has a line of its own: the docs put the notebook's download link to its right.
    badge = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({links[0]})"
    )
    assert badge in "".join(cells[0]["source"]).splitlines()


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_colab_installs_the_version_being_built(notebook):
    # Colab installs from PyPI, so a pin left behind by a release installs an older API.
    text = "".join("".join(c["source"]) for c in json.loads(notebook.read_text())["cells"])
    pins = re.findall(r"andrey-core(?:\[[\w,]+\])?==([\w.]+)", text)
    assert pins and set(pins) == {andrey.__version__}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_local_setup_never_invokes_pip(notebook, monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    commands = []
    namespace = run_setup(setup_source(notebook), commands)
    assert namespace["IN_COLAB"] is False
    assert commands == [], "a local or docs kernel must not install notebook dependencies"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_colab_installs_the_release_from_pypi(notebook, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace())
    # Preserve the test runner's device setting when the cell pins its runtime to CPU.
    monkeypatch.setenv("ANDREY_DEVICE", "cpu")
    commands = []
    run_setup(setup_source(notebook), commands)
    (command,) = commands
    words = shlex.split(command)
    assert words[:2] == ["pip", "install"]
    assert f"andrey-core[numba,viz]=={andrey.__version__}" in words
    assert not any(".whl" in word or " @ " in word for word in words)  # PyPI, not a file
    assert ("causal-learn==0.1.4.8" in words) == (notebook.stem == "speed_on_your_machine")
