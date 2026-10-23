"""Run the public facade docstring examples, and the simulators', as doctests.

A targeted runner, not a global ``--doctest-modules``: it executes only the ``andrey.api`` facade
examples (the runnable ``Examples`` the docstring standard guarantees) and those of the simulators
and the bootstrap the API reference lists, so collection stays bounded to the public surface and
does not pull in every internal module's docstring.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import andrey.api

_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def _facade_modules() -> list[str]:
    """Public facade modules under ``andrey.api`` (the private ``_`` adapters carry no examples)."""
    return [
        f"andrey.api.{info.name}"
        for info in pkgutil.iter_modules(andrey.api.__path__)
        if not info.name.startswith("_")
    ]


# The other modules whose examples the API reference shows: the simulators and the bootstrap.
SIMULATORS = ["andrey.data.discrete", "andrey.data.mixed", "andrey.resample"]


@pytest.mark.parametrize("modname", _facade_modules() + SIMULATORS)
def test_facade_doctests(modname):
    mod = importlib.import_module(modname)
    results = doctest.testmod(mod, optionflags=_FLAGS, verbose=False)
    assert results.attempted > 0, f"{modname}: no doctests found -- a facade lost its Examples"
    assert results.failed == 0, (
        f"{modname}: {results.failed} of {results.attempted} doctests failed"
    )
