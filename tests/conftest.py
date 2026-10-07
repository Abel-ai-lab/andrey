"""Package-wide independence guard."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _no_causallearn_import():
    """Check that package tests do not import the benchmark comparator."""
    yield
    assert "causallearn" not in sys.modules, "an engine imported causal-learn"
