"""PC with the nonlinear CI tests recovers a graph whose links are not monotone.

``x2 = |x0| + |x1| + noise`` and ``x3 = sin(1.5 x2) + noise``: ``x0`` and ``x1`` are uncorrelated
with ``x2``, so partial correlation cannot see those edges. The CPDAG is fully oriented: the
collider ``x0 -> x2 <- x1`` and, by Meek's first rule, ``x2 -> x3``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey

TRUE_EDGES = [(0, 2, "directed"), (1, 2, "directed"), (2, 3, "directed")]


def _nonmonotone_graph(seed: int = 0, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = np.abs(x0) + np.abs(x1) + 0.3 * rng.standard_normal(n)
    x3 = np.sin(1.5 * x2) + 0.3 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2, x3])


@pytest.fixture(autouse=True)
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


@pytest.mark.parametrize("indep_test", ["kci", "rcot", "cmiknn", "gpdc"])
def test_pc_recovers_the_nonmonotone_graph(indep_test):
    out = andrey.pc(_nonmonotone_graph(), indep_test=indep_test)
    assert out.structure.oriented_edges() == TRUE_EDGES


@pytest.mark.parametrize("indep_test", ["kci", "rcot", "cmiknn", "gpdc"])
def test_pc_finds_no_edges_between_independent_columns(indep_test):
    data = np.random.default_rng(9).standard_normal((300, 4))
    assert andrey.pc(data, indep_test=indep_test).structure.oriented_edges() == []
