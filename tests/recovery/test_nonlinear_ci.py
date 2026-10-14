"""PC with the nonlinear conditional-independence tests: fixed cases, ground truth, and recorded
p-values.

The fixed case ``nonmonotone_4v`` has links partial correlation cannot see: ``x2 = |x0| + |x1| +
noise`` and ``x3 = (x2 - 1)**2 + noise``. Its CPDAG is fully oriented: the collider
``x0 -> x2 <- x1`` and, by Meek's first rule, ``x2 -> x3``. Each test's p-values on a few queries of
that case are recorded outputs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import ci
from andrey.metrics import shd, to_cpdag
from tests.recovery.helpers import datagen, harness
from tests.recovery.helpers.ci_tests import CI_TESTS, recorded_pvalues

CASE = "nonmonotone_4v"
TRUE_EDGES = [(i, j, "directed") for i, j in datagen.CASES[CASE]["edges"]]


@pytest.fixture(autouse=True)
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


@pytest.fixture
def indep_test(request):
    """The ``indep_test`` name PC selects each test by, with the settings ``CI_TESTS`` gives it."""
    algorithm = request.param
    name = f"{CI_TESTS[algorithm][0]}_recovery"
    snapshot = dict(ci._CI_REGISTRY)
    ci.register_indep_test(name, CI_TESTS[algorithm][1])
    try:
        yield name
    finally:
        ci._CI_REGISTRY.clear()
        ci._CI_REGISTRY.update(snapshot)


# ---- fixed cases -------------------------------------------------------------------------------


def test_partial_correlation_misses_the_nonmonotone_links():
    edges = andrey.pc(datagen.generate(CASE)).structure.oriented_edges()
    assert {(0, 2), (1, 2)}.isdisjoint((i, j) for i, j, _ in edges)


@pytest.mark.parametrize("indep_test", CI_TESTS, indirect=True)
def test_pc_recovers_the_nonmonotone_graph(indep_test):
    out = andrey.pc(datagen.generate(CASE), indep_test=indep_test)
    assert out.structure.oriented_edges() == TRUE_EDGES


@pytest.mark.parametrize("indep_test", CI_TESTS, indirect=True)
def test_pc_finds_no_edges_between_independent_columns(indep_test):
    x = np.random.default_rng(0).standard_normal((300, 4))
    fisherz = ci.FisherZ(x)
    # No chance correlation: every pair is far from rejection.
    assert min(fisherz(i, j) for i in range(4) for j in range(i + 1, 4)) > 0.2
    assert andrey.pc(x, indep_test=indep_test).structure.oriented_edges() == []


# ---- ground truth ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def additive_noise_case():
    """Additive nonlinear noise on a 4-node Erdos-Renyi DAG, 200 rows, and its true CPDAG."""
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="additive_noise",
        d=4,
        n=200,
        seed=0,
        density=2.0,
        scale="standardize",
    )
    return ds.data, to_cpdag(ds.graph)


def test_partial_correlation_errs_on_the_additive_noise_case(additive_noise_case):
    x, truth = additive_noise_case
    assert shd(andrey.pc(x).structure, truth) >= 2


@pytest.mark.parametrize("indep_test", CI_TESTS, indirect=True)
def test_pc_recovers_the_additive_noise_graph(indep_test, additive_noise_case):
    x, truth = additive_noise_case
    assert shd(andrey.pc(x, indep_test=indep_test).structure, truth) <= 1


# ---- recorded outputs --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", CI_TESTS)
def test_pvalues_match_the_recorded_output(algorithm):
    candidate = recorded_pvalues(algorithm, datagen.generate(CASE), datagen.CASES[CASE]["queries"])
    harness.assert_baseline(algorithm, candidate, case=CASE)
