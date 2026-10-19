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
    edges = datagen.to_variables(CASE, andrey.pc(datagen.generate(CASE)).structure)
    assert {(0, 2), (1, 2)}.isdisjoint((i, j) for i, j, _ in edges.oriented_edges())


# RCoT is left out: at this case's 400 rows, below the 500 its warning names, a true independence
# falls below 0.05 for some feature seeds, so the graph follows the seed; the many-graph test
# below checks RCoT.
@pytest.mark.parametrize("indep_test", [a for a in CI_TESTS if a != "RCoT"], indirect=True)
def test_pc_recovers_the_nonmonotone_graph(indep_test):
    out = andrey.pc(datagen.generate(CASE), indep_test=indep_test)
    assert datagen.to_variables(CASE, out.structure).oriented_edges() == TRUE_EDGES


@pytest.mark.parametrize("indep_test", CI_TESTS, indirect=True)
def test_pc_finds_no_edges_between_independent_columns(indep_test):
    x = np.random.default_rng(0).standard_normal((300, 4))
    fisherz = ci.FisherZ(x)
    # No chance correlation: every pair is far from rejection.
    assert min(fisherz(i, j) for i in range(4) for j in range(i + 1, 4)) > 0.2
    assert andrey.pc(x, indep_test=indep_test).structure.oriented_edges() == []


# ---- ground truth ------------------------------------------------------------------------------


#: Each test's additive-noise case: rows and the graph's seed.
ADDITIVE_NOISE = {"KCI": (200, 0), "CMIknn": (200, 0), "GPDC": (200, 0)}

#: RCoT's random features make one graph's result follow its seed, so RCoT is scored over the
#: graphs of seeds 0-19 at 500 rows, the smallest sample size it is documented for. Over 100 graphs
#: and 10 RCoT seeds, a fit had at most one skeleton error with probability 0.97, so fewer than 17
#: of 20 such fits has probability 0.004; a random set of 20 fits erred in total at least as often
#: as Fisher-Z on its graphs with probability 0.006.
RCOT_GRAPHS = range(20)
RCOT_ROWS = 500


def _additive_noise_case(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Additive nonlinear noise on a 4-node Erdos-Renyi DAG, and its true CPDAG."""
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="additive_noise",
        d=4,
        n=n,
        seed=seed,
        density=2.0,
        scale="standardize",
    )
    return ds.data, to_cpdag(ds.graph)


@pytest.mark.parametrize("case", sorted(set(ADDITIVE_NOISE.values())))
def test_partial_correlation_errs_on_the_additive_noise_case(case):
    x, truth = _additive_noise_case(*case)
    assert shd(andrey.pc(x).structure, truth) >= 2


@pytest.mark.parametrize(
    ("indep_test", "case"), list(ADDITIVE_NOISE.items()), indirect=["indep_test"]
)
def test_pc_recovers_the_additive_noise_graph(indep_test, case):
    x, truth = _additive_noise_case(*case)
    assert shd(andrey.pc(x, indep_test=indep_test).structure, truth) <= 1


def _skeleton_errors(estimated, truth) -> int:
    """Pairs adjacent in one graph and not in the other."""
    a, b = estimated.to_numpy() != 0, truth.to_numpy() != 0
    return int(np.triu((a | a.T) != (b | b.T), 1).sum())


@pytest.mark.parametrize("indep_test", ["RCoT"], indirect=True)
def test_pc_with_rcot_recovers_most_additive_noise_skeletons(indep_test):
    errors, fisherz_errors = [], []
    for seed in RCOT_GRAPHS:
        x, truth = _additive_noise_case(RCOT_ROWS, seed)
        errors.append(_skeleton_errors(andrey.pc(x, indep_test=indep_test).structure, truth))
        fisherz_errors.append(_skeleton_errors(andrey.pc(x).structure, truth))
    assert sum(e <= 1 for e in errors) >= 17
    assert sum(errors) < sum(fisherz_errors)


# ---- recorded outputs --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", CI_TESTS)
def test_pvalues_match_the_recorded_output(algorithm):
    candidate = recorded_pvalues(algorithm, datagen.generate(CASE), CASE)
    harness.assert_baseline(algorithm, candidate, case=CASE)
