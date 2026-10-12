"""PCMCI+ recovery: planted lagged and same-step links, generated truth, and the null case."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, NULL, TAIL, GraphStructure, TemporalStructure
from andrey.data import functional, graphs, noise, temporal
from andrey.data.scm import SCM
from andrey.metrics import temporal_scores

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

# B0[effect, cause]: X0 -> X1 -> X2 <- X3 within each step.
B0 = np.zeros((5, 5))
B0[1, 0] = B0[2, 1] = B0[2, 3] = 0.6
# A_k[effect, cause]: every series depends on its own past; X1 -> X4 at lag 1, X4 -> X3 at lag 2.
LAG_1 = np.diag([0.5, 0.4, 0.3, 0.4, 0.5])
LAG_1[4, 1] = 0.5
LAG_2 = np.zeros((5, 5))
LAG_2[3, 4] = -0.4


def _lagged(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """Every ``(cause, effect, lag)`` link at a positive lag: an arrowhead at the effect."""
    return {
        (int(i), int(j), lag)
        for lag in out.structure.lags
        if lag > 0
        for j, i in np.argwhere(out.structure.lag(lag).to_numpy() == ARROW)
    }


def _same_step(out: andrey.StructureOutput) -> set[tuple[int, int, str]]:
    """Every lag-0 link as ``(i, j, "->")`` for ``i -> j``, or ``(i < j, "--")``."""
    m = out.structure.lag(0).to_numpy()
    links = set()
    for i, j in zip(*np.nonzero(np.triu(m != NULL))):
        i, j = int(i), int(j)
        if m[i, j] == TAIL and m[j, i] == TAIL:
            links.add((i, j, "--"))
        else:
            links.add((i, j, "->") if m[j, i] == ARROW else (j, i, "->"))
    return links


@pytest.fixture(scope="module")
def planted() -> np.ndarray:
    innovations = np.random.default_rng(0).standard_normal((1200, 5))
    ib_inv = np.linalg.inv(np.eye(5) - B0)
    return temporal.generate(ib_inv, np.stack([LAG_1, LAG_2]), None, innovations, burn_in=200)


INDEP_TESTS = ["student_t", "fisherz"]


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_recovers_the_planted_links(planted, indep_test):
    out = andrey.pcmciplus(planted, tau_max=2, indep_test=indep_test)
    assert _lagged(out) == {(i, i, 1) for i in range(5)} | {(1, 4, 1), (4, 3, 2)}
    # The collider X1 -> X2 <- X3 orients two links, and R1 orients X0 -> X1 from X0's past.
    assert _same_step(out) == {(0, 1, "->"), (1, 2, "->"), (3, 2, "->")}
    assert out.metadata["conflicts"] == []


def test_the_conservative_rule_recovers_the_planted_links(planted):
    out = andrey.pcmciplus(planted, tau_max=2, collider_rule="conservative")
    assert _same_step(out) == {(0, 1, "->"), (1, 2, "->"), (3, 2, "->")}


def test_the_none_rule_keeps_the_skeleton_and_reports_its_conflicts_as_undirected(planted):
    # The first step drops X3 one step back as a parent of X2, so the none rule reads an empty
    # separating set and calls X3_{t-1} -> X3_t <- X2_t a collider, against X3_t -> X2_t.
    out = andrey.pcmciplus(planted, tau_max=2, collider_rule="none")
    majority = andrey.pcmciplus(planted, tau_max=2)
    assert {(i, j) for i, j, _ in _same_step(out)} == {(0, 1), (1, 2), (2, 3)}
    assert _lagged(out) == _lagged(majority)
    assert out.metadata["conflicts"] == [[2, 3]]
    assert (2, 3, "--") in _same_step(out)


def test_same_result_on_every_run(planted):
    first, second = (andrey.pcmciplus(planted, tau_max=2) for _ in range(2))
    assert first.structure == second.structure
    assert first.metadata == second.metadata


def test_the_result_does_not_depend_on_the_variable_order(planted):
    order = [3, 0, 4, 2, 1]
    out = andrey.pcmciplus(planted, tau_max=2)
    permuted = andrey.pcmciplus(planted[:, order], tau_max=2)
    back = {(order[i], order[j], kind) for i, j, kind in _same_step(permuted)}
    back = {(min(i, j), max(i, j), k) if k == "--" else (i, j, k) for i, j, k in back}
    assert back == _same_step(out)
    assert {(order[i], order[j], k) for i, j, k in _lagged(permuted)} == _lagged(out)


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_generated_var_scores_well_above_the_empty_graph(indep_test):
    nodes = tuple(f"x{i}" for i in range(5))
    scm = SCM(
        graph=graphs.FixedDAG(nodes=nodes, arcs=(("x0", "x1"), ("x1", "x2"), ("x3", "x2"))),
        functional=functional.var(n_lags=2, lag_density=0.2),
        noise=noise.gaussian(),
    )
    dataset = scm.sample(1000, 0)
    out = andrey.pcmciplus(dataset.data, tau_max=2, indep_test=indep_test)
    scores = temporal_scores(out, dataset.graph)
    nothing = GraphStructure.from_numpy(np.zeros((5, 5), np.int8), kind="digraph")
    empty = TemporalStructure.from_lag_graphs([nothing] * 3, lags=[0, 1, 2])
    assert scores["lagged"]["precision"] >= 0.8
    assert scores["summary"]["f1"] >= 0.6
    assert temporal_scores(empty, dataset.graph)["summary"]["f1"] == 0.0
    # Every true lag-0 link is found, and every oriented one points the true way.
    found = out.structure.lag(0).to_numpy() != NULL
    assert np.array_equal(found, dataset.graph.lag(0).to_numpy() != NULL)
    assert scores["contemporaneous"]["precision"] == 1.0
    assert scores["contemporaneous"]["recall"] >= 0.6


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_independent_autocorrelated_series_give_only_self_loops(indep_test):
    rng = np.random.default_rng(1)
    innovations = rng.standard_normal((700, 4))
    data = temporal.generate(np.eye(4), 0.9 * np.eye(4)[None], None, innovations, burn_in=200)
    out = andrey.pcmciplus(data, tau_max=2, indep_test=indep_test)
    assert _lagged(out) == {(i, i, 1) for i in range(4)}
    assert _same_step(out) == set()
