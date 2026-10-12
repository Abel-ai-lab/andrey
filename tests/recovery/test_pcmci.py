"""PCMCI recovery: planted lagged links, generated VAR ground truth, and the null case."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, GraphStructure, TemporalStructure
from andrey.data import functional, graphs, noise, temporal
from andrey.data.scm import SCM
from andrey.metrics import temporal_scores

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

# A_k[effect, cause]: every series depends on its own past; five cross links at lags 1 and 2.
LAG_1 = np.array(
    [
        [0.6, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, -0.4],
        [0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5],
    ]
)
LAG_2 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _links(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """Every ``(cause, effect, lag)`` link of a PCMCI result: an arrowhead at the effect."""
    return {
        (int(i), int(j), lag)
        for lag in out.structure.lags
        for j, i in np.argwhere(out.structure.lag(lag).to_numpy() == ARROW)
    }


@pytest.fixture(scope="module")
def planted() -> tuple[np.ndarray, set[tuple[int, int, int]]]:
    a_stack = np.stack([LAG_1, LAG_2])
    innovations = np.random.default_rng(0).standard_normal((1200, 5))
    data = temporal.generate(np.eye(5), a_stack, None, innovations, burn_in=200)
    truth = {(int(i), int(j), k + 1) for k, j, i in np.argwhere(a_stack != 0)}
    return data, truth


INDEP_TESTS = ["student_t", "fisherz"]


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_recovers_the_planted_links(planted, indep_test):
    data, truth = planted
    assert len(truth) == 10
    out = andrey.pcmci(data, tau_max=2, alpha_level=0.01, indep_test=indep_test)
    assert _links(out) == truth


def test_the_tests_differ_in_p_values_and_agree_on_the_graph(planted):
    data, _ = planted
    student_t, fisherz = (
        andrey.pcmci(data, tau_max=2, alpha_level=0.01, indep_test=name) for name in INDEP_TESTS
    )
    assert student_t.structure == fisherz.structure
    assert student_t.metadata["p_values"] != fisherz.metadata["p_values"]


def test_same_result_on_every_run(planted):
    data, _ = planted
    first, second = (andrey.pcmci(data, tau_max=2) for _ in range(2))
    assert first.structure == second.structure
    assert first.metadata == second.metadata


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_generated_var_scores_well_above_the_empty_graph(indep_test):
    scm = SCM(
        graph=graphs.FixedDAG(nodes=tuple(f"x{i}" for i in range(5)), arcs=()),
        functional=functional.var(n_lags=2, lag_density=0.3),
        noise=noise.gaussian(),
    )
    dataset = scm.sample(1000, 0)
    out = andrey.pcmci(dataset.data, tau_max=2, alpha_level=0.01, indep_test=indep_test)
    lagged = temporal_scores(out, dataset.graph)["lagged"]
    nothing = GraphStructure.from_numpy(np.zeros((5, 5), np.int8), kind="digraph")
    empty = TemporalStructure.from_lag_graphs([nothing, nothing], lags=[1, 2])
    assert lagged["precision"] >= 0.9
    assert lagged["recall"] >= 0.8
    assert lagged["shd"] <= temporal_scores(empty, dataset.graph)["lagged"]["shd"] / 4


@pytest.mark.parametrize("indep_test", INDEP_TESTS)
def test_independent_autocorrelated_series_give_only_self_loops(indep_test):
    rng = np.random.default_rng(1)
    innovations = rng.standard_normal((700, 4))
    data = temporal.generate(np.eye(4), 0.9 * np.eye(4)[None], None, innovations, burn_in=200)
    out = andrey.pcmci(data, tau_max=2, alpha_level=0.01, indep_test=indep_test)
    assert _links(out) == {(i, i, 1) for i in range(4)}
