"""Window marks: the link strings both packages return, the scores taken off them, and their record.

PCMCI+ and LPCMCI return links at lags ``0..tau_max`` with a mark at each end. Every fact checked
here is one the campaign depends on: that a link string becomes the marks it names, that an
encoding the converter does not know fails the fit, that equal graphs hash equal and unequal ones
do not, and that the scores count what their names say on graphs small enough to count by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey_bench import datasets, rundir, scoring
from andrey_bench.adapters.baseline_empty import baseline_empty_window
from andrey_bench.integration import _truth, run_batch
from andrey_bench.window import (
    ARROW,
    CIRCLE,
    CONFLICT,
    TAIL,
    WindowGraph,
    marks_from_link_strings,
)


def _strings(n: int = 3, lags: int = 2, **links: str) -> np.ndarray:
    """A ``(n, n, lags)`` link-string graph; a keyword ``l<k>_<i>_<j>`` sets ``graph[i, j, k]``."""
    graph = np.full((n, n, lags), "", dtype="<U3")
    for name, link in links.items():
        k, i, j = (int(v) for v in name[1:].split("_"))
        graph[i, j, k] = link
    return graph


def _dag_truth(edges0=(), links=(), n: int = 3, lags: int = 2) -> WindowGraph:
    """A window truth with lag-0 edges ``i -> j`` and lagged links ``(k, i, j)``."""
    marks = np.zeros((lags, n, n), dtype=np.int8)
    for i, j in edges0:
        marks[0, i, j], marks[0, j, i] = TAIL, ARROW
    for k, i, j in links:
        marks[k, i, j] = TAIL
    return WindowGraph(marks, lag0="dag")


# --- link strings ----------------------------------------------------------------------------------


def test_a_link_string_becomes_the_marks_it_names():
    graph = _strings(l0_0_1="o->", l0_1_0="<-o", l0_1_2="x-x", l0_2_1="x-x", l1_2_0="<->")
    graph[1, 1, 1] = "-->"  # X1(t-1) -> X1(t)
    marks = marks_from_link_strings(graph)
    assert marks.shape == (2, 3, 3) and marks.dtype == np.int8
    assert (marks[0, 0, 1], marks[0, 1, 0]) == (CIRCLE, ARROW)
    assert (marks[0, 1, 2], marks[0, 2, 1]) == (CONFLICT, CONFLICT)
    # A lagged link keeps the mark at its earlier end; the later end is an arrowhead.
    assert (marks[1, 2, 0], marks[1, 1, 1]) == (ARROW, TAIL)
    assert np.count_nonzero(marks) == 6


@pytest.mark.parametrize(
    "links, message",
    [
        ({"l0_0_1": "-->"}, "mirror"),  # no "<--" at (1, 0)
        ({"l0_0_1": "-->", "l0_1_0": "-->"}, "mirror"),
        ({"l1_0_1": "<--"}, "back in time"),
        ({"l1_0_1": "o-o"}, "back in time"),
        ({"l1_0_1": "-?>"}, "unexpected"),
        ({"l1_0_1": "->"}, "unexpected"),
    ],
)
def test_a_link_string_it_does_not_know_fails_the_fit(links, message):
    with pytest.raises(RuntimeError, match=message):
        marks_from_link_strings(_strings(**links))


def test_a_window_graph_needs_a_square_stack_and_a_known_lag0():
    with pytest.raises(ValueError, match="shape"):
        WindowGraph(np.zeros((2, 3, 4), np.int8), lag0="pag")
    with pytest.raises(ValueError, match="lag0"):
        WindowGraph(np.zeros((2, 3, 3), np.int8), lag0="digraph")


# --- scores ----------------------------------------------------------------------------------------


def test_the_truth_scores_perfectly_and_the_empty_window_scores_two_per_link():
    truth = _dag_truth(edges0=[(0, 1)], links=[(1, 0, 0), (1, 1, 2)])
    perfect = scoring.score_result(WindowGraph(truth.marks, lag0="pag"), truth)
    assert perfect["family"] == "temporal_window"
    assert perfect["endpoint_shd"] == perfect["wrong_marks"] == 0.0
    assert perfect["adjacency_f1"] == 1.0 and perfect["n_true_links"] == 3.0
    assert perfect["lagged_tail"] == 2.0

    empty = scoring.score_result(WindowGraph(np.zeros_like(truth.marks), lag0="pag"), truth)
    assert empty["endpoint_shd"] == 6.0
    assert (empty["lag0_endpoint_shd"], empty["lagged_endpoint_shd"]) == (2.0, 4.0)
    assert empty["adjacency_recall"] == 0.0 and empty["n_est_links"] == 0.0


def test_an_unoriented_cpdag_link_costs_one_and_a_reversed_one_two():
    truth = _dag_truth(edges0=[(0, 1)])
    undirected = _strings(l0_0_1="o-o", l0_1_0="o-o")
    scores = scoring.score_result(WindowGraph(marks_from_link_strings(undirected), "cpdag"), truth)
    assert scores["endpoint_shd"] == 1.0 and scores["wrong_marks"] == 0.0
    assert scores["adjacency_f1"] == 1.0
    # A conflict is undirected in a CPDAG too.
    conflict = _strings(l0_0_1="x-x", l0_1_0="x-x")
    assert (
        scoring.score_result(WindowGraph(marks_from_link_strings(conflict), "cpdag"), truth)[
            "endpoint_shd"
        ]
        == 1.0
    )

    reversed_ = _strings(l0_0_1="<--", l0_1_0="-->")
    scores = scoring.score_result(WindowGraph(marks_from_link_strings(reversed_), "cpdag"), truth)
    assert scores["endpoint_shd"] == 2.0 and scores["wrong_marks"] == 2.0


def test_a_pag_circle_costs_its_open_end_and_claims_nothing():
    truth = _dag_truth(edges0=[(0, 1)], links=[(1, 2, 0)])
    pag = _strings(l0_0_1="o->", l0_1_0="<-o", l1_2_0="o->")
    scores = scoring.score_result(WindowGraph(marks_from_link_strings(pag), "pag"), truth)
    assert scores["endpoint_shd"] == 2.0
    assert scores["wrong_marks"] == 0.0
    assert (scores["lagged_tail"], scores["lagged_circle"]) == (0.0, 1.0)
    # A hidden-cause reading of a true lagged cause is one wrong mark at its earlier end.
    bidirected = _strings(l0_0_1="-->", l0_1_0="<--", l1_2_0="<->")
    scores = scoring.score_result(WindowGraph(marks_from_link_strings(bidirected), "pag"), truth)
    assert scores["endpoint_shd"] == scores["wrong_marks"] == 1.0
    assert scores["lagged_arrow"] == 1.0


def test_a_link_at_the_wrong_lag_is_a_miss_and_an_extra():
    truth = _dag_truth(links=[(1, 0, 1)], lags=3)
    wrong = WindowGraph(marks_from_link_strings(_strings(lags=3, l2_0_1="-->")), "cpdag")
    scores = scoring.score_result(wrong, truth)
    assert scores["endpoint_shd"] == 4.0
    assert scores["adjacency_precision"] == scores["adjacency_recall"] == 0.0


def test_a_window_is_scored_only_against_a_window_truth():
    estimate = WindowGraph(np.zeros((2, 3, 3), np.int8), "pag")
    with pytest.raises(TypeError):
        scoring.score_result(estimate, _truth(np.zeros((1, 3, 3), np.int8), "temporal"))
    with pytest.raises(ValueError, match="DAG"):
        scoring.score_result(estimate, estimate)


def test_the_structure_hash_tells_marks_apart():
    pag = _strings(l0_0_1="o->", l0_1_0="<-o")
    other = _strings(l0_0_1="-->", l0_1_0="<--")
    same = scoring.structure_hash(WindowGraph(marks_from_link_strings(pag), "pag"))
    assert same == scoring.structure_hash(WindowGraph(marks_from_link_strings(pag.copy()), "pag"))
    assert same != scoring.structure_hash(WindowGraph(marks_from_link_strings(other), "pag"))


def test_sortability_is_not_computed_on_a_window():
    assert scoring.sortability(np.zeros((10, 3)), _dag_truth(edges0=[(0, 1)])) == {}


def test_a_window_estimate_survives_the_run_that_measured_it(tmp_path):
    graph = _strings(l0_0_1="o->", l0_1_0="<-o", l1_2_2="x->")
    estimate = WindowGraph(marks_from_link_strings(graph), "pag")
    rundir.write_structure(tmp_path, "abc:andrey.lpcmci.student_t.tau1:0:def", estimate)
    restored = rundir.read_structure(tmp_path, "abc:andrey.lpcmci.student_t.tau1:0:def")
    assert isinstance(restored, WindowGraph) and restored.lag0 == "pag"
    np.testing.assert_array_equal(restored.marks, estimate.marks)
    assert scoring.structure_hash(restored) == scoring.structure_hash(estimate)


# --- the pipeline ----------------------------------------------------------------------------------


def test_a_same_step_run_records_its_window_and_link_scores(tmp_path):
    """Through the real pipeline: the empty window's endpoint SHD is twice the truth's links."""
    topology, functional, noise, standardize = datasets.REGIMES["var_same_step"]
    key = datasets.DatasetKey(
        topology=topology,
        functional=functional,
        noise=noise,
        standardize=standardize,
        density=datasets.TEMPORAL_REGIMES["var_same_step"],
        d=5,
        n=200,
        seed=0,
    )
    result = run_batch(
        tmp_path,
        [baseline_empty_window(1)],
        [key],
        machine_id="tiny-cpu",
        cap_wall_s=240.0,
        repeats=1,
        warmup=0,
        run_dir=tmp_path,
    )
    assert result.errors == []
    (record,) = result.records
    assert (record.status, record.output_type, record.tau_max) == ("ok", "temporal", 1)
    _, marks = datasets.load(tmp_path, key)
    truth = _truth(marks, datasets.WINDOW_KIND)
    lag0_links = int(np.count_nonzero(truth.marks[0])) // 2
    links = lag0_links + int(np.count_nonzero(truth.marks[1]))
    assert record.metrics["n_true_links"] == links
    assert record.metrics["endpoint_shd"] == 2 * links
    assert isinstance(rundir.read_structure(tmp_path, record.run_id), WindowGraph)
