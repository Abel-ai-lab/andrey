"""``TemporalStructure.summary_graph()`` and the ``SummaryGraph`` projection."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import (
    ARROW,
    EDGE_DTYPE,
    GraphStructure,
    StructureOutput,
    SummaryGraph,
    TemporalStructure,
)
from andrey.core.structure import TAIL
from andrey.metrics import score, temporal_scores


def _digraph(pairs, n, *, self_loops=(), kind="digraph"):
    """A ``GraphStructure`` from directed edges ``i -> j`` plus optional self-loops."""
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = ARROW
    for k in self_loops:
        M[k, k] = ARROW
    return GraphStructure.from_numpy(M, kind=kind, allow_self_loops=bool(self_loops))


def _undirected(pairs, n):
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = TAIL
    return GraphStructure.from_numpy(M, kind="cpdag")


def test_cross_lag_pair_collapses_to_a_two_cycle():
    # X -> Y at lag 1 and Y -> X at lag 2 collapse to a 2-cycle in the summary (not a DAG).
    t = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1)], 2), _digraph([(1, 0)], 2)], lags=[1, 2]
    )
    s = t.summary_graph()
    assert isinstance(s, SummaryGraph)
    assert s.kind == "digraph"
    s.validate()
    assert s.endpoints(0, 1) == (ARROW, ARROW)  # a 2-cycle, both arrowheads
    assert s.lags_of(0, 1) == (1,)  # 0 -> 1 came from lag 1
    assert s.lags_of(1, 0) == (2,)  # 1 -> 0 came from lag 2 (per-direction lag-sets)


def test_autoregression_collapses_to_a_self_loop():
    t = TemporalStructure.from_lag_graphs([_digraph([], 2, self_loops=(0,))], lags=[1])
    s = t.summary_graph()
    assert s.endpoints(0, 0) == (ARROW, ARROW)  # diagonal ARROW: a self-loop
    assert s.lags_of(0, 0) == (1,)
    g = s.to_networkx()  # inherited digraph view: a real loop edge
    assert g.has_edge(0, 0)


def test_lag_sets_union_across_lags_and_include_lag0():
    # the same edge at lags 1 and 3 is one summary edge carrying both lags.
    t = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1)], 2), _digraph([(0, 1)], 2)], lags=[1, 3]
    )
    assert t.summary_graph().lags_of(0, 1) == (1, 3)

    # include_lag0 strips the contemporaneous lag: an edge present only at lag 0 disappears, and a
    # shared edge loses 0 from its lag-set.
    t2 = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1), (1, 2)], 3, kind="dag"), _digraph([(0, 1)], 3)], lags=[0, 1]
    )
    assert t2.summary_graph(include_lag0=True).lags_of(0, 1) == (0, 1)
    assert t2.summary_graph(include_lag0=True).lags_of(1, 2) == (0,)
    no0 = t2.summary_graph(include_lag0=False)
    assert no0.lags_of(0, 1) == (1,)  # 0 stripped from the shared edge
    assert no0.lags_of(1, 2) == ()  # the lag-0-only edge is gone


def test_undirected_marks_carry_no_direction():
    # a cpdag lag-0 undirected edge makes no directional claim, so it never enters the summary.
    t = TemporalStructure.from_lag_graphs(
        [_undirected([(0, 1)], 3), _digraph([(1, 2)], 3)], lags=[0, 1]
    )
    s = t.summary_graph()
    assert s.lags_of(0, 1) == () and s.lags_of(1, 0) == ()  # the undirected edge is dropped
    assert s.lags_of(1, 2) == (1,)  # the directed lag-1 edge survives


def test_one_way_edge_orientation_is_faithful():
    # a lone directed edge keeps its direction in the summary marks AND the networkx view -- it does
    # not collapse to a 2-cycle or point backward (pins the marks-reconstruction orientation).
    s = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2)], lags=[1]).summary_graph()
    assert s.endpoints(0, 1) == (TAIL, ARROW)  # tail at source 0, arrowhead at target 1
    assert s.endpoints(1, 0) == (ARROW, TAIL)  # the reciprocal cell, not a separate 1 -> 0 edge
    g = s.to_networkx()
    assert g.has_edge(0, 1) and not g.has_edge(1, 0)
    assert s.lags_of(0, 1) == (1,) and s.lags_of(1, 0) == ()


def test_lag_sets_are_sorted_regardless_of_stack_order():
    # a non-ascending lag stack still yields sorted lag tuples (the documented 'sorted' contract).
    t = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1)], 2), _digraph([(0, 1)], 2)], lags=[3, 1]
    )
    assert t.summary_graph().lags_of(0, 1) == (1, 3)


def test_summary_graph_of_a_time_carrying_structure_uses_the_final_occasion():
    # for a (time, lag) structure, summary_graph collapses the representative (final) occasion.
    inst = _digraph([(0, 1)], 2, kind="dag")  # instantaneous 0 -> 1
    empty = _digraph([], 2)  # occasion 1: no lag-1 edge
    lag1 = _digraph([(1, 0)], 2)  # final occasion: lag-1 edge 1 -> 0
    t = TemporalStructure.from_time_lag_graphs([[inst, empty], [inst, lag1]], times=[1, 2])
    final = TemporalStructure.from_lag_graphs([inst, lag1], lags=t.lags)
    assert t.summary_graph() == final.summary_graph()


def test_summary_scores_as_a_digraph_including_self_loops():
    # a self-loop-bearing summary scores in the digraph family (diagonal-aware).
    true_t = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2, self_loops=(0,))], lags=[1])
    est_t = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2)], lags=[1])  # no self-loop
    result = score(est_t.summary_graph(), true_t.summary_graph())
    assert result["family"] == "digraph"
    assert result["shd"] == 1  # the missing self-loop is one structural edit
    assert result["skeleton_recall"] == 0.5  # one of the two true edges recovered


def test_summary_matches_temporal_scores_offdiagonal_and_detects_a_miss():
    from andrey.metrics._common import directed_adjacency

    true_t = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1)], 3), _digraph([(1, 2)], 3)], lags=[1, 2]
    )
    est_t = TemporalStructure.from_lag_graphs(  # misses the lag-2 edge 1 -> 2
        [_digraph([(0, 1)], 3), _digraph([], 3)], lags=[1, 2]
    )
    # not vacuous: temporal_scores' summary detects the missing edge (recall < 1).
    assert temporal_scores(est_t, true_t)["summary"]["recall"] < 1.0
    # the summary_graph projection and temporal_scores share one collapse: off-diagonal directed
    # support agrees (no lag-0 here, so no lag-0 delta between the two paths).
    off = ~np.eye(3, dtype=bool)
    for t in (est_t, true_t):
        via_summary = directed_adjacency(t.summary_graph().to_numpy()) & off
        via_union = np.zeros((3, 3), dtype=bool)
        for lag in t.lags:
            via_union |= directed_adjacency(t.lag(lag).to_numpy())
        assert np.array_equal(via_summary, via_union & off)


def test_equality_and_cross_type():
    t1 = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2)], lags=[1])
    t2 = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2)], lags=[2])  # same edge, lag 2
    a, a2, b = t1.summary_graph(), t1.summary_graph(), t2.summary_graph()
    assert a == a2  # identical projections
    assert a != b  # same topology, different lag-set -> not equal
    # a SummaryGraph never equals a plain GraphStructure with the same marks (type-guarded).
    plain = GraphStructure.from_numpy(a.to_numpy(), kind="digraph")
    assert a != plain
    assert plain != a


def test_labels_propagate_from_the_temporal_structure():
    t = TemporalStructure.from_lag_graphs([_digraph([(0, 1)], 2)], lags=[1], labels=("X", "Y"))
    assert t.summary_graph().labels == ("X", "Y")


def test_serialization_is_rejected_as_a_derived_projection(tmp_path):
    # a SummaryGraph is a derived projection: StructureOutput refuses to serialize it rather than
    # silently drop the lag-sets. The source TemporalStructure round-trips and re-derives it.
    t = TemporalStructure.from_lag_graphs(
        [_digraph([(0, 1)], 2), _digraph([(1, 0)], 2)], lags=[1, 2]
    )
    out = StructureOutput.new(t.summary_graph())  # an in-memory summary envelope is fine
    with pytest.raises(TypeError, match="SummaryGraph is a derived projection"):
        out.save(tmp_path / "summary.json")
    # the documented alternative: persist the source and re-derive the identical summary on load
    tp = tmp_path / "temporal.json"
    StructureOutput.new(t).save(tp)
    assert StructureOutput.load(tp).structure.summary_graph() == t.summary_graph()


@pytest.mark.parametrize(
    "build",
    [
        lambda: SummaryGraph.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="digraph"),
        lambda: SummaryGraph.from_edges(np.zeros(0, dtype=EDGE_DTYPE), n_nodes=2),
        lambda: SummaryGraph.from_networkx(pytest.importorskip("networkx").DiGraph([(0, 1)])),
    ],
)
def test_a_summary_graph_comes_only_from_a_temporal_structure(build):
    with pytest.raises(TypeError, match=r"summary_graph\(\)"):
        build()
