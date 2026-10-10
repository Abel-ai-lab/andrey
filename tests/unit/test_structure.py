"""Reference-equivalence and determinism tests for the graph types."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import EDGE_DTYPE, GraphStructure, Structure, TemporalStructure
from andrey.core.structure import ARROW, CIRCLE, NULL, TAIL


def directed(pairs, n, kind="dag", labels=None):
    """A ``GraphStructure`` from directed edges ``i -> j`` (mark TAIL at i, ARROW at j)."""
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = ARROW
    return GraphStructure.from_numpy(M, kind=kind, labels=labels)


def test_structure_base_is_abstract():
    with pytest.raises(TypeError):
        Structure(_n_nodes=3)


def test_from_numpy_to_numpy_roundtrip_is_exact():
    M = np.zeros((4, 4), dtype=np.int8)
    M[0, 1], M[1, 0] = TAIL, ARROW  # 0 -> 1
    M[2, 3], M[3, 2] = ARROW, ARROW  # 2 <-> 3 (bidirected -> pag)
    M[0, 2], M[2, 0] = CIRCLE, ARROW  # 0 o-> 2
    g = GraphStructure.from_numpy(M)
    assert g.kind == "pag"
    assert np.array_equal(g.to_numpy(), M)
    assert np.array_equal(g.adjacency, M)
    assert g.to_numpy().dtype == np.int8
    assert np.array_equal(g.to_numpy(dtype=np.int64), M.astype(np.int64))


@pytest.mark.parametrize(
    "build,expected_kind",
    [
        (lambda: directed([(0, 1), (1, 2)], 3), "dag"),
        (lambda: GraphStructure.from_numpy(_undirected([(0, 1)], 3)), "cpdag"),
    ],
)
def test_kind_inference(build, expected_kind):
    assert build().kind == expected_kind


def _undirected(pairs, n):
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = TAIL
    return M


def test_kind_override_beats_inference():
    # An all-directed matrix inferred as dag can be tagged cpdag by the producing algorithm.
    g = directed([(0, 1)], 2, kind="cpdag")
    assert g.kind == "cpdag"


def test_isolated_nodes_preserved():
    g = directed([(0, 1)], 5)  # nodes 2,3,4 isolated
    assert g.n_nodes == 5
    assert np.array_equal(g.to_numpy(), directed([(0, 1)], 5).to_numpy())


def test_edges_roundtrip():
    g = directed([(0, 1), (1, 2), (0, 3)], 4)
    edges = g.to_edges()
    assert edges.dtype.names == ("i", "j", "mark_i", "mark_j")
    assert GraphStructure.from_edges(edges, n_nodes=4) == g


@pytest.mark.parametrize("pair", [(0, -1), (-1, 0), (0, 3), (3, 0)])
def test_from_edges_rejects_a_node_outside_the_graph(pair):
    """A negative index would wrap to the last node and build a different graph."""
    edges = np.array([(*pair, TAIL, ARROW)], dtype=EDGE_DTYPE)
    with pytest.raises(IndexError, match="outside range"):
        GraphStructure.from_edges(edges, n_nodes=3)


def test_networkx_roundtrip_dag_and_pag():
    g = directed([(0, 1), (2, 1)], 3)
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"
    assert GraphStructure.from_networkx(G) == g

    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # bidirected
    M[1, 2], M[2, 1] = CIRCLE, CIRCLE  # o-o
    pag = GraphStructure.from_numpy(M)
    P = pag.to_networkx()
    assert type(P).__name__ == "MultiDiGraph"
    assert GraphStructure.from_networkx(P) == pag


def test_networkx_digraph_two_cycle_and_self_loop_roundtrip():
    import networkx as nx

    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1 (two directed edges, not bidirected)
    M[1, 2], M[2, 1] = TAIL, ARROW  # directed 1 -> 2
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"  # a digraph stays a plain DiGraph, not a MultiDiGraph
    assert G.has_edge(0, 1) and G.has_edge(1, 0)  # both directions of the 2-cycle are real edges
    assert G.has_edge(0, 0)  # self-loop present
    assert not nx.is_directed_acyclic_graph(G)
    assert GraphStructure.from_networkx(G, kind="digraph") == g


def test_networkx_digraph_three_cycle_is_topology_faithful():
    import networkx as nx

    # a 3-cycle 0->1->2->0 (includes the backward edge 2->0, arrowhead at the lower index): each
    # edge must orient by its marks so nx sees the cycle. Index-order emission renders it acyclic.
    g = directed([(0, 1), (1, 2), (2, 0)], 3, kind="digraph")
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"
    assert set(G.edges()) == {(0, 1), (1, 2), (2, 0)}  # oriented by marks, not by index order
    assert not nx.is_directed_acyclic_graph(G)
    assert GraphStructure.from_networkx(G, kind="digraph") == g


def test_from_networkx_bare_reciprocal_pair_is_a_two_cycle():
    import networkx as nx

    # a bare DiGraph (no endpoints attrs) carrying both 0->1 and 1->0 imports as a 2-cycle, not a
    # single edge collapsed by last-writer-wins.
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0)])
    g = GraphStructure.from_networkx(G, kind="digraph")
    assert g.endpoints(0, 1) == (ARROW, ARROW)
    g.validate()


def test_digraph_edges_roundtrip():
    # the documented to_edges / from_edges inverse for a digraph 2-cycle + self-loop; kind is passed
    # explicitly (a bare edge-list infers pag for a double arrowhead, exactly like from_networkx).
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1
    M[1, 2], M[2, 1] = TAIL, ARROW  # directed 1 -> 2
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    assert GraphStructure.from_edges(g.to_edges(), n_nodes=3, kind="digraph") == g


# ---- kind-aware compact mark storage (internal) -------------------------------------------------


def _sym_twin(g):
    """A byte-for-byte symmetric-layout twin of ``g`` (same graph, forced ``_layout='sym'``)."""
    from andrey.core.structure import _dense_to_csr, _frozen

    indptr, indices, marks = _dense_to_csr(g.to_numpy())
    return GraphStructure(
        _n_nodes=g.n_nodes,
        _labels=g.labels,
        _indptr=_frozen(indptr, np.int64),
        _indices=_frozen(indices, np.int32),
        _marks=_frozen(marks, np.int8),
        _kind=g.kind,
        _layout="sym",
        _node_types=g._node_types,
    )


def test_compact_layout_is_view_identical_to_symmetric():
    sl = np.zeros((3, 3), np.int8)
    sl[0, 0] = ARROW  # self-loop 0 -> 0
    sl[2, 1], sl[1, 2] = ARROW, TAIL  # directed 1 -> 2
    fixtures = [
        directed([(0, 1), (1, 2), (2, 0)], 3),  # dag, directed cycle-topology (backward edges)
        directed([(2, 0), (2, 1)], 3),  # dag fork
        GraphStructure.from_numpy(sl, kind="dag", allow_self_loops=True),  # dag + self-loop
        _undirected_graph([(0, 1), (1, 2)], 3, kind="cpdag"),  # cpdag, all undirected
        GraphStructure.from_numpy(  # cpdag mixed directed + undirected
            np.array([[0, 1, 0], [2, 0, 1], [0, 1, 0]], np.int8), kind="cpdag"
        ),
        directed([], 4),  # empty (isolated nodes)
    ]
    for g in fixtures:
        twin = _sym_twin(g)
        assert g._layout == "compact", g.kind  # dag/cpdag halve the store
        assert g._indices.size <= twin._indices.size  # never larger than symmetric
        # every public view is byte-identical to the symmetric twin.
        assert np.array_equal(g.to_numpy(), twin.to_numpy())
        assert np.array_equal(g.to_edges(), twin.to_edges())
        assert np.array_equal(g.to_scipy_sparse().toarray(), twin.to_scipy_sparse().toarray())
        for i in range(g.n_nodes):
            assert np.array_equal(g.neighbors(i), twin.neighbors(i))
            assert g.neighbors(i).dtype == np.int32
            for j in range(g.n_nodes):
                assert g.endpoints(i, j) == twin.endpoints(i, j)
        g_nx, t_nx = g.to_networkx(), twin.to_networkx()
        assert type(g_nx).__name__ == type(t_nx).__name__
        assert set(g_nx.edges()) == set(t_nx.edges())
        g.validate()
        assert g == twin and twin == g  # layout-normalizing equality


def test_pag_and_digraph_stay_symmetric():
    pag = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="pag")
    di = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")
    assert pag._layout == "sym" and di._layout == "sym"  # per-endpoint marks needed, no reduction


def test_compact_neighbors_are_read_only():
    g = directed([(0, 1), (1, 2)], 3)  # dag -> compact
    assert g._layout == "compact"
    with pytest.raises(ValueError):
        g.neighbors(1)[0] = 9  # the reconstructed neighbors array is a fresh read-only copy


def test_to_scipy_sparse_read_only_across_layouts():
    # to_scipy_sparse must not leak the layout: both a compact (dag) and a symmetric (digraph) store
    # hand back read-only buffers, so mutating them raises regardless of how the marks are stored.
    compact = directed([(0, 1), (1, 2)], 3)  # dag -> compact
    sym = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")  # sym
    assert compact._layout == "compact" and sym._layout == "sym"
    for g in (compact, sym):
        with pytest.raises(ValueError):
            g.to_scipy_sparse().data[:] = 9


def test_degenerate_store_stays_symmetric():
    # a store validate() would reject must NOT compact -- compaction would silently canonicalize it.
    half = GraphStructure._from_csr(2, indptr=[0, 1, 1], indices=[1], marks=[TAIL], kind="dag")
    assert half._layout == "sym"  # an asymmetric half-edge is left symmetric for validate() to flag
    with pytest.raises(ValueError, match="asymmetric"):
        half.validate()


def test_compact_store_serializes_symmetric_at_v0(tmp_path):
    import json

    from andrey.core import StructureOutput

    g = directed([(0, 1), (1, 2)], 3)  # dag -> compact (nnz 2), symmetric on disk (4 marks)
    out = StructureOutput.new(g)
    for fmt, name in (("json", "g.json"), ("npz", "g.npz")):
        out.save(tmp_path / name, fmt=fmt)
        assert StructureOutput.load(tmp_path / name) == out  # re-compacts on load
    doc = json.loads((tmp_path / "g.json").read_text())
    assert doc["format_version"] == 0  # the compact store never touches the on-disk format
    assert len(doc["marks"]) == 4  # symmetric: 2 edges x 2 endpoints, not the halved 2


def test_networkx_cpdag_multigraph_and_asymmetric_marks():
    # cpdag undirected -> DiGraph carrying (TAIL, TAIL); multigraph=True forces MultiDiGraph.
    cpdag = _undirected_graph([(0, 1)], 2, kind="cpdag")
    assert type(cpdag.to_networkx()).__name__ == "DiGraph"
    assert type(cpdag.to_networkx(multigraph=True)).__name__ == "MultiDiGraph"
    assert GraphStructure.from_networkx(cpdag.to_networkx(multigraph=True)) == cpdag

    # Asymmetric compound mark o-> (CIRCLE, ARROW) must round-trip through networkx.
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    pag = GraphStructure.from_numpy(M, kind="pag")
    assert GraphStructure.from_networkx(pag.to_networkx()) == pag


def test_networkx_labels_roundtrip():
    g = directed([(0, 1)], 2, labels=("A", "B"))
    G = g.to_networkx()
    assert list(G.nodes()) == ["A", "B"]
    back = GraphStructure.from_networkx(G)
    assert back == g
    assert back.labels == ("A", "B")


# Every mark pair on edge 0-1, and the (source, target, type) row it reads as.
ORIENTED = [
    (TAIL, ARROW, (0, 1, "directed")),
    (ARROW, TAIL, (1, 0, "directed")),
    (CIRCLE, ARROW, (0, 1, "partially_directed")),
    (ARROW, CIRCLE, (1, 0, "partially_directed")),
    (TAIL, CIRCLE, (0, 1, "partially_undirected")),
    (CIRCLE, TAIL, (1, 0, "partially_undirected")),
    (TAIL, TAIL, (0, 1, "undirected")),
    (ARROW, ARROW, (0, 1, "bidirected")),
    (CIRCLE, CIRCLE, (0, 1, "circle")),
]


@pytest.mark.parametrize(("mark_0", "mark_1", "row"), ORIENTED)
def test_oriented_edges_follow_the_marks(mark_0, mark_1, row):
    M = np.array([[0, mark_0], [mark_1, 0]])
    assert GraphStructure.from_numpy(M, kind="pag").oriented_edges() == [row]
    named = GraphStructure.from_numpy(M, kind="pag", labels=("a", "b"))
    source, target, edge_type = row
    assert named.oriented_edges() == [("ab"[source], "ab"[target], edge_type)]
    assert named.oriented_edges(index=True) == [row]


def test_oriented_edges_ignore_node_order():
    g = directed([(2, 0), (2, 1)], 3, labels=("wet", "slippery", "rain"))
    assert g.oriented_edges() == [("rain", "wet", "directed"), ("rain", "slippery", "directed")]
    # to_networkx keeps index order, so the arc for rain -> wet runs (wet, rain).
    assert list(g.to_networkx().edges) == [("wet", "rain"), ("slippery", "rain")]


def test_oriented_edges_self_loop_is_directed():
    M = np.array([[ARROW, TAIL], [ARROW, 0]])  # X(t-1) -> X(t), and 0 -> 1
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    assert g.oriented_edges() == [(0, 0, "directed"), (0, 1, "directed")]


def test_digraph_two_cycles_are_directed_edges():
    M = np.array([[ARROW, ARROW, 0], [ARROW, 0, ARROW], [0, TAIL, 0]], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="digraph", labels=("a", "b", "c"), allow_self_loops=True)
    assert g.oriented_edges() == [
        ("a", "a", "directed"),
        ("a", "b", "directed"),
        ("b", "a", "directed"),
        ("c", "b", "directed"),
    ]
    assert g.oriented_edges(index=True) == [
        (0, 0, "directed"),
        (0, 1, "directed"),
        (1, 0, "directed"),
        (2, 1, "directed"),
    ]
    assert repr(g).splitlines() == [
        "digraph  |  3 nodes  |  4 edges (4 directed)",
        "",
        "  a -> a",
        "  a -> b",
        "  b -> a",
        "  c -> b",
    ]
    assert {(s, t) for s, t, _ in g.oriented_edges()} == set(g.to_networkx().edges)


@pytest.mark.parametrize("limit", [0, 1, 2, 30, None])
def test_digraph_two_cycles_count_omitted_directions(limit):
    M = np.zeros((17, 17), dtype=np.int8)
    M[0, 1:] = M[1:, 0] = ARROW
    g = GraphStructure.from_numpy(M, kind="digraph")
    edges = [edge for j in range(1, 17) for edge in (f"0 -> {j}", f"{j} -> 0")]
    shown = edges[:limit]
    expected = ["digraph  |  17 nodes  |  32 edges (32 directed)", ""]
    expected += [f"  {edge}" for edge in shown]
    if len(shown) < 32:
        expected += [f"  ... and {32 - len(shown)} more"]
    assert g._summary(limit=limit) == expected
    if limit == 30:
        assert repr(g).splitlines() == expected


def test_temporal_two_cycles_and_summary_graph():
    M = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="digraph", labels=("x", "y"))
    temporal = TemporalStructure.from_lag_graphs([g], lags=[1], labels=("x", "y"))
    assert repr(temporal).splitlines() == [
        "temporal  |  2 nodes  |  1 lags",
        "",
        "  lag 1  |  2 edges (2 directed)",
        "    x -> y",
        "    y -> x",
    ]
    assert repr(temporal.summary_graph()) == repr(g)


def test_repr_lists_the_edges_by_name():
    g = directed([(0, 2), (1, 2), (2, 3)], 4, labels=("rain", "sprinkler", "wet", "slippery"))
    assert repr(g) == (
        "dag  |  4 nodes  |  3 edges (3 directed)\n"
        "\n"
        "  rain -> wet\n"
        "  sprinkler -> wet\n"
        "  wet -> slippery"
    )
    assert str(g) == repr(g)  # print() shows the same text


def test_repr_uses_each_mark_pair_symbol_and_counts_types_in_order():
    M = np.zeros((6, 6), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW  # 0 o-> 1
    M[2, 3], M[3, 2] = ARROW, ARROW  # 2 <-> 3
    M[4, 5], M[5, 4] = CIRCLE, CIRCLE  # 4 o-o 5
    M[0, 5], M[5, 0] = TAIL, CIRCLE  # 0 -o 5
    lines = repr(GraphStructure.from_numpy(M, kind="pag")).splitlines()
    assert lines[0] == (
        "pag  |  6 nodes  |  4 edges "
        "(1 partially_directed, 1 partially_undirected, 1 bidirected, 1 circle)"
    )
    assert lines[2:] == ["  0 o-> 1", "  0 -o 5", "  2 <-> 3", "  4 o-o 5"]


def test_repr_lists_at_most_thirty_edges():
    g = directed([(0, j) for j in range(1, 36)], 36)
    lines = repr(g).splitlines()
    assert lines[0] == "dag  |  36 nodes  |  35 edges (35 directed)"
    assert lines[2:32] == [f"  0 -> {j}" for j in range(1, 31)]
    assert lines[32:] == ["  ... and 5 more"]


def test_repr_of_an_empty_graph_and_latent_nodes():
    assert repr(directed([], 2)) == "dag  |  2 nodes  |  0 edges (no edges)"
    M = np.zeros((3, 3), dtype=np.int8)
    M[2, 0], M[0, 2] = TAIL, ARROW  # the latent L1 -> a
    types = np.array([0, 0, 1], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="dag", labels=("a", "b", "L1"), node_types=types)
    assert repr(g).splitlines()[-1] == "  latent: L1"


def test_repr_of_a_temporal_stack_and_its_summary_graph():
    lag0 = directed([(0, 1)], 2, labels=("x", "y"))
    loop = np.array([[ARROW, TAIL], [ARROW, 0]], dtype=np.int8)  # x(t-1) -> x(t), x(t-1) -> y(t)
    lag1 = GraphStructure.from_numpy(loop, kind="digraph", labels=("x", "y"), allow_self_loops=True)
    t = TemporalStructure.from_lag_graphs([lag0, lag1], labels=("x", "y"))
    assert repr(t) == (
        "temporal  |  2 nodes  |  2 lags\n"
        "\n"
        "  lag 0  |  1 edges (1 directed)\n"
        "    x -> y\n"
        "\n"
        "  lag 1  |  2 edges (2 directed)\n"
        "    x -> x\n"
        "    x -> y"
    )
    assert repr(t.summary_graph()).startswith("digraph  |  2 nodes  |  2 edges (2 directed)")


def test_repr_needs_a_summary_from_each_structure():
    class Bare(Structure):
        type = "graph"

    with pytest.raises(NotImplementedError, match="Bare"):
        repr(Bare(_n_nodes=1))


def test_type_counts_match_the_oriented_edges():
    from collections import Counter

    rng = np.random.default_rng(0)
    for _ in range(25):
        M = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.3:
                    M[i, j], M[j, i] = rng.integers(1, 4, size=2)
        g = GraphStructure.from_numpy(M, kind="pag")
        assert g._type_counts() == dict(Counter(t for *_, t in g.oriented_edges()))
        assert list(zip(*(a.tolist() for a in g._mark_pairs()))) == list(g._iter_edges())


def test_bare_digraph_defaults_to_arrows():
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge(0, 1)  # no endpoints attribute
    g = GraphStructure.from_networkx(G)
    assert g.endpoints(0, 1) == (TAIL, ARROW)


def test_neighbors_and_endpoints():
    g = directed([(0, 1), (0, 2)], 3)
    assert set(g.neighbors(0).tolist()) == {1, 2}
    assert g.endpoints(0, 1) == (TAIL, ARROW)
    assert g.endpoints(1, 0) == (ARROW, TAIL)
    assert g.endpoints(1, 2) == (NULL, NULL)


def test_to_scipy_sparse_matches_dense():
    g = directed([(0, 1), (1, 2)], 3)
    assert np.array_equal(g.to_scipy_sparse().toarray(), g.to_numpy())


def test_validate_accepts_valid_and_symmetric():
    directed([(0, 1), (1, 2)], 3).validate()
    _undirected_graph([(0, 1)], 2, kind="cpdag").validate()


def _undirected_graph(pairs, n, kind):
    return GraphStructure.from_numpy(_undirected(pairs, n), kind=kind)


def test_validate_rejects_circle_outside_pag():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_numpy(M, kind="cpdag").validate()


def test_validate_rejects_bidirected_outside_pag():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW
    with pytest.raises(ValueError, match="bidirected"):
        GraphStructure.from_numpy(M, kind="cpdag").validate()


def test_validate_rejects_undirected_in_dag():
    with pytest.raises(ValueError, match="undirected"):
        _undirected_graph([(0, 1)], 2, kind="dag").validate()


def test_validate_accepts_digraph_two_cycle_and_self_loop():
    # kind="digraph" reads ARROW/ARROW as two directed edges (a 2-cycle), not a bidirected PAG edge,
    # and carries autoregressive self-loops -- both are legal where dag/cpdag would reject them.
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1
    M[2, 2] = ARROW  # self-loop 2 -> 2
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    g.validate()
    assert g.kind == "digraph"


def test_validate_rejects_two_cycle_outside_pag_or_digraph():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW
    with pytest.raises(ValueError, match="digraph"):  # a dag double arrowhead names both escapes
        GraphStructure.from_numpy(M, kind="dag").validate()


def test_validate_rejects_undirected_in_digraph():
    with pytest.raises(ValueError, match="undirected"):
        _undirected_graph([(0, 1)], 2, kind="digraph").validate()


def test_validate_rejects_circle_in_digraph():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_numpy(M, kind="digraph").validate()


def test_from_numpy_rejects_bad_input():
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.zeros((2, 3), dtype=np.int8))  # non-square
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.array([[TAIL, 0], [0, 0]], dtype=np.int8))  # self-loop
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.array([[0, 9], [0, 0]], dtype=np.int8))  # out-of-range code
    with pytest.raises(ValueError):
        # negative endpoint codes are invalid
        GraphStructure.from_numpy(np.array([[0, -1], [ARROW, 0]], dtype=np.int8))


def test_from_numpy_rejects_asymmetric_support():
    # (0,1) has a mark but (1,0) does not - a malformed one-sided edge from a buggy remap.
    with pytest.raises(ValueError, match="symmetric"):
        GraphStructure.from_numpy(np.array([[0, TAIL], [0, 0]], dtype=np.int8))


def test_stored_arrays_are_read_only():
    g = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")
    assert g._layout == "sym"
    with pytest.raises(ValueError):
        g.neighbors(0)[0] = 5  # neighbors() is a view into the frozen store
    with pytest.raises(ValueError):
        g._marks[0] = 9


def test_labels_length_checked():
    with pytest.raises(ValueError):
        directed([(0, 1)], 2, labels=("only-one",))


def test_equality_and_unhashable():
    a = directed([(0, 1)], 3)
    b = directed([(0, 1)], 3)
    c = directed([(1, 0)], 3)
    assert a == b
    assert a != c
    assert a.__hash__ is None
    with pytest.raises(TypeError):
        {a}  # unhashable


def test_graph_not_equal_temporal():
    g = directed([(0, 1)], 2)
    t = TemporalStructure(_n_nodes=2, _lags=(g,))
    assert (g == t) is False
    assert (t == g) is False


def test_temporal_structure():
    g0 = directed([(0, 1)], 2)
    g1 = directed([], 2)
    t = TemporalStructure(_n_nodes=2, _lags=(g0, g1))
    assert t.n_lags == 2
    assert t.lag(0) == g0
    assert t.type == "temporal"
    assert t == TemporalStructure(_n_nodes=2, _lags=(g0, g1))
    with pytest.raises(ValueError):  # a lag with a mismatched node count
        TemporalStructure(_n_nodes=3, _lags=(g0,))


def test_to_numpy_is_deterministic():
    g = directed([(0, 1), (1, 2), (2, 0)], 3, kind="cpdag")
    first = g.to_numpy()
    for _ in range(3):
        assert np.array_equal(g.to_numpy(), first)


def test_node_types_latent_flagging(tmp_path):
    """Optional per-node types flag latents; they persist through equality and json."""
    from andrey.core import LATENT, OBSERVED, StructureOutput

    # a 3-node dag where L2 (latent) causes the two observed X0, X1.
    g = directed([(2, 0), (2, 1)], 3, kind="dag", labels=("X0", "X1", "L2"))
    node_types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    flagged = GraphStructure.from_numpy(
        g.to_numpy(), kind="dag", labels=("X0", "X1", "L2"), node_types=node_types
    )
    assert np.array_equal(flagged.node_types, [OBSERVED, OBSERVED, LATENT])
    # an all-observed graph reports None and is not equal to a latent-flagged one.
    assert g.node_types is None
    assert flagged != g
    # a wrong-length node_types is rejected.
    with pytest.raises(ValueError, match="node_types"):
        GraphStructure.from_numpy(g.to_numpy(), kind="dag", node_types=np.array([0, 1], np.int8))
    # the flags round-trip through both json and npz (a graph structure supports both).
    out = StructureOutput.new(flagged)
    for fmt, fname in (("json", "gin.json"), ("npz", "gin.npz")):
        p = tmp_path / fname
        out.save(p, fmt=fmt)
        assert StructureOutput.load(p) == out


def test_self_loop_round_trips_through_inverses():
    """A self-loop graph survives ``from_edges`` / ``from_networkx`` (its documented inverses)."""
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    M[2, 1], M[1, 2] = ARROW, TAIL  # directed edge 1 -> 2 (arrow at 2, tail at 1)
    g = GraphStructure.from_numpy(M, kind="dag", allow_self_loops=True)
    assert np.array_equal(GraphStructure.from_edges(g.to_edges(), n_nodes=3).to_numpy(), M)
    assert np.array_equal(GraphStructure.from_networkx(g.to_networkx()).to_numpy(), M)


def test_from_numpy_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="foo")


@pytest.mark.parametrize("mark", [TAIL, CIRCLE])
def test_a_self_loop_is_marked_with_an_arrowhead(mark):
    with pytest.raises(ValueError, match="self-loop is marked 2"):
        GraphStructure.from_numpy(np.diag([mark, 0]), kind="digraph", allow_self_loops=True)


def test_a_self_loop_reads_the_same_in_the_lag_graph_and_the_summary():
    empty = GraphStructure.from_numpy(np.zeros((2, 2), int))
    loop = GraphStructure.from_numpy(np.diag([ARROW, 0]), kind="digraph", allow_self_loops=True)
    summary = TemporalStructure.from_lag_graphs([empty, loop]).summary_graph()
    assert (
        loop.oriented_edges(index=True)
        == summary.oriented_edges(index=True)
        == [(0, 0, "directed")]
    )
