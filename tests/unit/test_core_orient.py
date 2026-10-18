"""Reference-free tests for the graph-orientation substrate.

Covers the four primitives over endpoint-mark matrices -- ``meek``, ``pdag2dag``, ``dag2cpdag``,
``orient_colliders`` -- plus the ``to_structure`` / ``from_structure`` interop. Exercises
hand-checked graphs whose CPDAGs are known by inspection, the invariants tying the primitives
together, inline goldens for fixed inputs, and every documented fail-fast path. Runs without any
external reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core.orient import (
    _cpdag_of_pdag,
    dag2cpdag,
    from_structure,
    meek,
    orient_colliders,
    pdag2dag,
    to_structure,
)
from andrey.core.structure import ARROW, NULL, TAIL

# ---- construction helpers -----------------------------------------------------------------------


def dag(pairs: list[tuple[int, int]], n: int) -> np.ndarray:
    """Endpoint-mark matrix of directed edges ``i -> j`` (tail at ``i``, arrowhead at ``j``)."""
    a = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        a[i, j] = TAIL
        a[j, i] = ARROW
    return a


def undirected(pairs: list[tuple[int, int]], n: int) -> np.ndarray:
    """Endpoint-mark matrix of undirected edges ``i -- j`` (tails at both ends)."""
    a = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        a[i, j] = TAIL
        a[j, i] = TAIL
    return a


def random_dag(rng: np.random.Generator, d: int, p: float = 0.4) -> np.ndarray:
    """A random DAG on ``d`` nodes: edges follow a random topological permutation."""
    perm = rng.permutation(d)
    a = np.zeros((d, d), dtype=np.int8)
    for lo in range(d):
        for hi in range(lo + 1, d):
            if rng.random() < p:
                u, v = int(perm[lo]), int(perm[hi])
                a[u, v] = TAIL
                a[v, u] = ARROW
    return a


def true_sepsets(adj: np.ndarray) -> dict[tuple[int, int], set[int]]:
    """Separating sets for a DAG: the union of the two endpoints' parent sets."""
    d = adj.shape[0]
    parents = adj == ARROW
    sepsets: dict[tuple[int, int], set[int]] = {}
    for x in range(d):
        for y in range(x + 1, d):
            if adj[x, y] == NULL:
                px = set(np.nonzero(parents[x])[0].tolist())
                py = set(np.nonzero(parents[y])[0].tolist())
                sepsets[(x, y)] = px | py
    return sepsets


# ---- hand-checked DAG-to-CPDAG cases -------------------------------------------------------------


def test_dag2cpdag_vstructure_stays_directed():
    # An unshielded collider 0 -> 1 <- 2 is compelled: its two arrowheads are shared by every
    # member of the equivalence class, so the CPDAG equals the input DAG.
    vstruct = dag([(0, 1), (2, 1)], 3)
    assert np.array_equal(dag2cpdag(vstruct), vstruct)


def test_dag2cpdag_chain_fully_unorients():
    # A chain 0 -> 1 -> 2 has no collider; both edges are reversible and become undirected.
    chain = dag([(0, 1), (1, 2)], 3)
    assert np.array_equal(dag2cpdag(chain), undirected([(0, 1), (1, 2)], 3))


def test_dag2cpdag_triangle_fully_unorients():
    # A shielded triple 0 -> 1, 0 -> 2, 1 -> 2 has no unshielded collider; the whole triangle is
    # reversible and unorients to 0 -- 1 -- 2 -- 0.
    triangle = dag([(0, 1), (0, 2), (1, 2)], 3)
    assert np.array_equal(dag2cpdag(triangle), undirected([(0, 1), (0, 2), (1, 2)], 3))


@pytest.mark.parametrize("mark", [TAIL, ARROW])
def test_dag2cpdag_rejects_an_edge_that_is_not_directed(mark):
    # An undirected (tail-tail) or bidirected edge next to a directed one is not a DAG.
    graph = dag([(1, 2)], 3)
    graph[0, 1] = graph[1, 0] = mark
    with pytest.raises(ValueError, match="requires a DAG"):
        dag2cpdag(graph)


def test_dag2cpdag_diamond_golden():
    # Diamond 0 -> {1, 2} -> 3: the collider at 3 (1 and 2 non-adjacent) is compelled, while the
    # two top edges are reversible.
    diamond = dag([(0, 1), (0, 2), (1, 3), (2, 3)], 4)
    expected = np.array(
        [
            [0, TAIL, TAIL, 0],
            [TAIL, 0, 0, TAIL],
            [TAIL, 0, 0, TAIL],
            [0, ARROW, ARROW, 0],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(dag2cpdag(diamond), expected)


def test_dag2cpdag_fully_directed_golden():
    # A v-structure whose collider propagates: 0 -> 1 <- 2 compels 1 -> 3 (else a new collider at 1)
    # and 3 -> 4, leaving the CPDAG fully directed. Inline golden, reference-checked at write time.
    graph = dag([(0, 1), (2, 1), (1, 3), (3, 4)], 5)
    expected = np.array(
        [
            [0, TAIL, 0, 0, 0],
            [ARROW, 0, ARROW, TAIL, 0],
            [0, TAIL, 0, 0, 0],
            [0, ARROW, 0, 0, TAIL],
            [0, 0, 0, ARROW, 0],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(dag2cpdag(graph), expected)
    # A fully-directed CPDAG extends back to itself.
    assert np.array_equal(pdag2dag(expected), expected)


def test_dag2cpdag_empty_and_no_edges():
    # A DAG with no edges has an all-zero CPDAG; a zero-node matrix is handled without error.
    assert np.array_equal(dag2cpdag(np.zeros((4, 4), dtype=np.int8)), np.zeros((4, 4), np.int8))
    assert dag2cpdag(np.zeros((0, 0), dtype=np.int8)).shape == (0, 0)


# ---- Meek completion -----------------------------------------------------------------------------


def test_meek_rule_r1_orients_undirected():
    # R1: 0 -> 1 -- 2 with 0, 2 non-adjacent forces 1 -> 2 (else a new unshielded collider at 1).
    pdag = np.zeros((3, 3), dtype=np.int8)
    pdag[0, 1], pdag[1, 0] = TAIL, ARROW  # 0 -> 1
    pdag[1, 2], pdag[2, 1] = TAIL, TAIL  # 1 -- 2
    completed = meek(pdag)
    assert completed[1, 2] == TAIL and completed[2, 1] == ARROW  # 1 -> 2


def test_meek_leaves_isolated_undirected_edge():
    # An undirected edge with no orienting context stays undirected: Meek adds no arbitrary arrows.
    edge = undirected([(0, 1)], 2)
    assert np.array_equal(meek(edge), edge)


@pytest.mark.parametrize("d", range(4, 9))
def test_meek_is_a_fixed_point_of_cpdags(d: int):
    # Every CPDAG is already Meek-complete: applying the rules to a ``dag2cpdag`` output is a no-op.
    rng = np.random.default_rng(d)
    for _ in range(40):
        cpdag = dag2cpdag(random_dag(rng, d))
        assert np.array_equal(meek(cpdag), cpdag)


# ---- PDAG extension ------------------------------------------------------------------------------


@pytest.mark.parametrize("d", range(4, 9))
def test_pdag2dag_then_dag2cpdag_reproduces_cpdag(d: int):
    # A consistent extension of a CPDAG is a DAG in the same equivalence class, so re-deriving the
    # essential graph returns the original CPDAG.
    rng = np.random.default_rng(100 + d)
    for _ in range(40):
        cpdag = dag2cpdag(random_dag(rng, d))
        extension = pdag2dag(cpdag)
        assert np.array_equal(dag2cpdag(extension), cpdag)


def test_pdag2dag_keeps_directed_edges():
    # The extension retains every directed edge of the input and only orients undirected ones.
    pdag = np.zeros((3, 3), dtype=np.int8)
    pdag[0, 1], pdag[1, 0] = TAIL, ARROW  # 0 -> 1 (directed, must survive)
    pdag[1, 2], pdag[2, 1] = TAIL, TAIL  # 1 -- 2 (undirected, gets oriented)
    out = pdag2dag(pdag)
    assert out[0, 1] == TAIL and out[1, 0] == ARROW  # 0 -> 1 kept
    assert out[1, 2] != NULL and out[2, 1] != NULL  # 1 -- 2 now directed some way


def _perturbed_cpdag(rng: np.random.Generator, d: int, p: float | None = None) -> np.ndarray:
    """A random DAG's CPDAG with some undirected edges directed and some directed edges undirected.

    ``p`` is the DAG's edge density, drawn from ``[0.1, 0.7]`` when not given. Some of these PDAGs
    have no consistent extension.
    """
    p = float(rng.uniform(0.1, 0.7)) if p is None else p
    pdag = dag2cpdag(random_dag(rng, d, p))
    for a, b in np.argwhere((pdag == TAIL) & (pdag.T == TAIL)):
        if a < b and rng.random() < 0.3:
            pdag[a, b], pdag[b, a] = (TAIL, ARROW) if rng.random() < 0.5 else (ARROW, TAIL)
    for a, b in np.argwhere((pdag == TAIL) & (pdag.T == ARROW)):
        if rng.random() < 0.15:
            pdag[a, b], pdag[b, a] = TAIL, TAIL
    return pdag


def test_cpdag_of_pdag_matches_extension_then_cpdag():
    # The bitmask rebuild GES runs after each move returns dag2cpdag(pdag2dag(pdag)), and raises
    # pdag2dag's error when the PDAG has no consistent extension.
    rng = np.random.default_rng(7)
    raised = 0
    for _ in range(600):
        d = int(rng.integers(1, 30))
        for pdag in (random_dag(rng, d), _perturbed_cpdag(rng, d)):
            try:
                expected = dag2cpdag(pdag2dag(pdag))
            except ValueError:
                raised += 1
                with pytest.raises(ValueError, match="no consistent"):
                    _cpdag_of_pdag(pdag)
                continue
            assert np.array_equal(_cpdag_of_pdag(pdag), expected)
    assert 0 < raised < 600


@pytest.mark.parametrize("d", [63, 64, 65, 128, 129])
def test_cpdag_of_pdag_across_word_boundary(d):
    # Node masks past 64 nodes span more than one 64-bit word; the rebuild still matches the
    # reference on sparse DAGs, their CPDAGs, and perturbed CPDAGs.
    rng = np.random.default_rng(d)
    for _ in range(4):
        p = float(rng.uniform(1.0, 4.0)) / d
        for pdag in (random_dag(rng, d, p), _perturbed_cpdag(rng, d, p)):
            try:
                expected = dag2cpdag(pdag2dag(pdag))
            except ValueError:
                with pytest.raises(ValueError, match="no consistent"):
                    _cpdag_of_pdag(pdag)
                continue
            assert np.array_equal(_cpdag_of_pdag(pdag), expected)


# ---- collider orientation ------------------------------------------------------------------------


def test_orient_colliders_marks_unshielded_collider():
    # Skeleton 0 -- 1 -- 2 with 0, 2 non-adjacent and 1 not in sep(0, 2): 1 is a collider.
    skeleton = undirected([(0, 1), (1, 2)], 3)
    out = orient_colliders(skeleton, {(0, 2): []})
    assert out[0, 1] == TAIL and out[1, 0] == ARROW  # 0 -> 1
    assert out[2, 1] == TAIL and out[1, 2] == ARROW  # 2 -> 1


def test_orient_colliders_respects_separating_set():
    # With 1 in sep(0, 2) the triple is a non-collider: every edge stays undirected.
    skeleton = undirected([(0, 1), (1, 2)], 3)
    assert np.array_equal(orient_colliders(skeleton, {(0, 2): [1]}), skeleton)


def test_orient_colliders_sepset_key_order_and_missing():
    # The separating set is looked up in either key order; a missing pair defaults to an empty set,
    # so both a reversed key and no key at all yield the collider orientation.
    skeleton = undirected([(0, 1), (1, 2)], 3)
    reversed_key = orient_colliders(skeleton, {(2, 0): [1]})  # non-collider via reversed key
    assert np.array_equal(reversed_key, skeleton)
    missing = orient_colliders(skeleton, {})  # empty default -> collider
    assert missing[1, 0] == ARROW and missing[1, 2] == ARROW


def test_orient_colliders_none_value_is_empty_set():
    # A ``None`` separating-set value is treated as the empty set, giving a collider.
    skeleton = undirected([(0, 1), (1, 2)], 3)
    out = orient_colliders(skeleton, {(0, 2): None})
    assert out[1, 0] == ARROW and out[1, 2] == ARROW


@pytest.mark.parametrize("d", range(4, 9))
def test_orient_colliders_plus_meek_reproduces_cpdag(d: int):
    # Recovering a DAG's CPDAG from its skeleton: orient the unshielded colliders using the DAG's
    # true separating sets, then Meek-complete -- this reconstructs ``dag2cpdag`` of the DAG.
    rng = np.random.default_rng(200 + d)
    for _ in range(40):
        adj = random_dag(rng, d)
        cpdag = dag2cpdag(adj)
        skeleton = np.where(adj != NULL, np.int8(TAIL), np.int8(NULL))
        np.fill_diagonal(skeleton, np.int8(NULL))
        recovered = meek(orient_colliders(skeleton, true_sepsets(adj)))
        assert np.array_equal(recovered, cpdag)


# ---- structure interop ---------------------------------------------------------------------------


def test_to_from_structure_roundtrip():
    # A validated endpoint-mark matrix survives a round-trip through ``GraphStructure`` unchanged.
    adj = dag([(0, 1), (2, 1), (1, 3), (3, 4)], 5)
    cpdag = dag2cpdag(adj)
    structure = to_structure(cpdag, kind="cpdag")
    assert structure.kind == "cpdag"
    restored = from_structure(structure)
    assert restored.dtype == np.int8
    assert np.array_equal(restored, cpdag)


def test_to_structure_validates_input():
    # The emitter validates before building the store: a non-zero diagonal is rejected up front.
    bad = np.eye(3, dtype=np.int8)
    with pytest.raises(ValueError):
        to_structure(bad, kind="dag")


# ---- determinism ---------------------------------------------------------------------------------


def test_primitives_are_deterministic():
    # Repeated calls on identical input return identical matrices (no RNG, no hash-order effects).
    adj = random_dag(np.random.default_rng(7), 7)
    cpdag = dag2cpdag(adj)
    assert np.array_equal(dag2cpdag(adj), cpdag)
    assert np.array_equal(meek(cpdag), meek(cpdag))
    assert np.array_equal(pdag2dag(cpdag), pdag2dag(cpdag))


def test_input_is_not_mutated():
    # The primitives own their output and never write through to the caller's array.
    adj = dag([(0, 1), (1, 2)], 3)
    snapshot = adj.copy()
    dag2cpdag(adj)
    meek(adj)
    pdag2dag(adj)
    assert np.array_equal(adj, snapshot)


# ---- fail-fast paths -----------------------------------------------------------------------------


def test_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        dag2cpdag(np.zeros((2, 3), dtype=np.int8))


def test_rejects_out_of_range_marks():
    with pytest.raises(ValueError, match="marks must be"):
        dag2cpdag(np.array([[0, 5], [5, 0]], dtype=np.int8))


def test_rejects_non_zero_diagonal():
    with pytest.raises(ValueError, match="zero diagonal"):
        dag2cpdag(np.array([[TAIL, 0], [0, 0]], dtype=np.int8))


def test_rejects_asymmetric_support():
    # A half-edge (a mark at i -> j with no j -> i mark) breaks the symmetric-support invariant.
    asym = np.array([[0, TAIL], [0, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match="symmetric"):
        dag2cpdag(asym)


def test_dag2cpdag_rejects_cyclic_input():
    # A directed cycle 0 -> 1 -> 2 -> 0 has no topological order.
    cycle = dag([(0, 1), (1, 2), (2, 0)], 3)
    with pytest.raises(ValueError, match="acyclic"):
        dag2cpdag(cycle)


def test_dag2cpdag_rejects_bidirected_edge():
    # A bidirected edge (arrowheads at both ends) is not a DAG.
    bidirected = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match="DAG"):
        dag2cpdag(bidirected)


def test_pdag2dag_rejects_inextensible_pdag():
    # A 4-cycle 0 -- 1 -> 2 -- 3 -> 0 admits no consistent extension: orienting either undirected
    # edge closes a directed cycle or creates a new collider, so no removable sink ever exists.
    pdag = np.zeros((4, 4), dtype=np.int8)
    pdag[0, 1], pdag[1, 0] = TAIL, TAIL  # 0 -- 1
    pdag[1, 2], pdag[2, 1] = TAIL, ARROW  # 1 -> 2
    pdag[2, 3], pdag[3, 2] = TAIL, TAIL  # 2 -- 3
    pdag[3, 0], pdag[0, 3] = TAIL, ARROW  # 3 -> 0
    with pytest.raises(ValueError, match="no consistent"):
        pdag2dag(pdag)
