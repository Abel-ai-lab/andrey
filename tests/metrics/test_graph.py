"""Graph metrics: hand cases, identity, adversarial, and CPDAG canonicalization."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.metrics import (
    confounder_pair_scores,
    markov_equivalent,
    orientation_scores,
    shd,
    skeleton_scores,
    to_cpdag,
)

EDGE_01 = [[0, 1], [0, 0]]  # 0 -> 1
EDGE_10 = [[0, 0], [1, 0]]  # 1 -> 0
CHAIN = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]  # 0 -> 1 -> 2
FORK = [[0, 0, 0], [1, 0, 1], [0, 0, 0]]  # 1 -> 0, 1 -> 2 (MEC-equivalent to CHAIN)
VSTRUCT = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]  # 0 -> 2 <- 1 (a collider)


# ---- SHD ----------------------------------------------------------------------------------------


def test_shd_identity_is_zero():
    for g in (EDGE_01, CHAIN, VSTRUCT, np.zeros((4, 4), dtype=int)):
        assert shd(g, g) == 0


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (EDGE_01, EDGE_10, 1),  # reversal
        (EDGE_01, [[0, 0], [0, 0]], 1),  # deletion
        ([[0, 0], [0, 0]], EDGE_01, 1),  # insertion
        (CHAIN, FORK, 1),  # one edge reversed
    ],
)
def test_shd_hand_cases(a, b, expected):
    assert shd(a, b) == expected


def test_shd_complete_vs_empty_counts_every_pair():
    complete = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]  # 3 edges
    assert shd(complete, np.zeros((3, 3), dtype=int)) == 3


def test_shd_all_reversed_chain():
    reversed_chain = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]  # 2 -> 1 -> 0
    assert shd(CHAIN, reversed_chain) == 2  # both edges reversed, one edit each


def test_shd_endpoint_aware_doubles_a_reversal(pag_of):
    # 0 -> 1 vs 1 -> 0 as PAG marks: both endpoints flip, so endpoint-aware costs 2, structural 1.
    forward = pag_of([[0, 1], [2, 0]])
    backward = pag_of([[0, 2], [1, 0]])
    assert shd(forward, backward) == 1
    assert shd(forward, backward, endpoint_aware=True) == 2


# ---- skeleton / orientation ---------------------------------------------------------------------


def test_skeleton_perfect_recovery():
    scores = skeleton_scores(CHAIN, CHAIN)
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_skeleton_ignores_orientation():
    assert skeleton_scores(EDGE_01, EDGE_10)["f1"] == 1.0  # same skeleton, reversed


def test_skeleton_empty_vs_empty_is_perfect():
    z = np.zeros((3, 3), dtype=int)
    assert skeleton_scores(z, z)["f1"] == 1.0


def test_skeleton_empty_estimate_vs_true_edge():
    scores = skeleton_scores([[0, 0], [0, 0]], EDGE_01)
    assert scores["precision"] == 1.0  # no false positives (vacuous)
    assert scores["recall"] == 0.0  # missed the only edge
    assert scores["f1"] == 0.0


def test_skeleton_spurious_estimate_vs_empty_truth():
    scores = skeleton_scores(EDGE_01, [[0, 0], [0, 0]])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 1.0  # vacuous, nothing to recall
    assert scores["f1"] == 0.0


def test_orientation_reversal_is_wrong():
    assert orientation_scores(EDGE_01, EDGE_10)["f1"] == 0.0


def test_orientation_perfect_on_vstruct():
    assert orientation_scores(VSTRUCT, VSTRUCT)["f1"] == 1.0


# ---- confounder pairs ---------------------------------------------------------------------------


def test_confounder_pairs_detect_bidirected(pag_of):
    bidirected = pag_of([[0, 2], [2, 0]])  # 0 <-> 1
    directed = pag_of([[0, 1], [2, 0]])  # 0 -> 1, no confounder
    assert confounder_pair_scores(bidirected, bidirected)["precision"] == 1.0
    scores = confounder_pair_scores(directed, bidirected)
    assert scores["recall"] == 0.0  # truth has a confounder, estimate does not


def test_standalone_metrics_reject_digraph_vs_pag():
    # a digraph 2-cycle and a PAG bidirected edge share ARROW/ARROW marks but mean opposite things;
    # the mark-semantic metrics would otherwise report a false exact match (SHD 0, arrowhead f1 1.0,
    # a perfect confounder). Every kind-semantic entry must fail fast on the pair.
    from andrey.core import GraphStructure

    marks = np.array([[0, 2], [2, 0]], dtype=np.int8)
    digraph = GraphStructure.from_numpy(marks, kind="digraph")
    pag = GraphStructure.from_numpy(marks, kind="pag")
    for metric in (shd, orientation_scores, confounder_pair_scores):
        with pytest.raises(ValueError, match="digraph to a PAG"):
            metric(digraph, pag)
        with pytest.raises(ValueError, match="digraph to a PAG"):
            metric(pag, digraph)  # rejected from either side
    # the skeleton is kind-agnostic (both share the 0-1 edge), so it stays answerable
    assert skeleton_scores(digraph, pag)["f1"] == 1.0


# ---- CPDAG canonicalization ---------------------------------------------------------------------


def test_to_cpdag_keeps_vstructure_oriented():
    cpdag = to_cpdag(VSTRUCT)
    assert cpdag.kind == "cpdag"
    assert int((cpdag.to_numpy() == 2).sum()) == 2  # both collider arrowheads survive


def test_to_cpdag_unorients_a_chain():
    cpdag = to_cpdag(CHAIN)
    assert int((cpdag.to_numpy() == 2).sum()) == 0  # a chain's MEC is fully undirected


def test_mec_equivalent_dags_share_a_cpdag():
    # CHAIN and FORK are Markov-equivalent: their CPDAGs, hence CPDAG-SHD, are identical.
    assert shd(to_cpdag(CHAIN), to_cpdag(FORK)) == 0
    assert shd(CHAIN, FORK) == 1  # but the raw directed graphs differ


def test_to_cpdag_rejects_digraph():
    from andrey.core import GraphStructure

    # a directed cycle has no essential graph, so to_cpdag raises rather than pass it through.
    two_cycle = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], dtype=np.int8), kind="digraph")
    with pytest.raises(ValueError, match="digraph"):
        to_cpdag(two_cycle)


# ---- markov_equivalent (skeleton + Verma-Pearl v-structures) ------------------------------------


def test_markov_equivalent_same_mec_no_vstructure():
    # CHAIN (0->1->2) and FORK (1->0, 1->2): same skeleton, neither has a v-structure -> same MEC.
    assert markov_equivalent(CHAIN, FORK)


def test_markov_equivalent_identity():
    assert markov_equivalent(VSTRUCT, VSTRUCT)
    assert markov_equivalent(CHAIN, CHAIN)


def test_markov_equivalent_different_skeletons_separate_mec():
    # CHAIN and VSTRUCT differ in skeleton (0-1-2 vs 0-2-1), so they are not Markov equivalent.
    assert not markov_equivalent(CHAIN, VSTRUCT)


def test_markov_equivalent_same_skeleton_different_vstructure():
    # Same skeleton {0-1, 1-2} but VSTRUCT-like collider vs a chain: different v-structures.
    chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0 -> 1 -> 2, no collider
    collider = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])  # 0 -> 1 <- 2, collider at 1
    assert not markov_equivalent(chain, collider)


def test_markov_equivalent_accepts_cpdag_structures():
    # A GES CPDAG (endpoint marks, ARROW codes) compares to itself and to its DAG source.
    from andrey.core import GraphStructure

    marks = to_cpdag(VSTRUCT)  # kind="cpdag" GraphStructure with an ARROW-coded collider
    assert markov_equivalent(marks, VSTRUCT)
    same = GraphStructure.from_numpy(marks.to_numpy(), kind="cpdag")
    assert markov_equivalent(marks, same)


def test_markov_equivalent_rejects_node_count_mismatch():
    with pytest.raises(ValueError, match="node-count mismatch"):
        markov_equivalent(np.zeros((2, 2), dtype=int), np.zeros((3, 3), dtype=int))
