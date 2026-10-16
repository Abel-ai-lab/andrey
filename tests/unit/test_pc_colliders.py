"""PC orients every collider at once, and an edge two colliders dispute ends undirected."""

import numpy as np
import pytest

from andrey.constraint.cdnod import _orient_with_context
from andrey.constraint.pc import _orient_colliders, _undirect_conflicts
from andrey.core import ARROW, NULL, TAIL
from andrey.core.orient import meek


def undirected(edges, d):
    skeleton = np.zeros((d, d), dtype=np.int8)
    for a, b in edges:
        skeleton[a, b] = skeleton[b, a] = TAIL
    return skeleton


def relabel(marks, order):
    """The marks of the graph whose node ``k`` is node ``order[k]`` of ``marks``."""
    return marks[np.ix_(order, order)]


# Chain 0 - 1 - 2 - 3 with empty separating sets: colliders at 1 (0, 1, 2) and at 2 (1, 2, 3)
# give the edge 1 - 2 an arrowhead at each end.
CHAIN = undirected([(0, 1), (1, 2), (2, 3)], 4)
CHAIN_SEPSETS = {(0, 2): (), (1, 3): (), (0, 3): ()}


def orient(skeleton, sepsets):
    return _undirect_conflicts(meek(_orient_colliders(skeleton, sepsets)))


def test_conflicting_colliders_mark_the_shared_edge_bidirected():
    pdag = _orient_colliders(CHAIN, CHAIN_SEPSETS)
    assert pdag[0, 1] == TAIL and pdag[1, 0] == ARROW  # 0 -> 1
    assert pdag[1, 2] == ARROW and pdag[2, 1] == ARROW  # 1 <-> 2
    assert pdag[3, 2] == TAIL and pdag[2, 3] == ARROW  # 3 -> 2


def test_a_conflicting_edge_stays_undirected_through_meek():
    # Undirected before Meek, R1 would orient 1 - 2 from 0 -> 1 or from 3 -> 2, whichever ran first.
    out = orient(CHAIN, CHAIN_SEPSETS)
    assert out[1, 2] == TAIL and out[2, 1] == TAIL
    assert out[1, 0] == ARROW and out[2, 3] == ARROW


@pytest.mark.parametrize("order", [(3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)])
def test_the_orientation_follows_the_nodes_not_their_order(order):
    order = np.array(order)
    position = {int(node): k for k, node in enumerate(order)}
    sepsets = {(position[a], position[b]): () for a, b in CHAIN_SEPSETS}
    relabeled = orient(relabel(CHAIN, order), sepsets)
    assert np.array_equal(relabeled, relabel(orient(CHAIN, CHAIN_SEPSETS), order))


def test_cdnod_points_context_edges_out_over_a_collider_at_the_context():
    # Context 2 adjacent to 0 and 1, which are separated by the empty set: a collider at 2.
    skeleton = undirected([(0, 2), (1, 2)], 3)
    pdag = _orient_with_context(skeleton, {(0, 1): ()}, context=2)
    assert pdag[2, 0] == TAIL and pdag[0, 2] == ARROW  # 2 -> 0
    assert pdag[2, 1] == TAIL and pdag[1, 2] == ARROW  # 2 -> 1
    assert not ((pdag == ARROW) & (pdag.T == ARROW)).any()
    assert np.array_equal(pdag != NULL, skeleton != NULL)
