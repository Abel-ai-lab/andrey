"""PC orients colliders from the highest separating p-value down; an exact tie leaves a dispute.

With ``collider_order="column"``, it orients them in column order and keeps the first of two that
conflict.
"""

import numpy as np
import pytest

import andrey
from andrey.constraint.pc import _orient_colliders, _undirect_conflicts
from andrey.core import ARROW, TAIL
from andrey.core.orient import meek


def marks(d, directed=(), undirected=()):
    """The endpoint marks of a graph on ``d`` nodes with the given edges."""
    out = np.zeros((d, d), dtype=np.int8)
    for a, b in directed:
        out[a, b], out[b, a] = TAIL, ARROW
    for a, b in undirected:
        out[a, b] = out[b, a] = TAIL
    return out


def relabel(m, order):
    """The marks of the graph whose node ``k`` is node ``order[k]`` of ``m``."""
    return m[np.ix_(order, order)]


# Chain 0 - 1 - 2 - 3 with empty separating sets: the colliders at 1 (0, 1, 2) and at 2 (1, 2, 3)
# would orient the edge 1 - 2 in opposite directions.
CHAIN = marks(4, undirected=[(0, 1), (1, 2), (2, 3)])
CHAIN_SEPSETS = {(0, 2): (), (1, 3): ()}


def orient(skeleton, sepsets, pvalues):
    return _undirect_conflicts(meek(_orient_colliders(skeleton, sepsets, pvalues)))


@pytest.mark.parametrize(
    ("pvalues", "expected"),
    [
        ({(0, 2): 0.9, (1, 3): 0.3}, marks(4, directed=[(0, 1), (2, 1)], undirected=[(2, 3)])),
        ({(0, 2): 0.3, (1, 3): 0.9}, marks(4, directed=[(1, 2), (3, 2)], undirected=[(0, 1)])),
    ],
    ids=["first-in-column-order-ranks-higher", "second-ranks-higher"],
)
def test_the_higher_ranked_collider_wins_a_conflict(pvalues, expected):
    np.testing.assert_array_equal(orient(CHAIN, CHAIN_SEPSETS, pvalues), expected)


def test_an_exact_tie_leaves_the_disputed_edge_undirected():
    pvalues = {(0, 2): 0.5, (1, 3): 0.5}
    pdag = _orient_colliders(CHAIN, CHAIN_SEPSETS, pvalues)
    assert pdag[1, 2] == ARROW and pdag[2, 1] == ARROW  # both colliders set: 1 <-> 2
    trace: list[dict] = []
    out = _undirect_conflicts(meek(pdag), _trace=trace)
    # Undirected before Meek, R1 would orient 1 - 2 from 0 -> 1 or from 3 -> 2.
    np.testing.assert_array_equal(out, marks(4, directed=[(0, 1), (3, 2)], undirected=[(1, 2)]))
    assert [r["edges"] for r in trace if r["step"] == "conflict"] == [[(1, 2)]]


@pytest.mark.parametrize("ranks", [(0.9, 0.3), (0.3, 0.9), (0.5, 0.5)])
@pytest.mark.parametrize("order", [(3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)])
def test_the_orientation_follows_the_nodes_not_their_order(ranks, order):
    pvalues = dict(zip(CHAIN_SEPSETS, ranks))
    order = np.array(order)
    position = {int(node): k for k, node in enumerate(order)}

    def moved(pairs):
        return {tuple(sorted((position[a], position[b]))): v for (a, b), v in pairs.items()}

    relabeled = orient(relabel(CHAIN, order), moved(CHAIN_SEPSETS), moved(pvalues))
    np.testing.assert_array_equal(relabeled, relabel(orient(CHAIN, CHAIN_SEPSETS, pvalues), order))


@pytest.mark.parametrize("ranks", [(0.9, 0.3), (0.3, 0.9), (0.5, 0.5)])
def test_column_order_keeps_the_collider_visited_first(ranks):
    # The collider at node 1 comes first in column order, whatever the ranks; the one at 2 is
    # skipped, and no edge is left disputed.
    pvalues = dict(zip(CHAIN_SEPSETS, ranks))
    pdag = _orient_colliders(CHAIN, CHAIN_SEPSETS, pvalues, order="column")
    out = _undirect_conflicts(meek(pdag))
    np.testing.assert_array_equal(out, marks(4, directed=[(0, 1), (2, 1)], undirected=[(2, 3)]))


def test_an_unknown_collider_order_is_rejected():
    x = np.random.default_rng(0).standard_normal((40, 2))
    with pytest.raises(ValueError, match="collider_order must be 'p_value' or 'column'"):
        andrey.pc(x, collider_order="first")
