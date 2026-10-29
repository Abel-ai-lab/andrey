"""PC orients colliders from the highest separating p-value down; an exact tie leaves a dispute.

With ``collider_order="column"``, it orients them in column order and keeps the first of two that
conflict.
"""

import numpy as np
import pytest

import andrey
from andrey.constraint.cdnod import _orient_with_context
from andrey.constraint.pc import _orient_colliders, _undirect_conflicts, collider_conflicts
from andrey.core import ARROW, NULL, TAIL
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


def test_a_conflict_is_the_edge_two_colliders_ask_to_orient_both_ways():
    assert collider_conflicts(CHAIN, CHAIN_SEPSETS) == {(1, 2)}
    assert collider_conflicts(CHAIN, {(0, 2): (1,), (1, 3): ()}) == set()  # 1 separates 0 and 2
    assert collider_conflicts(CHAIN, CHAIN_SEPSETS, ambiguous={(0, 1, 2)}) == set()


@pytest.mark.parametrize("method", ["pc", "cdnod"])
def test_pc_and_cdnod_report_the_conflicts_they_resolve(method):
    # A hidden common cause of 1 and 2 gives the chain 0 - 1 - 2 - 3 with empty separating sets.
    rng = np.random.default_rng(0)
    x0, x3, hidden = rng.standard_normal((3, 3000))
    x1 = x0 + hidden + 0.5 * rng.standard_normal(3000)
    x2 = x3 + hidden + 0.5 * rng.standard_normal(3000)
    data = np.column_stack([x0, x1, x2, x3])
    args = (data,) if method == "pc" else (data, np.zeros(3000))
    assert getattr(andrey, method)(*args).metadata["conflicts"] == [[1, 2]]
    assert andrey.pc(data[:, :2]).metadata["conflicts"] == []


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


def test_column_order_keeps_the_collider_visited_first():
    # The collider at node 1 comes first in column order, though these ranks pick the one at 2
    # under "p_value"; the one at 2 is skipped, and no edge is left disputed.
    pvalues = dict(zip(CHAIN_SEPSETS, (0.3, 0.9)))
    pdag = _orient_colliders(CHAIN, CHAIN_SEPSETS, pvalues, order="column")
    out = _undirect_conflicts(meek(pdag))
    np.testing.assert_array_equal(out, marks(4, directed=[(0, 1), (2, 1)], undirected=[(2, 3)]))


@pytest.mark.parametrize("method", ["pc", "cdnod"])
def test_an_unknown_collider_order_is_rejected(method):
    x = np.random.default_rng(0).standard_normal((40, 2))
    args = (x, np.repeat([0.0, 1.0], 20)) if method == "cdnod" else (x,)
    with pytest.raises(ValueError, match="collider_order must be one of 'p_value', 'column'"):
        getattr(andrey, method)(*args, collider_order="first")


@pytest.mark.parametrize("order", ["p_value", "column"])
def test_cdnod_directs_context_edges_before_ranking_colliders(order):
    # Context 3 is adjacent to 0 and 1, which are separated: a collider at the context, which
    # "p_value" ranks first. Directing the context's edges first skips it in either order, so the
    # collider 2 -> 0 <- 3 is kept.
    skeleton = marks(4, undirected=[(0, 3), (1, 3), (0, 2)])
    sepsets = {(0, 1): (), (2, 3): ()}
    pvalues = {(0, 1): 0.9, (2, 3): 0.5}
    pdag = _orient_with_context(skeleton, sepsets, pvalues, context=3, order=order)
    np.testing.assert_array_equal(pdag, marks(4, directed=[(3, 0), (3, 1), (2, 0)]))
    assert np.array_equal(pdag != NULL, skeleton != NULL)
