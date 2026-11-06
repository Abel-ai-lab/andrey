"""GRaSP: greedy relaxation of the sparsest permutation over a BIC, BDeu, or mixed-data score.

A depth-bounded search of tucks (:func:`_dfs`) relaxes a variable order; each variable's parents
come from its persistent grow-shrink tree, BOSS's :class:`~andrey.search.boss._GrowShrinkTree`.
Scores are negated local scores, so the search maximizes. The search runs on the columns sorted by
tie rank (:func:`~andrey.search.boss.ranked_score`), so a variable's index is its tie rank. The
algorithm and its reference are described on :func:`andrey.grasp`.
"""

from __future__ import annotations

import random

import numpy as np

from andrey.core import GraphStructure
from andrey.core.knowledge import Constraints
from andrey.search.boss import (
    _SCORE_EPS,
    _GrowShrinkTree,
    permutation_search,
    precedence_sort,
)


def _settled(tree: _GrowShrinkTree, prefix: set[int], parents: list[int]) -> float:
    """Trace ``tree`` from ``prefix`` into ``parents``; return the traced set's full negated score.

    The leaf is scored in full (:meth:`~andrey.search.boss._Leaf.settle`), so the DFS compares full
    local scores.
    """
    leaf = tree.trace(prefix, parents)
    leaf.settle()
    return leaf.score


class _Order:
    """A variable permutation with each variable's parent set, local score, and total edge count.

    ``order`` is the permutation; ``parents``/``local_scores`` map a variable to its grow-shrink
    parent set and that set's higher-is-better local score; ``edges`` tracks the running edge total
    the DFS reports. Positions (variable -> index) are cached and rebuilt lazily after a mutation.
    The seeded shuffle starts the order; with ``before`` (``before[a, b]``: ``a`` precedes ``b``) it
    is then rearranged as little as possible to keep ``before``.
    """

    __slots__ = ("order", "parents", "local_scores", "edges", "_positions", "_positions_dirty")

    def __init__(
        self,
        n_features: int,
        rng: random.Random,
        gsts: list[_GrowShrinkTree],
        before: np.ndarray | None = None,
    ) -> None:
        self.order = list(range(n_features))
        self.parents: dict[int, list[int]] = {}
        self.local_scores: dict[int, float] = {}
        self.edges = 0
        self._positions: dict[int, int] = {}
        self._positions_dirty = False

        rng.shuffle(self.order)
        if before is not None:
            self.order = precedence_sort(self.order, before)
        self._rebuild_positions()
        for y in self.order:
            self.parents[y] = []
            self.local_scores[y] = gsts[y].root.grow_score

    def get(self, i: int) -> int:
        return self.order[i]

    def index(self, y: int) -> int:
        if self._positions_dirty:
            self._rebuild_positions()
        return self._positions[y]

    def insert(self, i: int, y: int) -> None:
        self.order.insert(i, y)
        self._positions_dirty = True

    def pop(self, i: int = -1) -> int:
        y = self.order.pop(i)
        self._positions_dirty = True
        return y

    def get_parents(self, y: int) -> list[int]:
        return self.parents[y]

    def get_local_score(self, y: int) -> float:
        return self.local_scores[y]

    def set_local_score(self, y: int, local_score: float) -> None:
        self.local_scores[y] = local_score

    def bump_edges(self, bump: int) -> None:
        self.edges += bump

    def len(self) -> int:
        return len(self.order)

    def _rebuild_positions(self) -> None:
        self._positions = {y: i for i, y in enumerate(self.order)}
        self._positions_dirty = False


def _collect_ancestors(y: int, order: _Order, ancestors: list[int], ancestor_set: set[int]) -> None:
    """Depth-first collect ``y`` and its ancestors (current parent sets) into ``ancestors``."""
    ancestors.append(y)
    ancestor_set.add(y)
    for x in order.get_parents(y):
        if x not in ancestor_set:
            _collect_ancestors(x, order, ancestors, ancestor_set)


def _tuck(i: int, j: int, order: _Order) -> None:
    """Tuck the variable at position ``i`` back to position ``j`` (``j < i``).

    Every variable strictly between ``j`` and ``i`` that is an ancestor of ``order[i]`` is shifted
    to just before position ``j``, reversing the covered edge while keeping the order else intact.
    """
    ancestors: list[int] = []
    ancestor_set: set[int] = set()
    _collect_ancestors(order.get(i), order, ancestors, ancestor_set)
    shift = 0
    for k in range(j + 1, i + 1):
        if order.get(k) in ancestor_set:
            order.insert(j + shift, order.pop(k))
            shift += 1


def _update(i: int, j: int, order: _Order, gsts: list[_GrowShrinkTree]) -> tuple[int, float]:
    """Re-score the span ``[j, i]`` after a tuck; return the edge-count and total-score deltas."""
    edge_bump = 0
    old_score = 0.0
    new_score = 0.0
    prefix = set(order.order[:j])
    for k in range(j, i + 1):
        z = order.get(k)
        z_parents = order.get_parents(z)
        edge_bump -= len(z_parents)
        old_score += order.get_local_score(z)
        z_parents.clear()
        local_score = _settled(gsts[z], prefix, z_parents)
        order.set_local_score(z, local_score)
        edge_bump += len(z_parents)
        new_score += local_score
        prefix.add(z)
    return edge_bump, new_score - old_score


def _is_covered(x: int, y_parent_set: set[int], order: _Order) -> bool:
    """True when ``x -> y`` is covered: ``Pa(x) + {x}`` equals ``Pa(y)`` (``y`` the tuck target)."""
    x_parent_set = set(order.get_parents(x))
    x_parent_set.add(x)
    return x_parent_set == y_parent_set


def _snapshot(order: _Order, start: int, stop: int) -> tuple:
    """Snapshot the order span ``[start, stop]`` -- variables, parent sets, scores, edge total."""
    vertices: list[int] = []
    parent_snapshots: list[list[int]] = []
    local_scores: list[float] = []
    for k in range(start, stop + 1):
        z = order.order[k]
        vertices.append(z)
        parent_snapshots.append(order.parents[z][:])
        local_scores.append(order.local_scores[z])
    return vertices, parent_snapshots, local_scores, order.edges


def _restore(order: _Order, start: int, snapshot: tuple) -> None:
    """Restore an order span from :func:`_snapshot`, marking cached positions stale."""
    vertices, parent_snapshots, local_scores, edges = snapshot
    for offset, z in enumerate(vertices):
        order.order[start + offset] = z
        order.parents[z] = parent_snapshots[offset]
        order.local_scores[z] = local_scores[offset]
    order.edges = edges
    order._positions_dirty = True


def _dag_key(order: _Order) -> frozenset[tuple[int, int]]:
    """The DAG the order induces, as its set of ``(parent, child)`` edges."""
    return frozenset((x, y) for y, parents in order.parents.items() for x in parents)


def _dfs(
    depth: int,
    chain: set[frozenset[tuple[int, int]]],
    order: _Order,
    gsts: list[_GrowShrinkTree],
    rng: random.Random,
    before: np.ndarray | None = None,
) -> bool:
    """Search tucks for a score-improving chain; return ``True`` when one is applied.

    ``chain`` holds the DAGs of the chain of tucks that led to this order, its own DAG included.
    Visits variables and their parents in shuffled order: any edge at the first level, covered
    edges below. A tuck that raises the total negated score by more than ``_SCORE_EPS`` is kept
    (returns ``True``); one that changes it by an amount in ``(-_SCORE_EPS, _SCORE_EPS]``, to a DAG
    not yet in the chain, is recursed through (bounded by ``depth``) before it is rolled back.
    A tuck that puts ``b`` before ``a`` where ``before[a, b]`` is rolled back unscored. Returns
    ``False`` when no chain improves the score.
    """
    indices = list(range(order.len()))
    rng.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y)
        rng.shuffle(y_parents)
        y_parent_set = set(y_parents)
        # A copy: re-scoring a tuck refills y's stored parent list.
        for x in list(y_parents):
            if len(chain) > 1 and not _is_covered(x, y_parent_set, order):
                continue
            j = order.index(x)
            cache = _snapshot(order, j, i)
            _tuck(i, j, order)
            if before is not None:  # a tuck reorders the span [j, i] only
                span = order.order[j : i + 1]
                if np.tril(before[np.ix_(span, span)], -1).any():
                    _restore(order, j, cache)
                    continue
            edge_bump, score_bump = _update(i, j, order, gsts)
            if score_bump > _SCORE_EPS:
                order.bump_edges(edge_bump)
                return True
            if depth > 0 and score_bump > -_SCORE_EPS:
                key = _dag_key(order)
                if key not in chain:
                    chain.add(key)
                    if _dfs(depth - 1, chain, order, gsts, rng, before):
                        return True
                    chain.remove(key)
            _restore(order, j, cache)
    return False


def _search_from(
    n_features: int,
    rng: random.Random,
    gsts: list[_GrowShrinkTree],
    depth: int,
    before: np.ndarray | None,
) -> _Order:
    """Grow-shrink every variable of an ``rng``-shuffled order, then tuck until none improves."""
    order = _Order(n_features, rng, gsts, before)
    prefix: set[int] = set()
    for i in range(n_features):
        y = order.get(i)
        y_parents = order.get_parents(y)
        local_score = _settled(gsts[y], prefix, y_parents)
        order.set_local_score(y, local_score)
        order.bump_edges(len(y_parents))
        prefix.add(y)

    while _dfs(depth - 1, {_dag_key(order)}, order, gsts, rng, before):
        pass
    return order


def grasp(
    data: np.ndarray,
    *,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    depth: int = 3,
    random_state: int = 0,
    n_starts: int = 1,
    score_func: str = "bic",
    max_parents: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float]:
    """Learn a CPDAG from ``data`` by GRaSP over a linear-Gaussian or mixed-data BIC score.

    Runs the greedy relaxation of the sparsest permutation: seeds a permutation, grow-shrinks each
    variable's parents from its predecessors, then relaxes the order through depth-bounded covered
    tucks until no tuck improves the total score. The settled parent sets form a DAG, returned as
    its CPDAG. The search is stochastic and seeded from ``random_state`` for a reproducible run. It
    runs on the columns sorted by :func:`~andrey.search.boss.ranked_score`, so the starting order,
    the visiting order, and every tie follow the columns' data, and reordering the columns of
    ``data`` reorders the CPDAG's nodes.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    penalty_discount : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    equivalent_sample_size : float
        Prior weight of ``"bdeu"``, the equivalent sample size of its uniform Dirichlet prior
        (:class:`~andrey.core.score.BDeuScore`); positive. The other scores ignore it.
    depth : int
        Most tucks in one chain of moves, at least ``1``.
    random_state : int
        Seed for the private RNG of each start, which fixes its initial permutation and DFS
        shuffles (:func:`~andrey.core.seeding.start_rng`).
    n_starts : int
        Number of searches, each from its own seeded initial permutation, at least ``1``. The
        returned DAG is the best-scoring one; a later start replaces an earlier one only when it
        scores better by more than ``_SCORE_EPS``.
    score_func : str
        Local score, one of :data:`~andrey.core.score.SCORES`.
    max_parents : int or None
        Caps each node's parent set, required parents included: the grow step adds no parent to a
        set that holds ``max_parents`` members. ``None`` sets no cap. With a cap, ``constraints``
        may not require more than ``max_parents`` parents of a node (``ValueError``).
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; BDeu and the multinomial BIC read
        every column as categorical.
    constraints : Constraints or None
        Background knowledge: each variable's parents exclude the forbidden ones and include the
        required ones, and every order the search visits keeps
        :meth:`~andrey.core.knowledge.Constraints.precedence`: the seeded starting order is
        rearranged as little as possible to keep it, and a tuck that breaks it is not taken. The
        CPDAG is then narrowed by the knowledge (:func:`~andrey.core.knowledge.mpdag`).

    Returns
    -------
    tuple[GraphStructure, float]
        The learned CPDAG, and the total score of the settled DAG (the sum of its local scores;
        lower is better).
    """

    def run_start(
        start: int, rng: random.Random, gsts: list[_GrowShrinkTree], before: np.ndarray | None
    ) -> dict[int, list[int]]:
        return _search_from(len(gsts), rng, gsts, depth, before).parents

    return permutation_search(
        data,
        run_start,
        score_func=score_func,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        categorical=categorical,
        constraints=constraints,
        max_parents=max_parents,
        n_starts=n_starts,
        random_state=random_state,
    )
