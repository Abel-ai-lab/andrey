"""GRaSP: greedy relaxation of the sparsest permutation over a linear-Gaussian BIC score.

A depth-bounded search of tucks (:func:`_dfs`) relaxes a variable order; each variable's parents
come from its persistent grow-shrink tree (:class:`_GST`). Scores are negated
:class:`~andrey.core.score.BICScore` local scores, so the search maximizes. The algorithm and its
reference are described on :func:`andrey.grasp`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore

# The depth-first search of covered tucks descends at most this many score-preserving moves before
# backtracking.
_DEPTH = 3

# DFS accepts a tuck when the total negated BIC increases by more than this tolerance;
# changes in (-tolerance, tolerance] are explored as ties. Grow recomputes full BIC scores
# when a candidate's score is this close to the score of the current parent set or another
# candidate, so batch rounding does not change strict improvement decisions or branch order.
_SCORE_EPS = 1e-6


class _GSTNode:
    """One node of a grow-shrink tree: a grow step and its cached higher-is-better local score.

    ``grow`` expands the node into the improving single-variable additions (children, best-scoring
    first); ``shrink`` records the removals that improve the score once no addition helps.
    Both are computed lazily on first visit and cached, so a later trace reaching this node reuses
    them. ``score`` is the local score after this node's grow step, in the search's higher-is-better
    convention.
    """

    __slots__ = ("tree", "add", "grow_score", "shrink_score", "branches", "remove")

    def __init__(self, tree: _GST, add: int | None = None, score: float | None = None) -> None:
        if score is None:
            score = tree.empty_score
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches: list[_GSTNode] | None = None
        self.remove: list[int] | None = None

    def __lt__(self, other: _GSTNode) -> bool:
        return self.grow_score < other.grow_score

    def grow(self, available: list[int], parents: list[int]) -> None:
        """Populate ``branches`` with the additions that improve this node's score (sorted).

        Scores the single-variable additions in one batch against the shared ``parents`` base.
        Rechecks near-tied candidates and the base with full BIC scores before the strict
        improvement comparison, and rechecks near-tied siblings before ordering them.
        ``parents`` is left unchanged.
        """
        self.branches = []
        vertex = self.tree.vertex
        deltas = [("add", add) for add in available]
        values = self.tree.delta.score_many_with_base(vertex, parents, deltas)
        base_rechecked = False
        for add, value in zip(available, values):
            candidate_score = -value
            if abs(candidate_score - self.grow_score) <= _SCORE_EPS:
                if not base_rechecked:
                    self.grow_score = -self.tree.score.score(vertex, parents)
                    # Grow precedes shrink, so both cached base scores must agree.
                    self.shrink_score = self.grow_score
                    base_rechecked = True
                candidate_score = -self.tree.score.score(vertex, [*parents, add])
            if candidate_score > self.grow_score:
                self.branches.append(_GSTNode(self.tree, add, candidate_score))
        ranked = sorted(self.branches, reverse=True)
        recheck: set[_GSTNode] = set()
        for left, right in zip(ranked, ranked[1:]):
            if left.grow_score - right.grow_score <= _SCORE_EPS:
                recheck.update((left, right))
        for branch in recheck:
            assert branch.add is not None
            branch.grow_score = -self.tree.score.score(vertex, [*parents, branch.add])
            branch.shrink_score = branch.grow_score
        # `trace` takes the first branch allowed by the prefix, so branch order determines the
        # search path. Sort the original candidate order so exact ties retain that order.
        self.branches.sort(reverse=True)

    def shrink(self, parents: list[int]) -> None:
        """Record in ``remove`` the parents whose deletion improves the score, greedily.

        Repeatedly removes the single parent that most improves ``shrink_score`` until none does,
        mutating ``parents`` to the shrunk set and appending each removed parent to ``remove``.
        """
        self.remove = []
        vertex = self.tree.vertex
        score_fn = self.tree.score
        # The cached grow score may be approximate even when there are no grow candidates left.
        self.shrink_score = -score_fn.score(vertex, parents)
        while True:
            best: int | None = None
            for candidate in list(parents):
                parents.remove(candidate)
                candidate_score = -score_fn.score(vertex, parents)
                parents.append(candidate)
                if candidate_score > self.shrink_score:
                    self.shrink_score = candidate_score
                    best = candidate
            if best is None:
                break
            self.remove.append(best)
            parents.remove(best)

    def trace(
        self,
        available: list[int],
        parents: list[int],
        prefix_set: set[int],
    ) -> float:
        """Walk the grow-shrink tree along ``prefix_set``, returning the settled local score.

        Descends into the first improving addition whose variable is an allowed predecessor (in
        ``prefix_set``), appending it to ``parents``; variables passed over are removed from
        ``available``. When no further addition is allowed, applies the shrink step and returns the
        higher-is-better local score of the settled parent set.
        """
        if self.branches is None:
            self.grow(available, parents)
        branches = self.branches
        assert branches is not None  # grow() above always assigns a list
        for branch in branches:
            add = branch.add
            assert add is not None  # only the root has add=None, and it is never a branch
            available.remove(add)
            if add in prefix_set:
                parents.append(add)
                return branch.trace(available, parents, prefix_set)
        if self.remove is None:
            self.shrink(parents)
            return self.shrink_score
        for removed in self.remove:
            parents.remove(removed)
        return self.shrink_score


class _GST:
    """Persistent grow-shrink tree for one target variable over a fixed score.

    Holds the root node and the empty-parent-set score, both reused across every permutation the
    search visits. :meth:`trace` grows/shrinks the target's parents from the variables that precede
    it (``prefix``), reusing any cached path the tree already holds.
    """

    __slots__ = ("vertex", "score", "delta", "n_features", "empty_score", "root")

    def __init__(self, vertex: int, score: BICScore, delta: DeltaBICScore, n_features: int) -> None:
        self.vertex = vertex
        self.score = score
        self.delta = delta
        self.n_features = n_features
        self.empty_score = -score.score(vertex, [])
        self.root = _GSTNode(self)

    def trace(self, prefix: Sequence[int], parents: list[int]) -> float:
        """Grow-shrink the target's parents from the allowed predecessors ``prefix``.

        ``parents`` (mutated in place) receives the chosen parent set; the return is its
        higher-is-better local score. ``available`` is every variable except the target.
        """
        available = [i for i in range(self.n_features) if i != self.vertex]
        return self.root.trace(available, parents, set(prefix))


class _Order:
    """A variable permutation with each variable's parent set, local score, and total edge count.

    ``order`` is the permutation; ``parents``/``local_scores`` map a variable to its grow-shrink
    parent set and that set's higher-is-better local score; ``edges`` tracks the running edge total
    the DFS reports. Positions (variable -> index) are cached and rebuilt lazily after a mutation.
    """

    __slots__ = ("order", "parents", "local_scores", "edges", "_positions", "_positions_dirty")

    def __init__(self, n_features: int, rng: random.Random, gsts: list[_GST]) -> None:
        self.order = list(range(n_features))
        self.parents: dict[int, list[int]] = {}
        self.local_scores: dict[int, float] = {}
        self.edges = 0
        self._positions: dict[int, int] = {}
        self._positions_dirty = False

        rng.shuffle(self.order)
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


def _update(i: int, j: int, order: _Order, gsts: list[_GST]) -> tuple[int, float]:
    """Re-score the span ``[j, i]`` after a tuck; return the edge-count and total-score deltas."""
    edge_bump = 0
    old_score = 0.0
    new_score = 0.0
    for k in range(j, i + 1):
        z = order.get(k)
        z_parents = order.get_parents(z)
        edge_bump -= len(z_parents)
        old_score += order.get_local_score(z)
        z_parents.clear()
        local_score = gsts[z].trace(order.order[:k], z_parents)
        order.set_local_score(z, local_score)
        edge_bump += len(z_parents)
        new_score += local_score
    return edge_bump, new_score - old_score


def _is_covered(x: int, y_parent_set: set[int], order: _Order) -> bool:
    """True when ``x -> y`` is covered: ``Pa(x) + {x}`` equals ``Pa(y)`` (``y`` the tuck target)."""
    x_parent_set = set(order.get_parents(x))
    x_parent_set.add(x)
    return x_parent_set == y_parent_set


def _flipped_pairs(x: int, i: int, order: _Order) -> set[tuple[int, int]]:
    """Sorted pairs ``(x, z)`` for parents ``z`` of ``x`` that precede position ``i``."""
    return {(x, z) if x <= z else (z, x) for z in order.get_parents(x) if order.index(z) < i}


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


def _dfs(
    depth: int,
    flipped: set[tuple[int, int]],
    history: list[set[tuple[int, int]]],
    order: _Order,
    gsts: list[_GST],
    rng: random.Random,
    history_keys: set[frozenset[tuple[int, int]]] | None = None,
) -> bool:
    """Search covered tucks for a score-improving move; return ``True`` when one is applied.

    Visits variables and their parents in shuffled order. Each covered tuck is applied and the span
    re-scored: a strictly-improving move is kept (returns ``True``); a score-preserving move is
    recursed through (bounded by ``depth``, de-duplicated by the set of flipped edge pairs) before
    the tuck is rolled back. Returns ``False`` when no move improves the score.
    """
    if history_keys is None:
        history_keys = {frozenset(item) for item in history}

    indices = list(range(order.len()))
    rng.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y)
        rng.shuffle(y_parents)
        y_parent_set = set(y_parents)
        for x in y_parents:
            covered = _is_covered(x, y_parent_set, order)
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            cache = _snapshot(order, j, i)
            _tuck(i, j, order)
            edge_bump, score_bump = _update(i, j, order, gsts)
            if score_bump > _SCORE_EPS:
                order.bump_edges(edge_bump)
                return True
            if score_bump > -_SCORE_EPS:
                flipped = flipped ^ _flipped_pairs(x, i, order)
                if len(flipped) > 0:
                    flipped_key = frozenset(flipped)
                    if flipped_key not in history_keys:
                        history.append(flipped)
                        history_keys.add(flipped_key)
                        if depth > 0 and _dfs(
                            depth - 1, flipped, history, order, gsts, rng, history_keys
                        ):
                            return True
                        del history[-1]
                        history_keys.remove(flipped_key)
            _restore(order, j, cache)
    return False


def grasp(
    data: np.ndarray, *, lambda_value: float = 1.0, random_state: int = 0
) -> tuple[GraphStructure, float]:
    """Learn a CPDAG from ``data`` by GRaSP over the linear-Gaussian BIC score.

    Runs the greedy relaxation of the sparsest permutation: seeds a permutation, grow-shrinks each
    variable's parents from its predecessors, then relaxes the order through depth-bounded covered
    tucks until no tuck improves the total score. The settled parent sets form a DAG, returned as
    its CPDAG. The search is stochastic and seeded from ``random_state`` for a reproducible run.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.
    random_state : int
        Seed for the private RNG that fixes the initial permutation and DFS shuffles.

    Returns
    -------
    tuple[GraphStructure, float]
        The learned CPDAG, and the total BIC score of the settled DAG (sum of local
        ``local_score_BIC_from_cov`` values; lower is better).
    """
    X = np.asarray(data, dtype=np.float64)
    n_features = X.shape[1]
    if n_features <= 1:  # no candidate parents -- the CPDAG is the edgeless graph
        empty = np.zeros((n_features, n_features), dtype=np.int8)
        return to_structure(empty, kind="cpdag"), 0.0

    score = BICScore(X, lambda_value=lambda_value)
    delta = DeltaBICScore(score)
    rng = random.Random(random_state)
    gsts = [_GST(v, score, delta, n_features) for v in range(n_features)]

    order = _Order(n_features, rng, gsts)
    for i in range(n_features):
        y = order.get(i)
        y_parents = order.get_parents(y)
        local_score = gsts[y].trace(order.order[:i], y_parents)
        order.set_local_score(y, local_score)
        order.bump_edges(len(y_parents))

    while _dfs(_DEPTH - 1, set(), [], order, gsts, rng):
        pass

    adj = np.zeros((n_features, n_features), dtype=np.int8)
    total_bic = 0.0
    for y in range(n_features):
        parents = order.get_parents(y)
        total_bic += score.score(y, parents)
        for x in parents:
            adj[x, y] = TAIL
            adj[y, x] = ARROW

    cpdag = dag2cpdag(adj)
    return to_structure(cpdag, kind="cpdag"), float(total_bic)
