"""Best-Order Score Search (BOSS) over a linear-Gaussian BIC score.

Each variable owns a grow-shrink tree; sweeps slide variables to their best order position
(:func:`_better_mutation`) until no move improves the total. Scores are negated
:class:`~andrey.core.score.BICScore` local scores, so the search maximizes. The algorithm and its
reference are described on :func:`andrey.boss`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore

# Fixed absolute BIC-score tolerance for numerical ties.
_SCORE_EPS = 1e-6


class _GSTNode:
    """One node of a grow-shrink tree: a parent set reached along a grow path from the root.

    ``grow_score`` is the negated local score of the node's parent set; ``branches`` are the
    single-variable additions that improve it by more than ``_SCORE_EPS`` (best tie group first,
    ascending variable index within a group; built lazily by :meth:`_grow`). ``remove`` is the
    shrink sequence -- the variables dropped, best first -- built lazily by :meth:`_shrink`, with
    ``shrink_score`` the negated score after all removals.
    """

    __slots__ = ("tree", "add", "grow_score", "shrink_score", "branches", "remove")

    def __init__(self, tree: _GrowShrinkTree, add: int | None = None, score: float | None = None):
        if score is None:
            score = tree.empty_score
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches: list[_GSTNode] | None = None
        self.remove: list[int] | None = None

    def _grow(self, available: list[int], parents: list[int]) -> None:
        """Keep additions that improve ``grow_score`` by more than ``_SCORE_EPS``.

        Group improving additions within ``_SCORE_EPS`` of each group's best score. Visit groups
        best first, with ascending variable index within each group.
        """
        self.branches = []
        vertex = self.tree.vertex
        base = list(parents)
        deltas = [("add", int(u)) for u in available]
        values = self.tree.delta.score_many_with_base(vertex, base, deltas)
        candidates = sorted(
            (float(value), add)
            for add, value in zip(available, values)
            if -float(value) > self.grow_score + _SCORE_EPS
        )
        # Anchor each tie group to its best score: pairwise approximate comparisons are not
        # transitive. `trace` takes the first allowed branch, so each group uses variable order.
        start = 0
        while start < len(candidates):
            end = start + 1
            while end < len(candidates) and candidates[end][0] - candidates[start][0] <= _SCORE_EPS:
                end += 1
            for value, add in sorted(candidates[start:end], key=lambda candidate: candidate[1]):
                self.branches.append(_GSTNode(self.tree, add, -value))
            start = end

    def _shrink(self, parents: list[int]) -> None:
        """Remove parents while the score improves by more than ``_SCORE_EPS``.

        Each step selects the lowest-index parent among improving removals within ``_SCORE_EPS``
        of the best score. Record the removal order and cache the final score in ``shrink_score``.
        """
        self.remove = []
        vertex = self.tree.vertex
        while True:
            current = list(parents)
            if not current:
                break
            deltas = [("remove", int(u)) for u in current]
            values = self.tree.delta.score_many_with_base(vertex, current, deltas)
            candidates = [
                (u, -float(value))
                for u, value in zip(current, values)
                if -float(value) > self.shrink_score + _SCORE_EPS
            ]
            if not candidates:
                break
            best_score = max(gain for _, gain in candidates)
            best, self.shrink_score = min(
                (u, gain) for u, gain in candidates if best_score - gain <= _SCORE_EPS
            )
            self.remove.append(best)
            parents.remove(best)

    def trace(self, available: list[int], parents: list[int], prefix_set: set[int]) -> float:
        """Walk the grow path allowed by ``prefix_set``, then shrink; return the node's score.

        Appends each taken grow variable to ``parents`` and, on reaching the leaf, removes the
        shrink variables, so ``parents`` ends as the traced parent set. Mutates ``available`` to
        drop every branch variable it passes.
        """
        if self.branches is None:
            self._grow(available, parents)
        branches = self.branches or []
        for branch in branches:
            add = branch.add
            assert add is not None  # branch nodes always carry the variable they add
            available.remove(add)
            if add in prefix_set:
                parents.append(add)
                return branch.trace(available, parents, prefix_set)
        if self.remove is None:
            self._shrink(parents)
            return self.shrink_score
        for remove in self.remove:
            parents.remove(remove)
        return self.shrink_score


class _GrowShrinkTree:
    """A variable's grow-shrink tree: caches the best parent set per allowed prefix.

    :meth:`trace` returns ``vertex``'s best-scoring parent set drawn from ``prefix`` (writing it
    into the caller's list) and its negated local score. Repeated traces reuse the grow/shrink work
    cached in the shared node tree.
    """

    __slots__ = ("vertex", "score", "delta", "n_features", "empty_score", "root")

    def __init__(self, vertex: int, score: BICScore, delta: DeltaBICScore, n_features: int) -> None:
        self.vertex = vertex
        self.score = score
        self.delta = delta
        self.n_features = n_features
        self.empty_score = -score.score(vertex, [])
        self.root = _GSTNode(self)

    def _available(self) -> list[int]:
        """Every variable that may parent ``vertex`` -- all others."""
        vertex = self.vertex
        return [i for i in range(self.n_features) if i != vertex]

    def trace(self, prefix: Sequence[int], parents: list[int] | None = None) -> float:
        """Return ``vertex``'s best score for parents drawn from ``prefix``; fill ``parents``."""
        if parents is None:
            parents = []
        return self.root.trace(self._available(), parents, set(prefix))


def _better_mutation(
    v: int, order: list[int], gsts: list[_GrowShrinkTree], positions: dict[int, int]
) -> bool:
    """Move ``v`` to the order position that most improves the total score; report whether it moved.

    Scores every insertion point for ``v`` by sweeping the prefix scores forward with ``v`` removed
    and backward with ``v`` reinserted, so each candidate position costs one GST trace per variable.
    Only positions that improve the total by more than ``_SCORE_EPS`` are eligible. Selects the
    leftmost eligible position within ``_SCORE_EPS`` of the highest score. Refreshes ``positions``
    after a move; returns ``False`` when no position improves by more than the tolerance.
    """
    i = positions[v]
    p = len(order)
    scores = [0.0] * (p + 1)

    prefix: list[int] = []
    acc = 0.0
    for j in range(0, i):
        w = order[j]
        scores[j] = gsts[v].trace(prefix) + acc
        acc += gsts[w].trace(prefix)
        prefix.append(w)
    scores[i] = gsts[v].trace(prefix) + acc
    for j in range(i + 1, p):
        w = order[j]
        scores[j] = gsts[v].trace(prefix) + acc
        acc += gsts[w].trace(prefix)
        prefix.append(w)
    scores[p] = gsts[v].trace(prefix) + acc

    prefix.append(v)
    acc = 0.0
    for j in range(p - 1, i, -1):
        w = order[j]
        prefix.pop(j - 1)
        acc += gsts[w].trace(prefix)
        scores[j] += acc
    scores[i] += acc
    for j in range(i - 1, -1, -1):
        w = order[j]
        prefix.pop(j)
        acc += gsts[w].trace(prefix)
        scores[j] += acc

    best_score = max(scores)
    if best_score <= scores[i] + _SCORE_EPS:
        return False
    best = next(
        j
        for j in range(p + 1)
        if best_score - scores[j] <= _SCORE_EPS and scores[j] > scores[i] + _SCORE_EPS
    )
    insert_at = best - (1 if best > i else 0)
    order.insert(insert_at, order.pop(i))
    positions.clear()
    positions.update((node, idx) for idx, node in enumerate(order))
    return True


def boss(
    data: np.ndarray, *, lambda_value: float = 1.0, random_state: int = 0
) -> tuple[GraphStructure, float]:
    """Learn a CPDAG from ``data`` by best-order score search and return it with its total BIC.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.
    random_state : int
        Seed for the sweep-order RNG.

    Returns
    -------
    GraphStructure
        The learned CPDAG (``kind="cpdag"``).
    float
        The total negated BIC of the learned order's DAG -- the sum of each variable's local score
        given its traced parents (lower is better, matching :meth:`BICScore.score`).

    Notes
    -----
    See :func:`andrey.boss` for the score tolerance.
    """
    X = np.asarray(data, dtype=np.float64)
    p = X.shape[1]
    if p == 0:
        return to_structure(np.zeros((0, 0), dtype=np.int8), kind="cpdag"), 0.0

    score = BICScore(X, lambda_value=lambda_value)
    delta = DeltaBICScore(score)
    rng = random.Random(random_state)

    order = list(range(p))
    gsts = [_GrowShrinkTree(v, score, delta, p) for v in order]
    positions = {v: idx for idx, v in enumerate(order)}
    variables = list(order)

    while True:
        improved = False
        rng.shuffle(variables)
        for v in variables:
            improved |= _better_mutation(v, order, gsts, positions)
        if not improved:
            break

    parents: dict[int, list[int]] = {}
    for i, v in enumerate(order):
        pa: list[int] = []
        gsts[v].trace(order[:i], pa)
        parents[v] = pa

    adj = np.zeros((p, p), dtype=np.int8)
    total = 0.0
    for child in range(p):
        pa = parents[child]
        total += score.score(child, pa)
        for parent in pa:
            adj[parent, child] = TAIL
            adj[child, parent] = ARROW

    cpdag = dag2cpdag(adj)
    return to_structure(cpdag, kind="cpdag"), float(total)
