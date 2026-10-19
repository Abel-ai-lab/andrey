"""Best-Order Score Search (BOSS) over a linear-Gaussian BIC, a categorical, or a mixed-data score.

Each variable owns a grow-shrink tree; sweeps slide variables to their best order position
(:func:`_better_mutation`) until no move improves the total. Scores are negated local scores, so
the search maximizes; BIC moves take the Schur update, the categorical and mixed-data scores two
full scores. The search runs on the columns sorted by tie rank (:func:`ranked_score`), so a
variable's index is its tie rank. The algorithm and its reference are described on
:func:`andrey.boss`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import DISCRETE_SCORES, MIXED_SCORES, BICScore, LocalScore, make_score
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta
from andrey.search.hc import tie_ranks

#: The local scores :func:`boss` and :func:`~andrey.search.grasp.grasp` accept.
PERMUTATION_SCORES: tuple[str, ...] = (
    "local_score_BIC_from_cov",
    "local_score_BDeu",
    "local_score_BIC_categorical",
    "local_score_degen_gauss",
    "local_score_cond_gauss",
)

# Fixed absolute score tolerance for numerical ties.
_SCORE_EPS = 1e-6


def ranked_score(
    x: np.ndarray, score_func: str, lambda_value: float, categorical: npt.ArrayLike | None
) -> tuple[LocalScore, np.ndarray]:
    """The local score over the columns of ``x`` sorted by :func:`~andrey.search.hc.tie_ranks`.

    A permutation search over this score makes every choice that goes by variable index by each
    column's data instead of its position: its starting order, the orders its seed shuffles, and
    its ties between candidate parents and between order moves. The sorted data are C-contiguous
    whatever the layout of ``x``, so every column order of the same data gives the same score, bit
    for bit, and reordering the columns of ``x`` reorders the learned graph's nodes and changes
    nothing else, unless two columns are identical. Under BDeu or the multinomial BIC every column
    ranks by which rows share a category, and under a mixed-data score the categorical columns do,
    with the mask sorted with them; the linear-Gaussian BIC ranks every column as continuous.

    Returns the score and ``ranks``: column ``i`` of ``x`` is variable ``ranks[i]`` of the score,
    so a matrix ``m`` over the score's variables maps back to the columns of ``x`` as
    ``m[np.ix_(ranks, ranks)]``. A ``ValueError`` from the score, such as one naming linearly
    dependent columns, names the columns of ``x``.
    """
    mixed = score_func in MIXED_SCORES
    if mixed:
        mask = column_mask(categorical, x.shape[1])
    else:
        mask = np.full(x.shape[1], score_func in DISCRETE_SCORES)
    ranks = tie_ranks(x, mask)
    by_rank = np.argsort(ranks)
    try:
        score = make_score(
            score_func,
            np.ascontiguousarray(x[:, by_rank]),
            lambda_value=lambda_value,
            categorical=mask[by_rank] if mixed else None,
        )
    except ValueError:
        # The error names positions in the sorted data; the same check on ``x`` names its columns.
        make_score(score_func, x, lambda_value=lambda_value, categorical=categorical)
        raise
    return score, ranks


def precedence_sort(order: Sequence[int], before: np.ndarray) -> list[int]:
    """``order`` rearranged so that ``a`` precedes ``b`` wherever ``before[a, b]`` (acyclic).

    Each step places the first variable of ``order`` whose predecessors are all placed, so an order
    that already keeps ``before`` is returned unchanged.
    """
    pending = list(order)
    placed: list[int] = []
    while pending:
        k = next(k for k, v in enumerate(pending) if not before[pending, v].any())
        placed.append(pending.pop(k))
    return placed


def parent_knowledge(constraints: Constraints, v: int) -> tuple[list[int], list[int]]:
    """Variable ``v``'s candidate parents (neither forbidden nor required) and required parents."""
    forbidden, required = constraints.forbidden[:, v], constraints.required[:, v]
    candidates = np.flatnonzero(~forbidden & ~required).tolist()
    return [u for u in candidates if u != v], np.flatnonzero(required).tolist()


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

    def _shrink(self, parents: list[int], required: Sequence[int] = ()) -> None:
        """Remove parents while the score improves by more than ``_SCORE_EPS``.

        Each step selects the lowest-index parent among improving removals within ``_SCORE_EPS``
        of the best score. Record the removal order and cache the final score in ``shrink_score``.
        A ``required`` parent is never removed.
        """
        self.remove = []
        vertex = self.tree.vertex
        while True:
            current = list(parents)
            removable = [u for u in current if u not in required] if required else current
            if not removable:
                break
            deltas = [("remove", int(u)) for u in removable]
            values = self.tree.delta.score_many_with_base(vertex, current, deltas)
            candidates = [
                (u, -float(value))
                for u, value in zip(removable, values)
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
            self._shrink(parents, self.tree.required)
            return self.shrink_score
        for remove in self.remove:
            parents.remove(remove)
        return self.shrink_score


class _GrowShrinkTree:
    """A variable's grow-shrink tree: caches the best parent set per allowed prefix.

    :meth:`trace` returns ``vertex``'s best-scoring parent set drawn from ``prefix`` (writing it
    into the caller's list) and its negated local score. Repeated traces reuse the grow/shrink work
    cached in the shared node tree.

    ``candidates`` limits the parents the tree grows from (``None``: every other variable), and
    ``required`` parents are in every parent set: the root scores them, grow adds to them, and
    shrink never removes them. The caller keeps each required parent in ``prefix``.
    """

    __slots__ = (
        "vertex",
        "score",
        "delta",
        "n_features",
        "empty_score",
        "root",
        "candidates",
        "required",
    )

    def __init__(
        self,
        vertex: int,
        score: LocalScore,
        delta: DeltaBICScore | FullScoreDelta,
        n_features: int,
        *,
        candidates: list[int] | None = None,
        required: list[int] | None = None,
    ) -> None:
        self.vertex = vertex
        self.score = score
        self.delta = delta
        self.n_features = n_features
        self.candidates = candidates
        self.required = [] if required is None else required
        self.empty_score = -score.score(vertex, self.required)
        self.root = _GSTNode(self)

    def _available(self) -> list[int]:
        """Every variable that may parent ``vertex`` -- all others, or the ``candidates``."""
        if self.candidates is not None:
            return list(self.candidates)
        vertex = self.vertex
        return [i for i in range(self.n_features) if i != vertex]

    def trace(self, prefix: Sequence[int], parents: list[int] | None = None) -> float:
        """Return ``vertex``'s best score for parents drawn from ``prefix``; fill ``parents``."""
        if parents is None:
            parents = []
        parents.extend(self.required)
        return self.root.trace(self._available(), parents, set(prefix))


def _better_mutation(
    v: int,
    order: list[int],
    gsts: list[_GrowShrinkTree],
    positions: dict[int, int],
    before: np.ndarray | None = None,
) -> bool:
    """Move ``v`` to the order position that most improves the total score; report whether it moved.

    Scores every insertion point for ``v`` by sweeping the prefix scores forward with ``v`` removed
    and backward with ``v`` reinserted, so each candidate position costs one GST trace per variable.
    Only positions that improve the total by more than ``_SCORE_EPS`` are eligible. Selects the
    leftmost eligible position within ``_SCORE_EPS`` of the highest score. Refreshes ``positions``
    after a move; returns ``False`` when no position improves by more than the tolerance. With
    ``before`` (``before[a, b]``: ``a`` precedes ``b``), only positions after every predecessor of
    ``v`` and before every successor are eligible.
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

    if before is not None:  # scores[j] places v just before order[j]
        lo = max((positions[w] + 1 for w in np.flatnonzero(before[:, v]).tolist()), default=0)
        hi = min((positions[w] for w in np.flatnonzero(before[v]).tolist()), default=p)
        scores = [s if lo <= j <= hi else -np.inf for j, s in enumerate(scores)]
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
    data: np.ndarray,
    *,
    lambda_value: float = 1.0,
    random_state: int = 0,
    score_func: str = "local_score_BIC_from_cov",
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float]:
    """Learn a CPDAG from ``data`` by best-order score search and return it with its total score.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    random_state : int
        Seed for the sweep-order RNG.
    score_func : str
        Local score, one of :data:`PERMUTATION_SCORES`: the linear-Gaussian BIC, BDeu or the
        multinomial BIC on category codes (:class:`~andrey.core.score.BDeuScore`,
        :class:`~andrey.core.score.CategoricalBICScore`), or a mixed-data BIC
        (:class:`~andrey.core.score.DGScore`, :class:`~andrey.core.score.CGScore`).
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; BDeu and the multinomial BIC read
        every column as categorical.
    constraints : Constraints or None
        Background knowledge: each variable's parents exclude the forbidden ones and include the
        required ones, and every order the search visits keeps
        :meth:`~andrey.core.knowledge.Constraints.precedence`, starting from the tie-rank order
        rearranged as little as possible. The CPDAG is then narrowed by the knowledge
        (:func:`~andrey.core.knowledge.mpdag`).

    Returns
    -------
    GraphStructure
        The learned CPDAG (``kind="cpdag"``).
    float
        The total score of the learned order's DAG -- the sum of each variable's local score
        given its traced parents (lower is better, matching :meth:`BICScore.score`).

    Notes
    -----
    The search runs on the columns sorted by :func:`ranked_score`, so the starting order, the sweep
    order, and every tie follow the columns' data, and reordering the columns of ``data`` reorders
    the CPDAG's nodes. See :func:`andrey.boss` for the score tolerance.
    """
    X = np.asarray(data, dtype=np.float64)
    p = X.shape[1]
    if p == 0:
        return to_structure(np.zeros((0, 0), dtype=np.int8), kind="cpdag"), 0.0

    if score_func not in PERMUTATION_SCORES:
        raise NotImplementedError(
            f"unsupported score_func {score_func!r}; use one of {PERMUTATION_SCORES}"
        )
    score, ranks = ranked_score(X, score_func, lambda_value, categorical)
    delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
    rng = random.Random(random_state)

    order = list(range(p))
    before = None
    if constraints is None:
        gsts = [_GrowShrinkTree(v, score, delta, p) for v in order]
    else:
        ranked = constraints.permuted(np.argsort(ranks))  # score variable k is column by_rank[k]
        before = ranked.precedence()
        gsts = []
        for v in order:
            candidates, required = parent_knowledge(ranked, v)
            gsts.append(
                _GrowShrinkTree(v, score, delta, p, candidates=candidates, required=required)
            )
        order = precedence_sort(order, before)
    positions = {v: idx for idx, v in enumerate(order)}
    variables = list(order)

    while True:
        improved = False
        rng.shuffle(variables)
        for v in variables:
            improved |= _better_mutation(v, order, gsts, positions, before)
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

    cpdag = dag2cpdag(adj[np.ix_(ranks, ranks)])
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    return to_structure(cpdag, kind="cpdag"), float(total)
