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
from collections.abc import Callable, Sequence

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
_EPS = float(np.finfo(np.float64).eps)


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


def _near_threshold(
    values: list[float], bounds: list[float], score: float, bound: float
) -> tuple[list[int], list[int]]:
    """Split the candidates that ``-value > score + _SCORE_EPS`` keeps from the ones it could flip.

    ``values`` are local scores (lower is better) within ``bounds`` of their full scores, and
    ``score`` is within ``bound`` of its own. Returns ``(kept, near)``: ``kept`` passes the test
    for every value the bounds allow, and ``near`` could fall on either side. A rounding term of a
    few units in the last place covers the sum and the difference the test computes.
    """
    threshold = score + _SCORE_EPS
    reach = max(bounds, default=0.0) + bound
    if reach == 0.0:
        return [k for k, value in enumerate(values) if -value > threshold], []
    rounding = 8 * _EPS * abs(threshold)
    low = threshold - reach - rounding
    kept: list[int] = []
    near: list[int] = []
    for k in [k for k, value in enumerate(values) if -value > low]:
        margin = -values[k] - threshold
        slack = bounds[k] + bound
        if slack > 0.0 and abs(margin) <= slack + rounding:
            near.append(k)
        elif margin > 0.0:
            kept.append(k)
    return kept, near


def _settle_groups(
    values: list[float],
    bounds: list[float],
    rescore: Callable[[int], float],
    tighten: Callable[[int], float],
    *,
    first_only: bool = False,
) -> None:
    """Tighten, then rescore, the candidates whose ``_SCORE_EPS`` tie group full scores could
    change.

    The candidates sort by value (lower is better) into groups, each holding the candidates within
    ``_SCORE_EPS`` of its best. Full scores give the same groups when, in each group, every
    member's highest possible value is within the tolerance of every other member's lowest possible
    value, and every later candidate's lowest possible value is beyond it from the highest possible
    best. The candidates of a group that could break this, and the later ones that could join it,
    get their bounds tightened by ``tighten`` first, and are scored in full by ``rescore`` (bound
    0) if that is not enough; the check repeats after each step. ``first_only`` checks only the
    first group. Updates ``values`` and ``bounds`` in place.
    """
    tightened: set[int] = set()
    while True:
        order = sorted(range(len(values)), key=values.__getitem__)
        slack = [bounds[k] + 8 * _EPS * abs(values[k]) if bounds[k] > 0.0 else 0.0 for k in order]
        lo = [values[k] - e for k, e in zip(order, slack)]
        hi = [values[k] + e for k, e in zip(order, slack)]
        later_lo = lo[:]
        for t in range(len(order) - 2, -1, -1):
            later_lo[t] = min(later_lo[t], later_lo[t + 1])
        flagged: set[int] = set()
        start = 0
        while start < len(order):
            best = values[order[start]]
            end = start + 1
            while end < len(order) and values[order[end]] - best <= _SCORE_EPS:
                end += 1
            group_hi = min(hi[start:end])
            for t in range(start, end):
                if any(hi[t] - lo[s] > _SCORE_EPS for s in range(start, end) if s != t):
                    flagged.update(order[start:end])
                    break
            if end < len(order) and later_lo[end] - group_hi <= _SCORE_EPS:
                flagged.update(order[start:end])
                flagged.update(
                    order[t] for t in range(end, len(order)) if lo[t] - group_hi <= _SCORE_EPS
                )
            if first_only:
                break
            start = end
        flagged = {k for k in flagged if bounds[k] > 0.0}
        if not flagged:
            return
        if not flagged <= tightened:
            for k in flagged - tightened:
                bounds[k] = tighten(k)
            tightened |= flagged
            continue
        for k in flagged:
            values[k] = rescore(k)
            bounds[k] = 0.0


def _apart(candidates: list[tuple[float, int, float]]) -> bool:
    """Whether each of the sorted ``(value, variable, bound)`` candidates is a tie group of its own
    under every value the bounds allow: every gap between neighbors passes the tolerance by more
    than twice the largest bound. All-zero bounds also pass: the values are full scores."""
    reach = max(candidate[2] for candidate in candidates)
    if reach == 0.0:
        return True
    reach = 2 * reach + 16 * _EPS * max(abs(candidates[0][0]), abs(candidates[-1][0])) + _SCORE_EPS
    return all(high[0] - low[0] > reach for low, high in zip(candidates, candidates[1:]))


def _settle_kept(
    kept: list[int],
    values: list[float],
    bounds: list[float],
    rescore: Callable[[int], float],
    tighten: Callable[[int], float],
    *,
    first_only: bool = False,
) -> None:
    """Run :func:`_settle_groups` over the ``kept`` candidates of ``values`` and ``bounds``."""
    sub_values = [values[k] for k in kept]
    sub_bounds = [bounds[k] for k in kept]
    _settle_groups(
        sub_values,
        sub_bounds,
        lambda j: rescore(kept[j]),
        lambda j: tighten(kept[j]),
        first_only=first_only,
    )
    for j, k in enumerate(kept):
        values[k] = sub_values[j]
        bounds[k] = sub_bounds[j]


def _tightened(
    tree: _GrowShrinkTree, base: Sequence[int], op: str, u: int | None, bound: float
) -> float:
    """``bound``, or the componentwise bound of the batched move ``(base, op, u)`` that gave the
    score (:meth:`~andrey.core.score_delta.DeltaBICScore.move_bound`) when it is smaller. No
    score is computed; a full score (bound 0) stays as it is."""
    if bound == 0.0 or u is None:
        return bound
    return min(bound, tree.delta.move_bound(tree.vertex, base, op, u))


class _Leaf:
    """A parent set a grow-shrink tree settles on, with its negated local score.

    Every node whose grow path and shrink end at the same parent set shares one leaf, so equal
    parent sets carry equal scores wherever the search compares them. ``bound`` bounds the score's
    distance from the full local score: 0 for a full score, the batch's error bound otherwise.
    ``move`` is the batched move ``(base, op, u)`` that gave the score, until :meth:`tighten` has
    used it.
    """

    __slots__ = ("tree", "parents", "score", "bound", "move")

    def __init__(
        self,
        tree: _GrowShrinkTree,
        parents: tuple[int, ...],
        score: float,
        bound: float,
        move: tuple[list[int], str, int | None] | None = None,
    ) -> None:
        self.tree = tree
        self.parents = parents
        self.score = score
        self.bound = bound
        self.move = move if bound > 0.0 else None

    def tighten(self) -> None:
        """Tighten ``bound`` to the componentwise bound of ``move`` (:func:`_tightened`)."""
        if self.move is not None:
            base, op, u = self.move
            self.bound = _tightened(self.tree, base, op, u, self.bound)
            self.move = None

    def settle(self) -> None:
        """Make ``score`` the full local score of ``parents``."""
        if self.bound > 0.0:
            self.score = -self.tree.score.score(self.tree.vertex, self.parents)
            self.bound = 0.0
            self.move = None


class _GSTNode:
    """One node of a grow-shrink tree: a parent set reached along a grow path from the root.

    ``grow_score`` is the negated local score of the node's parent set; ``branches`` are the
    single-variable additions that improve it by more than ``_SCORE_EPS`` (best tie group first,
    ascending variable index within a group; built lazily by :meth:`_grow`). ``remove`` is the
    shrink sequence -- the variables dropped, best first -- built lazily by :meth:`_shrink`, with
    ``shrink_score`` the negated score after all removals and ``leaf`` the :class:`_Leaf` of the
    parent set it ends at. ``grow_bound`` and ``shrink_bound`` bound each score's distance from the
    full local score: 0 for a full score, the Schur update's error bound for a batched one.
    """

    __slots__ = (
        "tree",
        "add",
        "grow_score",
        "shrink_score",
        "grow_bound",
        "shrink_bound",
        "branches",
        "remove",
        "leaf",
    )

    def __init__(
        self,
        tree: _GrowShrinkTree,
        add: int | None = None,
        score: float | None = None,
        bound: float = 0.0,
    ):
        if score is None:
            score = tree.empty_score
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.grow_bound = bound
        self.shrink_bound = bound
        self.branches: list[_GSTNode] | None = None
        self.remove: list[int] | None = None
        self.leaf: _Leaf | None = None

    def _grow(self, available: list[int], parents: list[int]) -> None:
        """Keep additions that improve ``grow_score`` by more than ``_SCORE_EPS``.

        Group improving additions within ``_SCORE_EPS`` of each group's best score. Visit groups
        best first, with ascending variable index within each group. A comparison that the score
        bounds leave open is decided by tighter bounds, or by full scores of the parent set and
        of the candidates.
        """
        self.branches = []
        tree = self.tree
        vertex = tree.vertex
        base = list(parents)
        deltas = [("add", int(u)) for u in available]
        bounds: list[float] = []
        values = tree.delta.score_many_with_base(vertex, base, deltas, bounds=bounds)

        def rescore(k: int) -> float:
            return tree.score.score(vertex, [*base, available[k]])

        def tighten(k: int) -> float:
            return _tightened(tree, base, "add", available[k], bounds[k])

        kept, near = _near_threshold(values, bounds, self.grow_score, self.grow_bound)
        if near:
            # Tighten the bounds the comparisons depend on before scoring anything in full: the
            # candidates' and the node's own, from the move of its parent's batch.
            for k in near:
                bounds[k] = tighten(k)
            own_base = [p for p in base if p != self.add]
            self.grow_bound = _tightened(tree, own_base, "add", self.add, self.grow_bound)
            self.shrink_bound = self.grow_bound
            kept, near = _near_threshold(values, bounds, self.grow_score, self.grow_bound)
        if near:
            if self.grow_bound > 0.0:
                # Grow precedes shrink, so both scores of the node become the full score.
                self.grow_score = self.shrink_score = -tree.score.score(vertex, base)
                self.grow_bound = self.shrink_bound = 0.0
                kept, near = _near_threshold(values, bounds, self.grow_score, 0.0)
            for k in near:
                values[k] = rescore(k)
                bounds[k] = 0.0
                if -values[k] > self.grow_score + _SCORE_EPS:
                    kept.append(k)
        candidates = sorted((values[k], available[k], bounds[k]) for k in kept)
        if len(candidates) > 1 and not _apart(candidates):
            _settle_kept(kept, values, bounds, rescore, tighten)
            candidates = sorted((values[k], available[k], bounds[k]) for k in kept)
        # Anchor each tie group to its best score: pairwise approximate comparisons are not
        # transitive. `trace` takes the first allowed branch, so each group uses variable order.
        start = 0
        while start < len(candidates):
            end = start + 1
            while end < len(candidates) and candidates[end][0] - candidates[start][0] <= _SCORE_EPS:
                end += 1
            for value, add, bound in sorted(candidates[start:end], key=lambda c: c[1]):
                self.branches.append(_GSTNode(self.tree, add, -value, bound))
            start = end

    def _shrink(self, parents: list[int], required: Sequence[int] = ()) -> None:
        """Remove parents while the score improves by more than ``_SCORE_EPS``.

        Each step selects the lowest-index parent among improving removals within ``_SCORE_EPS``
        of the best score. Record the removal order, cache the final score in ``shrink_score``
        and the final parent set's :class:`_Leaf` in ``leaf``. A ``required`` parent is never
        removed. A comparison that the score bounds leave open is decided by tighter bounds, or
        by full scores.
        """
        self.remove = []
        tree = self.tree
        vertex = tree.vertex
        # The batched move that gave ``shrink_score``: the node's own add until a removal, and
        # whether it can still tighten the bound.
        removal: tuple[list[int], int] | None = None
        tightenable = True

        def own_move() -> tuple[list[int], str, int | None]:
            if removal is not None:
                return removal[0], "remove", removal[1]
            return [p for p in parents if p != self.add], "add", self.add

        while True:
            current = list(parents)
            removable = [u for u in current if u not in required] if required else current
            if not removable:
                break
            deltas = [("remove", int(u)) for u in removable]
            bounds: list[float] = []
            values = tree.delta.score_many_with_base(vertex, current, deltas, bounds=bounds)

            def rescore(k: int, current: list[int] = current, removable: list[int] = removable):
                return tree.score.score(vertex, [u for u in current if u != removable[k]])

            def tighten(k: int, current: list[int] = current, removable: list[int] = removable):
                return _tightened(tree, current, "remove", removable[k], bounds[k])

            kept, near = _near_threshold(values, bounds, self.shrink_score, self.shrink_bound)
            if near:
                for k in near:
                    bounds[k] = tighten(k)
                if tightenable:
                    self.shrink_bound = _tightened(tree, *own_move(), self.shrink_bound)
                    tightenable = False
                kept, near = _near_threshold(values, bounds, self.shrink_score, self.shrink_bound)
            if near and self.shrink_bound > 0.0:
                self.shrink_score = -tree.score.score(vertex, current)
                self.shrink_bound = 0.0
                kept, near = _near_threshold(values, bounds, self.shrink_score, 0.0)
            for k in near:
                values[k] = rescore(k)
                bounds[k] = 0.0
                if -values[k] > self.shrink_score + _SCORE_EPS:
                    kept.append(k)
            if not kept:
                break
            if len(kept) > 1 and not _apart(sorted((values[k], k, bounds[k]) for k in kept)):
                _settle_kept(kept, values, bounds, rescore, tighten, first_only=True)
            best_value = min(values[k] for k in kept)
            best, k = min((removable[k], k) for k in kept if values[k] - best_value <= _SCORE_EPS)
            self.shrink_score = -values[k]
            self.shrink_bound = bounds[k]
            removal, tightenable = (current, best), True
            self.remove.append(best)
            parents.remove(best)
        key = tuple(sorted(parents))
        leaf = tree.leaves.get(key)
        if leaf is None:
            move = own_move() if tightenable and self.shrink_bound > 0.0 else None
            leaf = _Leaf(tree, key, self.shrink_score, self.shrink_bound, move)
            tree.leaves[key] = leaf
        self.leaf = leaf

    def trace(self, excluded: set[int], parents: list[int], prefix_set: set[int]) -> _Leaf:
        """Walk the grow path allowed by ``prefix_set``, then shrink; return the :class:`_Leaf`.

        Appends each taken grow variable to ``parents`` and, on reaching the leaf, removes the
        shrink variables, so ``parents`` ends as the traced parent set, the leaf's ``parents``.
        Adds every branch variable it passes to ``excluded``; a node grows from the tree's
        ``available`` variables outside it.
        """
        if self.branches is None:
            self._grow([u for u in self.tree.available if u not in excluded], parents)
        branches = self.branches or []
        for branch in branches:
            add = branch.add
            assert add is not None  # branch nodes always carry the variable they add
            excluded.add(add)
            if add in prefix_set:
                parents.append(add)
                return branch.trace(excluded, parents, prefix_set)
        if self.remove is None:
            self._shrink(parents, self.tree.required)
        else:
            for remove in self.remove:
                parents.remove(remove)
        assert self.leaf is not None  # set by the first shrink
        return self.leaf


class _GrowShrinkTree:
    """A variable's grow-shrink tree: caches the best parent set per allowed prefix.

    :meth:`trace` returns the :class:`_Leaf` of ``vertex``'s best-scoring parent set drawn from
    ``prefix`` (writing the set into the caller's list), one per distinct parent set (``leaves``).
    Repeated traces reuse the grow/shrink work cached in the shared node tree.

    ``candidates`` limits the parents the tree grows from (``None``: every other variable), and
    ``required`` parents are in every parent set: the root scores them, grow adds to them, and
    shrink never removes them. The caller keeps each required parent in ``prefix``.
    """

    __slots__ = (
        "vertex",
        "score",
        "delta",
        "empty_score",
        "root",
        "available",
        "required",
        "leaves",
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
        # Every variable that may parent `vertex` -- all others, or the `candidates`.
        self.available = tuple(
            [i for i in range(n_features) if i != vertex] if candidates is None else candidates
        )
        self.required = [] if required is None else required
        self.empty_score = -score.score(vertex, self.required)
        self.root = _GSTNode(self)
        self.leaves: dict[tuple[int, ...], _Leaf] = {}

    def trace(self, prefix: set[int], parents: list[int] | None = None) -> _Leaf:
        """Return the leaf of ``vertex``'s best parents drawn from ``prefix``; fill ``parents``.

        ``prefix`` is the set of variables allowed as parents; the trace only reads it.
        """
        if parents is None:
            parents = []
        parents.extend(self.required)
        return self.root.trace(set(), parents, prefix)


def _sweep(
    v: int, i: int, order: list[int], gsts: list[_GrowShrinkTree]
) -> tuple[list[float], list[_Leaf], list[_Leaf | None], list[_Leaf | None], float, float]:
    """Score every insertion point of ``v``, keeping the leaves each score adds up.

    Sweeps the prefix scores forward with ``v`` removed and backward with ``v`` reinserted, so each
    candidate position costs one GST trace per variable; ``scores[j]`` places ``v`` just before
    ``order[j]``. Returns ``(scores, v_leaves, without, with_v, bound_sum, abs_sum)``:
    ``v_leaves[j]`` is ``v``'s leaf at position ``j``, ``without[m]`` and ``with_v[m]`` the leaves
    of ``order[m]`` with ``v`` after it and before it (``None`` at ``v``'s index ``i``), and the
    sums add every traced leaf's score bound and absolute score.
    """
    p = len(order)
    scores = [0.0] * (p + 1)
    v_leaves: list[_Leaf] = []
    without: list[_Leaf | None] = [None] * p
    with_v: list[_Leaf | None] = [None] * p
    bound_sum = abs_sum = 0.0

    prefix: set[int] = set()
    acc = 0.0
    for j in range(0, i):
        w = order[j]
        leaf = gsts[v].trace(prefix)
        v_leaves.append(leaf)
        scores[j] = leaf.score + acc
        node = without[j] = gsts[w].trace(prefix)
        acc += node.score
        bound_sum += leaf.bound + node.bound
        abs_sum += abs(leaf.score) + abs(node.score)
        prefix.add(w)
    leaf = gsts[v].trace(prefix)
    v_leaves.append(leaf)
    scores[i] = leaf.score + acc
    bound_sum += leaf.bound
    abs_sum += abs(leaf.score)
    for j in range(i + 1, p):
        w = order[j]
        leaf = gsts[v].trace(prefix)
        v_leaves.append(leaf)
        scores[j] = leaf.score + acc
        node = without[j] = gsts[w].trace(prefix)
        acc += node.score
        bound_sum += leaf.bound + node.bound
        abs_sum += abs(leaf.score) + abs(node.score)
        prefix.add(w)
    leaf = gsts[v].trace(prefix)
    v_leaves.append(leaf)
    scores[p] = leaf.score + acc
    bound_sum += leaf.bound
    abs_sum += abs(leaf.score)

    prefix.add(v)
    acc = 0.0
    for j in range(p - 1, i, -1):
        w = order[j]
        prefix.remove(w)
        node = with_v[j] = gsts[w].trace(prefix)
        acc += node.score
        bound_sum += node.bound
        abs_sum += abs(node.score)
        scores[j] += acc
    scores[i] += acc
    for j in range(i - 1, -1, -1):
        w = order[j]
        prefix.remove(w)
        node = with_v[j] = gsts[w].trace(prefix)
        acc += node.score
        bound_sum += node.bound
        abs_sum += abs(node.score)
        scores[j] += acc
    return scores, v_leaves, without, with_v, bound_sum, abs_sum


def _decided(scores: list[float], i: int, reach: float) -> bool:
    """Whether every comparison :func:`_better_mutation` makes stays on its side when each score
    moves by up to ``reach``: a quick check before the per-pair bounds of :func:`_unsettled`.

    Holds when every position is beyond ``reach`` of the current score plus the tolerance, at most
    one eligible position is within the tolerance of the best, and every other eligible position is
    beyond ``reach`` of that.
    """
    threshold = scores[i] + _SCORE_EPS
    if min(abs(score - threshold) for score in scores) <= reach:
        return False
    best_score = max(scores)
    if best_score <= threshold:
        return True
    near_best = 0
    for score in scores:
        if score > threshold:
            gap = best_score - score - _SCORE_EPS
            if gap <= 0.0:
                near_best += 1
            if abs(gap) <= reach or near_best > 1:
                return False
    return True


def _unsettled(
    scores: list[float],
    i: int,
    v_leaves: list[_Leaf],
    without: list[_Leaf | None],
    with_v: list[_Leaf | None],
) -> list[_Leaf]:
    """The leaves to score in full before :func:`_better_mutation` decides, or ``[]``.

    Two positions' scores share every leaf but ``v``'s own and those of the variables between the
    positions whose leaf changes with ``v`` before them, so their difference is within the bounds
    of those leaves of the difference full scores give, plus the rounding of the four sums
    (``2 (p + 2) eps`` times the absolute scores a position adds up). A comparison of the decision
    -- each position against the current one, and the positions within ``_SCORE_EPS`` of the best
    against each other and against the other eligible ones -- that this bound leaves open returns
    the bounded leaves it depends on, or every bounded leaf when only the rounding leaves it open.
    """
    p = len(without)
    changed = [0.0] * (p + 1)  # prefix sums of the bounds of leaves that change with v
    total = 0.0
    magnitude = max(abs(leaf.score) for leaf in v_leaves)
    for m in range(p):
        a, b = without[m], with_v[m]
        if a is not None and b is not None:
            magnitude += max(abs(a.score), abs(b.score))
            if a is not b:
                total += a.bound + b.bound
        changed[m + 1] = total
    rounding = 2 * (p + 2) * _EPS * magnitude

    def gap_bound(j: int, k: int) -> float:
        if j > k:
            j, k = k, j
        bound = changed[k] - changed[j] + rounding
        if v_leaves[j] is not v_leaves[k]:
            bound += v_leaves[j].bound + v_leaves[k].bound
        return bound

    open_pairs: list[tuple[int, int]] = []
    threshold = scores[i] + _SCORE_EPS
    for j, score in enumerate(scores):
        if score != -np.inf and abs(score - threshold) <= gap_bound(j, i):
            open_pairs.append((j, i))
    eligible = [j for j, score in enumerate(scores) if score > threshold]
    if eligible:
        best_score = max(scores)
        anchor = scores.index(best_score)
        members = [j for j in eligible if best_score - scores[j] <= _SCORE_EPS]
        for a in members:
            for m in members:
                if a != m and abs(scores[a] - scores[m] - _SCORE_EPS) <= gap_bound(a, m):
                    open_pairs.append((a, m))
        for r in eligible:
            if r not in members and abs(scores[anchor] - scores[r] - _SCORE_EPS) <= gap_bound(
                anchor, r
            ):
                open_pairs.append((anchor, r))
    if not open_pairs:
        return []
    nodes: list[_Leaf | None] = []
    for j, k in open_pairs:
        j, k = min(j, k), max(j, k)
        nodes += (v_leaves[j], v_leaves[k])
        for m in range(j, k):
            if without[m] is not with_v[m]:
                nodes += (without[m], with_v[m])
    unsettled = {id(node): node for node in nodes if node is not None and node.bound > 0}
    if not unsettled:  # only the rounding leaves a comparison open
        leaves = [*v_leaves, *without, *with_v]
        unsettled = {id(node): node for node in leaves if node is not None and node.bound > 0}
    return list(unsettled.values())


def _better_mutation(
    v: int,
    order: list[int],
    gsts: list[_GrowShrinkTree],
    positions: dict[int, int],
    before: np.ndarray | None = None,
) -> bool:
    """Move ``v`` to the order position that most improves the total score; report whether it moved.

    Scores every insertion point for ``v`` (:func:`_sweep`). Only positions that improve the total
    by more than ``_SCORE_EPS`` are eligible. Selects the leftmost eligible position within
    ``_SCORE_EPS`` of the highest score. Refreshes ``positions`` after a move; returns ``False``
    when no position improves by more than the tolerance. With ``before`` (``before[a, b]``: ``a``
    precedes ``b``), only positions after every predecessor of ``v`` and before every successor are
    eligible. When the leaves' score bounds leave a comparison open (:func:`_unsettled`), those
    leaves' bounds are tightened (:meth:`_Leaf.tighten`), then, if that is not enough, the leaves
    are scored in full, and the positions scored again, so the move is the one full local scores
    choose.
    """
    i = positions[v]
    p = len(order)
    lo, hi = 0, p
    if before is not None:  # scores[j] places v just before order[j]
        lo = max((positions[w] + 1 for w in np.flatnonzero(before[:, v]).tolist()), default=0)
        hi = min((positions[w] for w in np.flatnonzero(before[v]).tolist()), default=p)
    while True:
        scores, v_leaves, without, with_v, bound_sum, abs_sum = _sweep(v, i, order, gsts)
        if before is not None:
            scores = [s if lo <= j <= hi else -np.inf for j, s in enumerate(scores)]
        if bound_sum == 0.0 or _decided(scores, i, bound_sum + 2 * (p + 2) * _EPS * abs_sum):
            break
        unsettled = _unsettled(scores, i, v_leaves, without, with_v)
        if not unsettled:
            break
        if any(leaf.move is not None for leaf in unsettled):
            for leaf in unsettled:
                leaf.tighten()
        else:
            for leaf in unsettled:
                leaf.settle()

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
    prefix: set[int] = set()
    for v in order:
        pa: list[int] = []
        gsts[v].trace(prefix, pa)
        parents[v] = pa
        prefix.add(v)

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
