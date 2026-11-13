"""LOAD [1]_: the causal relation of two variables and the optimal adjustment set for the effect of
one on the other, from local structures learned one Markov blanket at a time.

:func:`local_optimal_adjustment` runs Algorithm 3 of [1]_ on a
:class:`~andrey.constraint.blanket.LocalSearch`. Step 1 (Algorithm 1) learns the edges at each of
the two variables with MB-by-MB as [1]_ runs it (:data:`~andrey.constraint.sequential.LOAD`, one
:class:`~andrey.constraint.sequential.Cache` for every run over the same search) and reads their
relation: explicit ancestry by Theorem 3 of [2]_ (Algorithm 8, :func:`explicit_ancestor`), possible
ancestry by Theorem 2 of [2]_ (Algorithm 9, :func:`possible_ancestor`). Step 2 tests amenability
from each sibling of the treatment (Lemma 4.3, Algorithm 2, :func:`amenable_at`). Step 3 finds the
possible descendants of the treatment by Algorithm 9 for every other variable. Step 4 runs
MB-by-MB at the outcome over the other variables, the treatment, and the outcome alone
(:class:`ColumnView`), and the outcome's parents there, the treatment left out, are the optimal
set (Lemma 4.4). Where the effect is not identifiable, the sets are the locally valid parent sets
of the treatment (Algorithm 10, :func:`local_valid_sets`).

References
----------
.. [1] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
   statistically efficient causal inference. Proceedings of the 29th International Conference on
   Artificial Intelligence and Statistics.
.. [2] Fang, Z., Liu, Y., Geng, Z., Zhu, S., and He, Y. (2022). A local method for identifying
   causal relations under Markov equivalence. Artificial Intelligence 305, 103669.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from andrey.constraint.blanket import LocalSearch
from andrey.constraint.pc import undirect_conflicts
from andrey.constraint.sequential import LOAD, Cache, Network, sequential
from andrey.core import ARROW, NULL, TAIL
from andrey.core.ci import pins_blas


@dataclass(frozen=True)
class Edges:
    """The parents, children, and siblings (undirected neighbors) of one variable in a grown
    graph, each in the order the variables joined it."""

    parents: list[int]
    children: list[int]
    siblings: list[int]


def edges_at(network: Network, v: int) -> Edges:
    """``v``'s parents, children, and siblings in ``network``."""
    parents, children, siblings = [], [], []
    for u in network.neighbors(v):
        near, far = network.mark(v, u), network.mark(u, v)
        if near == ARROW and far == TAIL:
            parents.append(u)
        elif near == TAIL and far == ARROW:
            children.append(u)
        else:
            siblings.append(u)
    return Edges(parents, children, siblings)


def _dependent(search: LocalSearch, a: int, b: int, cond: Iterable[int]) -> bool:
    """Whether the test finds ``a`` and ``b`` dependent given ``cond``: a p-value at or below
    ``alpha``."""
    return bool(search.scan(a, [b], sorted(set(cond)))[1][0] <= search.alpha)


def explicit_ancestor(search: LocalSearch, x: int, y: int, network: Network) -> bool:
    """Whether ``x`` is an explicit ancestor of ``y`` (a directed path in the CPDAG), from ``x``'s
    edges in ``network`` (Algorithm 8 of [1]_): a child is one, a parent or a sibling is not, and
    any other variable is one when the test finds it dependent on ``x`` given ``x``'s parents and
    siblings ([2]_ Theorem 3)."""
    at = edges_at(network, x)
    if y in at.children:
        return True
    if y in at.parents or y in at.siblings:
        return False
    return _dependent(search, x, y, at.parents + at.siblings)


def possible_ancestor(search: LocalSearch, x: int, y: int, network: Network) -> bool:
    """Whether ``x`` is a possible ancestor of ``y`` (a path in the CPDAG with no edge into ``x``'s
    side), from ``x``'s edges in ``network`` (Algorithm 9 of [1]_): a child or a sibling is one, a
    parent is not, and any other variable is one when the test finds it dependent on ``x`` given
    ``x``'s parents ([2]_ Theorem 2)."""
    at = edges_at(network, x)
    if y in at.children or y in at.siblings:
        return True
    if y in at.parents:
        return False
    return _dependent(search, x, y, at.parents)


def amenable_at(search: LocalSearch, x: int, y: int, v: int, network: Network) -> bool:
    """Whether the sibling ``v`` of the treatment ``x`` leaves the CPDAG amenable relative to
    ``(x, y)``, from ``v``'s edges in ``network`` (Algorithm 2 of [1]_): not when ``v`` is adjacent
    to the outcome ``y``, else when the test finds ``v`` independent of ``y`` given ``v``'s parents
    and ``x`` (Lemma 4.3)."""
    if network.mark(v, y) != NULL:
        return False
    return not _dependent(search, v, y, [*edges_at(network, v).parents, x])


def local_valid_sets(network: Network, x: int, y: int) -> list[list[int]]:
    """The locally valid parent sets of ``x`` in ``network`` (Algorithm 10 of [1]_, after
    Algorithm 3 of Maathuis et al. 2009): ``x``'s parents with each subset of its siblings other
    than ``y`` whose every member is adjacent to every other candidate parent, so that no new
    v-structure forms at ``x``. ``y`` is left out of a set that holds it. Each set ascending."""
    at = edges_at(network, x)
    siblings = [s for s in at.siblings if s != y]
    found = []
    for k in range(len(siblings) + 1):
        for new in combinations(siblings, k):
            candidates = {*at.parents, *new}
            if all(network.mark(s, c) != NULL for s in new for c in candidates if c != s):
                found.append(sorted(candidates - {y}))
    return found


class ColumnView:
    """A conditional-independence test read over a subset of its columns: column ``k`` of the view
    is column ``columns[k]`` of ``test``. Every answer is ``test``'s answer to the same query, so a
    view over a test that keeps its answers reads them, and the view of a test with batched or
    partial-correlation calls has them too.
    """

    statistic_pvalue: Callable[..., Any] | None = None
    statistic_pvalues: Callable[..., Any] | None = None
    pvalues_given: Callable[..., Any] | None = None
    batched_call: Callable[..., Any] | None = None
    _bounded_pvalue: Callable[..., Any] | None = None

    def __init__(self, test: Any, columns: Sequence[int]) -> None:
        self.test = test
        self.columns = np.asarray(columns, dtype=np.intp)
        self.pin_blas = pins_blas(test)
        c = self.columns

        def cols(s: Iterable[int]) -> list[int]:
            return [int(c[v]) for v in s]

        if (one := getattr(test, "statistic_pvalue", None)) is not None:
            self.statistic_pvalue = lambda x, y, s=(): one(int(c[x]), int(c[y]), cols(s))
        if (many := getattr(test, "statistic_pvalues", None)) is not None:
            self.statistic_pvalues = lambda xs, ys, sets: many(
                c[np.asarray(xs, dtype=np.intp)],
                c[np.asarray(ys, dtype=np.intp)],
                c[np.asarray(sets, dtype=np.intp)],
            )
        if (given := getattr(test, "pvalues_given", None)) is not None:
            self.pvalues_given = lambda t, xs, s: given(int(c[t]), cols(xs), cols(s))
        if (batched := getattr(test, "batched_call", None)) is not None:
            self.batched_call = lambda x, y, sets: batched(
                int(c[x]), int(c[y]), [cols(s) for s in sets]
            )
        if (bounded := getattr(test, "_bounded_pvalue", None)) is not None:
            self._bounded_pvalue = lambda x, y, s, **kw: bounded(
                int(c[x]), int(c[y]), cols(s), **kw
            )

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        c = self.columns
        return self.test(int(c[x]), int(c[y]), [int(c[v]) for v in condition_set])


def column_search(search: LocalSearch, columns: Sequence[int]) -> LocalSearch:
    """A :class:`~andrey.constraint.blanket.LocalSearch` over ``columns`` of ``search``'s test, the
    same level, cap, and ranks, so a choice between tied candidates goes as in ``search``."""
    levels = None if search.levels is None else search.levels[list(columns)]
    return LocalSearch(
        ColumnView(search.test, columns),
        [search.ranks[k] for k in columns],
        search.alpha,
        max_depth=search.max_depth,
        levels=levels,
        n=search.n,
        total_alpha=search.total_alpha,
    )


@dataclass
class Answer:
    """What the search says of the total effect of ``treatment`` on ``outcome``: the fields of an
    ``effects`` record (:func:`~andrey.core.effects.effect_record`), nodes by position."""

    treatment: int
    outcome: int
    nonzero: bool | None
    identifiable: bool
    adjustment_sets: list[list[int]] = field(default_factory=list)
    set_kind: str | None = None


@dataclass
class Result:
    """One run of :func:`local_optimal_adjustment`.

    ``answers`` holds one :class:`Answer` per ordered pair.
    ``marks`` holds the edges at the two variables as their own runs grew them, over every column.
    ``learned`` lists the variables whose blanket the search learned, in order, Step 4's included
    (over its columns, named by position in the data). ``possible_descendants`` holds Step 3's
    answer, ``None`` when the search stopped before it.
    """

    answers: list[Answer]
    marks: np.ndarray
    learned: list[int]
    possible_descendants: list[int] | None


def _zero(network: Network, a: int, b: int) -> Answer:
    """``a``'s effect on ``b`` when ``a`` is a definite non-ancestor of ``b``: zero, and identified
    by an adjustment set unless the two are adjacent, when the edge ``b -> a`` is a back-door path
    that no set blocks; then the sets are ``a``'s locally valid parent sets."""
    if network.mark(a, b) == NULL:
        return Answer(a, b, nonzero=False, identifiable=True)
    return Answer(a, b, False, False, local_valid_sets(network, a, b), "local_parents")


def _union(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The edges of both, an arrowhead kept at an end where either has one, an edge with
    arrowheads at both ends made undirected."""
    marks = np.where(first != NULL, first, second)
    both = (first != NULL) & (second != NULL)
    marks[both & ((first == ARROW) | (second == ARROW))] = ARROW
    return undirect_conflicts(marks)


def local_optimal_adjustment(
    search: LocalSearch, pair: tuple[int, int], blanket: str = "gs"
) -> Result:
    """Answer for both ordered pairs of the two variables ``pair`` (Algorithm 3 of [1]_).

    The variable whose rank comes first is ``X`` of [1]_, so the answer does not depend on the
    column order; the treatment's siblings in Step 2 go in rank order too. ``blanket`` names the
    Markov blanket learner of every MB-by-MB run, Grow-Shrink in [1]_.
    """
    x, y = search.ordered({v: 0.0 for v in pair})
    d = search.d
    cache = Cache()
    runs = {v: sequential(search, v, blanket, LOAD, cache) for v in (x, y)}
    learned = [v for r in runs.values() for v in r.learned]
    graph = {v: r.network for v, r in runs.items()}
    marks = _union(graph[x].star(x, d), graph[y].star(y, d))

    def result(answers: list[Answer], possde: list[int] | None = None) -> Result:
        return Result(answers, marks, learned, possde)

    # Step 1: the relation of the two variables (Algorithm 1).
    if explicit_ancestor(search, x, y, graph[x]):
        t, o = x, y
    elif explicit_ancestor(search, y, x, graph[y]):
        t, o = y, x
    else:
        answers = []
        for a, b in ((x, y), (y, x)):
            if possible_ancestor(search, a, b, graph[a]):
                sets = local_valid_sets(graph[a], a, b)
                answers.append(Answer(a, b, None, False, sets, "local_parents"))
            else:
                answers.append(_zero(graph[a], a, b))
        return result(answers)
    reverse = _zero(graph[o], o, t)

    # Step 2: amenability, from each sibling of the treatment (Algorithm 2).
    for v in search.ordered({s: 0.0 for s in edges_at(graph[t], t).siblings}):
        run = sequential(search, v, blanket, LOAD, cache)
        learned += run.learned
        if not amenable_at(search, t, o, v, run.network):
            sets = local_valid_sets(graph[t], t, o)
            return result([Answer(t, o, True, False, sets, "local_parents"), reverse])

    # Step 3: the possible descendants of the treatment (Algorithm 9 for every other variable).
    at = edges_at(graph[t], t)
    near = set(at.parents) | set(at.children) | set(at.siblings)
    others = [v for v in range(d) if v not in (t, o) and v not in near]
    possde = sorted((set(at.children) | set(at.siblings)) - {o})
    if others:
        pvalues = search.scan(t, others, sorted(at.parents))[1]
        possde = sorted([*possde, *(v for v, p in zip(others, pvalues) if p <= search.alpha)])

    # Step 4: the outcome's parents over the rest, the optimal set (Lemma 4.4).
    keep = [v for v in range(d) if v not in set(possde)]
    view = column_search(search, keep)
    step = sequential(view, keep.index(o), blanket, LOAD, Cache())
    learned += [keep[v] for v in step.learned]
    parents = edges_at(step.network, keep.index(o)).parents
    optimal = sorted(keep[v] for v in parents if keep[v] != t)
    return result([Answer(t, o, True, True, [optimal], "optimal"), reverse], possde)
