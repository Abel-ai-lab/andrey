"""The permutation search of IGSP and UT-IGSP over minimal I-MAPs, the engine behind
:func:`andrey.igsp` and :func:`andrey.ut_igsp`.

A DAG is the minimal I-MAP of a variable order: ``i -> j`` for ``i`` before ``j`` unless ``x_i``
and ``x_j`` are independent given every other variable before ``j``, by the conditional-independence
test on the observational rows (Solus, Wang, and Uhler 2021). The search moves between such DAGs by
reversing I-covered arrows, a depth-first search from a start DAG with restarts (Wang et al. 2017;
Yang, Katcoff, and Uhler 2018; Squires, Wang, and Uhler 2020):

1. The start is the minimal I-MAP of a minimum-degree order: the undirected graph of the pairs
   dependent given all other variables, eliminated a vertex of least degree at a time (its
   neighbors joined), the elimination order reversed.
2. A DAG's children are the reversals of its I-covered arrows, each updated to the minimal I-MAP of
   the swapped order by retesting the arrows into the two ends (:func:`reverse`), less the DAGs
   visited since the last restart and those the method does not admit.
3. When some child passes the method's restart test, the search restarts from the one with the
   lowest objective, the first in visiting order on a tie, with the visited set and the trace
   cleared. Otherwise it descends into the first child while the trace is shorter than ``depth``,
   and at a dead end returns to the parent and its next child not yet visited: the children of a
   DAG ``depth`` reversals from the last restart are built and tested, never descended into. A
   start ends when the root has no child left.
4. A search runs ``n_starts`` starts and returns the DAG with the lowest objective it scored, the
   first on a tie, so a later start replaces an earlier one only with a strictly lower objective.

Every choice that goes by variable index follows the engine's variable order: the caller sorts the
columns by :func:`~andrey.core.ties.tie_ranks` first. Start ``0`` breaks the minimum-degree ties by
the lowest index and visits children in the order of their arrows ``(i, j)``; start ``s >= 1``
draws both from :func:`~andrey.core.seeding.start_rng`.

A method plugs in through :class:`Rules`: which covered arrows are I-covered, the objective, which
children the search admits, the restart test, and the visiting priority. :func:`igsp_rules` gives
IGSP's and :func:`ut_igsp_rules` UT-IGSP's. Background knowledge
(:class:`~andrey.core.knowledge.Constraints`) holds every required arrow, never puts a forbidden
one in an I-MAP, eliminates a vertex only after every vertex that must come after it, and never
reverses an arrow it decides.

References
----------
Squires, C., Wang, Y., and Uhler, C. (2020). Permutation-based causal structure learning with
unknown intervention targets. UAI, PMLR 124, 1039-1048.

Wang, Y., Solus, L., Yang, K. D., and Uhler, C. (2017). Permutation-based causal inference
algorithms with interventions. NeurIPS 30.

Yang, K. D., Katcoff, A., and Uhler, C. (2018). Characterizing and learning equivalence classes of
causal DAGs under interventions. ICML, PMLR 80, 5537-5546.

Solus, L., Wang, Y., and Uhler, C. (2021). Consistency guarantees for greedy permutation-based
causal inference algorithms. Biometrika 108(4), 795-814.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core._bitset import nodes
from andrey.core.ci import MIXED_INDEP_TESTS, make_indep_test
from andrey.core.invariance import InvarianceTest, make_invariance_test
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.orient import to_structure
from andrey.core.seeding import start_rng
from andrey.core.ties import tie_mask, tie_ranks
from andrey.search.gies import cut_pairs, replace_unprotected

__all__ = [
    "DEPTH",
    "UT_IGSP_OBJECTIVE",
    "Rules",
    "igsp",
    "igsp_rules",
    "igsp_search",
    "reverse",
    "ut_igsp",
    "ut_igsp_rules",
]

#: The depth of the depth-first search: reversals from the last restart past which it does not
#: descend.
DEPTH = 4
#: UT-IGSP's objective: ``"sum"``, the number of arrows plus the number of targets over the
#: settings, or ``"lexicographic"``, the arrows first and the targets on a tie.
UT_IGSP_OBJECTIVE = "sum"

#: A DAG: each variable's parents as a bitmask.
Parents = tuple[int, ...]
#: A conditional-independence query in the engine's variable order: ``independent(i, j, S)``.
Independent = Callable[[int, int, tuple[int, ...]], bool]
#: The knowledge as the search reads it: the required and forbidden arrows and the precedence of
#: :class:`~andrey.core.knowledge.Constraints`, or ``None``.
_Know = tuple[np.ndarray, np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class Rules:
    """What a method changes in the shared search: five functions of the engine's DAGs.

    ``objective(parents)`` scores a DAG, lower is better, as any comparable value.
    ``icovered(parents, i, j)`` says whether the covered arrow ``i -> j`` is I-covered.
    ``admits(child, current)`` says whether a child with objective ``child`` joins the search from a
    DAG with objective ``current``, and ``restarts(child, current)`` whether it restarts the
    search. ``priority(parents, i, j)``, when given, orders the children, lower first; children
    with equal priority keep the visiting order.
    """

    objective: Callable[[Parents], Any]
    icovered: Callable[[Parents, int, int], bool]
    admits: Callable[[Any, Any], bool]
    restarts: Callable[[Any, Any], bool]
    priority: Callable[[Parents, int, int], int] | None = None


def igsp_search(
    n_vars: int,
    independent: Independent,
    rules: Rules,
    *,
    n_starts: int,
    seed: int,
    depth: int = DEPTH,
    constraints: Constraints | None = None,
) -> tuple[Parents, Any]:
    """Search minimal I-MAPs for the DAG with the lowest objective; return it and its objective.

    ``independent(i, j, S)`` is the conditional-independence test of ``i`` and ``j`` given the
    sorted tuple ``S``, ``True`` for independent; it is asked each query once. ``rules`` are the
    method's (:class:`Rules`), ``constraints`` the knowledge in the engine's variable order. Raises
    :class:`~andrey.core.knowledge.KnowledgeError` when the knowledge admits no order.
    """
    test = _memoized(independent)
    know = None
    if constraints is not None:
        know = (constraints.required, constraints.forbidden, constraints.precedence())
    every = (1 << n_vars) - 1
    adjacent = [
        {j for j in range(n_vars) if j != i and not test(i, j, every & ~(1 << i) & ~(1 << j))}
        for i in range(n_vars)
    ]
    best: Parents | None = None
    best_value: Any = None
    for s in range(n_starts):
        rng = None if s == 0 else start_rng(seed, s)
        order = _min_degree_order(adjacent, None if know is None else know[2], rng)
        start = _imap(order, test, know)
        found, value = _run(start, test, rules, depth, rng, know)
        if best is None or value < best_value:
            best, best_value = found, value
    return best, best_value  # ty: ignore[invalid-return-type]  # n_starts >= 1


def _memoized(independent: Independent) -> Callable[[int, int, int], bool]:
    """``independent`` asked by a bitmask conditioning set, each query once."""
    memo: dict[tuple[int, int, int], bool] = {}

    def test(i: int, j: int, mask: int) -> bool:
        key = (i, j, mask) if i < j else (j, i, mask)
        if key not in memo:
            memo[key] = bool(independent(key[0], key[1], tuple(nodes(mask))))
        return memo[key]

    return test


def _min_degree_order(
    adjacent: Sequence[set[int]], before: np.ndarray | None, rng: random.Random | None
) -> list[int]:
    """The reversed minimum-degree elimination order of the undirected graph ``adjacent``.

    A vertex can be eliminated only when no remaining vertex must come after it (``before``), as
    the eliminated vertex goes last among the remaining ones. Ties go to the lowest vertex without
    ``rng`` and to a draw from ``rng`` with it.
    """
    n = len(adjacent)
    neighbors = [set(a) for a in adjacent]
    remaining = set(range(n))
    eliminated: list[int] = []
    while remaining:
        eligible = sorted(
            v
            for v in remaining
            if before is None or not any(before[v, u] for u in remaining if u != v)
        )
        least = min(len(neighbors[v]) for v in eligible)
        tied = [v for v in eligible if len(neighbors[v]) == least]
        v = tied[0] if rng is None else rng.choice(tied)
        for a in neighbors[v]:
            neighbors[a] |= neighbors[v] - {a}
            neighbors[a].discard(v)
        remaining.discard(v)
        eliminated.append(v)
    return eliminated[::-1]


def _imap(order: Sequence[int], test: Callable[[int, int, int], bool], know: _Know) -> Parents:
    """The minimal I-MAP of ``order``: ``i -> j`` for ``i`` before ``j`` when required, never when
    forbidden, otherwise unless ``i`` and ``j`` are independent given ``j``'s other predecessors."""
    parents = [0] * len(order)
    earlier = 0
    for j in order:
        for i in nodes(earlier):
            if know is not None and know[0][i, j]:
                parents[j] |= 1 << i
            elif know is not None and know[1][i, j]:
                continue
            elif not test(i, j, earlier & ~(1 << i)):
                parents[j] |= 1 << i
        earlier |= 1 << j
    return tuple(parents)


def reverse(
    parents: Parents,
    i: int,
    j: int,
    test: Callable[[int, int, int], bool],
    required: np.ndarray | None = None,
) -> Parents:
    """The DAG after reversing the covered arrow ``i -> j`` of ``parents``, as a minimal I-MAP.

    ``j``'s parents become ``i``'s, and ``i``'s gain ``j``; then each parent ``q`` of ``i`` loses
    ``q -> i`` when ``i`` and ``q`` are independent given ``i``'s other new parents, and ``q -> j``
    when ``j`` and ``q`` are independent given ``i``'s other old parents (Solus, Wang, and Uhler
    2021). A required arrow stays. ``test(a, b, mask)`` takes the conditioning set as a bitmask.
    """
    out = list(parents)
    old = parents[i]
    new_i = old | (1 << j)
    out[i], out[j] = new_i, old
    for q in nodes(old):
        bit = 1 << q
        if (required is None or not required[q, i]) and test(i, q, new_i & ~bit):
            out[i] &= ~bit
        if (required is None or not required[q, j]) and test(j, q, old & ~bit):
            out[j] &= ~bit
    return tuple(out)


def _run(
    start: Parents,
    test: Callable[[int, int, int], bool],
    rules: Rules,
    depth: int,
    rng: random.Random | None,
    know: _Know,
) -> tuple[Parents, Any]:
    """One start from the DAG ``start``; returns the lowest-objective DAG it scored."""
    current, value = start, rules.objective(start)
    best, best_value = current, value
    visited: set[Parents] = set()
    trace: list[list[tuple[Parents, Any]]] = []  # each ancestor's children not yet tried
    while True:
        visited.add(current)
        children = []
        for i, j in _moves(current, rules, rng, know):
            child = reverse(current, i, j, test, None if know is None else know[0])
            if child in visited:
                continue
            child_value = rules.objective(child)
            if child_value < best_value:
                best, best_value = child, child_value
            if rules.admits(child_value, value):
                children.append((child, child_value))
        lower = [c for c in children if rules.restarts(c[1], value)]
        if lower:
            current, value = min(lower, key=lambda c: c[1])
            visited, trace = set(), []
            continue
        if children and len(trace) < depth:
            trace.append(children[1:])
            current, value = children[0]
            continue
        while trace:
            left = [c for c in trace[-1] if c[0] not in visited]
            if left:
                trace[-1] = left[1:]
                current, value = left[0]
                break
            trace.pop()
        else:
            return best, best_value


def _moves(
    parents: Parents, rules: Rules, rng: random.Random | None, know: _Know
) -> list[tuple[int, int]]:
    """The I-covered arrows of ``parents`` in visiting order.

    An arrow ``i -> j`` is covered when ``j``'s parents are ``i`` and ``i``'s parents; the knowledge
    keeps a required arrow, one whose reverse is forbidden, and one it orders ``i`` before ``j``.
    The order is by ``(i, j)`` without ``rng``, shuffled by ``rng`` with it, then sorted by the
    rules' priority.
    """
    moves = []
    for j, pa in enumerate(parents):
        for i in nodes(pa):
            if pa != parents[i] | (1 << i):
                continue
            if know is not None and (know[0][i, j] or know[1][j, i] or know[2][i, j]):
                continue
            moves.append((i, j))
    moves.sort()
    if rng is not None:
        rng.shuffle(moves)
    moves = [m for m in moves if rules.icovered(parents, *m)]
    priority = rules.priority
    if priority is not None:
        moves.sort(key=lambda m: priority(parents, *m))
    return moves


def ut_igsp_rules(
    invariant: Callable[[int, tuple[int, ...], int], bool],
    known: Sequence[frozenset[int]],
    *,
    objective: str | None = None,
) -> tuple[Rules, Callable[[Parents], list[frozenset[int]]]]:
    """UT-IGSP's rules, and the function that gives a DAG's targets in each setting.

    ``invariant(j, S, k)`` is the invariance test of ``x_j`` given the sorted tuple ``S`` between
    setting ``k`` and the observational rows, ``True`` for invariant; it is asked each query once.
    ``known[k]`` holds setting ``k``'s known targets. A DAG's targets in setting ``k`` are the known
    ones and every variable whose conditional given its parents is not invariant. ``objective``
    (``None`` reads :data:`UT_IGSP_OBJECTIVE`) is ``"sum"``, the number of arrows plus the number
    of targets over the settings, or ``"lexicographic"``, the arrows then the targets; the search
    admits children with an objective at most the current one and restarts from a strictly lower
    one. A covered arrow ``i -> j`` is I-covered when ``x_j`` given its parents is not invariant in
    every setting whose known targets hold ``i``.
    """
    rule = UT_IGSP_OBJECTIVE if objective is None else objective
    if rule not in ("sum", "lexicographic"):
        raise ValueError(f"objective must be 'sum' or 'lexicographic', got {rule!r}")
    memo: dict[tuple[int, int, int], bool] = {}

    def variant(j: int, mask: int, k: int) -> bool:
        if (j, mask, k) not in memo:
            memo[j, mask, k] = not invariant(j, tuple(nodes(mask)), k)
        return memo[j, mask, k]

    terms: dict[tuple[int, int], int] = {}

    def targets_of(j: int, mask: int) -> int:
        if (j, mask) not in terms:
            terms[j, mask] = sum(j in kn or variant(j, mask, k) for k, kn in enumerate(known))
        return terms[j, mask]

    def score(parents: Parents) -> tuple[int, ...]:
        arrows = sum(pa.bit_count() for pa in parents)
        count = sum(targets_of(j, pa) for j, pa in enumerate(parents))
        return (arrows + count,) if rule == "sum" else (arrows, count)

    def icovered(parents: Parents, i: int, j: int) -> bool:
        return all(variant(j, parents[j], k) for k, kn in enumerate(known) if i in kn)

    def targets(parents: Parents) -> list[frozenset[int]]:
        return [
            kn | {j for j, pa in enumerate(parents) if j not in kn and variant(j, pa, k)}
            for k, kn in enumerate(known)
        ]

    rules = Rules(
        objective=score,
        icovered=icovered,
        admits=lambda child, current: child <= current,
        restarts=lambda child, current: child < current,
    )
    return rules, targets


def igsp_rules(
    invariant: Callable[[int, tuple[int, ...], int], bool], family: Sequence[frozenset[int]]
) -> Rules:
    """IGSP's rules for the settings whose targets are ``family`` (Yang, Katcoff, and Uhler 2018).

    ``invariant(j, S, k)`` is the invariance test of ``x_j`` given the sorted tuple ``S`` between
    setting ``k`` and the observational rows, ``True`` for invariant; it is asked each query once.

    A covered arrow ``i -> j`` is I-covered unless ``x_j`` is not invariant in some setting whose
    targets are exactly ``{i}`` (Definition 4.1). When every setting has one target, ``i -> j`` is
    I-contradicting (Definition 4.3) if some setting has the targets ``{i}`` and ``x_j`` is
    invariant in every such setting, or ``x_i`` is not invariant in some setting with the targets
    ``{j}``. Otherwise (Definition 4.2) it is I-contradicting if some settings hold ``i`` and not
    ``j`` and, given some subset of ``j``'s other neighbors, ``x_j`` is invariant in all of them;
    or if some settings hold ``j`` and not ``i`` and, given every subset of ``i``'s other
    neighbors, ``x_i`` is not invariant in one of them. The objective is the number of arrows, then
    the number of I-contradicting ones; every child joins the search, a child with fewer arrows
    restarts it, and the reversals of I-contradicting arrows are visited first.
    """
    memo: dict[tuple[int, int, int], bool] = {}

    def variant(j: int, mask: int, k: int) -> bool:
        if (j, mask, k) not in memo:
            memo[j, mask, k] = not invariant(j, tuple(nodes(mask)), k)
        return memo[j, mask, k]

    single = all(len(t) <= 1 for t in family)
    alone: dict[int, list[int]] = {}  # the settings whose targets are exactly {v}, by v
    for k, t in enumerate(family):
        if len(t) == 1:
            alone.setdefault(next(iter(t)), []).append(k)
    holding: dict[tuple[int, int], list[int]] = {}  # the settings that hold i and not j
    first: dict[tuple[int, int, int], bool] = {}
    second: dict[tuple[int, int, int], bool] = {}

    def without(i: int, j: int) -> list[int]:
        if (i, j) not in holding:
            holding[i, j] = [k for k, t in enumerate(family) if i in t and j not in t]
        return holding[i, j]

    def invariant_given_some(i: int, j: int, others: int) -> bool:
        if (i, j, others) not in first:
            ks = without(i, j)
            first[i, j, others] = bool(ks) and any(
                not any(variant(j, s, k) for k in ks) for s in _subsets(others)
            )
        return first[i, j, others]

    def variant_given_every(i: int, j: int, others: int) -> bool:
        if (i, j, others) not in second:
            ks = without(j, i)
            second[i, j, others] = bool(ks) and all(
                any(variant(i, s, k) for k in ks) for s in _subsets(others)
            )
        return second[i, j, others]

    last: list[Any] = [None, None]

    def neighbors(parents: Parents) -> list[int]:
        if last[0] != parents:
            adjacent = list(parents)
            for j, pa in enumerate(parents):
                for i in nodes(pa):
                    adjacent[i] |= 1 << j
            last[:] = [parents, adjacent]
        return last[1]

    def contradicting(parents: Parents, i: int, j: int) -> bool:
        if single:
            if i in alone and not any(variant(j, 0, k) for k in alone[i]):
                return True
            return any(variant(i, 0, k) for k in alone.get(j, ()))
        ne = neighbors(parents)
        return invariant_given_some(i, j, ne[j] & ~(1 << i)) or variant_given_every(
            i, j, ne[i] & ~(1 << j)
        )

    def objective(parents: Parents) -> tuple[int, int]:
        arrows = [(i, j) for j, pa in enumerate(parents) for i in nodes(pa)]
        return len(arrows), sum(contradicting(parents, i, j) for i, j in arrows)

    return Rules(
        objective=objective,
        icovered=lambda parents, i, j: not any(variant(j, 0, k) for k in alone.get(i, ())),
        admits=lambda child, current: True,
        restarts=lambda child, current: child[0] < current[0],
        priority=lambda parents, i, j: 0 if contradicting(parents, i, j) else 1,
    )


def _subsets(mask: int) -> Iterator[int]:
    """Every subset of the bitmask ``mask``, the empty one first."""
    subset = 0
    while True:
        yield subset
        if subset == mask:
            return
        subset = (subset - mask) & mask


def _prepared(
    X: np.ndarray,
    codes: np.ndarray,
    by_code: Mapping[int, frozenset[int] | None],
    *,
    indep_test: str,
    degrees_of_freedom: str = "mean",
    invariance_test: str,
    invariance_alpha: float,
    categorical: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], Any, Callable[[int, tuple[int, ...], int], bool]]:
    """The tests both methods search with, over the columns sorted by their tie ranks.

    Returns each column's rank, the column at each rank, the codes of the settings (every
    environment whose targets are not ``frozenset()``, sorted), the conditional-independence test
    on the pooled observational rows, and the invariance query ``invariant(j, S, k)`` between
    setting ``k`` and those rows at ``invariance_alpha``, each setting's test built on first use.
    """
    d = X.shape[1]
    mixed = indep_test in MIXED_INDEP_TESTS
    mask = tie_mask(indep_test, categorical, d)
    ranks = tie_ranks(X, mask)
    by_rank = np.argsort(ranks)
    sorted_x = np.ascontiguousarray(X[:, by_rank])
    sorted_mask = mask[by_rank] if mixed else None
    observational = np.isin(codes, [c for c, t in by_code.items() if t == frozenset()])
    settings = [c for c in sorted(by_code) if by_code[c] != frozenset()]
    try:
        ci = make_indep_test(
            indep_test,
            sorted_x[observational],
            categorical=sorted_mask,
            seed=seed,
            degrees_of_freedom=degrees_of_freedom,
        )
    except ValueError:
        # The error names positions in the sorted data; the same test on ``X`` names its columns.
        make_indep_test(
            indep_test,
            X[observational],
            categorical=categorical,
            seed=seed,
            degrees_of_freedom=degrees_of_freedom,
        )
        raise
    tests: dict[int, InvarianceTest] = {}

    def invariant(j: int, condition_set: tuple[int, ...], k: int) -> bool:
        if k not in tests:
            groups = np.where(observational, 0, np.where(codes == settings[k], 1, -1))
            tests[k] = make_invariance_test(
                invariance_test,
                sorted_x,
                groups,
                indep_test=indep_test,
                degrees_of_freedom=degrees_of_freedom,
                categorical=sorted_mask,
                seed=seed,
            )
        return tests[k](j, condition_set) > invariance_alpha

    return ranks, by_rank, settings, ci, invariant


def _essential(
    found: Parents,
    ranks: np.ndarray,
    family: Sequence[frozenset[int]],
    constraints: Constraints | None,
) -> GraphStructure:
    """The ``cpdag`` interventional essential graph of the DAG ``found`` (in rank space) under the
    observational setting and each setting's targets ``family`` (column positions), the most
    oriented under ``constraints``."""
    d = len(found)
    dag = np.zeros((d, d), dtype=np.int8)
    for j, pa in enumerate(found):
        for i in nodes(pa):
            dag[i, j], dag[j, i] = TAIL, ARROW
    dag = dag[np.ix_(ranks, ranks)]
    rows = np.zeros((len(family) + 1, d), dtype=bool)
    for row, t in enumerate(family, start=1):
        rows[row, sorted(t)] = True
    graph = replace_unprotected(dag, cut_pairs(rows))
    if constraints is not None:
        graph = mpdag(graph, constraints)
    return to_structure(graph, kind="cpdag")


def igsp(
    X: np.ndarray,
    codes: np.ndarray,
    targets_by_code: Mapping[int, frozenset[int] | None],
    *,
    alpha: float,
    invariance_alpha: float,
    indep_test: str,
    degrees_of_freedom: str = "mean",
    invariance_test: str,
    categorical: np.ndarray | None,
    n_starts: int,
    seed: int,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, tuple[int, int]]:
    """Run IGSP on the checked data; return the I-essential graph and the objective.

    ``codes[i]`` is row ``i``'s environment and ``targets_by_code`` each environment's targets by
    column position: ``frozenset()`` for an observational one, whose rows are pooled, and a
    nonempty set for a setting. The conditional-independence test ``indep_test`` at ``alpha``
    reads the observational rows; the invariance test ``invariance_test`` at ``invariance_alpha``
    compares each setting with them. ``categorical`` is the mixed-data tests' mask. The search runs
    over the columns sorted by :func:`~andrey.core.ties.tie_ranks` (:func:`igsp_search`,
    :func:`igsp_rules`).

    Returns the ``cpdag`` interventional essential graph of the DAG found under the observational
    setting and each setting's targets, the most oriented under ``constraints``, and the DAG's
    objective: its number of arrows and of I-contradicting arrows. A
    :class:`~andrey.core.score.CopiedColumnsError` names the columns of ``X``.
    """
    ranks, by_rank, settings, ci, invariant = _prepared(
        X,
        codes,
        targets_by_code,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        invariance_test=invariance_test,
        invariance_alpha=invariance_alpha,
        categorical=categorical,
        seed=seed,
    )
    family = [frozenset(targets_by_code[c] or ()) for c in settings]
    found, value = igsp_search(
        X.shape[1],
        lambda i, j, s: ci(i, j, s) > alpha,
        igsp_rules(invariant, [frozenset(int(ranks[v]) for v in t) for t in family]),
        n_starts=n_starts,
        seed=seed,
        constraints=None if constraints is None else constraints.permuted(by_rank),
    )
    return _essential(found, ranks, family, constraints), value


def ut_igsp(
    X: np.ndarray,
    codes: np.ndarray,
    known_by_code: Mapping[int, frozenset[int] | None],
    *,
    alpha: float,
    invariance_alpha: float,
    indep_test: str,
    degrees_of_freedom: str = "mean",
    invariance_test: str,
    categorical: np.ndarray | None,
    n_starts: int,
    seed: int,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, dict[int, frozenset[int]], tuple[int, ...]]:
    """Run UT-IGSP on the checked data; return the I-essential graph, the targets, and the score.

    ``codes[i]`` is row ``i``'s environment and ``known_by_code`` each environment's known targets
    by column position: ``frozenset()`` for an observational one, whose rows are pooled, and a set,
    possibly empty only through ``None`` (unknown), for a setting. The conditional-independence test
    ``indep_test`` at ``alpha`` reads the observational rows; the invariance test
    ``invariance_test`` at ``invariance_alpha`` compares each setting with them. ``categorical`` is
    the mixed-data tests' mask. The search runs over the columns sorted by
    :func:`~andrey.core.ties.tie_ranks` (:func:`igsp_search`, :func:`ut_igsp_rules`).

    Returns the ``cpdag`` interventional essential graph of the DAG found, under the
    observational setting and each setting's targets, the most oriented under ``constraints``;
    each setting's targets by code, as column positions, known and learned; and the DAG's
    objective. A :class:`~andrey.core.score.CopiedColumnsError` names the columns of ``X``.
    """
    ranks, by_rank, settings, ci, invariant = _prepared(
        X,
        codes,
        known_by_code,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        invariance_test=invariance_test,
        invariance_alpha=invariance_alpha,
        categorical=categorical,
        seed=seed,
    )
    known = [frozenset(int(ranks[v]) for v in known_by_code[c] or ()) for c in settings]
    rules, targets_of = ut_igsp_rules(invariant, known)
    found, value = igsp_search(
        X.shape[1],
        lambda i, j, s: ci(i, j, s) > alpha,
        rules,
        n_starts=n_starts,
        seed=seed,
        constraints=None if constraints is None else constraints.permuted(by_rank),
    )
    targets = {
        c: frozenset(int(by_rank[r]) for r in t)
        for c, t in zip(settings, targets_of(found), strict=True)
    }
    return _essential(found, ranks, list(targets.values()), constraints), targets, value
