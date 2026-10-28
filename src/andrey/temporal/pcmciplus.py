"""PCMCI+: the lagged and contemporaneous causal links of a multivariate time series.

PCMCI+ extends PCMCI to links within one time step (lag 0). It works on the copies
``X^i_{t-tau}`` of the variables, ``tau = 0 .. tau_max``, in four phases:

1. **Lagged skeleton.** :func:`~andrey.temporal.pcmci.pc1` selects the lagged parents ``B(j)``
   of each variable ``X^j_t`` among the lags ``1 .. tau_max``.
2. **Contemporaneous skeleton.** The graph starts with the links from ``B(j)`` and a link between
   every two variables at lag 0. Round ``p = 0, 1, ...`` tests each linked pair
   ``(X^i_{t-tau}, X^j_t)`` given each subset ``S`` of ``p`` lag-0 neighbors of ``X^j_t``, with
   ``B(j)`` (without ``X^i_{t-tau}``) and ``B(i)`` shifted ``tau`` steps back added to every
   conditioning set. The first p-value above ``pc_alpha`` removes the link and stores ``S`` as its
   separating set. A lag-0 pair is tested from both ends, the ordered pairs ``(X^i_t, X^j_t)`` in
   order of ``(i, j)``. The neighbor sets stay fixed during a round and are re-sorted after it,
   strongest link first, by the smallest absolute statistic each link has had. The rounds stop
   when no linked pair has ``p`` neighbors left to condition on.
3. **Colliders.** Each unshielded triple ``X^i_{t-tau} - X^k_t - X^j_t`` with a lag-0 link
   ``X^k_t - X^j_t`` is a collider, ``X^i_{t-tau} -> X^k_t <- X^j_t``, by ``collider_rule``:
   ``"none"`` when ``X^k_t`` is not in the separating set; ``"majority"`` and ``"conservative"``
   test ``X^i_{t-tau}`` and ``X^j_t`` again given every subset of the lag-0 neighbors of
   ``X^j_t`` (and, at lag 0, of ``X^i_t``), with the phase-2 conditions added, and count the
   separating subsets that hold ``X^k_t``. ``"majority"`` calls the triple a collider when under
   half of them hold it, a non-collider when over half do, and ambiguous at exactly half;
   ``"conservative"`` requires none and all. A triple with no separating subset is ambiguous.
4. **Rules.** Meek's rules orient more lag-0 links: R1 (``X^i_{t-tau} -> X^k_t - X^j_t``, an
   unambiguous unshielded triple, gives ``X^k_t -> X^j_t``), R2 (``X^i_t -> X^k_t -> X^j_t``
   and ``X^i_t - X^j_t`` give ``X^i_t -> X^j_t``), and R3 (``X^i_t - X^k_t -> X^j_t`` and
   ``X^i_t - X^l_t -> X^j_t`` with ``X^k_t``, ``X^l_t`` not linked, ``X^i_t - X^j_t``, and the
   triple ``X^k_t - X^i_t - X^l_t`` unambiguous give ``X^i_t -> X^j_t``), in turn, until none
   applies.

Lagged links always point forward in time. Phases 3 and 4 collect the orientations of a pass from
the graph as it stood before the pass, then apply them together; a link that gets both orientations
is marked as a conflict and takes no part in later rules. Every test reads the lagged design of
:func:`~andrey.temporal.pcmci.lagged_design`, as in PCMCI.

References
----------
Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated
nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial
Intelligence (UAI), PMLR 124, 1388-1397. https://proceedings.mlr.press/v124/runge20a.html
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import collider_verdict
from andrey.constraint.pc import check_knowledge_cycle
from andrey.core import backend
from andrey.core.knowledge import orient as orient_known
from andrey.core.structure import ARROW, NULL, TAIL
from andrey.search.hc import tie_ranks
from andrey.temporal.pcmci import INDEP_TESTS, Link, LinkTest, design_test, pc1

if TYPE_CHECKING:
    from andrey.core.knowledge import LaggedConstraints

COLLIDER_RULES: tuple[str, ...] = ("majority", "conservative", "none")

# A triple ``(i, lag, k, j)``: ``X^i_{t-lag} - X^k_t - X^j_t``.
Triple = tuple[int, int, int, int]


@dataclass(frozen=True)
class PCMCIPlusResult:
    """The phases' results of one PCMCI+ run over ``n`` variables.

    ``parents[j]`` holds the lagged ``(i, lag)`` parents phase 1 selected for variable ``j``,
    strongest first. ``adjacency``, ``p_values``, and ``statistics`` have shape
    ``(tau_max + 1, n, n)``, indexed ``[lag, i, j]`` for ``X^i_{t-lag}`` and ``X^j_t``; lag 0 is
    symmetric. ``adjacency`` holds the links after phase 2. A link's p-value is the largest of its
    phase-2 tests, and its statistic the partial correlation of that test; a lagged link phase 1
    removed reports its largest phase-1 test instead, and the lag-0 diagonal is ``1`` and ``0``,
    as is a link the knowledge keeps apart; a required link reads ``0`` and ``1``.
    ``marks`` holds the lag-0 endpoint marks: ``marks[i, j]`` is the mark at ``i`` on the link
    ``i - j``, ``TAIL``/``ARROW`` for ``i -> j``, ``TAIL`` at both ends for an unoriented link, and
    ``ARROW`` at both ends for a conflict. ``sepsets[(i, lag, j)]`` is the lag-0 separating set
    phase 2 stored for a removed link, under both orders at lag 0. ``ambiguous_triples`` lists the
    ``(i, lag, k, j)`` triples phase 3 left ambiguous, a lag-0 triple in both orders.
    """

    parents: tuple[tuple[Link, ...], ...]
    adjacency: np.ndarray
    marks: np.ndarray
    p_values: np.ndarray
    statistics: np.ndarray
    sepsets: dict[tuple[int, int, int], tuple[int, ...]]
    ambiguous_triples: tuple[Triple, ...]

    @property
    def conflicts(self) -> tuple[tuple[int, int], ...]:
        """The lag-0 links with conflicting orientations, as ``(i, j)`` pairs with ``i < j``."""
        both = (self.marks == ARROW) & (self.marks.T == ARROW)
        return tuple((int(i), int(j)) for i, j in np.argwhere(np.triu(both)))


class _Recorder:
    """Passes queries to ``test``; keeps each column pair's largest p-value and its statistic."""

    def __init__(self, test: LinkTest) -> None:
        self.test = test
        self.largest: dict[tuple[int, int], tuple[float, float]] = {}

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        statistics, p_values = self.test.statistic_pvalues(x, y, conditions)
        for pair, statistic, p_value in zip(
            zip(np.asarray(x).tolist(), np.asarray(y).tolist()),
            statistics.tolist(),
            p_values.tolist(),
        ):
            best = self.largest.get(pair)
            if best is None or p_value > best[1]:
                self.largest[pair] = (statistic, p_value)
        return statistics, p_values


def lagged_parents(
    test: LinkTest,
    n_vars: int,
    tau_max: int,
    pc_alpha: float,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
) -> tuple[tuple[Link, ...], ...]:
    """Phase 1: the PC1 lagged parents of every variable, strongest first; ``max_depth`` caps
    PC1's conditioning sets."""
    return tuple(
        pc1(test, n_vars, j, tau_max, pc_alpha, ranks, constraints, max_depth)
        for j in range(n_vars)
    )


def _conditions(
    parents: tuple[tuple[Link, ...], ...],
    n_vars: int,
    i: int,
    lag: int,
    j: int,
    subset: Iterable[int],
) -> list[int]:
    """Design columns of ``S``, ``B(j)`` without ``X^i_{t-lag}``, and ``B(i)`` shifted ``lag``."""
    columns = set(subset)  # the lag-0 column of variable k is k
    columns.update(s * n_vars + k for k, s in parents[j] if (k, s) != (i, lag))
    columns.update((s + lag) * n_vars + k for k, s in parents[i])
    return sorted(columns)


def _test_subsets(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    i: int,
    lag: int,
    j: int,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """Test ``X^i_{t-lag}`` against ``X^j_t`` given each subset, all one size, in one call."""
    rows = [_conditions(parents, n_vars, i, lag, j, subset) for subset in subsets]
    conditions = np.array(rows, dtype=np.intp).reshape(len(rows), len(rows[0]))
    x = np.full(len(rows), lag * n_vars + i, dtype=np.intp)
    return test.statistic_pvalues(x, np.full_like(x, j), conditions)


def skeleton(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    tau_max: int,
    pc_alpha: float,
    constraints: LaggedConstraints | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int, int], tuple[int, ...]]]:
    """Phase 2: return ``(adjacency, largest p-values, their statistics, sepsets)``.

    The p-value array holds ``-1`` for a link phase 2 never tested. With ``constraints``, the lag-0
    links start from the pairs they allow, and no required link is tested or removed.
    """
    shape = (tau_max + 1, n_vars, n_vars)
    adjacency = np.zeros(shape, dtype=bool)
    for j, links in enumerate(parents):
        for i, lag in links:
            adjacency[lag, i, j] = True
    adjacency[0] = ~np.eye(n_vars, dtype=bool)
    kept = np.zeros(shape, dtype=bool)  # the required links, both orders at lag 0
    if constraints is not None:
        adjacency[0] &= constraints.contemporaneous().support()
        kept = constraints.required.copy()
        kept[0] |= kept[0].T
    p_values = np.full(shape, -1.0)
    statistics = np.zeros(shape)
    strength = np.full(shape, np.inf)
    sepsets: dict[tuple[int, int, int], tuple[int, ...]] = {}
    neighbors = [tuple(np.flatnonzero(adjacency[0, :, j]).tolist()) for j in range(n_vars)]
    p = 0
    while True:
        pairs = [
            (i, lag, j)
            for j in range(n_vars)
            for i, lag in parents[j]
            if adjacency[lag, i, j] and not kept[lag, i, j] and len(neighbors[j]) >= p
        ] + [
            (i, 0, j)  # in order of (i, j): a pair first meets its higher variable's neighbors
            for i in range(n_vars)
            for j in range(n_vars)
            if adjacency[0, i, j] and not kept[0, i, j] and len(neighbors[j]) - 1 >= p
        ]
        if not pairs:
            break
        for i, lag, j in pairs:
            if not adjacency[lag, i, j]:
                continue  # removed from its other end in this round
            subsets = list(combinations([k for k in neighbors[j] if lag or k != i], p))
            tested_statistics, tested_p_values = _test_subsets(
                test, n_vars, parents, i, lag, j, subsets
            )
            # The subsets count as tested in order up to the first that separates.
            separating = np.flatnonzero(tested_p_values > pc_alpha)
            tested = separating[0] + 1 if separating.size else len(subsets)
            largest = int(np.argmax(tested_p_values[:tested]))
            ends = ((lag, i, j), (0, j, i)) if lag == 0 else ((lag, i, j),)
            for index in ends:
                strength[index] = min(strength[index], np.abs(tested_statistics[:tested]).min())
                if tested_p_values[largest] > p_values[index]:
                    p_values[index] = tested_p_values[largest]
                    statistics[index] = tested_statistics[largest]
            if separating.size:
                for index in ends:
                    adjacency[index] = False
                sepsets[(i, lag, j)] = subsets[separating[0]]
                if lag == 0:
                    sepsets[(j, 0, i)] = subsets[separating[0]]
        p += 1
        neighbors = [
            tuple(
                sorted(
                    np.flatnonzero(adjacency[0, :, j]).tolist(), key=lambda k: -strength[0, k, j]
                )
            )
            for j in range(n_vars)
        ]
    return adjacency, p_values, statistics, sepsets


def _neighbors(adjacency: np.ndarray, j: int) -> list[int]:
    """The lag-0 neighbors of variable ``j``."""
    return np.flatnonzero(adjacency[0, j]).tolist()


def _separating_subsets(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    adjacency: np.ndarray,
    pc_alpha: float,
    i: int,
    lag: int,
    j: int,
) -> list[frozenset[int]]:
    """Every distinct subset of the lag-0 neighbors of ``j`` (and of ``i`` at lag 0) that separates.

    A subset separates ``X^i_{t-lag}`` and ``X^j_t`` when its test, with the phase-2 conditions
    added, gives a p-value above ``pc_alpha``.
    """
    families = [_neighbors(adjacency, j)] + ([_neighbors(adjacency, i)] if lag == 0 else [])
    separating = []
    for size in range(max(len(family) for family in families) + 1):
        subsets = list(
            dict.fromkeys(
                tuple(sorted(subset))
                for family in families
                for subset in combinations(family, size)
            )
        )
        if subsets:
            _, p_values = _test_subsets(test, n_vars, parents, i, lag, j, subsets)
            separating += [frozenset(subsets[k]) for k in np.flatnonzero(p_values > pc_alpha)]
    return separating


def apply_orientations(marks: np.ndarray, requests: Iterable[tuple[int, int]]) -> bool:
    """Orient each requested ``(source, target)`` lag-0 link; a link asked both ways is a conflict.

    Return whether any link changed.
    """
    wanted: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for source, target in requests:
        wanted.setdefault((min(source, target), max(source, target)), set()).add((source, target))
    for (a, b), directions in wanted.items():
        if len(directions) == 2:
            marks[a, b] = marks[b, a] = ARROW
        else:
            ((source, target),) = directions
            marks[source, target], marks[target, source] = TAIL, ARROW
    return bool(wanted)


def colliders(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    adjacency: np.ndarray,
    sepsets: dict[tuple[int, int, int], tuple[int, ...]],
    pc_alpha: float,
    collider_rule: str,
    constraints: LaggedConstraints | None = None,
) -> tuple[np.ndarray, frozenset[Triple]]:
    """Phase 3: return the lag-0 marks with the colliders oriented, and the ambiguous triples.

    With ``constraints``, the lag-0 links they decide are directed first, and no collider reverses
    one. With ``collider_rule="none"``, a triple over a pair they keep apart has no separating set
    to read and is ambiguous.
    """
    tau_max = adjacency.shape[0] - 1
    marks = np.where(adjacency[0], TAIL, NULL).astype(np.int8)
    apart = np.zeros(adjacency.shape, dtype=bool)
    if constraints is not None:
        lag0 = constraints.contemporaneous()
        marks = orient_known(marks, lag0)
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW))
        apart = constraints.forbidden.copy()
        apart[0] = ~lag0.support()
    cache: dict[tuple[int, int, int], list[frozenset[int]]] = {}
    requests: list[tuple[int, int]] = []
    ambiguous: set[Triple] = set()
    for k in range(n_vars):
        for j in _neighbors(adjacency, k):
            far = [(i, 0) for i in _neighbors(adjacency, k) if i < j and not adjacency[0, i, j]]
            far += [
                (i, lag)
                for lag in range(1, tau_max + 1)
                for i in range(n_vars)
                if adjacency[lag, i, k] and not adjacency[lag, i, j]
            ]
            for i, lag in far:
                if collider_rule == "none" and apart[lag, i, j]:
                    verdict = "ambiguous"
                elif collider_rule == "none":
                    verdict = "non-collider" if k in sepsets.get((i, lag, j), ()) else "collider"
                else:
                    key = (min(i, j), 0, max(i, j)) if lag == 0 else (i, lag, j)
                    if key not in cache:
                        cache[key] = _separating_subsets(
                            test, n_vars, parents, adjacency, pc_alpha, *key
                        )
                    verdict = collider_verdict(cache[key], k, collider_rule)
                if verdict == "collider":
                    requests.append((j, k))
                    if lag == 0:
                        requests.append((i, k))
                elif verdict == "ambiguous":
                    ambiguous.add((i, lag, k, j))
                    if lag == 0:
                        ambiguous.add((j, 0, k, i))
    if constraints is not None:
        directed = (marks == TAIL) & (marks.T == ARROW)
        requests = [(source, target) for source, target in requests if not directed[target, source]]
    apply_orientations(marks, requests)
    return marks, frozenset(ambiguous)


def _directed(marks: np.ndarray, a: int, b: int) -> bool:
    return bool(marks[a, b] == TAIL and marks[b, a] == ARROW)


def _undirected(marks: np.ndarray, a: int, b: int) -> bool:
    return bool(marks[a, b] == TAIL and marks[b, a] == TAIL)


def _rule_1(
    marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]
) -> list[tuple[int, int]]:
    """R1 orientations: ``X^i_{t-lag} -> X^k_t - X^j_t``, unshielded and unambiguous."""
    n_vars, tau_max = marks.shape[0], adjacency.shape[0] - 1
    requests = []
    for k in range(n_vars):
        for j in range(n_vars):
            if not _undirected(marks, k, j):
                continue
            into = [(i, 0) for i in range(n_vars) if _directed(marks, i, k)]
            into += [
                (i, lag)
                for lag in range(1, tau_max + 1)
                for i in range(n_vars)
                if adjacency[lag, i, k]
            ]
            if any(not adjacency[lag, i, j] and (i, lag, k, j) not in ambiguous for i, lag in into):
                requests.append((k, j))
    return requests


def _rule_2(marks: np.ndarray) -> list[tuple[int, int]]:
    """R2 orientations: ``X^i_t -> X^k_t -> X^j_t`` with ``X^i_t - X^j_t``."""
    n_vars = marks.shape[0]
    return [
        (i, j)
        for i in range(n_vars)
        for j in range(n_vars)
        if _undirected(marks, i, j)
        and any(_directed(marks, i, k) and _directed(marks, k, j) for k in range(n_vars))
    ]


def _rule_3(
    marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]
) -> list[tuple[int, int]]:
    """R3 orientations: ``X^i_t - X^k_t -> X^j_t`` and ``X^i_t - X^m_t -> X^j_t``, k, m apart."""
    n_vars = marks.shape[0]
    requests = []
    for i in range(n_vars):
        for j in range(n_vars):
            if not _undirected(marks, i, j):
                continue
            middles = [
                k for k in range(n_vars) if _undirected(marks, i, k) and _directed(marks, k, j)
            ]
            if any(
                not adjacency[0, k, m] and (k, 0, i, m) not in ambiguous
                for k, m in combinations(middles, 2)
            ):
                requests.append((i, j))
    return requests


def orient(marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]) -> None:
    """Phase 4: apply R1, R2, and R3 in turn to ``marks`` in place until no rule orients a link."""
    changed = True
    while changed:
        changed = apply_orientations(marks, _rule_1(marks, adjacency, ambiguous))
        changed |= apply_orientations(marks, _rule_2(marks))
        changed |= apply_orientations(marks, _rule_3(marks, adjacency, ambiguous))


def pcmciplus(
    data: np.ndarray,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.01,
    collider_rule: str = "majority",
    indep_test: str = "student_t",
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
) -> PCMCIPlusResult:
    """Run the four PCMCI+ phases on a ``(T, n)`` series with the ``indep_test`` test.

    ``constraints``, background knowledge over lags ``0`` to ``tau_max``, reach every phase
    (:func:`run_phases`). Raises :class:`ValueError` when ``indep_test`` is not a key of
    :data:`~andrey.temporal.pcmci.INDEP_TESTS` or ``collider_rule`` is not in
    :data:`COLLIDER_RULES`.
    """
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"indep_test must be one of {sorted(INDEP_TESTS)}, got {indep_test!r}")
    if collider_rule not in COLLIDER_RULES:
        raise ValueError(
            f"collider_rule must be one of {list(COLLIDER_RULES)}, got {collider_rule!r}"
        )
    data = np.asarray(data, dtype=np.float64)
    test = design_test(indep_test, data, tau_max)
    ranks = tie_ranks(data).tolist()
    with backend.one_blas_thread():  # the stacked inversions then give the same bits on any host
        return run_phases(
            test, data.shape[1], tau_max, pc_alpha, collider_rule, ranks, constraints, max_depth
        )


def run_phases(
    test: LinkTest,
    n_vars: int,
    tau_max: int,
    pc_alpha: float,
    collider_rule: str,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
) -> PCMCIPlusResult:
    """Run the four phases with ``test`` over the columns of the lagged design.

    ``ranks`` holds each variable's :func:`~andrey.search.hc.tie_ranks`, which orders PC1's equal
    strengths. With ``constraints``, no phase tests a forbidden or a required link: phase 1 leaves
    out the forbidden lagged links and keeps the required ones, phase 2 starts the lag-0 links
    from the allowed pairs and removes no required link, and phase 3 directs the lag-0 links the
    knowledge decides before the colliders, which do not reverse them. A link no phase tested
    reports p-value 1 and statistic 0, or 0 and 1 when it is required. ``max_depth`` caps phase 1's
    conditioning sets.
    """
    recorder = _Recorder(test)
    parents = lagged_parents(recorder, n_vars, tau_max, pc_alpha, ranks, constraints, max_depth)
    adjacency, p_values, statistics, sepsets = skeleton(
        test, n_vars, parents, tau_max, pc_alpha, constraints
    )
    for (x, j), (statistic, p_value) in recorder.largest.items():
        lag, i = divmod(x, n_vars)
        if p_values[lag, i, j] < 0:  # removed in phase 1
            p_values[lag, i, j], statistics[lag, i, j] = p_value, statistic
    p_values[p_values < 0] = 1.0  # the lag-0 diagonal, and the links the knowledge forbids
    if constraints is not None:
        required = constraints.required.copy()
        required[0] |= required[0].T
        p_values[required], statistics[required] = 0.0, 1.0
    marks, ambiguous = colliders(
        test, n_vars, parents, adjacency, sepsets, pc_alpha, collider_rule, constraints
    )
    orient(marks, adjacency, ambiguous)
    return PCMCIPlusResult(
        parents=parents,
        adjacency=adjacency,
        marks=marks,
        p_values=p_values,
        statistics=statistics,
        sepsets=sepsets,
        ambiguous_triples=tuple(sorted(ambiguous)),
    )
