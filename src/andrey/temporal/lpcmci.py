"""LPCMCI: lagged and same-step causal links of a time series with hidden common causes.

LPCMCI learns a partial ancestral graph over the copies ``X^i_{t-tau}`` of the variables,
``tau = 0 .. tau_max``. The graph is stationary: every link stands for all its copies shifted in
time, so the link ``X^i_{t-tau} - X^j_t`` is one entry ``(i, j, tau)``. Each link has an edge mark
at each end and a middle mark. An edge mark is a tail, an arrowhead, a circle (not decided yet), or
a conflict (rules oriented it both ways). A tail at ``A`` says ``A`` is an ancestor of the other
end; an arrowhead says it is not. Nodes are ordered by time, then by index:
``X^i_{t-tau} < X^j_t`` when ``tau > 0`` or ``tau = 0`` and ``i < j``. For a link ``A - B`` with
``A < B``, the middle mark records which separating-set searches remain: ``"?"`` both, ``"L"``
the search among the adjacencies of ``B`` (the one at ``A`` is done or not needed), ``"R"`` the
search among the adjacencies of ``A``, ``"!"`` neither, though the link can still be a false one
between two variables that are not ancestors of each other, and ``""`` neither, with the link
certain.

The method runs Algorithm 1 of the paper:

1. **Initialize** a complete graph: every lagged link ``X^i_{t-tau} o-> X^j_t`` with middle mark
   ``"L"`` (time order already rules out ``X^j_t`` as an ancestor of ``X^i_{t-tau}``), every
   same-step link ``o-o`` with ``"?"``.
2. **Preliminary phase**, ``n_preliminary_iterations`` times: run the ancestral removal phase
   (below), keep the links it found directed, re-initialize the graph, and orient the kept links
   as ``A -> B`` with the middle mark MMR gives them. Separating sets and test memories reset.
3. **Final phase**: the ancestral removal phase, then the non-ancestral removal phase.

The **ancestral removal phase** (Algorithm S2) runs rounds ``p = 0, 1, ...``. Each round tests the
autodependency links first, then the cross links at lag ``0``, ``1``, ..., ``tau_max``, one group at
a time. A link ``A - B``, ``A < B``, is tested given ``S`` plus the default conditions ``S_def``,
the parents of ``A`` and ``B``. ``S`` runs over the ``p``-subsets of the adjacencies of ``B`` that
have no arrowhead at themselves (``apds_t``), minus ``S_def``, sorted by the smallest absolute test
statistic of their link with ``B``, largest first, as it stood when the group started, so the tests
of one link do not reorder the search of another link in its group and the order does not depend on
the order of the variables; statistics equal to ``_KEY_DP`` decimals go to the earlier node, then to
the lower :func:`~andrey.search.hc.tie_ranks` of its variable. A same-step link is also tested from
``A``'s side; the earlier end of a lagged link is searched with the empty set only. The first
p-value above ``alpha`` marks the link for removal and adds ``S + S_def`` to its separating sets;
marked links leave the graph when their group ends. A side with fewer than ``p`` candidates is done,
and its letter is added to the middle mark (``"?"`` + ``"R"`` = ``"R"``, ``"L"`` + ``"R"`` =
``"!"``). When a round removes a link, the rules ``APR, R8', R2', R1', R9', R10'`` orient lagged
links and ``p`` restarts at 0. The phase ends when every middle mark is ``"!"`` or ``""``, and then
applies the full rule list ``APR, R8', R2', R1', R0'd, R0'c, R3', R4, R9', R10', R0'b, R0'a``.

The **non-ancestral removal phase** (Algorithm S3) tests every link whose middle mark is not
empty given subsets of ``napds_t``, a superset of the D-Sep set that uses the orientations found
so far, with ``S_def`` plus the nodes that have been parents of either end since the last
re-initialization; a link whose search set is smaller than ``p`` gets the empty middle mark. A
round that removes a link applies the full rule list and restarts ``p``; the phase ends when every
middle mark is empty and applies the full rule list once more.

The rules (Algorithm S4) run in list order: each collects its orientations and removals over the
whole graph as it stood, applies them together, and the list restarts after any change; a rule
that removes links applies only the removals, and its orientations wait for the next pass. The
middle-mark rule MMR runs once the list is through. An edge mark asked to be both a tail and an
arrowhead, or asked to change a tail or an arrowhead it already has, becomes a conflict. Each
unshielded triple is read once, in the window whose latest node is at time ``t``. Rules ``R0'a``
and ``R0'b`` test a link again given each stored separating set of the outer pair plus the
parents of both ends, and remove it when one of the tests separates. Every set that separates,
from every triple, is stored, each made weakly minimal: while some node that is not a known
ancestor of either end can be dropped with the rest still separating, the one whose drop leaves
the largest p-value goes. Neither step depends on the order of the variables. Whether a node
``B`` is outside the separating set of ``A`` and ``C`` is a majority vote over the distinct
separating sets among the stored ones and the subsets of the ``apds_t`` sets of ``A`` and ``C``,
given the parents of both; the earlier node of a lagged pair contributes the empty subset only.
Whether ``B`` is inside reads the vote when both links to ``B`` have empty middle marks, and
otherwise requires ``B`` in every stored separating set. The paths of R9' and R10' leave out links
whose middle mark is ``"?"``, and R4 reads only links with empty middle marks.

Every test reads the lagged design of :func:`~andrey.temporal.pcmci.lagged_design` over
``T - 2 * tau_max`` time steps, through any :class:`~andrey.temporal.pcmci.LinkTest`, and is
written with its later node at time ``t``. Every node a step uses lies within ``tau_max`` steps of
the later node of the pair or triple it concerns.

References
----------
Gerhardus, A., and Runge, J. (2020). High-recall causal discovery for autocorrelated time series
with latent confounders. Advances in Neural Information Processing Systems 33 (NeurIPS 2020),
12615-12625. https://arxiv.org/abs/2007.01884
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, islice

import numpy as np

from andrey.core import backend
from andrey.core.structure import ARROW, CIRCLE, NULL, TAIL
from andrey.search._parallel import _KEY_DP
from andrey.search.hc import tie_ranks
from andrey.temporal.pcmci import INDEP_TESTS, LinkTest, design_test

CONFLICT = 4  # an edge mark the rules oriented both as a tail and as an arrowhead
BATCH = 4096  # tests per call to the test object: bounds its stack of correlation submatrices
CACHE_SIZE = 1 << 18  # test results kept for reuse, least recently used dropped first
CACHED_DEPTH = 3  # larger searched sets rarely repeat: their batches skip the cache

# A node ``(i, lag)``: variable ``i`` at time ``t - lag``.
Node = tuple[int, int]
# A link ``(i, j, lag)``: ``X^i_{t-lag} - X^j_t``, with ``i < j`` at lag 0.
Key = tuple[int, int, int]
# An edge: the mark at the earlier node, the middle mark, the mark at the later node.
Edge = list

_SYMBOL = {TAIL: "-", ARROW: ">", CIRCLE: "o", CONFLICT: "x"}
_SYMBOL_START = {TAIL: "-", ARROW: "<", CIRCLE: "o", CONFLICT: "x"}


@dataclass(frozen=True)
class LPCMCIResult:
    """The partial ancestral graph LPCMCI returns over ``n`` variables.

    ``start_marks`` and ``end_marks`` have shape ``(tau_max + 1, n, n)``, indexed ``[lag, i, j]``
    for the link ``X^i_{t-lag} - X^j_t``: ``start_marks`` holds the mark at ``X^i_{t-lag}`` and
    ``end_marks`` the mark at ``X^j_t``, ``NULL`` where there is no link. Marks are ``TAIL``,
    ``ARROW``, ``CIRCLE``, or :data:`CONFLICT`. Lag 0 holds both orders of a link, so
    ``start_marks[0, i, j] == end_marks[0, j, i]``. ``statistics`` and ``p_values`` have the same
    shape; a pair's entry is its final-phase test with the largest p-value, and ``0`` and ``1``
    where no test ran. ``sepsets[(i, j, lag)]`` lists the
    separating sets of a removed link (``i < j`` at lag 0) as ``(variable, lag)`` nodes relative
    to ``X^j_t``. ``parents[j]`` holds the ``(i, lag)`` nodes the last preliminary iteration found
    to be parents of ``X^j_t``, kept as directed links for the final phase.
    """

    start_marks: np.ndarray
    end_marks: np.ndarray
    statistics: np.ndarray
    p_values: np.ndarray
    sepsets: dict[Key, tuple[frozenset[Node], ...]]
    parents: tuple[tuple[Node, ...], ...]

    @property
    def adjacency(self) -> np.ndarray:
        """``(tau_max + 1, n, n)``: whether ``X^i_{t-lag}`` and ``X^j_t`` are linked."""
        return self.start_marks != NULL

    def link(self, i: int, j: int, lag: int) -> str:
        """The link ``X^i_{t-lag} - X^j_t`` as three characters, ``""`` when absent.

        The first character is the mark at ``X^i_{t-lag}`` (``-``, ``<``, ``o``, or ``x``), the
        second ``-``, and the third the mark at ``X^j_t`` (``-``, ``>``, ``o``, or ``x``).
        """
        start = int(self.start_marks[lag, i, j])
        if start == NULL:
            return ""
        return _SYMBOL_START[start] + "-" + _SYMBOL[int(self.end_marks[lag, i, j])]


def _order(node: Node) -> tuple[int, int]:
    """Sort key of the total order: earlier time first, then lower index."""
    return (-node[1], node[0])


def _key(a: Node, b: Node) -> tuple[Key, bool]:
    """The link between nodes ``a`` and ``b``, and whether ``a`` is its earlier (start) node."""
    (va, la), (vb, lb) = a, b
    if la > lb or (la == lb and va < vb):
        return (va, vb, la - lb), True
    return (vb, va, lb - la), False


def _shift(nodes: Iterable[Node], by: int) -> frozenset[Node]:
    """``nodes`` moved ``by`` steps into the past."""
    return frozenset((v, lag + by) for v, lag in nodes)


def _add_middle(current: str, letter: str) -> str:
    """Combine a middle mark with a finished search side: ``?+R=R``, ``L+R=!``, ``x+''=''``."""
    if letter == "" or current == "":
        return ""
    if current == "?":
        return letter
    if current in ("L", "R") and letter != current:
        return "!"
    return current


class _Search:
    """The state of one LPCMCI run: the graph, the test memories, and the separating sets.

    ``ranks`` holds each variable's :func:`~andrey.search.hc.tie_ranks`. The search keeps what it
    reads often until a change can alter it: each node's neighbors until a link is removed, its
    parents and children until an edge mark changes or a link is removed, its path steps until any
    mark changes or a link is removed, and the stored separating sets moved into a frame until a
    set is stored. Callers read these and do not change them.
    """

    def __init__(
        self,
        test: LinkTest,
        n_vars: int,
        tau_max: int,
        alpha: float,
        ranks: Sequence[int],
        max_depth: int | None = None,
    ) -> None:
        self.test, self.n, self.tau_max, self.alpha = test, n_vars, tau_max, alpha
        self.max_depth = max_depth
        self.ranks = ranks
        self.window = [(v, lag) for lag in range(tau_max + 1) for v in range(n_vars)]
        self.edges: dict[Key, Edge] = {}
        self.neighbors_of: dict[int, set[tuple[int, int]]] = {}
        self.near: dict[Node, tuple[list[Node], frozenset[Node]]] = {}
        self.roles_of: dict[Node, tuple[set[Node], list[Node], list[Node], list[Node]]] = {}
        self.steps_of: dict[Node, list[Node]] = {}
        self.shifted: dict[tuple[Key, int], list[frozenset[Node]]] = {}
        self.imin: dict[Key, float] = {}
        self.best: dict[Key, tuple[float, float]] = {}
        self.sepsets: dict[Key, list[frozenset[Node]]] = {}
        self.ever_directed: set[tuple[Key, bool]] = set()
        self.version = 0
        self.cache: OrderedDict[tuple[int, int, tuple[int, ...]], tuple[float, float]]
        self.cache = OrderedDict()
        self.votes: dict[Key, tuple[int, Counter[Node]]] = {}
        self.votes_version = -1

    # ---- the graph -----------------------------------------------------------------------

    def initialize(self, directed: Iterable[tuple[Key, bool]] = ()) -> None:
        """Algorithm 1, line 1: the complete graph, with ``directed`` links oriented ``->``.

        ``directed`` holds ``(key, start_is_tail)`` pairs; their middle mark is ``"?"`` updated by
        MMR. Test memories and separating sets reset.
        """
        self.edges = {}
        for lag in range(self.tau_max + 1):
            for i in range(self.n):
                for j in range(self.n):
                    if lag == 0 and i >= j:
                        continue
                    self.edges[(i, j, lag)] = [CIRCLE, "L", ARROW] if lag else [CIRCLE, "?", CIRCLE]
        for key, start_is_tail in directed:
            self.edges[key] = [TAIL, "L", ARROW] if start_is_tail else [ARROW, "R", TAIL]
        self.neighbors_of = {v: set() for v in range(self.n)}
        for i, j, lag in self.edges:
            self.neighbors_of[j].add((i, lag))
            self.neighbors_of[i].add((j, -lag))
        self.imin, self.best, self.sepsets, self.shifted = {}, {}, {}, {}
        self.ever_directed = set()
        self.version += 1
        self.near = {}
        self._marks_changed()
        self._record_directed()

    def remove(self, key: Key) -> None:
        """Remove the link ``key``."""
        i, j, lag = key
        del self.edges[key]
        self.neighbors_of[j].discard((i, lag))
        self.neighbors_of[i].discard((j, -lag))
        self.version += 1
        self.near = {}
        self._marks_changed()

    def _marks_changed(self) -> None:
        """Drop the parents, children, and path steps kept for the old edge marks."""
        self.roles_of, self.steps_of = {}, {}

    def set_middle(self, key: Key, middle: str) -> None:
        """Set the middle mark of the link ``key``."""
        self.edges[key][1] = middle
        self.steps_of = {}

    def adjacent(self, a: Node, b: Node) -> bool:
        (va, la), (vb, lb) = a, b  # the link as ``_key`` finds it
        if la > lb or (la == lb and va < vb):
            return la - lb <= self.tau_max and (va, vb, la - lb) in self.edges
        return lb - la <= self.tau_max and (vb, va, lb - la) in self.edges

    def mark(self, a: Node, b: Node) -> int:
        """The edge mark at ``a`` on its link with ``b``."""
        (va, la), (vb, lb) = a, b  # the link as ``_key`` finds it
        if la > lb or (la == lb and va < vb):
            return self.edges[(va, vb, la - lb)][0]
        return self.edges[(vb, va, lb - la)][2]

    def middle(self, a: Node, b: Node) -> str:
        """The middle mark of the link between ``a`` and ``b``, read for the earlier node first."""
        return self.edges[_key(a, b)[0]][1]

    def link(self, a: Node, b: Node) -> tuple[int, str, int]:
        """The mark at ``a``, the middle mark, and the mark at ``b`` of their link."""
        key, a_start = _key(a, b)
        start, middle, end = self.edges[key]
        return (start, middle, end) if a_start else (end, middle, start)

    def neighbors(self, a: Node) -> list[Node]:
        """The nodes linked to ``a`` within the window ``t - tau_max .. t``, in the total order."""
        found = self.near.get(a)
        return (found if found is not None else self._near(a))[0]

    def linked(self, a: Node) -> frozenset[Node]:
        """The nodes linked to ``a`` within the window, as a set."""
        found = self.near.get(a)
        return (found if found is not None else self._near(a))[1]

    def _near(self, a: Node) -> tuple[list[Node], frozenset[Node]]:
        v, lag = a
        found = [(k, lag + s) for k, s in self.neighbors_of[v] if 0 <= lag + s <= self.tau_max]
        found.sort(key=_order)
        self.near[a] = kept = (found, frozenset(found))
        return kept

    def directed(self, a: Node, b: Node) -> bool:
        """``a -> b``: a tail at ``a`` and an arrowhead at ``b``."""
        key, a_start = _key(a, b)
        edge = self.edges[key]
        if a_start:
            return edge[0] == TAIL and edge[2] == ARROW
        return edge[2] == TAIL and edge[0] == ARROW

    def _roles(self, a: Node) -> tuple[set[Node], list[Node], list[Node], list[Node]]:
        """The neighbors of ``a`` by their marks, each list in the order of :meth:`neighbors`.

        Return the parents of ``a`` as a set and as a list, its children, and the neighbors with no
        arrowhead at themselves.
        """
        found = self.roles_of.get(a)
        if found is None:
            parents, children, open_ends = [], [], []
            for b in self.neighbors(a):
                at_b, _, at_a = self.link(b, a)
                if at_b == TAIL and at_a == ARROW:
                    parents.append(b)
                elif at_b == ARROW and at_a == TAIL:
                    children.append(b)
                if at_b != ARROW:
                    open_ends.append(b)
            found = self.roles_of[a] = ({b for b in parents}, parents, children, open_ends)
        return found

    def parents(self, a: Node) -> set[Node]:
        """The parents of ``a`` within the window."""
        return self._roles(a)[0]

    def ancestors(self, nodes: Iterable[Node]) -> set[Node]:
        """The known ancestors of ``nodes`` within the window: directed paths into them."""
        found: set[Node] = set()
        frontier = list(nodes)
        while frontier:
            node = frontier.pop()
            for parent in self.parents(node):
                if parent not in found:
                    found.add(parent)
                    frontier.append(parent)
        return found

    def apds(self, b: Node, a: Node) -> set[Node]:
        """``apds_t(b, a)``: the neighbors of ``b`` but ``a`` with no arrowhead at themselves."""
        return {c for c in self._roles(b)[3] if c != a}

    def _record_directed(self) -> None:
        for key, (start, _, end) in self.edges.items():
            if start == TAIL and end == ARROW:
                self.ever_directed.add((key, True))
            elif start == ARROW and end == TAIL:
                self.ever_directed.add((key, False))

    def ever_parents(self, a: Node) -> set[Node]:
        """Nodes in the window that have been parents of ``a`` since the last initialization."""
        found = set()
        for b in self.window:
            if b == a or abs(b[1] - a[1]) > self.tau_max:
                continue
            key, b_start = _key(b, a)
            if (key, b_start) in self.ever_directed:
                found.add(b)
        return found

    # ---- tests ---------------------------------------------------------------------------

    def _column(self, node: Node) -> int:
        return node[1] * self.n + node[0]

    def run_tests(
        self,
        a: Node,
        b: Node,
        conditions: list[frozenset[Node]],
        *,
        rank: bool = True,
        report: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Test ``a`` against ``b`` given each set, all one size; return statistics and p-values.

        The pair is moved so its later node is at time ``t`` first. At most :data:`BATCH` sets
        are tested in one call, and the last :data:`CACHE_SIZE` results are kept for reuse.
        ``rank`` updates the smallest absolute statistic that orders search sets, ``report`` the
        largest p-value the result reports.
        """
        shift = -min(a[1], b[1])
        a, b = (a[0], a[1] + shift), (b[0], b[1] + shift)
        x, y = self._column(a), self._column(b)
        rows = [tuple(sorted(self._column((v, lag + shift)) for v, lag in s)) for s in conditions]
        found: dict[tuple[int, ...], tuple[float, float]] = {}
        for row in rows:
            cached = self.cache.get((x, y, row))
            if cached is not None:
                self.cache.move_to_end((x, y, row))
                found[row] = cached
        missing = sorted({row for row in rows if row not in found})
        for start in range(0, len(missing), BATCH):
            chunk = missing[start : start + BATCH]
            statistics, p_values = self.test.statistic_pvalues(
                np.full(len(chunk), x, dtype=np.intp),
                np.full(len(chunk), y, dtype=np.intp),
                np.array(chunk, dtype=np.intp).reshape(len(chunk), len(chunk[0])),
            )
            for row, statistic, p_value in zip(chunk, statistics.tolist(), p_values.tolist()):
                found[row] = (statistic, p_value)
                self.cache[(x, y, row)] = (statistic, p_value)
                if len(self.cache) > CACHE_SIZE:
                    self.cache.popitem(last=False)
        results = [found[row] for row in rows]
        statistics = [r[0] for r in results]
        p_values = [r[1] for r in results]
        if rank or report:
            key, _ = _key(a, b)
            for statistic, p_value in results:
                self._record(key, statistic, p_value, rank=rank, report=report)
        return statistics, p_values

    def _record(
        self, key: Key, statistic: float, p_value: float, *, rank: bool, report: bool
    ) -> None:
        """Update the link's smallest absolute statistic and its largest p-value."""
        if rank:
            self.imin[key] = min(abs(statistic), self.imin.get(key, np.inf))
        if report and (key not in self.best or p_value > self.best[key][1]):
            self.best[key] = (statistic, p_value)

    def separating(
        self, a: Node, b: Node, conditions: Iterable[frozenset[Node]]
    ) -> frozenset[Node] | None:
        """Test the sets, all one size, in order until one separates; return it.

        The tests made, up to the separating one, update the link's memories. The sets are tested
        in batches that double in size up to :data:`BATCH`, so a search that separates early stops
        early.
        """
        key, _ = _key(a, b)
        sets = iter(conditions)
        size = 16
        while batch := list(islice(sets, size)):
            statistics, p_values = self.run_tests(a, b, batch, rank=False, report=False)
            first = next((k for k, p in enumerate(p_values) if p > self.alpha), None)
            made = len(batch) if first is None else first + 1
            for statistic, p_value in zip(statistics[:made], p_values[:made]):
                self._record(key, statistic, p_value, rank=True, report=True)
            if first is not None:
                return batch[first]
            size = min(2 * size, BATCH)
        return None

    def separating_subsets(
        self, a: Node, b: Node, search: list[Node], sdef: frozenset[Node] | set[Node], p: int
    ) -> frozenset[Node] | None:
        """:meth:`separating` over ``S | sdef`` for the ``p``-subsets ``S`` of ``search``, in the
        order of :func:`itertools.combinations`.

        The sets are built as rows of column numbers, so a batch costs one array operation rather
        than one Python set per test; the tests, their order, and the memories they update are
        those of :meth:`separating`.
        """
        key, _ = _key(a, b)
        shift = -min(a[1], b[1])
        x, y = self._column((a[0], a[1] + shift)), self._column((b[0], b[1] + shift))
        columns = np.array([self._column((v, lag + shift)) for v, lag in search], dtype=np.intp)
        fixed = np.array([self._column((v, lag + shift)) for v, lag in sdef], dtype=np.intp)
        subsets = combinations(range(len(search)), p)
        size = 16
        while batch := list(islice(subsets, size)):
            picks = np.array(batch, dtype=np.intp).reshape(len(batch), p)
            fixed_rows = np.broadcast_to(fixed, (len(batch), len(fixed)))
            rows = np.concatenate([columns[picks], fixed_rows], axis=1)
            rows.sort(axis=1)
            statistics, p_values = self._test_rows(x, y, rows, cached=p <= CACHED_DEPTH)
            above = np.flatnonzero(p_values > self.alpha)
            made = len(batch) if len(above) == 0 else int(above[0]) + 1
            self._record_batch(key, statistics[:made], p_values[:made])
            if len(above):
                return frozenset(search[k] for k in batch[int(above[0])]) | frozenset(sdef)
            size = min(2 * size, BATCH)
        return None

    def _test_rows(
        self, x: int, y: int, rows: np.ndarray, cached: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Statistics and p-values of ``x`` against ``y`` given each sorted row of columns, through
        the cache that :meth:`run_tests` keeps when ``cached``, else in one call."""
        if not cached:
            n = len(rows)
            return self.test.statistic_pvalues(
                np.full(n, x, dtype=np.intp), np.full(n, y, dtype=np.intp), rows
            )
        keys = [(x, y, row) for row in map(tuple, rows.tolist())]
        statistics, p_values = np.empty(len(keys)), np.empty(len(keys))
        missing: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
        for k, cache_key in enumerate(keys):
            hit = self.cache.get(cache_key)
            if hit is None:
                missing.setdefault(cache_key, []).append(k)
            else:
                self.cache.move_to_end(cache_key)
                statistics[k], p_values[k] = hit
        if missing:
            order = sorted(missing)
            found_s, found_p = self.test.statistic_pvalues(
                np.full(len(order), x, dtype=np.intp),
                np.full(len(order), y, dtype=np.intp),
                np.array([k[2] for k in order], dtype=np.intp).reshape(len(order), rows.shape[1]),
            )
            for cache_key, statistic, p_value in zip(order, found_s.tolist(), found_p.tolist()):
                statistics[missing[cache_key]], p_values[missing[cache_key]] = statistic, p_value
                self.cache[cache_key] = (statistic, p_value)
                if len(self.cache) > CACHE_SIZE:
                    self.cache.popitem(last=False)
        return statistics, p_values

    def _record_batch(self, key: Key, statistics: np.ndarray, p_values: np.ndarray) -> None:
        """:meth:`_record` of each test in order, with ``rank`` and ``report``."""
        if not len(p_values):
            return
        self.imin[key] = min(float(np.abs(statistics).min()), self.imin.get(key, np.inf))
        best = int(np.argmax(p_values))  # the first of equal largest p-values, as _record keeps
        if key not in self.best or p_values[best] > self.best[key][1]:
            self.best[key] = (float(statistics[best]), float(p_values[best]))

    def add_sepset(self, key: Key, sepset: frozenset[Node], frame: int) -> None:
        """Store ``sepset``, given in a frame whose later node is ``frame`` steps back."""
        stored = _shift(sepset, -frame)
        found = self.sepsets.setdefault(key, [])
        if stored not in found:
            found.append(stored)
            self.shifted = {}

    def stored_sepsets(self, a: Node, c: Node) -> list[frozenset[Node]]:
        """The stored separating sets of ``a`` and ``c``, in the frame of the window."""
        key, _ = _key(a, c)
        frame = min(a[1], c[1])
        found = self.shifted.get((key, frame))
        if found is None:
            found = [_shift(s, frame) for s in self.sepsets.get(key, [])]
            self.shifted[(key, frame)] = found
        return found

    # ---- separating-set queries -------------------------------------------------------------

    def vote(self, key: Key) -> tuple[int, Counter[Node]]:
        """The separating sets of the pair ``key`` for a majority vote, counted.

        The sets are the stored ones and the separating sets among the subsets of ``apds_t`` of
        each node, given the parents of both; the earlier node of a lagged pair contributes the
        empty subset only. Each distinct set counts once. Return how many sets there are and, for
        each node, how many hold it, in the frame of ``key``. The counts last until the graph
        changes.
        """
        if self.votes_version != self.version:
            self.votes, self.votes_version = {}, self.version
        counted = self.votes.get(key)
        if counted is not None:
            return counted
        i, j, lag = key
        start, end = (i, lag), (j, 0)
        sdef = (self.parents(start) | self.parents(end)) - {start, end}
        stored = set(self.sepsets.get(key, []))
        total, holding = len(stored), Counter(node for s in stored for node in s)
        tested: frozenset[Node] | None = None  # subsets of these were tested from ``start``
        for anchor, other in ((start, end), (end, start)):
            candidates = sorted(self.apds(anchor, other) - sdef, key=_order)
            if anchor == start and lag > 0:
                candidates = []
            fresh = (
                frozenset(s)
                for size in range(self._depth(len(candidates)) + 1)
                for s in combinations(candidates, size)
                if tested is None or not tested.issuperset(s)
            )
            while batch := [s | sdef for s in islice(fresh, BATCH)]:
                by_size: dict[int, list[frozenset[Node]]] = {}
                for s in batch:
                    by_size.setdefault(len(s), []).append(s)
                for sets in by_size.values():
                    _, p_values = self.run_tests(start, end, sets, rank=False)
                    for s, p_value in zip(sets, p_values):
                        if p_value > self.alpha and s not in stored:
                            total += 1
                            holding.update(s)
            if anchor == start:
                tested = frozenset(candidates)
        self.votes[key] = (total, holding)
        return total, holding

    def share(self, a: Node, b: Node, c: Node) -> float:
        """The share of the voting separating sets of ``a`` and ``c`` that contain ``b``."""
        key, _ = _key(a, c)
        total, holding = self.vote(key)
        frame = min(a[1], c[1])
        return holding[(b[0], b[1] - frame)] / total if total else 0.5

    def outside_sepset(self, a: Node, b: Node, c: Node) -> bool:
        """Whether ``b`` is outside the separating set of ``a`` and ``c``, by majority."""
        return self.share(a, b, c) < 0.5

    def inside_sepset(self, a: Node, b: Node, c: Node) -> bool:
        """Whether ``b`` is inside a weakly minimal separating set of ``a`` and ``c``."""
        if (
            self.adjacent(a, b)
            and self.adjacent(b, c)
            and self.middle(a, b) == ""
            and self.middle(b, c) == ""
        ):
            return self.share(a, b, c) > 0.5
        stored = self.stored_sepsets(a, c)
        return bool(stored) and all(b in s for s in stored)

    # ---- the removal phases ------------------------------------------------------------------

    def _groups(self) -> list[Callable[[Key], bool]]:
        """Autodependency links first, then cross links by lag."""
        groups: list[Callable[[Key], bool]] = [lambda key: key[0] == key[1]]
        for m in range(self.tau_max + 1):
            groups.append(lambda key, m=m: key[2] == m and key[0] != key[1])
        return groups

    def _ranking(self) -> Callable[[set[Node], Node], list[Node]]:
        """A sort of candidates by the smallest statistics as they stand now.

        The sort puts ``nodes`` in order of the smallest statistic of their link with ``anchor``,
        largest first. Statistics equal to ``_KEY_DP`` decimals go to the earlier node, then to
        the lower rank. Each node's sort key is computed once per anchor.
        """
        imin = dict(self.imin)
        keys: dict[Node, dict[Node, tuple[float, int, int]]] = {}

        def ranked(nodes: set[Node], anchor: Node) -> list[Node]:
            known = keys.setdefault(anchor, {})
            for c in nodes:
                if c not in known:
                    statistic = round(float(imin.get(_key(anchor, c)[0], np.inf)), _KEY_DP)
                    known[c] = (-statistic, -c[1], self.ranks[c[0]])
            return sorted(nodes, key=known.__getitem__)

        return ranked

    def ancestral_phase(self, rules: list[Callable[[], tuple[dict, dict, dict]]]) -> None:
        """Algorithm S2, then the full rule list."""
        p = 0
        while True:
            removed = False
            for in_group in self._groups():
                ranked = self._ranking()
                marked = []
                for key in sorted(k for k in self.edges if in_group(k)):
                    i, j, lag = key
                    start, end = (i, lag), (j, 0)
                    sdef = (self.parents(start) | self.parents(end)) - {start, end}
                    separated = False
                    for forward in (True, False):
                        letters = ("?", "L") if forward else ("?", "R")
                        if self.edges[key][1] not in letters:
                            continue
                        anchor, other = (end, start) if forward else (start, end)
                        search = ranked(self.apds(anchor, other) - sdef, anchor)
                        if not forward and lag > 0:
                            search = search[:0]  # time order: the later node is no ancestor
                        if len(search) < p or self._past_depth(p):
                            letter = "R" if forward else "L"
                            self.set_middle(key, _add_middle(self.edges[key][1], letter))
                            continue
                        found = self.separating_subsets(start, end, search, sdef, p)
                        if found is not None:
                            self.add_sepset(key, found, 0)
                            separated = True
                    if separated:
                        marked.append(key)
                for key in marked:
                    self.remove(key)
                removed |= bool(marked)
            if removed:
                self.apply_rules(rules, lagged_only=True)
                p = 0
            else:
                p += 1
            if all(edge[1] in ("!", "") for edge in self.edges.values()):
                break
        self.apply_rules(self.full_rules(), lagged_only=False)

    def _past_depth(self, p: int) -> bool:
        """Whether ``p`` is above ``max_depth``: a search of that size is not run."""
        return self.max_depth is not None and p > self.max_depth

    def _depth(self, n_candidates: int) -> int:
        """The largest search-set size among ``n_candidates`` that ``max_depth`` allows."""
        return n_candidates if self.max_depth is None else min(n_candidates, self.max_depth)

    def napds(self, b: Node, a: Node) -> set[Node]:
        """``napds_t(b, a)``: candidates for a separating set of ``a`` and ``b``, neither ancestral.

        The neighbors in ``apds_t(b, a)`` not linked to ``a`` by a tail at ``a``, and the ends
        of the paths from ``b`` that Definition S5 of the paper allows. The first node of such a
        path comes before ``a`` in the total order; the later ones are not in the future of both
        ``a`` and ``b``.
        """
        first = {
            c for c in self.apds(b, a) if not (self.adjacent(a, c) and self.mark(a, c) == TAIL)
        }

        def allowed_inner(c: Node) -> bool:
            for end in (a, b):
                if self.adjacent(c, end) and self.mark(end, c) == TAIL:
                    return False
            if (
                self.adjacent(c, a)
                and self.adjacent(c, b)
                and self.mark(c, a) == ARROW
                and self.mark(c, b) == ARROW
            ):
                return False
            return not (c[1] < a[1] and c[1] < b[1])  # not in the future of both

        second: set[Node] = set()
        stack: list[tuple[Node, Node]] = []
        for c in self.neighbors(b):
            if c == a or self.mark(b, c) == TAIL:
                continue
            if self.adjacent(c, a) and self.mark(c, a) == ARROW:
                continue
            if c[1] < a[1]:
                continue  # after ``a``
            second.add(c)
            if self.mark(c, b) != TAIL:
                stack.append((b, c))
        visited = set(stack)
        while stack:
            before, last = stack.pop()
            for c in self.neighbors(last):
                if c in (a, b, before) or not allowed_inner(c):
                    continue
                if not self.adjacent(before, c) and not (
                    self.mark(last, before) == ARROW and self.mark(last, c) == ARROW
                ):
                    continue  # an unshielded triple needs a collider in the middle
                if self.mark(last, c) == TAIL:
                    continue
                second.add(c)
                if self.mark(c, last) != TAIL and (last, c) not in visited:
                    visited.add((last, c))
                    stack.append((last, c))
        return first | second

    def non_ancestral_phase(self) -> None:
        """Algorithm S3, then the full rule list."""
        p = 0
        while True:
            removed = False
            for in_group in self._groups():
                ranked = self._ranking()
                marked = []
                for key in sorted(k for k in self.edges if in_group(k)):
                    if self.edges[key][1] == "":
                        continue
                    i, j, lag = key
                    start, end = (i, lag), (j, 0)
                    sdef1 = (self.parents(start) | self.parents(end)) - {start, end}
                    sdef2 = (self.ever_parents(start) | self.ever_parents(end)) - {start, end}
                    near_end = self.napds(end, start)
                    near_start = self.napds(start, end)
                    search1 = ranked(near_end - sdef1 - sdef2, end)
                    search2 = ranked(near_start - sdef1 - sdef2, start)
                    if len(search1) < p or (lag == 0 and len(search2) < p) or self._past_depth(p):
                        self.set_middle(key, "")
                        continue
                    separated = False
                    sides = [(search1, near_end)] + ([(search2, near_start)] if lag == 0 else [])
                    for search, near in sides:
                        sdef = sdef1 | (sdef2 & near)
                        found = self.separating_subsets(start, end, search, sdef, p)
                        if found is not None:
                            self.add_sepset(key, found, 0)
                            separated = True
                    if separated:
                        marked.append(key)
                for key in marked:
                    self.remove(key)
                removed |= bool(marked)
            if removed:
                self.apply_rules(self.full_rules(), lagged_only=False)
                p = 0
            else:
                p += 1
            if all(edge[1] == "" for edge in self.edges.values()):
                break
        self.apply_rules(self.full_rules(), lagged_only=False)

    # ---- the rules ---------------------------------------------------------------------------

    def apply_rules(
        self, rules: list[Callable[[], tuple[dict, dict, dict]]], lagged_only: bool
    ) -> None:
        """Algorithm S4: run ``rules`` in order, restarting the list after any change.

        The middle-mark rule MMR runs once the list is through.
        """
        index = 0
        while index < len(rules):
            marks, middles, removals = rules[index]()
            if self._apply(marks, middles, removals, lagged_only):
                index = 0
            else:
                index += 1
        self._apply(*self.rule_mmr(), lagged_only=False)

    def _apply(
        self,
        marks: dict[tuple[Key, bool], set[int]],
        middles: dict[Key, str],
        removals: dict[Key, set[tuple[Node, Node, frozenset[Node]]]],
        lagged_only: bool,
    ) -> bool:
        """Apply a rule's proposals; return whether the graph changed.

        Removals go first: when a rule removes links, its orientations wait for the next pass. A
        removed link stores every separating set the rule found for it, each made weakly minimal.
        """
        changed = False
        if any(not (lagged_only and key[2] == 0) and key in self.edges for key in removals):
            marks, middles = {}, {}
        for (key, at_start), proposed in marks.items():
            if lagged_only and key[2] == 0 or key not in self.edges or key in removals:
                continue
            edge = self.edges[key]
            position = 0 if at_start else 2
            current = edge[position]
            new = CONFLICT if len(proposed) > 1 else next(iter(proposed))
            if current == CIRCLE:
                pass
            elif current != new:
                new = CONFLICT
            if new != current:
                edge[position] = new
                self._marks_changed()
                changed = True
        for key, middle in middles.items():
            if lagged_only and key[2] == 0 or key not in self.edges:
                continue
            if self.edges[key][1] != middle:
                self.set_middle(key, middle)
                changed = True
        for key, found in removals.items():
            if lagged_only and key[2] == 0 or key not in self.edges:
                continue
            self.remove(key)
            for a, b, sepset in found:
                self.add_sepset(key, self.weakly_minimal(a, b, sepset), min(a[1], b[1]))
            changed = True
        if changed:
            self.version += 1
            self._record_directed()
        return changed

    def weakly_minimal(self, a: Node, b: Node, sepset: frozenset[Node]) -> frozenset[Node]:
        """Drop the nodes of ``sepset`` that are not known ancestors while it still separates.

        Each step drops the node whose removal leaves the largest p-value above ``alpha``, so the
        result does not depend on the order of the variables. The known ancestors are read in the
        window whose later node is the later of ``a`` and ``b``.
        """
        later = min(a[1], b[1])
        known = _shift(self.ancestors([(a[0], a[1] - later), (b[0], b[1] - later)]), later)
        current = frozenset(sepset)
        while droppable := sorted(current - known, key=_order):
            trials = [current - {node} for node in droppable]
            _, p_values = self.run_tests(a, b, trials, rank=False, report=False)
            best = max(range(len(trials)), key=lambda k: p_values[k])
            if p_values[best] <= self.alpha:
                break
            current = trials[best]
        return current

    def full_rules(self) -> list[Callable[[], tuple[dict, dict, dict]]]:
        return [
            self.rule_apr,
            self.rule_r8,
            self.rule_r2,
            self.rule_r1,
            self.rule_r0d,
            self.rule_r0c,
            self.rule_r3,
            self.rule_r4,
            self.rule_r9,
            self.rule_r10,
            self.rule_r0b,
            self.rule_r0a,
        ]

    def preliminary_rules(self) -> list[Callable[[], tuple[dict, dict, dict]]]:
        return [
            self.rule_apr,
            self.rule_r8,
            self.rule_r2,
            self.rule_r1,
            self.rule_r9,
            self.rule_r10,
        ]

    @staticmethod
    def _propose(marks: dict, a: Node, b: Node, mark_a: int | None, mark_b: int | None) -> None:
        """Ask for ``mark_a`` at ``a`` and ``mark_b`` at ``b`` on their link."""
        key, a_start = _key(a, b)
        if mark_a is not None:
            marks.setdefault((key, a_start), set()).add(mark_a)
        if mark_b is not None:
            marks.setdefault((key, not a_start), set()).add(mark_b)

    def _middle_ok(self, b: Node, c: Node) -> bool:
        """``b o-* c`` with middle ``!``, or ``R`` with ``b < c``, or ``L`` with ``b > c``."""
        middle = self.middle(b, c)
        b_first = _order(b) < _order(c)
        return middle == "!" or (middle == "R" and b_first) or (middle == "L" and not b_first)

    def _unshielded(self) -> Iterable[tuple[Node, Node, Node]]:
        """Every unshielded triple ``(a, b, c)``, both orders of ``a`` and ``c``.

        Each triple appears once, moved so its latest node is at time ``t``.
        """
        for b in self.window:
            around = self.neighbors(b)
            for a in around:
                near_a = self.linked(a)
                for c in around:
                    if a != c and c not in near_a and min(a[1], b[1], c[1]) == 0:
                        yield a, b, c

    def rule_apr(self) -> tuple[dict, dict, dict]:
        """APR: ``A -!> B``, ``A -L> B`` (``A > B``), ``A -R> B`` (``A < B``) become ``A -> B``."""
        middles = {}
        for key, (start, middle, end) in self.edges.items():
            if start == TAIL and end == ARROW and middle in ("!", "R"):
                middles[key] = ""
            elif start == ARROW and end == TAIL and middle in ("!", "L"):
                middles[key] = ""
        return {}, middles, {}

    def rule_mmr(self) -> tuple[dict, dict, dict]:
        """MMR: an arrowhead at the later node adds ``L``, at the earlier node ``R``."""
        middles = {}
        for key, (start, middle, end) in self.edges.items():
            if end == ARROW and middle in ("?", "R"):
                middles[key] = "L" if middle == "?" else "!"
            elif start == ARROW and middle in ("?", "L"):
                middles[key] = "R" if middle == "?" else "!"
        return {}, middles, {}

    def rule_r0a(self) -> tuple[dict, dict, dict]:
        """R0'a: an unshielded triple whose middle node is outside the separating set collides."""
        marks: dict = {}
        removals: dict = {}
        for a, b, c in self._unshielded():
            if _order(a) > _order(c):
                continue
            dependent = True
            for x in (a, c):
                if self.middle(x, b) != "":
                    found = self._retest(x, b, a, c)
                    if found:
                        removals.setdefault(_key(x, b)[0], set()).update((x, b, s) for s in found)
                        dependent = False
            if not dependent:
                continue
            at_b = (self.mark(b, a), self.mark(b, c))
            if TAIL in at_b or CONFLICT in at_b or at_b == (ARROW, ARROW):
                continue
            if self.outside_sepset(a, b, c):
                self._propose(marks, a, b, None, ARROW)
                self._propose(marks, c, b, None, ARROW)
        return marks, {}, removals

    def _retest(self, x: Node, b: Node, a: Node, c: Node) -> list[frozenset[Node]]:
        """Test ``x`` and ``b`` given each separating set of ``a`` and ``c`` plus their parents.

        The sets are read in the window of the triple; nodes in the future of both ``x`` and
        ``b`` are left out. Return every conditioning set that separates them.
        """
        later = min(x[1], b[1])
        parents = (self.parents(x) | self.parents(b)) - {x, b}
        found = []
        for sepset in self.stored_sepsets(a, c) or [frozenset()]:
            nodes = frozenset(n for n in sepset | parents if later <= n[1] <= self.tau_max)
            _, (p_value,) = self.run_tests(x, b, [nodes - {x, b}])
            if p_value > self.alpha and nodes - {x, b} not in found:
                found.append(nodes - {x, b})
        return found

    def rule_r0b(self) -> tuple[dict, dict, dict]:
        """R0'b: ``A *-> B o-* C``, unshielded, ``B`` outside the separating set: ``B <-* C``."""
        marks: dict = {}
        removals: dict = {}
        for a, b, c in self._unshielded():
            if self.mark(b, a) != ARROW or self.mark(b, c) != CIRCLE:
                continue
            if self.mark(c, b) == CONFLICT or not self._middle_ok(b, c):
                continue
            if self.middle(a, b) != "":
                found = self._retest(a, b, a, c)
                if found:
                    removals.setdefault(_key(a, b)[0], set()).update((a, b, s) for s in found)
                    continue
            if self.outside_sepset(a, b, c):
                self._propose(marks, b, c, ARROW, None)
        return marks, {}, removals

    def rule_r0c(self) -> tuple[dict, dict, dict]:
        """R0'c: ``A *-* B o-* C`` with a certain ``A - B``, ``B`` outside the set: ``B <-* C``."""
        marks: dict = {}
        for a, b, c in self._unshielded():
            if self.middle(a, b) != "" or self.mark(b, c) != CIRCLE:
                continue
            if self.mark(c, b) == CONFLICT or not self._middle_ok(b, c):
                continue
            if self.outside_sepset(a, b, c):
                self._propose(marks, b, c, ARROW, None)
        return marks, {}, {}

    def rule_r0d(self) -> tuple[dict, dict, dict]:
        """R0'd: FCI's collider rule on triples whose two links have empty middle marks."""
        marks: dict = {}
        for a, b, c in self._unshielded():
            if _order(a) > _order(c):
                continue
            if self.middle(a, b) != "" or self.middle(b, c) != "":
                continue
            at_b = (self.mark(b, a), self.mark(b, c))
            if at_b not in ((CIRCLE, CIRCLE), (ARROW, CIRCLE), (CIRCLE, ARROW)):
                continue
            if self.outside_sepset(a, b, c):
                self._propose(marks, a, b, None, ARROW)
                self._propose(marks, c, b, None, ARROW)
        return marks, {}, {}

    def rule_r1(self) -> tuple[dict, dict, dict]:
        """R1': ``A *-> B o-* C``, unshielded, ``B`` in the separating set: ``B -> C``."""
        marks: dict = {}
        for a, b, c in self._unshielded():
            if self.mark(b, a) != ARROW or self.mark(b, c) != CIRCLE:
                continue
            if self.mark(c, b) == CONFLICT:
                continue
            if self.inside_sepset(a, b, c):
                self._propose(marks, b, c, TAIL, ARROW)
        return marks, {}, {}

    def rule_r2(self) -> tuple[dict, dict, dict]:
        """R2': ``A -> B *-> C`` or ``A *-> B -> C`` with ``A *-o C``: ``A *-> C``."""
        marks: dict = {}
        for a in self.window:
            around = self.neighbors(a)
            sides: list[tuple[Node, bool, bool]] | None = None  # b, b <-* a, a -> b
            for c in around:
                if self.mark(c, a) != CIRCLE or self.mark(a, c) == CONFLICT:
                    continue
                if sides is None:
                    sides = [(b, self.mark(b, a) == ARROW, self.directed(a, b)) for b in around]
                near_c = self.linked(c)
                for b, into_b, a_to_b in sides:
                    if b == c or b not in near_c:
                        continue
                    into_c = self.mark(c, b) == ARROW
                    if (a_to_b and into_c) or (into_b and self.directed(b, c)):
                        self._propose(marks, a, c, None, ARROW)
        return marks, {}, {}

    def rule_r3(self) -> tuple[dict, dict, dict]:
        """R3': ``A *-> B <-* C`` unshielded, ``A *-o D o-* C``, ``D *-o B``, ``D`` in the set."""
        marks: dict = {}
        for a, b, c in self._unshielded():
            if _order(a) > _order(c):
                continue
            if self.mark(b, a) != ARROW or self.mark(b, c) != ARROW:
                continue
            for d in self.neighbors(b):
                if d in (a, c) or self.mark(b, d) != CIRCLE or self.mark(d, b) == CONFLICT:
                    continue
                if not (self.adjacent(d, a) and self.adjacent(d, c)):
                    continue
                if self.mark(d, a) != CIRCLE or self.mark(d, c) != CIRCLE:
                    continue
                if self.mark(a, d) == CONFLICT or self.mark(c, d) == CONFLICT:
                    continue
                if self.inside_sepset(a, d, c):
                    self._propose(marks, d, b, None, ARROW)
        return marks, {}, {}

    def rule_r4(self) -> tuple[dict, dict, dict]:
        """R4: FCI's discriminating path rule.

        On a discriminating path ``D, ..., A, B, C`` for ``B`` with ``B o-* C``: ``B -> C`` when
        ``B`` is in the separating set of ``D`` and ``C``, ``A <-> B <-> C`` when it is outside.
        Every link the rule reads has an empty middle mark.
        """
        marks: dict = {}
        for b in self.window:
            for c in self.neighbors(b):
                if self.mark(b, c) != CIRCLE:
                    continue
                for a in self.neighbors(b):
                    if a == c or not self.adjacent(a, c):
                        continue
                    if self.mark(a, b) != ARROW or not self.directed(a, c):
                        continue
                    if any(self.middle(*pair) != "" for pair in ((b, c), (a, b), (a, c))):
                        continue
                    for d in self._discriminating_ends(a, b, c):
                        if self.inside_sepset(d, b, c):
                            self._propose(marks, b, c, TAIL, ARROW)
                        elif self.outside_sepset(d, b, c):
                            self._propose(marks, a, b, ARROW, ARROW)
                            self._propose(marks, b, c, ARROW, ARROW)
        return marks, {}, {}

    def _discriminating_ends(self, a: Node, b: Node, c: Node) -> list[Node]:
        """The far ends ``D`` of the discriminating paths ``D, ..., A, B, C`` for ``B``.

        Every node between ``D`` and ``B`` is a collider on the path and a parent of ``C``; the
        caller checks this for ``A``.
        """
        ends = set()
        seen = {a, b, c}
        frontier = [a]
        while frontier:
            node = frontier.pop()
            for d in self.neighbors(node):
                if d in seen or self.mark(node, d) != ARROW or self.middle(node, d) != "":
                    continue
                if not self.adjacent(d, c):
                    ends.add(d)
                elif (
                    self.directed(d, c) and self.middle(d, c) == "" and self.mark(d, node) == ARROW
                ):
                    seen.add(d)
                    frontier.append(d)
        return sorted(ends, key=_order)

    def rule_r8(self) -> tuple[dict, dict, dict]:
        """R8': ``A -> B -> C`` with ``A o-* C``: ``A -> C``."""
        marks: dict = {}
        for a in self.window:
            for c in self.neighbors(a):
                if self.mark(a, c) != CIRCLE or self.mark(c, a) == CONFLICT:
                    continue
                near_c = self.linked(c)
                if any(b != c and b in near_c and self.directed(b, c) for b in self._roles(a)[2]):
                    self._propose(marks, a, c, TAIL, ARROW)
        return marks, {}, {}

    def _paths(self, start: Node, step_ok: Callable[[list[Node], Node], bool]) -> Iterable[list]:
        """Uncovered potentially directed paths from ``start``; every step passes ``step_ok``.

        A step ``last - node`` is potentially directed: not into ``last`` and not out of ``node``.
        """
        stack = [[start]]
        while stack:
            path = stack.pop()
            yield path
            last = path[-1]
            near_before = self.linked(path[-2]) if len(path) >= 2 else frozenset()
            for node in self._steps(last):
                if node not in path and node not in near_before and step_ok(path, node):
                    stack.append(path + [node])

    def _steps(self, a: Node) -> list[Node]:
        """The neighbors ``b`` of ``a``, in order, whose link can be a path step ``a - b``.

        The link is potentially directed, not into ``a`` and not out of ``b``, and its middle mark
        is not ``"?"``.
        """
        found = self.steps_of.get(a)
        if found is None:
            found = []
            for b in self.neighbors(a):
                at_a, middle, at_b = self.link(a, b)
                if at_a in (CIRCLE, TAIL) and at_b in (CIRCLE, ARROW) and middle != "?":
                    found.append(b)
            self.steps_of[a] = found
        return found

    def rule_r9(self) -> tuple[dict, dict, dict]:
        """R9': ``A1 o-> An`` with an uncovered potentially directed path ``A1, ..., An``."""
        marks: dict = {}
        for a1 in self.window:
            for an in self.neighbors(a1):
                if self.mark(a1, an) != CIRCLE or self.mark(an, a1) != ARROW:
                    continue

                def step_ok(path: list[Node], node: Node, an: Node = an) -> bool:
                    k = path[-1]
                    if node == an:
                        return len(path) >= 2 and (
                            self.directed(k, node) or self.inside_sepset(path[-2], k, node)
                        )
                    before = path[-2] if len(path) >= 2 else an
                    if len(path) == 1 and self.adjacent(node, an):
                        return False
                    return self.directed(k, node) or (
                        not self.adjacent(before, node) and self.inside_sepset(before, k, node)
                    )

                for path in self._paths(a1, step_ok):
                    if path[-1] == an:
                        self._propose(marks, a1, an, TAIL, None)
                        break
        return marks, {}, {}

    def rule_r10(self) -> tuple[dict, dict, dict]:
        """R10': ``A o-> D`` with ``B -> D <- C`` reached from ``A`` by two such paths."""
        marks: dict = {}
        for a in self.window:
            for d in self.neighbors(a):
                if self.mark(a, d) != CIRCLE or self.mark(d, a) != ARROW:
                    continue
                into = [x for x in self._roles(d)[1] if x != a]
                if len(into) < 2:
                    continue

                def step_ok(path: list[Node], node: Node, d: Node = d) -> bool:
                    if node == d:
                        return False
                    if len(path) == 1:
                        return True
                    k = path[-1]
                    return self.directed(k, node) or self.inside_sepset(path[-2], k, node)

                reach: dict[Node, set[Node]] = {}  # end node -> first nodes after A
                for path in self._paths(a, step_ok):
                    if len(path) >= 2 and path[-1] in into:
                        reach.setdefault(path[-1], set()).add(path[1])
                found = False
                for x, y in combinations(sorted(reach, key=_order), 2):
                    for first_x in reach[x]:
                        for first_y in reach[y]:
                            if first_x != first_y and not self.adjacent(first_x, first_y):
                                if self.inside_sepset(first_x, a, first_y):
                                    found = True
                if found:
                    self._propose(marks, a, d, TAIL, None)
        return marks, {}, {}

    # ---- the whole method ------------------------------------------------------------------

    def directed_links(self) -> list[tuple[Key, bool]]:
        """The links that are directed now, as ``(key, start_is_tail)``."""
        found = []
        for key, (start, _, end) in self.edges.items():
            if start == TAIL and end == ARROW:
                found.append((key, True))
            elif start == ARROW and end == TAIL:
                found.append((key, False))
        return found


def lpcmci(
    data: np.ndarray,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    indep_test: str = "student_t",
    max_depth: int | None = None,
) -> LPCMCIResult:
    """Run LPCMCI on a ``(T, n)`` series with the ``indep_test`` partial-correlation test.

    ``max_depth`` caps the size of the searched subsets ``S`` in both removal phases and in the
    majority vote; the default conditions ``S_def`` come on top. ``None`` sets no cap.

    Raises :class:`ValueError` when ``indep_test`` is not a key of
    :data:`~andrey.temporal.pcmci.INDEP_TESTS`.
    """
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"indep_test must be one of {sorted(INDEP_TESTS)}, got {indep_test!r}")
    data = np.asarray(data, dtype=np.float64)
    test = design_test(indep_test, data, tau_max)
    ranks = tie_ranks(data).tolist()
    with backend.one_blas_thread():
        return run(
            test, data.shape[1], tau_max, pc_alpha, n_preliminary_iterations, ranks, max_depth
        )


def run(
    test: LinkTest,
    n_vars: int,
    tau_max: int,
    alpha: float,
    n_preliminary_iterations: int,
    ranks: Sequence[int],
    max_depth: int | None = None,
) -> LPCMCIResult:
    """Run Algorithm 1 with ``test`` over the columns of the lagged design.

    ``ranks`` holds each variable's :func:`~andrey.search.hc.tie_ranks`; ``max_depth`` caps the
    searched subset size (see :func:`lpcmci`).
    """
    search = _Search(test, n_vars, tau_max, alpha, ranks, max_depth)
    search.initialize()
    kept: list[tuple[Key, bool]] = []
    for _ in range(n_preliminary_iterations):
        search.ancestral_phase(search.preliminary_rules())
        kept = search.directed_links()
        search.initialize(kept)
    search.ancestral_phase(search.preliminary_rules())
    search.non_ancestral_phase()
    return _result(search, kept)


def _result(search: _Search, kept: list[tuple[Key, bool]]) -> LPCMCIResult:
    n, tau_max = search.n, search.tau_max
    shape = (tau_max + 1, n, n)
    start_marks = np.zeros(shape, dtype=np.int8)
    end_marks = np.zeros(shape, dtype=np.int8)
    for (i, j, lag), (start, _, end) in search.edges.items():
        start_marks[lag, i, j], end_marks[lag, i, j] = start, end
        if lag == 0:
            start_marks[0, j, i], end_marks[0, j, i] = end, start
    statistics = np.zeros(shape)
    p_values = np.ones(shape)
    for (i, j, lag), (statistic, p_value) in search.best.items():
        statistics[lag, i, j], p_values[lag, i, j] = statistic, p_value
        if lag == 0:
            statistics[0, j, i], p_values[0, j, i] = statistic, p_value
    parents: list[list[Node]] = [[] for _ in range(n)]
    for (i, j, lag), start_is_tail in kept:
        if start_is_tail:
            parents[j].append((i, lag))
        else:
            parents[i].append((j, 0))
    return LPCMCIResult(
        start_marks=start_marks,
        end_marks=end_marks,
        statistics=statistics,
        p_values=p_values,
        sepsets={key: tuple(sets) for key, sets in sorted(search.sepsets.items())},
        parents=tuple(tuple(sorted(found, key=_order)) for found in parents),
    )
