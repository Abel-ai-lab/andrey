"""Extremely Greedy Equivalence Search, the engine behind :func:`andrey.xges`.

The base search applies the best improving Delete, else Reverse, else Insert, and stops when none
improves the score: the linear-Gaussian BIC, the categorical BDeu and BIC, or a mixed-data BIC. The
extended search forces each Delete of the converged CPDAG,
forbids re-inserting the deleted edge, reruns the base search, and keeps lower-scoring results.

:class:`_Search` keeps candidates in a priority queue per operator type. A popped operator is
dropped when it no longer applies to the current CPDAG. After a move, only the operators it can
affect are rescored, and the CPDAG is completed locally. The update rules and the local completion
follow the authors' reference implementation.

Scores are deltas of the deviance the local score minimizes; a negative delta improves it. Insert
deltas come from :func:`~andrey.core.score_delta.add_deltas` under the BIC, which scores every
candidate parent against one parent set at once by a rank-one update, and from
:func:`~andrey.core.score_delta.full_add_deltas`, two full scores per candidate, under the other
scores. A Delete and the parent removed by a Reverse each use the difference of two memoized local
scores.

References
----------
Nazaret and Blei (2024). Extremely Greedy Equivalence Search. UAI 2024.
https://arxiv.org/abs/2502.19551

Chickering (2002). Optimal Structure Identification With Greedy Search.
https://jmlr.org/papers/v3/chickering02b.html

Hauser and Buhlmann (2012). Characterization and Greedy Learning of Interventional Markov
Equivalence Classes of Directed Acyclic Graphs. https://jmlr.org/papers/v13/hauser12a.html
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from itertools import chain, count

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, backend
from andrey.core.knowledge import Constraints, admits, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import to_structure
from andrey.core.score import (
    DISCRETE_SCORES,
    MIXED_SCORES,
    SCORES,
    BICScore,
    LocalScore,
    make_score,
)
from andrey.core.score_delta import add_deltas, full_add_deltas
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP
from andrey.search.ges import _recompute_objective, _reform_cpdag, required_dag
from andrey.search.hc import tie_ranks
from andrey.search.operators import exceeds_parent_cap

# The extended search keeps a branch when it lowers the deviance by more than this times the sample
# count (1e-7 times the sample count in log-likelihood units).
_ACCEPT_PER_SAMPLE = 2e-7

# The state of the pair (a, b), a < b.
_NONE, _UNDIRECTED, _LOW_TO_HIGH, _HIGH_TO_LOW = 0, 1, 2, 3

_ADD_UNDIRECTED, _ADD_DIRECTED, _DROP_UNDIRECTED, _ORIENT, _DROP_DIRECTED, _UNORIENT, _REVERSE = (
    range(1, 8)
)


def _is_directed(state: int) -> bool:
    return state in (_LOW_TO_HIGH, _HIGH_TO_LOW)


def _change_kind(old: int, new: int) -> int:
    """Classify the change of a pair's state from ``old`` to ``new``."""
    if old == _NONE:
        return _ADD_UNDIRECTED if new == _UNDIRECTED else _ADD_DIRECTED
    if old == _UNDIRECTED:
        return _DROP_UNDIRECTED if new == _NONE else _ORIENT
    if new == _NONE:
        return _DROP_DIRECTED
    return _UNORIENT if new == _UNDIRECTED else _REVERSE


_MASK64 = (1 << 64) - 1


def _edge_key(a: int, b: int, state: int) -> int:
    """A 128-bit key of the pair ``(a, b)`` in ``state``; the XOR of a graph's keys fingerprints it.

    Equal graphs have equal fingerprints, but XOR is linear, so distinct graphs can share one.
    Tuple hashes of ints do not depend on ``PYTHONHASHSEED``, so the keys are the same every run.
    """
    if state == _NONE:
        return 0
    return (hash((a, b, state)) & _MASK64) | (hash((state, b, a)) & _MASK64) << 64


class _Changes(dict):
    """The net change of each pair since the last :meth:`clear`: ``{(a, b): [old, new]}``, a < b.

    A pair whose state returns to where it started is dropped.
    """

    def record(self, a: int, b: int, old: int, new: int) -> None:
        entry = self.get((a, b))
        if entry is None:
            self[(a, b)] = [old, new]
        elif entry[0] == new:
            del self[(a, b)]
        else:
            entry[1] = new

    def rekey(self, fingerprint: int) -> int:
        """A graph's ``fingerprint`` (the XOR of its :func:`_edge_key`) after these changes."""
        for (a, b), (old, new) in self.items():
            fingerprint ^= _edge_key(a, b, old) ^ _edge_key(a, b, new)
        return fingerprint


def _returns(moves: Iterable[_Changes]) -> bool:
    """Whether ``moves``, applied in order, restore every pair's initial state."""
    net = _Changes()
    for move in moves:
        for (a, b), (old, new) in move.items():
            net.record(a, b, old, new)
    return all(old == new for old, new in net.values())


class _Graph:
    """A PDAG as per-node sets of parents, children, undirected neighbors, and adjacent nodes.

    ``forbidden`` holds the ``(x, y)`` pairs whose Insert ``x -> y`` the extended search forbids.
    A :meth:`branch` shares a snapshot's sets until it changes a node: its first change copies
    that node's four sets from ``source``, the graph it branched from, and adds the node to
    ``copied``. ``version`` counts the changes to the graph's edges, so a branch can check that its
    snapshot still matches the graph.
    """

    __slots__ = ("pa", "ch", "ne", "ad", "forbidden", "source", "copied", "version")

    def __init__(self, d: int) -> None:
        self.pa: list[set[int]] = [set() for _ in range(d)]
        self.ch: list[set[int]] = [set() for _ in range(d)]
        self.ne: list[set[int]] = [set() for _ in range(d)]
        self.ad: list[set[int]] = [set() for _ in range(d)]
        self.forbidden: set[tuple[int, int]] = set()
        self.source: _Graph | None = None
        self.copied: set[int] = set()
        self.version = 0

    def copy(self) -> _Graph:
        out = _Graph.__new__(_Graph)
        out.pa = [set(s) for s in self.pa]
        out.ch = [set(s) for s in self.ch]
        out.ne = [set(s) for s in self.ne]
        out.ad = [set(s) for s in self.ad]
        out.forbidden = set(self.forbidden)
        out.source, out.copied = None, set()
        out.version = self.version
        return out

    def branch(self, snapshot: _Graph) -> _Graph:
        """A copy of this graph that shares the sets of ``snapshot``, a :meth:`copy` of this graph
        that neither graph has changed since.

        A set's iteration order depends on its history, and the search's ties follow that order.
        The snapshot's sets iterate as fresh copies do, and the branch copies a node's sets from
        this graph, as :meth:`copy` does, before changing them. A snapshot taken at another version
        than this graph's now, or changed since, raises ``RuntimeError``.
        """
        if snapshot.version != self.version:
            raise RuntimeError("the snapshot no longer matches the graph it was copied from")
        out = _Graph.__new__(_Graph)
        out.pa, out.ch = snapshot.pa.copy(), snapshot.ch.copy()
        out.ne, out.ad = snapshot.ne.copy(), snapshot.ad.copy()
        out.forbidden = set(self.forbidden)
        out.source, out.copied = self, set()
        out.version = self.version
        return out

    def detach(self) -> None:
        """Stop copying from ``source``; the caller drops ``source`` and the snapshot."""
        self.source, self.copied = None, set()

    def _own(self, v: int, source: _Graph) -> None:
        self.pa[v], self.ch[v] = set(source.pa[v]), set(source.ch[v])
        self.ne[v], self.ad[v] = set(source.ne[v]), set(source.ad[v])
        self.copied.add(v)

    def state(self, a: int, b: int) -> int:
        """The state of the pair ``(a, b)``, ``a < b``."""
        if b in self.ne[a]:
            return _UNDIRECTED
        if b in self.ch[a]:
            return _LOW_TO_HIGH
        if a in self.ch[b]:
            return _HIGH_TO_LOW
        return _NONE

    def set_state(self, a: int, b: int, state: int) -> None:
        """Replace the edge between ``a < b`` by ``state``."""
        self.version += 1
        source = self.source
        if source is not None:
            if a not in self.copied:
                self._own(a, source)
            if b not in self.copied:
                self._own(b, source)
        self.ne[a].discard(b)
        self.ne[b].discard(a)
        self.ch[a].discard(b)
        self.pa[b].discard(a)
        self.ch[b].discard(a)
        self.pa[a].discard(b)
        self.ad[a].discard(b)
        self.ad[b].discard(a)
        if state == _NONE:
            return
        self.ad[a].add(b)
        self.ad[b].add(a)
        if state == _UNDIRECTED:
            self.ne[a].add(b)
            self.ne[b].add(a)
        elif state == _LOW_TO_HIGH:
            self.ch[a].add(b)
            self.pa[b].add(a)
        else:
            self.ch[b].add(a)
            self.pa[a].add(b)

    def change(self, x: int, y: int, state: int, log: _Changes) -> None:
        """Set the edge between ``x`` and ``y`` to ``state`` (``_LOW_TO_HIGH`` means ``x -> y``)."""
        a, b = (x, y) if x < y else (y, x)
        if x > y and _is_directed(state):
            state = _HIGH_TO_LOW if state == _LOW_TO_HIGH else _LOW_TO_HIGH
        old = self.state(a, b)
        self.set_state(a, b, state)
        log.record(a, b, old, state)

    def is_clique(self, nodes: Iterable[int]) -> bool:
        nodes = set(nodes)
        return all(nodes - {v} <= self.ad[v] for v in nodes)

    def marks(self) -> np.ndarray:
        """The PDAG as an Andrey mark matrix."""
        d = len(self.pa)
        adj = np.zeros((d, d), dtype=np.int8)
        for x in range(d):
            for y in self.ch[x]:
                adj[x, y] = TAIL
                adj[y, x] = ARROW
            for y in self.ne[x]:
                adj[x, y] = TAIL
        return adj


class _Paths(dict[tuple[int, int], set[tuple[int, int]]]):
    """Per edge ``(u, v)``, the Inserts ``(x, y)`` whose recorded open path uses it.

    A :meth:`branch` shares a snapshot's sets until it adds to one: its first :meth:`add` to an
    edge copies that edge's set from ``source``, the records it branched from, and adds the edge to
    ``copied``. As for :meth:`_Graph.branch`, the snapshot is a :meth:`copy` of ``source`` and
    neither has changed since.
    """

    __slots__ = ("source", "copied")

    def __init__(self, items: dict | None = None, source: _Paths | None = None) -> None:
        super().__init__(items or {})
        self.source = source
        self.copied: set[tuple[int, int]] = set()

    def add(self, edge: tuple[int, int], insert: tuple[int, int]) -> None:
        inserts = self.get(edge)
        if inserts is None:
            inserts = self[edge] = set()
            if self.source is not None:
                self.copied.add(edge)
        elif self.source is not None and edge not in self.copied:
            inserts = self[edge] = set(self.source[edge])
            self.copied.add(edge)
        inserts.add(insert)

    def copy(self) -> _Paths:
        return _Paths({k: set(v) for k, v in self.items()})

    def branch(self, snapshot: _Paths) -> _Paths:
        return _Paths(snapshot, source=self)

    def detach(self) -> None:
        """Stop copying from ``source``; the caller drops ``source`` and the snapshot."""
        self.source, self.copied = None, set()


def _extension_parents(
    pa: list[set[int]], ne: list[set[int]], nodes: Iterable[int]
) -> dict[int, set[int]]:
    """Each node's parents in a consistent DAG extension of the CPDAG that the graph alone picks.

    Covers the chain components (nodes joined by undirected edges) of ``nodes``. Each component is
    oriented along a maximum cardinality search from its lowest node, ties to the lowest node. A
    chain component is chordal, so each node's earlier neighbors form a clique and the orientation
    adds no v-structure; nodes in other components keep their parents.
    """
    out: dict[int, set[int]] = {}
    for root in nodes:
        if root in out:
            continue
        if not ne[root]:
            out[root] = pa[root]
            continue
        component, stack = {root}, [root]
        while stack:
            for u in ne[stack.pop()] - component:
                component.add(u)
                stack.append(u)
        weight = dict.fromkeys(component, 0)
        heap = [(0, v) for v in component]
        heapq.heapify(heap)
        rank: dict[int, int] = {}
        while heap:
            w, v = heapq.heappop(heap)
            if v in rank or -w != weight[v]:
                continue
            rank[v] = len(rank)
            for u in ne[v]:
                if u not in rank:
                    weight[u] += 1
                    heapq.heappush(heap, (-weight[u], u))
        for v in component:
            out[v] = pa[v] | {u for u in ne[v] if rank[u] < rank[v]}
    return out


def _extension_scores(
    pa: list[set[int]], ne: list[set[int]], nodes: Iterable[int], score: LocalScore
) -> dict[int, float]:
    """The memoized local score of each node :func:`_extension_parents` covers.

    The sum over all nodes is the CPDAG's total score, independent of the moves that reached it,
    even when singular parent sets make Schur updates and full-score differences disagree. A
    singular parent set divides by zero on purpose, so the caller runs this under
    ``np.errstate(divide="ignore", invalid="ignore")``, once for all the calls of a search.
    """
    return {v: score.score(v, tuple(ps)) for v, ps in _extension_parents(pa, ne, nodes).items()}


def _local(score: LocalScore, v: int, parents: Iterable[int]) -> float:
    """The local score of ``v`` given a parent set."""
    return score.score(v, sorted(parents))


# ---- local CPDAG completion ---------------------------------------------------------------------


def _in_v_structure(g: _Graph, x: int, y: int) -> bool:
    """``x -> y`` with another parent of ``y`` not adjacent to ``x``."""
    return x in g.pa[y] and any(z != x and z not in g.ad[x] for z in g.pa[y])


def _meek_orients(g: _Graph, x: int, y: int) -> bool:
    """True when Meek's rule 1, 2, or 3 orients the edge between ``x`` and ``y`` as ``x -> y``."""
    if any(y not in g.ad[z] for z in g.pa[x]):  # rule 1: z -> x -- y, z and y non-adjacent
        return True
    if any(y in g.ch[z] for z in g.ch[x]):  # rule 2: x -> z -> y
        return True
    mid = sorted(g.ne[x] & g.pa[y])  # rule 3: x -- z -> y and x -- w -> y, z and w non-adjacent
    return any(w not in g.ad[z] for i, z in enumerate(mid) for w in mid[i + 1 :])


class _EdgeQueue:
    """A FIFO of edges to re-check, each queued at most once at a time."""

    def __init__(self) -> None:
        self._queue: deque[tuple[int, int, bool]] = deque()
        self._queued: set[tuple[int, int, bool]] = set()

    def push(self, x: int, y: int, directed: bool) -> None:
        edge = (x, y, True) if directed else (min(x, y), max(x, y), False)
        if edge not in self._queued:
            self._queue.append(edge)
            self._queued.add(edge)

    def pop(self) -> tuple[int, int, bool]:
        edge = self._queue.popleft()
        self._queued.discard(edge)
        return edge

    def __bool__(self) -> bool:
        return bool(self._queue)


def _queue_around(g: _Graph, x: int, y: int, queue: _EdgeQueue) -> None:
    """Queue every edge at ``x`` or ``y`` other than the one between them."""
    for u, v in ((x, y), (y, x)):
        for z in g.ch[v]:
            if z != u:
                queue.push(v, z, True)
        for z in g.pa[v]:
            if z != u:
                queue.push(z, v, True)
        for z in g.ne[v]:
            if z != u:
                queue.push(v, z, False)


def _complete(g: _Graph, queue: _EdgeQueue, log: _Changes) -> None:
    """Re-check the queued edges until each directed edge is compelled and no undirected one is."""
    while queue:
        x, y, directed = queue.pop()
        if directed:
            if y not in g.ch[x]:
                continue
            if _in_v_structure(g, x, y) or _meek_orients(g, x, y):
                continue
            g.change(x, y, _UNDIRECTED, log)
        else:
            if y not in g.ne[x]:
                continue
            if _meek_orients(g, x, y):
                pass
            elif _meek_orients(g, y, x):
                x, y = y, x
            else:
                continue
            g.change(x, y, _LOW_TO_HIGH, log)
        _queue_around(g, x, y, queue)


# ---- the operators ------------------------------------------------------------------------------


def _apply_insert(g: _Graph, x: int, y: int, T: frozenset[int], log: _Changes) -> None:
    """Insert ``x -> y``, orient ``t -> y`` for each ``t`` in ``T``, and complete the CPDAG."""
    g.change(x, y, _LOW_TO_HIGH, log)
    for t in T:
        g.change(t, y, _LOW_TO_HIGH, log)
    queue = _EdgeQueue()
    queue.push(x, y, True)
    _queue_around(g, x, y, queue)
    for t in T:
        _queue_around(g, t, y, queue)
    _complete(g, queue, log)


def _apply_reverse(g: _Graph, x: int, y: int, T: frozenset[int], log: _Changes) -> None:
    """Remove ``y -> x``, then insert ``x -> y`` with ``T``."""
    g.change(x, y, _NONE, log)
    _apply_insert(g, x, y, T, log)


def _apply_delete(g: _Graph, x: int, y: int, C: frozenset[int], log: _Changes) -> None:
    """Delete the edge ``x -> y`` or ``x -- y``, keeping ``C`` undirected; complete the CPDAG.

    Every ``h`` in ``H = (Ne(y) & Ad(x)) - C`` gets ``y -> h``, and ``x -> h`` where ``x -- h``.
    """
    g.change(x, y, _NONE, log)
    H = (g.ne[y] & g.ad[x]) - C
    for h in H:
        g.change(y, h, _LOW_TO_HIGH, log)
        if h in g.ne[x]:
            g.change(x, h, _LOW_TO_HIGH, log)
    queue = _EdgeQueue()
    _queue_around(g, x, y, queue)
    for h in H:
        _queue_around(g, h, y, queue)
        _queue_around(g, x, h, queue)
    _complete(g, queue, log)


def _blocks_paths(
    g: _Graph, y: int, x: int, blocked: set[int], paths: _Paths, skip_direct: bool
) -> bool:
    """True when every semi-directed path from ``y`` to ``x`` passes through ``blocked``.

    ``skip_direct`` ignores ``y -> x`` itself (a Reverse). Each edge of an open path maps to
    ``(x, y)`` in ``paths`` so edge changes requeue the Insert.
    """
    if y == x:
        return False
    parent: dict[int, int] = {}
    visited = {y}
    queue = deque([y])
    while queue:
        node = queue.popleft()
        for nxt in chain(g.ch[node], g.ne[node]):
            if nxt in visited or nxt in blocked or (skip_direct and node == y and nxt == x):
                continue
            parent[nxt] = node
            if nxt == x:
                cur = x
                while cur != y:
                    paths.add((parent[cur], cur), (x, y))
                    cur = parent[cur]
                return False
            visited.add(nxt)
            queue.append(nxt)
    return True


def _insert_valid(
    g: _Graph,
    x: int,
    y: int,
    T: frozenset[int],
    E: frozenset[int],
    paths: _Paths,
    reverse: bool = False,
) -> bool:
    """Whether Insert ``x -> y`` with ``T`` applies to ``g`` and still has effective parents ``E``.

    With ``reverse``, the Insert is the second half of reversing ``y -> x``.
    """
    if (x, y) in g.forbidden:
        return False
    ad_x = g.ad[x]
    if (y not in g.pa[x]) if reverse else (y in ad_x):
        return False
    ne_y = g.ne[y]
    if not (T <= ne_y and T.isdisjoint(ad_x)):
        return False
    na_t = (ne_y & ad_x) | T
    if E != na_t | g.pa[y]:
        return False
    if not g.is_clique(na_t):
        return False
    blocked = na_t | g.ne[x] if reverse else na_t
    return _blocks_paths(g, y, x, blocked, paths, skip_direct=reverse)


def _pinned(h0: Iterable[int], x: int, y: int, constraints: Constraints) -> set[int]:
    """The ``h`` of ``h0`` a Delete of ``x - y`` must keep: those whose orientation ``y -> h`` or
    ``x -> h`` is forbidden."""
    f = constraints.forbidden
    return {h for h in h0 if f[y, h] or f[x, h]}


def _delete_valid(
    g: _Graph,
    x: int,
    y: int,
    C: frozenset[int],
    E: frozenset[int],
    constraints: Constraints | None = None,
) -> bool:
    """Whether Delete ``x - y`` keeping ``C`` applies to ``g`` and still has effective parents
    ``E``; with ``constraints``, whether ``C`` also keeps every :func:`_pinned` neighbor.
    """
    if y not in g.ch[x] and y not in g.ne[x]:
        return False
    if not (C <= g.ne[y] and C <= g.ad[x]):
        return False
    if E != C | g.pa[y] | {x}:
        return False
    if constraints is not None and not _pinned(g.ne[y] & g.ad[x], x, y, constraints) <= C:
        return False
    return g.is_clique(C)


# ---- the search ---------------------------------------------------------------------------------


# Bytes the cached per-(target, parent set) score arrays may hold; each is one float per variable.
_ADD_DELTAS_BYTES = 256 * 1024**2


class _AddDeltas:
    """Bounded LRU cache of the add deltas per ``(y, E)``.

    :func:`~andrey.core.score_delta.add_deltas` gives them under the BIC and
    :func:`~andrey.core.score_delta.full_add_deltas` under the other scores.

    ``self(y, E)[x]`` is the score change of adding ``x`` to ``y``'s parents ``E``.
    """

    def __init__(self, score: LocalScore, d: int) -> None:
        self.score = score
        self.d = d
        self.cap = max(1024, _ADD_DELTAS_BYTES // (8 * max(d, 1)))
        self.cache: OrderedDict[tuple[int, frozenset[int]], np.ndarray] = OrderedDict()

    def __call__(self, y: int, E: frozenset[int]) -> np.ndarray:
        key = (y, E)
        deltas = self.cache.get(key)
        if deltas is None:
            if isinstance(self.score, BICScore):
                deltas = add_deltas(self.score, y, E)
            else:
                deltas = full_add_deltas(self.score, y, E, self.d)
            self.cache[key] = deltas
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return deltas


class _Search:
    """XGES over a :class:`_Graph`, with a priority queue per operator type.

    Queue entries are ``(delta, seq, x, y, subset, E[, Pa(x)])``: ``seq`` orders entries of one
    operator found more than once, ``subset`` is an Insert's or Reverse's ``T`` and a Delete's
    ``C``, and ``E`` the target's parent set the delta was scored with. :meth:`tied` takes entries
    by delta per sample rounded to :data:`~andrey.search._parallel._KEY_DP` decimals, then by the
    ranks of ``x``, of ``y``, and of the subset, ranking only the entries whose rounded deltas tie;
    ``ranks`` holds each column's :func:`~andrey.search.hc.tie_ranks`.

    With ``constraints``, no operator adds a forbidden edge or removes or reverses a required one,
    and a move whose class holds no DAG that satisfies them is undone. With ``cap``, no Insert or
    Reverse that gives its target more than ``cap`` parents is queued.
    """

    def __init__(
        self,
        score: LocalScore,
        d: int,
        ranks: list[int],
        constraints: Constraints | None = None,
        cap: int | None = None,
    ) -> None:
        self.add_deltas = _AddDeltas(score, d)
        self.d = d
        self.cap = cap
        self.n_samples = score.n
        self.tie_gap = score.n * 10.0**-_KEY_DP
        self.ranks = ranks
        self.constraints = constraints
        # Per target, the nodes forbidden into it; None without knowledge.
        self.into = (
            None
            if constraints is None
            else [set(np.flatnonzero(constraints.forbidden[:, y]).tolist()) for y in range(d)]
        )
        self.g = _Graph(d)
        self.paths = _Paths()
        self.fingerprint = 0  # the XOR of _edge_key over the edges of g
        self.inserts: list[tuple] = []
        self.reverses: list[tuple] = []
        self.deletes: list[tuple] = []
        self.seq = count()

    def branch(self, snapshot: tuple[_Graph, _Paths]) -> _Search:
        """A copy of the graph and of the path records, each branched from its ``snapshot``, with
        empty queues and the same score cache."""
        out = _Search.__new__(_Search)
        out.add_deltas, out.d, out.seq, out.cap = self.add_deltas, self.d, self.seq, self.cap
        out.n_samples, out.tie_gap, out.ranks = self.n_samples, self.tie_gap, self.ranks
        out.constraints, out.into = self.constraints, self.into
        out.g = self.g.branch(snapshot[0])
        out.paths = self.paths.branch(snapshot[1])
        out.fingerprint = self.fingerprint
        out.inserts, out.reverses, out.deletes = [], [], []
        return out

    # -- finding operators ----------------------------------------------------------------------

    def tied(self, heap: list[tuple], entry: tuple) -> tuple:
        """The entry to take after popping ``entry`` from ``heap``, by rounded delta and rank.

        The heap orders entries by raw delta, and the caller has checked that the next delta lies
        within ``tie_gap`` of ``entry``'s. The entries whose deltas per sample round to ``entry``'s
        at ``_KEY_DP`` decimals are ranked by ``ranks[x]``, ``ranks[y]``, and the sorted ranks of
        the subset, then by ``seq``; the others go back on the heap. Entries of one operator alone
        keep the heap's order: at most one of them matches the current parents.
        """
        n, gap, ranks = self.n_samples, self.tie_gap, self.ranks
        lowest = round(entry[0] / n, _KEY_DP)
        tied = [entry]
        while heap and heap[0][0] - entry[0] < gap and round(heap[0][0] / n, _KEY_DP) == lowest:
            tied.append(heapq.heappop(heap))
        if any(other[2:5] != entry[2:5] for other in tied[1:]):

            def rank(e: tuple) -> tuple:
                return (ranks[e[2]], ranks[e[3]], sorted(ranks[v] for v in e[4]), e[1])

            tied.sort(key=rank)
        for other in tied[1:]:
            heapq.heappush(heap, other)
        return tied[0]

    def subsets(
        self, t0: list[int], na: set[int], E: frozenset[int]
    ) -> Iterable[tuple[frozenset[int], frozenset[int]]]:
        """Each ``T`` in ``t0`` with ``na | T`` a clique, and the parent set ``E | T``."""
        ad = self.g.ad
        stack = [(frozenset(), 0, E)]
        while stack:
            T, start, E = stack.pop()
            yield T, E
            for i in range(start, len(t0)):
                z = t0[i]
                if T <= ad[z] and na <= ad[z]:
                    stack.append((T | {z}, i + 1, E | {z}))

    def find_inserts_to_y(
        self, y: int, only_x: int | None = None, out: list[tuple] | None = None
    ) -> None:
        """Queue each improving Insert into ``y`` (from ``only_x`` alone, when given).

        With ``out``, append every Insert from ``only_x`` to ``out`` instead, improving or not.
        With ``cap``, an Insert whose parent set ``E`` already holds ``cap`` members is left out.
        """
        g, cap = self.g, self.cap
        ne_y = g.ne[y]
        free = ne_y  # the undirected neighbors T may hold
        if self.into is not None:
            free = ne_y - self.into[y]
        if only_x is not None:
            near = {only_x}
        else:
            # A source adjacent to no neighbor of y has NA empty and T ranging over the cliques in
            # Ne(y), so those sources share their (T, E) pairs and are scored as one array.
            near = set().union(*(g.ad[z] for z in ne_y)) - g.ad[y] - {y}
            outside = np.ones(self.d, dtype=bool)
            outside[list(near | g.ad[y] | {y})] = False
            if self.constraints is not None:
                outside &= ~self.constraints.forbidden[:, y]
            far = np.flatnonzero(outside)
            if far.size:
                for T, E in self.subsets(sorted(free), set(), frozenset(g.pa[y])):
                    if cap is not None and len(E) >= cap:
                        continue
                    deltas = self.add_deltas(y, E)[far]
                    keep = deltas < -_EPS_IMPROVE
                    for x, dl in zip(far[keep].tolist(), deltas[keep].tolist(), strict=True):
                        heapq.heappush(self.inserts, (dl, next(self.seq), x, y, T, E))
        if self.into is not None:
            near -= self.into[y]
        for x in near:
            na = ne_y & g.ad[x]
            if not g.is_clique(na):
                continue
            for T, E in self.subsets(sorted(free - g.ad[x]), na, frozenset(na | g.pa[y])):
                if cap is not None and len(E) >= cap:
                    continue
                delta = float(self.add_deltas(y, E)[x])
                if out is not None:
                    out.append((delta, x, y, T, E))
                elif delta < -_EPS_IMPROVE:
                    heapq.heappush(self.inserts, (delta, next(self.seq), x, y, T, E))

    def drop_delta(self, y: int, E: frozenset[int], x: int) -> float:
        """Score change of removing parent ``x`` of ``y`` from ``E``, from two memoized scores."""
        score = self.add_deltas.score
        return _local(score, y, E - {x}) - _local(score, y, E)

    def find_delete(self, y: int, x: int, out: list[tuple] | None = None) -> None:
        """Queue each improving Delete of the edge ``x -> y`` or ``x -- y``.

        With ``out``, push every Delete onto the heap ``out`` instead, improving or not.
        """
        g = self.g
        if y not in g.ch[x] and y not in g.ne[x]:
            return
        h0 = g.ne[y] & g.ad[x]
        E = frozenset(g.pa[y] | {x})
        pinned: set[int] = set()  # the neighbors every C holds
        if self.constraints is not None:
            if self.constraints.required[x, y] or self.constraints.required[y, x]:
                return
            pinned = _pinned(h0, x, y, self.constraints)
            if not g.is_clique(pinned):
                return
            h0, E = h0 - pinned, E | pinned
        for C, E in self.subsets(sorted(h0), pinned, E):
            if pinned:
                C = C | pinned
            delta = self.drop_delta(y, E, x)
            entry = (delta, next(self.seq), x, y, C, E)
            if out is not None:
                heapq.heappush(out, entry)
            elif delta < -_EPS_IMPROVE:
                heapq.heappush(self.deletes, entry)

    def find_reverse(self, y: int, x: int) -> None:
        """Queue each improving Reverse of ``y -> x``."""
        g = self.g
        if x not in g.ch[y]:
            return
        if self.constraints is not None and self.constraints.forbidden[x, y]:
            return  # x -> y is forbidden, or y -> x is required
        pa_x = frozenset(g.pa[x])
        drop = self.drop_delta(x, pa_x, y)
        inserts: list[tuple] = []
        self.find_inserts_to_y(y, x, inserts)
        for delta, _x, _y, T, E in inserts:
            delta += drop
            if delta < -_EPS_IMPROVE:
                entry = (delta, next(self.seq), x, y, T, E, pa_x)
                heapq.heappush(self.reverses, entry)

    def seed(self, adj: np.ndarray) -> None:
        """Start from the CPDAG ``adj``, the class of the required edges, and queue every Insert.

        No Delete or Reverse applies yet: each edge of ``adj`` joins a required pair.
        """
        log = _Changes()
        for x, y in zip(*np.nonzero((adj == TAIL) & (adj.T == ARROW)), strict=True):
            self.g.change(int(x), int(y), _LOW_TO_HIGH, log)
        for x, y in zip(*np.nonzero(np.triu((adj == TAIL) & (adj.T == TAIL))), strict=True):
            self.g.change(int(x), int(y), _UNDIRECTED, log)
        self.fingerprint = log.rekey(self.fingerprint)
        for y in range(self.d):
            self.find_inserts_to_y(y)

    def inserts_on_empty_graph(self) -> None:
        """Queue every improving single-edge Insert; none under a cap of 0."""
        if self.cap == 0:
            return
        empty: frozenset[int] = frozenset()
        for y in range(self.d):
            deltas = self.add_deltas(y, empty)
            keep = np.nonzero(deltas < -_EPS_IMPROVE)[0]
            self.inserts.extend(
                (dl, next(self.seq), x, y, empty, empty)
                for x, dl in zip(keep.tolist(), deltas[keep].tolist(), strict=True)
            )
        heapq.heapify(self.inserts)

    # -- updating after a move ------------------------------------------------------------------

    def update(self, log: _Changes) -> None:
        """Queue the operators that the edge changes in ``log`` can create or rescore.

        Replays the changes one at a time from the state before the move and collects, for each,
        the targets (and sources) whose operators it can affect, then rescans those.
        """
        g, paths = self.g, self.paths
        for (a, b), (old, _new) in log.items():
            g.set_state(a, b, old)

        full_insert: set[int] = set()
        partial_insert: defaultdict[int, set[int]] = defaultdict(set)
        full_delete_to: set[int] = set()
        full_delete_from: set[int] = set()
        delete_pairs: set[tuple[int, int]] = set()
        full_reverse_to: set[int] = set()
        full_reverse_from: set[int] = set()
        reverse_pairs: set[tuple[int, int]] = set()

        def pop_paths(*keys: tuple[int, int]) -> None:
            for key in keys:
                if key in paths:
                    reverse_pairs.update(paths.pop(key))

        def requeue_paths(*keys: tuple[int, int]) -> None:
            for key in keys:
                for x, y in paths.get(key, ()):
                    partial_insert[y].add(x)

        for (lo, hi), (old, new) in log.items():
            if _is_directed(old):
                a, b = (lo, hi) if old == _LOW_TO_HIGH else (hi, lo)
            elif _is_directed(new):
                a, b = (lo, hi) if new == _LOW_TO_HIGH else (hi, lo)
            else:
                a, b = lo, hi
            kind = _change_kind(old, new)

            # Inserts.
            if kind in (_ADD_UNDIRECTED, _ADD_DIRECTED):
                if kind == _ADD_UNDIRECTED:
                    full_insert.add(a)
                full_insert.add(b)
                full_insert.update(g.ne[a] & g.ne[b])
                for t in g.ne[b]:
                    partial_insert[t].add(a)
                for t in g.ne[a]:
                    partial_insert[t].add(b)
            elif kind == _DROP_UNDIRECTED:
                for t in g.ne[b]:
                    if t != a:
                        partial_insert[t].add(a)
                partial_insert[b].add(a)
                partial_insert[a].update(g.ad[b])
                for t in g.ne[a]:
                    if t != b:
                        partial_insert[t].add(b)
                partial_insert[a].add(b)
                partial_insert[b].update(g.ad[a])
                requeue_paths((a, b), (b, a))
            elif kind == _ORIENT:
                partial_insert[a].update(g.ad[b])
                full_insert.add(b)
                requeue_paths((b, a))
            elif kind == _DROP_DIRECTED:
                for t in g.ne[b]:
                    partial_insert[t].add(a)
                partial_insert[b].add(a)
                for t in g.ne[a]:
                    partial_insert[t].add(b)
                partial_insert[a].add(b)
                full_insert.add(b)
                requeue_paths((a, b))
            elif kind == _UNORIENT:
                full_insert.update((a, b))
            else:  # _REVERSE
                full_insert.update((a, b))
                requeue_paths((a, b))

            # Deletes.
            if kind in (_ADD_UNDIRECTED, _ADD_DIRECTED):
                if kind == _ADD_UNDIRECTED:
                    full_delete_to.add(a)
                full_delete_to.add(b)
                full_delete_from.update((a, b))
                common = g.ad[a] & g.ad[b]
                if common:
                    targets = g.ne[a] & g.ne[b]
                    delete_pairs.update((x, y) for x in common for y in targets)
            elif kind in (_ORIENT, _DROP_DIRECTED):
                full_delete_to.add(b)
            elif kind in (_UNORIENT, _REVERSE):
                full_delete_to.update((a, b))

            # Reverses.
            if kind in (_ADD_UNDIRECTED, _ADD_DIRECTED):
                if kind == _ADD_UNDIRECTED:
                    full_reverse_to.add(a)
                full_reverse_to.add(b)
                full_reverse_to.update(g.ne[a] & g.ne[b])
                full_reverse_from.update((a, b))
            elif kind == _DROP_UNDIRECTED:
                full_reverse_to.update((a, b))
                full_reverse_from.update((a, b))
                pop_paths((a, b), (b, a))
            elif kind == _ORIENT:
                full_reverse_to.update((a, b))
                full_reverse_from.add(b)
                pop_paths((b, a))
            elif kind == _DROP_DIRECTED:
                full_reverse_to.add(b)
                full_reverse_from.update((a, b))
                pop_paths((a, b))
            elif kind == _UNORIENT:
                full_reverse_to.update((a, b))
                full_reverse_from.add(b)
            else:  # _REVERSE
                full_reverse_to.update((a, b))
                full_reverse_from.update((a, b))
                pop_paths((a, b))

            g.set_state(lo, hi, new)

        for y in full_insert:
            partial_insert.pop(y, None)
        for y, xs in partial_insert.items():
            for x in xs:
                if x != y and x not in g.ad[y]:
                    self.find_inserts_to_y(y, x)
        for y in full_insert:
            self.find_inserts_to_y(y)

        for x in full_delete_from:
            delete_pairs.update((x, y) for y in g.ne[x] | g.ch[x])
        for y in full_delete_to:
            delete_pairs.update((x, y) for x in g.pa[y] | g.ne[y])
        for x, y in delete_pairs:
            if x != y:
                self.find_delete(y, x)

        for x in full_reverse_from:
            reverse_pairs.update((x, y) for y in g.pa[x])
        for y in full_reverse_to:
            reverse_pairs.update((x, y) for x in g.ch[y])
        for x, y in reverse_pairs:
            if x != y:
                self.find_reverse(y, x)

    # -- The base search and the extended search ------------------------------------------------

    def base_search(self) -> None:
        """Apply the best improving Delete, else Reverse, else Insert, until none is left.

        A move back to a graph this run has visited is undone and dropped. On a singular parent set
        the Schur update and two full scores can disagree, so each move of a cycle can look
        improving; skipping revisits ends the run on any data. A visited graph with the move's
        fingerprint is a revisit only when the moves made since it, and this one, cancel out.
        """
        g, paths = self.g, self.paths
        moves: list[_Changes] = []
        visits = {self.fingerprint: [0]}  # fingerprint -> len(moves) at each graph with it
        pop, gap = heapq.heappop, self.tie_gap
        while self.deletes or self.reverses or self.inserts:
            log = _Changes()
            heap = self.deletes or self.reverses or self.inserts
            entry = pop(heap)
            if heap and heap[0][0] - entry[0] < gap:
                entry = self.tied(heap, entry)
            if heap is self.deletes:
                _, _, x, y, C, E = entry
                if not _delete_valid(g, x, y, C, E, self.constraints):
                    continue
                _apply_delete(g, x, y, C, log)
            elif heap is self.reverses:
                _, _, x, y, T, E, pa_x = entry
                if g.pa[x] != pa_x or not _insert_valid(g, x, y, T, E, paths, reverse=True):
                    continue
                _apply_reverse(g, x, y, T, log)
            else:
                _, _, x, y, T, E = entry
                if not _insert_valid(g, x, y, T, E, paths):
                    continue
                _apply_insert(g, x, y, T, log)
            fingerprint = log.rekey(self.fingerprint)
            if any(_returns(chain(moves[k:], [log])) for k in visits.get(fingerprint, ())) or (
                self.constraints is not None and not admits(g.marks(), self.constraints)
            ):
                for (a, b), (old, _new) in log.items():
                    g.set_state(a, b, old)
                continue
            # The history grows with the pairs the moves change, about 1.3 per move: 4,190 records
            # (2.2 MB, under 1% of peak memory) for a search on 1000 variables at mean degree 4.
            moves.append(log)
            visits.setdefault(fingerprint, []).append(len(moves))
            self.fingerprint = fingerprint
            self.update(log)

    def extend(self, n_samples: int) -> float:
        """Force each Delete of the converged graph, rerun the base search, keep a lower score, and
        return the final graph's total of :func:`_extension_scores`.

        Deletes are tried best first. After a branch is kept, the remaining Deletes of the earlier
        graph are still tried where they apply; the Deletes are listed again once they run out, and
        the search stops when a fresh list is exhausted with no branch kept. Branches are compared
        by the sum of :func:`_extension_scores`, a function of the graph, so every kept branch
        lowers it and no graph is kept twice. A branch rescores only the chain components of the
        nodes whose parents or neighbors it changed; the other nodes keep their extension parents.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            tolerance = _ACCEPT_PER_SAMPLE * n_samples
            score = self.add_deltas.score
            scores = _extension_scores(self.g.pa, self.g.ne, range(self.d), score)
            local = [scores[v] for v in range(self.d)]
            total = sum(local)
            pending: list[tuple] = []
            fresh = False
            snapshot: tuple[_Graph, _Paths] | None = None  # copies of self.g and self.paths
            while pending or not fresh:
                if not pending:
                    for y in range(self.d):
                        for x in self.g.pa[y] | self.g.ne[y]:
                            self.find_delete(y, x, pending)
                    if not pending:
                        return total
                    fresh = True
                entry = heapq.heappop(pending)
                if pending and pending[0][0] - entry[0] < self.tie_gap:
                    entry = self.tied(pending, entry)
                _, _, x, y, C, E = entry
                if not _delete_valid(self.g, x, y, C, E, self.constraints):
                    continue
                if snapshot is None:
                    snapshot = self.g.copy(), self.paths.copy()
                trial = self.branch(snapshot)
                log = _Changes()
                _apply_delete(trial.g, x, y, C, log)
                if self.constraints is not None and not admits(trial.g.marks(), self.constraints):
                    continue
                trial.g.forbidden.add((x, y))
                trial.fingerprint = log.rekey(trial.fingerprint)
                trial.update(log)
                trial.base_search()
                g, h = self.g, trial.g
                # A node the branch never changed keeps its parents and neighbors.
                changed = [v for v in sorted(h.copied) if h.pa[v] != g.pa[v] or h.ne[v] != g.ne[v]]
                trial_local = local.copy()
                for v, value in _extension_scores(h.pa, h.ne, changed, score).items():
                    trial_local[v] = value
                trial_total = sum(trial_local)
                if trial_total < total - tolerance:
                    h.detach()
                    trial.paths.detach()
                    self.g, self.paths, self.fingerprint = h, trial.paths, trial.fingerprint
                    snapshot = None
                    local, total = trial_local, trial_total
                    fresh = False
            return total


def xges(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    extended_search: bool = True,
    max_parents: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Extremely Greedy Equivalence Search over a decomposable score.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Rows are observations; columns are variables.
    score_func : str
        Local score: ``"bic"`` (linear-Gaussian BIC), ``"bdeu"`` or ``"discrete_bic"`` (BDeu or
        the multinomial BIC on category codes), or a mixed-data BIC, ``"degen_gauss"`` or
        ``"cond_gauss"`` (:func:`~andrey.core.score.make_score`).
    penalty_discount : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    extended_search : bool
        Run the extended search after the base search; ``False`` stops after the base search.
    max_parents : int or None
        A cap on the most parents any DAG of the class gives a node. No Insert or Reverse that
        gives its target more than ``max_parents`` parents is queued; every DAG of a class has the
        same largest number of parents, so the cap holds for the whole class after every move, and
        a Delete never breaks it. ``None`` sets no cap. With a cap, ``constraints`` may not require
        edges that already break it.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.
    constraints : Constraints or None
        Background knowledge. No operator adds a forbidden edge or removes or reverses a required
        one, a move whose class holds no DAG that satisfies the knowledge is undone, and the
        search starts from the class of the required edges. The result is that class narrowed by
        the knowledge (:func:`~andrey.core.knowledge.mpdag`).

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total score (lower is better).

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not one of :data:`~andrey.core.score.SCORES`.
    ValueError
        If the required edges give a node more than ``max_parents`` parents.
    """
    if score_func not in SCORES:
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use one of {SCORES}")
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score = make_score(
        score_func,
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        categorical=categorical,
    )
    columns = np.full(d, score_func in DISCRETE_SCORES)
    if score_func in MIXED_SCORES:
        columns = column_mask(categorical, d)
    ranks = tie_ranks(X, columns).tolist()
    # One BLAS thread for reproducible scores and move ties.
    start = None if constraints is None else _reform_cpdag(required_dag(constraints))
    if max_parents is not None and start is not None and exceeds_parent_cap(start, max_parents):
        raise ValueError(
            f"the required edges give a node more than max_parents={max_parents} parents "
            "in some graph of their class"
        )
    with backend.one_blas_thread():
        search = _Search(score, d, ranks, constraints, max_parents)
        if start is None:
            search.inserts_on_empty_graph()
        else:
            search.seed(start)
        search.base_search()
        if extended_search:
            search.extend(X.shape[0])
        adj = search.g.marks()
        total = _recompute_objective(adj, score, d)
    if constraints is not None:
        adj = mpdag(adj, constraints)
    return to_structure(adj, kind="cpdag"), total
