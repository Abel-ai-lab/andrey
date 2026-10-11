"""Extremely Greedy Equivalence Search over the linear-Gaussian BIC, the engine behind
:func:`andrey.xges`.

XGES searches the CPDAGs that GES searches, with GES's Insert and Delete operators and the Reverse
operator of GES's turning phase, but interleaves them. Each step applies the best improving Delete
if there is one, else the best improving Reverse, else the best improving Insert, and the search
stops when no operator improves the score (XGES-0). The extended search then forces each Delete of
the converged CPDAG in turn, forbids re-inserting the deleted edge, reruns XGES-0 from there, and
keeps the result when its score is lower.

:class:`_Search` keeps every candidate operator in a priority queue per operator type. A popped
operator is checked against the current CPDAG and dropped when it no longer applies; after a move,
only the operators the move's edge changes can affect are rescored, and the CPDAG is completed
locally around the changed edges. The update rules and the local completion follow the authors'
Python package (``xges`` 0.1.6). :func:`_rescan_search` is the same search with every operator
rescanned from :mod:`andrey.search.operators` after each move and the CPDAG re-formed globally; the
tests compare the two.

Scores are deltas of the deviance :class:`~andrey.core.score.BICScore` minimizes, so a move improves
when its delta is negative. The package scores in log-likelihood units, ``-0.5`` times Andrey's
plus terms that cancel in every delta, so its ``alpha`` is Andrey's ``lambda_value``.

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
from array import array
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterable
from itertools import chain, count

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.core import ARROW, NULL, TAIL, GraphStructure
from andrey.core.orient import to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import add_deltas
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.ges import _reach_factory, _recompute_objective, _reform_cpdag

# The extended search keeps a branch when it lowers the deviance by more than this times the sample
# count: the package's 1e-7 * n in log-likelihood units, which are -0.5 times the deviance.
_ACCEPT_PER_SAMPLE = 2e-7

# The state of the pair (a, b), a < b.
_NONE, _UNDIRECTED, _LOW_TO_HIGH, _HIGH_TO_LOW = 0, 1, 2, 3

# What an edge change does, keyed by (old state kind, new state kind); the update rules are written
# per change, following the package's numbering.
_ADD_UNDIRECTED, _ADD_DIRECTED, _DROP_UNDIRECTED, _ORIENT, _DROP_DIRECTED, _UNORIENT, _REVERSE = (
    range(1, 8)
)

PathMap = defaultdict[tuple[int, int], set[tuple[int, int]]]


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


class _History:
    """The net pair changes of each accepted move, in order, in flat arrays of 10 bytes a change."""

    def __init__(self) -> None:
        self.a, self.b = array("i"), array("i")
        self.old, self.new = array("b"), array("b")

    def __len__(self) -> int:
        return len(self.a)

    def append(self, log: _Changes) -> None:
        for (a, b), (old, new) in log.items():
            self.a.append(a)
            self.b.append(b)
            self.old.append(old)
            self.new.append(new)

    def returns(self, start: int, log: _Changes) -> bool:
        """Whether the changes from ``start`` on, then ``log``, leave every pair as at ``start``."""
        net = _Changes()
        since = zip(self.a[start:], self.b[start:], self.old[start:], self.new[start:])
        for a, b, old, new in since:
            net.record(a, b, old, new)
        for (a, b), (old, new) in log.items():
            net.record(a, b, old, new)
        return all(old == new for old, new in net.values())


class _Graph:
    """A PDAG as per-node sets of parents, children, undirected neighbors, and adjacent nodes.

    ``forbidden`` holds the ``(x, y)`` pairs whose Insert ``x -> y`` the extended search forbids.
    """

    __slots__ = ("pa", "ch", "ne", "ad", "forbidden")

    def __init__(self, d: int) -> None:
        self.pa: list[set[int]] = [set() for _ in range(d)]
        self.ch: list[set[int]] = [set() for _ in range(d)]
        self.ne: list[set[int]] = [set() for _ in range(d)]
        self.ad: list[set[int]] = [set() for _ in range(d)]
        self.forbidden: set[tuple[int, int]] = set()

    def copy(self) -> _Graph:
        out = _Graph.__new__(_Graph)
        out.pa = [set(s) for s in self.pa]
        out.ch = [set(s) for s in self.ch]
        out.ne = [set(s) for s in self.ne]
        out.ad = [set(s) for s in self.ad]
        out.forbidden = set(self.forbidden)
        return out

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
    pa: list[set[int]], ne: list[set[int]], nodes: Iterable[int], score: BICScore
) -> dict[int, float]:
    """The memoized local score of each node :func:`_extension_parents` covers.

    The sum over all nodes is the CPDAG's total score. Unlike a sum of the search's deltas it does
    not depend on the moves that reached the graph, even where a singular parent set makes the Schur
    update and two full scores disagree.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return {v: score.score(v, tuple(ps)) for v, ps in _extension_parents(pa, ne, nodes).items()}


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
    g: _Graph, y: int, x: int, blocked: set[int], paths: PathMap, skip_direct: bool
) -> bool:
    """True when every semi-directed path from ``y`` to ``x`` passes through ``blocked``.

    ``skip_direct`` ignores the edge ``y -> x`` itself (a Reverse). An open path is recorded in
    ``paths``: each of its edges maps to ``(x, y)``, so a change to that edge requeues the Insert.
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
                    paths[(parent[cur], cur)].add((x, y))
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
    paths: PathMap,
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


def _delete_valid(g: _Graph, x: int, y: int, C: frozenset[int], E: frozenset[int]) -> bool:
    """Whether Delete ``x - y`` keeping ``C`` applies to ``g`` and still has effective parents
    ``E``.
    """
    if y not in g.ch[x] and y not in g.ne[x]:
        return False
    if not (C <= g.ne[y] and C <= g.ad[x]):
        return False
    if E != C | g.pa[y] | {x}:
        return False
    return g.is_clique(C)


# ---- the search ---------------------------------------------------------------------------------


# Bytes the cached per-(target, parent set) score arrays may hold; each is one float per variable.
_ADD_DELTAS_BYTES = 256 * 1024**2


class _AddDeltas:
    """Bounded LRU cache of :func:`~andrey.core.score_delta.add_deltas` per ``(y, E)``.

    ``self(y, E)[x]`` is the score change of adding ``x`` to ``y``'s parents ``E``.
    """

    def __init__(self, score: BICScore, d: int) -> None:
        self.score = score
        self.cap = max(1024, _ADD_DELTAS_BYTES // (8 * max(d, 1)))
        self.cache: OrderedDict[tuple[int, frozenset[int]], np.ndarray] = OrderedDict()

    def __call__(self, y: int, E: frozenset[int]) -> np.ndarray:
        key = (y, E)
        deltas = self.cache.get(key)
        if deltas is None:
            deltas = add_deltas(self.score, y, E)
            self.cache[key] = deltas
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return deltas


class _Search:
    """XGES over a :class:`_Graph`, with a priority queue per operator type.

    Queue entries are ``(delta, seq, x, y, subset, E[, Pa(x)])``: ``seq`` breaks ties in the order
    the operators were found, ``subset`` is an Insert's or Reverse's ``T`` and a Delete's ``C``,
    and ``E`` the target's parent set the delta was scored with.
    """

    def __init__(self, score: BICScore, d: int) -> None:
        self.add_deltas = _AddDeltas(score, d)
        self.d = d
        self.g = _Graph(d)
        self.paths: PathMap = defaultdict(set)
        self.fingerprint = 0  # the XOR of _edge_key over the edges of g
        self.inserts: list[tuple] = []
        self.reverses: list[tuple] = []
        self.deletes: list[tuple] = []
        self.seq = count()

    def branch(self) -> _Search:
        """A copy of the graph and path records, with empty queues and the same score cache."""
        out = _Search.__new__(_Search)
        out.add_deltas, out.d, out.seq = self.add_deltas, self.d, self.seq
        out.g = self.g.copy()
        out.paths = defaultdict(set, {k: set(v) for k, v in self.paths.items()})
        out.fingerprint = self.fingerprint
        out.inserts, out.reverses, out.deletes = [], [], []
        return out

    # -- finding operators ----------------------------------------------------------------------

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
        """
        g = self.g
        ne_y = g.ne[y]
        # A source adjacent to no neighbor of y has NA empty and T ranging over the cliques in
        # Ne(y), so those sources share their (T, E) pairs and are scored as one array.
        if only_x is not None:
            near = set() if ne_y.isdisjoint(g.ad[only_x]) else {only_x}
            far = np.array([] if near else [only_x], dtype=np.intp)
        else:
            near = set().union(*(g.ad[z] for z in ne_y)) - g.ad[y] - {y}
            outside = np.ones(self.d, dtype=bool)
            outside[list(near | g.ad[y] | {y})] = False
            far = np.flatnonzero(outside)
        if far.size:
            for T, E in self.subsets(sorted(ne_y), set(), frozenset(g.pa[y])):
                deltas = self.add_deltas(y, E)[far]
                if out is not None:
                    out.extend((dl, x, y, T, E) for x, dl in zip(far.tolist(), deltas.tolist()))
                    continue
                keep = deltas < -_EPS_IMPROVE
                for x, dl in zip(far[keep].tolist(), deltas[keep].tolist(), strict=True):
                    heapq.heappush(self.inserts, (dl, next(self.seq), x, y, T, E))
        for x in near:
            na = ne_y & g.ad[x]
            if not g.is_clique(na):
                continue
            for T, E in self.subsets(sorted(ne_y - g.ad[x]), na, frozenset(na | g.pa[y])):
                delta = float(self.add_deltas(y, E)[x])
                if out is not None:
                    out.append((delta, x, y, T, E))
                elif delta < -_EPS_IMPROVE:
                    heapq.heappush(self.inserts, (delta, next(self.seq), x, y, T, E))

    def drop_delta(self, y: int, E: frozenset[int], x: int) -> float:
        """Score change of removing parent ``x`` of ``y`` from ``E``."""
        return -float(self.add_deltas(y, E - {x})[x])

    def find_delete(self, y: int, x: int, out: list[tuple] | None = None) -> None:
        """Queue each improving Delete of the edge ``x -> y`` or ``x -- y``.

        With ``out``, push every Delete onto the heap ``out`` instead, improving or not.
        """
        g = self.g
        if y not in g.ch[x] and y not in g.ne[x]:
            return
        for C, E in self.subsets(sorted(g.ne[y] & g.ad[x]), set(), frozenset(g.pa[y] | {x})):
            delta = self.drop_delta(y, E, x)
            if out is not None:
                heapq.heappush(out, (delta, next(self.seq), x, y, C, E))
            elif delta < -_EPS_IMPROVE:
                heapq.heappush(self.deletes, (delta, next(self.seq), x, y, C, E))

    def find_reverse(self, y: int, x: int) -> None:
        """Queue each improving Reverse of ``y -> x``."""
        g = self.g
        if x not in g.ch[y]:
            return
        pa_x = frozenset(g.pa[x])
        drop = self.drop_delta(x, pa_x, y)
        inserts: list[tuple] = []
        self.find_inserts_to_y(y, x, inserts)
        for delta, _x, _y, T, E in inserts:
            delta += drop
            if delta < -_EPS_IMPROVE:
                heapq.heappush(self.reverses, (delta, next(self.seq), x, y, T, E, pa_x))

    def inserts_on_empty_graph(self) -> None:
        """Queue every improving single-edge Insert."""
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

    # -- XGES-0 and the extended search ---------------------------------------------------------

    def xges0(self) -> None:
        """Apply the best improving Delete, else Reverse, else Insert, until none is left.

        A move back to a graph this run has visited is undone and dropped. On a singular parent set
        the Schur update and two full scores can disagree, so each move of a cycle can look
        improving; skipping revisits ends the run on any data. A visited graph with the move's
        fingerprint is a revisit only when the moves made since it, and this one, cancel out.
        """
        g, paths = self.g, self.paths
        history = _History()
        visits = {self.fingerprint: [0]}  # fingerprint -> len(history) at each graph with it
        while self.deletes or self.reverses or self.inserts:
            log = _Changes()
            if self.deletes:
                _, _, x, y, C, E = heapq.heappop(self.deletes)
                if not _delete_valid(g, x, y, C, E):
                    continue
                _apply_delete(g, x, y, C, log)
            elif self.reverses:
                _, _, x, y, T, E, pa_x = heapq.heappop(self.reverses)
                if g.pa[x] != pa_x or not _insert_valid(g, x, y, T, E, paths, reverse=True):
                    continue
                _apply_reverse(g, x, y, T, log)
            else:
                _, _, x, y, T, E = heapq.heappop(self.inserts)
                if not _insert_valid(g, x, y, T, E, paths):
                    continue
                _apply_insert(g, x, y, T, log)
            fingerprint = log.rekey(self.fingerprint)
            if any(history.returns(k, log) for k in visits.get(fingerprint, ())):
                for (a, b), (old, _new) in log.items():
                    g.set_state(a, b, old)
                continue
            history.append(log)
            visits.setdefault(fingerprint, []).append(len(history))
            self.fingerprint = fingerprint
            self.update(log)

    def extend(self, n_samples: int) -> float:
        """Force each Delete of the converged graph, rerun XGES-0, keep a lower score, and return
        the final graph's total of :func:`_extension_scores`.

        Deletes are tried best first. After a branch is kept, the remaining Deletes of the earlier
        graph are still tried where they apply; the Deletes are listed again once they run out, and
        the search stops when a fresh list is exhausted with no branch kept. Branches are compared
        by the sum of :func:`_extension_scores`, a function of the graph, so every kept branch
        lowers it and no graph is kept twice. A branch rescores only the chain components of the
        nodes whose parents or neighbors it changed; the other nodes keep their extension parents.
        No branch scores below a total of nan or -inf, so then none is tried.
        """
        tolerance = _ACCEPT_PER_SAMPLE * n_samples
        score = self.add_deltas.score
        scores = _extension_scores(self.g.pa, self.g.ne, range(self.d), score)
        local = [scores[v] for v in range(self.d)]
        total = sum(local)
        if np.isnan(total) or total == -np.inf:
            return total
        pending: list[tuple] = []
        fresh = False
        while pending or not fresh:
            if not pending:
                for y in range(self.d):
                    for x in self.g.pa[y] | self.g.ne[y]:
                        self.find_delete(y, x, pending)
                if not pending:
                    return total
                fresh = True
            _, _, x, y, C, E = heapq.heappop(pending)
            if not _delete_valid(self.g, x, y, C, E):
                continue
            trial = self.branch()
            log = _Changes()
            _apply_delete(trial.g, x, y, C, log)
            trial.g.forbidden.add((x, y))
            trial.fingerprint = log.rekey(trial.fingerprint)
            trial.update(log)
            trial.xges0()
            g, h = self.g, trial.g
            changed = [v for v in range(self.d) if h.pa[v] != g.pa[v] or h.ne[v] != g.ne[v]]
            trial_local = local.copy()
            for v, value in _extension_scores(h.pa, h.ne, changed, score).items():
                trial_local[v] = value
            trial_total = sum(trial_local)
            if trial_total < total - tolerance:
                self.g, self.paths, self.fingerprint = h, trial.paths, trial.fingerprint
                local, total = trial_local, trial_total
                fresh = False
        return total


# ---- the rescanning reference -------------------------------------------------------------------


def _local(score: BICScore, v: int, parents: Iterable[int]) -> float:
    """The local score of ``v`` given a parent set."""
    return score.score(v, sorted(parents))


def _sets(adj: np.ndarray, v: int) -> tuple[set[int], set[int], set[int]]:
    """Parents, undirected neighbors, and adjacent nodes of ``v`` in a mark matrix."""
    return (
        set(ops.parents(adj, v).tolist()),
        set(ops.undirected_neighbors(adj, v).tolist()),
        set(ops.adjacent(adj, v).tolist()),
    )


def _all_deletes(adj: np.ndarray, score: BICScore) -> list[tuple]:
    """Every Delete of a CPDAG as ``(delta, x, y, C, E)``, ``C`` the neighbors kept undirected."""
    out = []
    for y in range(adj.shape[0]):
        pa_y, ne_y, _ = _sets(adj, y)
        for x in sorted(pa_y | ne_y):
            h0 = sorted(ne_y & set(ops.adjacent(adj, x).tolist()))
            for _, H in ops.clique_residuals(adj, h0):
                C = frozenset(h0) - set(H)
                E = frozenset(C | pa_y | {x})
                delta = _local(score, y, E - {x}) - _local(score, y, E)
                out.append((delta, x, y, C, E))
    return out


def _all_inserts(
    adj: np.ndarray, score: BICScore, forbidden: set[tuple[int, int]], reachable: Callable
) -> list[tuple]:
    """Every valid Insert ``x -> y`` of a CPDAG as ``(delta, x, y, T)``."""
    out = []
    d = adj.shape[0]
    for y in range(d):
        pa_y, ne_y, ad_y = _sets(adj, y)
        for x in range(d):
            if x == y or x in ad_y or (x, y) in forbidden:
                continue
            ad_x = set(ops.adjacent(adj, x).tolist())
            na = ne_y & ad_x
            for _, T in ops.clique_extensions(adj, na, sorted(ne_y - ad_x)):
                if reachable(y, x, na | set(T)):
                    continue
                E = pa_y | na | set(T)
                out.append((_local(score, y, E | {x}) - _local(score, y, E), x, y, frozenset(T)))
    return out


def _all_reverses(adj: np.ndarray, score: BICScore, forbidden: set[tuple[int, int]]) -> list[tuple]:
    """Every valid Reverse of a ``y -> x`` in a CPDAG as ``(delta, x, y, T)``."""
    out = []
    for x in range(adj.shape[0]):
        pa_x, ne_x, ad_x = _sets(adj, x)
        for y in sorted(pa_x):
            if (x, y) in forbidden:
                continue
            cut = adj.copy()
            cut[x, y] = cut[y, x] = NULL
            succ = ops.tail_successor_lists(cut)
            pa_y, ne_y, _ = _sets(adj, y)
            na = ne_y & ad_x
            drop = _local(score, x, pa_x - {y}) - _local(score, x, pa_x)
            for _, T in ops.clique_extensions(adj, na, sorted(ne_y - ad_x)):
                if not ops.blocks_semi_directed_paths(succ, y, x, na | set(T) | ne_x):
                    continue
                E = pa_y | na | set(T)
                delta = _local(score, y, E | {x}) - _local(score, y, E) + drop
                out.append((delta, x, y, frozenset(T)))
    return out


def _improving(candidates: list[tuple]) -> list[tuple]:
    """The improving candidates, lowest delta first (in the order found among equals)."""
    return sorted((c for c in candidates if c[0] < -_EPS_IMPROVE), key=lambda c: c[0])


def _delete_marks(adj: np.ndarray, x: int, y: int, C: frozenset[int]) -> np.ndarray:
    """Apply Delete ``x - y`` keeping ``C`` to a mark matrix and re-form the CPDAG."""
    adj = adj.copy()
    _, ne_y, _ = _sets(adj, y)
    H = (ne_y & set(ops.adjacent(adj, x).tolist())) - C
    ops.apply_delete(adj, x, y, H)
    return _reform_cpdag(adj)


def _insert_marks(adj: np.ndarray, x: int, y: int, T: frozenset[int]) -> np.ndarray:
    """Apply Insert ``x -> y`` with ``T`` to a copy of a mark matrix and re-form the CPDAG."""
    adj = adj.copy()
    ops.apply_insert(adj, x, y, T)
    return _reform_cpdag(adj)


def _rescan_moves(
    adj: np.ndarray, score: BICScore, forbidden: set[tuple[int, int]], reach: Callable
) -> Iterable[np.ndarray]:
    """The graphs XGES-0's improving moves on ``adj`` lead to, in the order it tries them."""
    for _, x, y, C, _E in _improving(_all_deletes(adj, score)):
        yield _delete_marks(adj, x, y, C)
    for _, x, y, T in _improving(_all_reverses(adj, score, forbidden)):
        cut = adj.copy()
        cut[x, y] = cut[y, x] = NULL
        yield _insert_marks(cut, x, y, T)
    for _, x, y, T in _improving(_all_inserts(adj, score, forbidden, reach(adj))):
        yield _insert_marks(adj, x, y, T)


def _rescan_xges0(adj: np.ndarray, score: BICScore, forbidden: set[tuple[int, int]]) -> np.ndarray:
    """XGES-0 with every operator rescanned after each move, skipping moves to visited graphs."""
    build_state = _reach_factory(adj.shape[0])
    seen = {adj.tobytes()}
    while True:
        for moved in _rescan_moves(adj, score, forbidden, build_state):
            if moved.tobytes() not in seen:
                seen.add(moved.tobytes())
                adj = moved
                break
        else:
            return adj


def _marks_total(adj: np.ndarray, score: BICScore) -> float:
    """The total of :func:`_extension_scores` for a CPDAG given as a mark matrix."""
    d = adj.shape[0]
    sets = [_sets(adj, v) for v in range(d)]
    scores = _extension_scores([s[0] for s in sets], [s[1] for s in sets], range(d), score)
    return sum(scores[v] for v in range(d))


def _rescan_search(score: BICScore, d: int, extended: bool) -> np.ndarray:
    """XGES with every operator rescanned and the CPDAG re-formed globally after each move."""
    adj = _rescan_xges0(np.zeros((d, d), dtype=np.int8), score, set())
    if not extended:
        return adj
    tolerance = _ACCEPT_PER_SAMPLE * score.n
    total = _marks_total(adj, score)
    if np.isnan(total) or total == -np.inf:
        return adj
    forbidden: set[tuple[int, int]] = set()
    pending: list[tuple] = []
    fresh = False
    while pending or not fresh:
        if not pending:
            pending = sorted(_all_deletes(adj, score), key=lambda c: c[0])
            if not pending:
                break
            fresh = True
        _, x, y, C, E = pending.pop(0)
        pa_y, ne_y, _ = _sets(adj, y)
        ad_x = set(ops.adjacent(adj, x).tolist())
        valid = (x in pa_y or x in ne_y) and C <= ne_y & ad_x and E == C | pa_y | {x}
        if not (valid and ops.is_clique(adj, C)):
            continue
        trial_forbidden = forbidden | {(x, y)}
        trial = _rescan_xges0(_delete_marks(adj, x, y, C), score, trial_forbidden)
        trial_total = _marks_total(trial, score)
        if trial_total < total - tolerance:
            adj, total, forbidden = trial, trial_total, trial_forbidden
            fresh = False
    return adj


def xges(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    extended_search: bool = True,
    _rescan: bool = False,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Extremely Greedy Equivalence Search over the linear-Gaussian BIC.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score; ``"local_score_BIC"`` (linear-Gaussian BIC) is supported.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes; the package's
        ``alpha``.
    extended_search : bool
        Run the extended search after XGES-0 (XGES); ``False`` stops at XGES-0.
    _rescan : bool
        Private test lever: rescan every operator after each move instead of updating the queues.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total BIC score (lower is better).
    """
    if score_func != "local_score_BIC":
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use 'local_score_BIC'")
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score = BICScore(X, lambda_value=lambda_value)
    # One BLAS thread keeps every score, and so every tie between moves, reproducible.
    with threadpool_limits(limits=1):
        if _rescan:
            adj = _rescan_search(score, d, extended_search)
        else:
            search = _Search(score, d)
            search.inserts_on_empty_graph()
            search.xges0()
            if extended_search:
                search.extend(X.shape[0])
            adj = search.g.marks()
        total = _recompute_objective(adj, score, d)
    return to_structure(adj, kind="cpdag"), total
