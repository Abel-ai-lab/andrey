"""Parallel GES candidate scans using :mod:`andrey.search._parallel`.

Each greedy pass scans Insert or Delete candidates and applies the best improving move. For each
target, candidates sharing ``base = sorted(NA u T u Pa)`` are scored in one batched Schur
``delta_many_with_base`` call. Fixed target ownership preserves arithmetic and reuses worker
caches across passes.

The move key is :func:`move_key`: ``stable_key(delta, n)``, which quantizes ``delta / n``, then the
:func:`~andrey.search.hc.tie_ranks` of ``x``, of ``y``, and of the subset ``T`` or ``H``. Moves
with equal quantized deltas are chosen by ranks that follow the columns, so reordering the columns
reorders the CPDAG the same way. Each key is unique, so the CPDAG is byte-identical across
partitions and worker assignments.

The forward phase keeps each target's Insert candidates between passes and enumerates them again
only for the targets a move touches (:class:`_ForwardScan`); it selects the move a full scan
(:func:`forward_chunk_best`) selects.

``ges()`` uses :func:`run_ges`; ``workers=1`` scans all targets in-process. The objective is
recomputed from the final CPDAG by :func:`andrey.search.ges._recompute_objective`.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

import numpy as np

from andrey.core import _bitset, env
from andrey.core.knowledge import Constraints, admits
from andrey.core.score import BICScore, LocalScore
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta
from andrey.search import _parallel as par
from andrey.search import calibrate
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP, stable_key
from andrey.search.ges import _reachable

_pick_context = par.pick_context

# A chunk best, or ``None`` when the chunk has no improving operator. ``key`` is the total-order
# reduce key from :func:`move_key`; ``T`` / ``H`` carry the move to apply.
ForwardResult = tuple[tuple, int, int, tuple[int, ...]] | None  # (key, x, y, T)
BackwardResult = tuple[tuple, int, int, tuple[int, ...]] | None  # (key, x, y, H)
# Moves a scan skips, by the move part of their :func:`move_key`, ``key[1:]``: under background
# knowledge, the moves whose class admits no DAG that satisfies it on the current graph.
Excluded = set[tuple[int, int, tuple[int, ...]]] | None


def move_key(
    delta: float, n: int, x: int, y: int, subset: Iterable[int], ranks: Sequence[int]
) -> tuple:
    """The reduce key of the move of edge ``x``-``y`` with ``subset`` (``T`` or ``H``); lower wins.

    ``(stable_key(delta, n), ranks[x], ranks[y], sorted ranks of subset)``, with ``ranks`` from
    :func:`~andrey.search.hc.tie_ranks`. A candidate's ``(x, y, subset)`` is unique in a pass, so
    the key is too.
    """
    return (stable_key(delta, n), ranks[x], ranks[y], tuple(sorted(ranks[v] for v in subset)))


def reduce_best(
    results: Sequence[ForwardResult | BackwardResult],
) -> ForwardResult | BackwardResult:
    """Select the best chunk by its :func:`move_key` in ``r[0]``.

    Uses :func:`andrey.search._parallel.reduce_best`. Each ``(x, y, subset)`` belongs to one target
    partition; unique keys make strict-``<`` reduction independent of completion order.
    """
    return par.reduce_best(results, key=lambda r: r[0])


class DeltaEngine(Protocol):
    """A move scorer: raw local-score deltas for single add/remove moves off a shared base.

    Satisfied by the PURE Schur :class:`~andrey.core.score_delta.DeltaBICScore` (BIC) and by
    :class:`~andrey.core.score_delta.FullScoreDelta`, which differences two full scores: BIC's
    reference arm, which the quality gate pairs against the Schur path, and BDeu.
    """

    @property
    def score_obj(self) -> LocalScore:
        """The local score whose deltas this engine computes."""
        ...

    def delta_many_with_base(
        self,
        i: int,
        base_parents: Sequence[int],
        deltas: Sequence[tuple[str, int]],
        *,
        bounds: list[float] | None = None,
    ) -> list[float]:
        """Return ``[score(i, base +/- u_k) - score(i, base)]`` for the requested moves.

        ``bounds``, when given, receives per move the most its delta can differ when another batch
        over the same base scores it (:func:`andrey.core.score_delta.delta_many_with_base`).
        """
        ...


# Per-graph-state neighborhood precompute shared by every chunk of a pass.
NeighborState = tuple[list[set[int]], list[np.ndarray], list[set[int]]]


def neighborhood_state(adj: np.ndarray) -> NeighborState:
    """The per-node adjacency sets, directed parents, and undirected neighbors for a graph state.

    Computed once per graph state and shared read-only across chunks. Each chunk reads all node
    arrays but iterates only its targets ``[y0, y1)``. Node ``v``'s entries equal ``ops.adjacent``,
    ``ops.parents``, and ``ops.undirected_neighbors`` of ``v``.
    """
    adj_sets = [set(row.tolist()) for row in _rows(adj != ops.NULL)]
    par_of = _rows(adj == ops.ARROW)
    nbr = [set(row.tolist()) for row in _rows((adj == ops.TAIL) & (adj.T == ops.TAIL))]
    return adj_sets, par_of, nbr


def _rows(mask: np.ndarray) -> list[np.ndarray]:
    """The column indices of each row's ``True`` entries, ascending, one array per row."""
    rows, cols = np.nonzero(mask)
    return np.split(cols, np.searchsorted(rows, np.arange(1, mask.shape[0])))


def _insert_candidates(
    y: int, adj: np.ndarray, state: NeighborState, forbidden: np.ndarray | None = None
) -> Iterator[tuple[int, int, tuple[int, ...], set[int]]]:
    """Yield ``(x, t_index, T, clique)`` for every Insert into ``y`` whose ``NA u T`` is a clique.

    ``clique = NA u T`` is the node set every semi-directed ``y ~> x`` path must meet for the
    Insert to be valid; the caller checks that. Sources come in ascending ``x``, and each source's
    subsets in :func:`~andrey.search.operators.clique_extensions` order. With ``forbidden``
    (``[i, j]``: no edge ``i -> j``), neither ``x`` nor a member of ``T`` is forbidden into ``y``.
    """
    adj_sets, _par_of, nbr = state
    nbrs_y = nbr[y]
    sources = np.flatnonzero((adj[:, y] == ops.NULL) & (adj[y] == ops.NULL))
    free = nbrs_y  # the undirected neighbors T may hold
    if forbidden is not None:
        sources = sources[~forbidden[sources, y]]
        free = nbrs_y - set(np.flatnonzero(forbidden[:, y]).tolist())
    for x in sources.tolist():
        if x == y:
            continue
        adj_x = adj_sets[x]
        na = nbrs_y & adj_x
        t0 = sorted(free - adj_x)
        for t_index, T in ops.clique_extensions(adj, na, t0):
            yield x, t_index, T, na | set(T)


def forward_chunk_best(
    targets: Sequence[int],
    adj: np.ndarray,
    delta_engine: DeltaEngine,
    n: int,
    max_parents: float | None,
    state: NeighborState,
    reachable: Callable[[int, int, set[int]], bool],
    ranks: Sequence[int],
    forbidden: np.ndarray | None = None,
    excluded: Excluded = None,
) -> ForwardResult:
    """The best-improving Insert whose target is in ``targets``, by :func:`move_key`.

    Enumerates the chunk's valid Insert candidates score-free (clique + reachability), groups them
    per target by ``base = sorted(NA u T u Pa)``, and scores each group with one batched
    ``delta_many_with_base`` call over its ``("add", x)`` moves -- so one shared ``inv(cov[base,
    base])`` amortizes across all sources sharing a base. A candidate is improving iff its raw delta
    is below ``-_EPS_IMPROVE``; the argmin is over the quantized key, so the visitation order
    cannot change the result. Grouping is per target, and a target lives wholly in one
    chunk, so the batch shape (hence each delta) is identical however the targets are partitioned.
    ``forbidden`` filters the candidates as in :func:`_insert_candidates`; a move in ``excluded``
    is never selected.
    """
    par_of = state[1]
    best: ForwardResult = None
    best_key: tuple | None = None
    for y in targets:
        paj = par_of[y]
        if max_parents is not None and len(paj) > max_parents:
            continue  # y already carries the maximum admissible parent count.
        pa_set = set(int(p) for p in paj)
        # Group valid candidates by their base parent set; each base scores its adds in one batch.
        groups: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
        for x, _t_index, T, clique in _insert_candidates(y, adj, state, forbidden):
            if reachable(y, x, clique):
                continue  # some semi-directed y ~> x path misses the clique: invalid Insert.
            base = tuple(sorted(pa_set | clique))
            groups.setdefault(base, []).append((x, T))
        for base, cands in groups.items():
            moves = [("add", x) for x, _ in cands]
            for (x, T), delta in zip(
                cands, delta_engine.delta_many_with_base(y, base, moves), strict=True
            ):
                if delta < -_EPS_IMPROVE:
                    key = move_key(delta, n, x, y, T, ranks)
                    if best_key is None or key < best_key:
                        if excluded and key[1:] in excluded:
                            continue
                        best_key = key
                        best = (key, x, y, T)
    return best


def backward_chunk_best(
    targets: Sequence[int],
    adj: np.ndarray,
    delta_engine: DeltaEngine,
    n: int,
    state: NeighborState,
    ranks: Sequence[int],
    kept: dict[tuple[int, tuple[int, ...], int], float] | None = None,
    constraints: Constraints | None = None,
    excluded: Excluded = None,
) -> BackwardResult:
    """The best-improving Delete whose target is in ``targets``, by :func:`move_key`.

    Enumerates the chunk's valid Delete candidates score-free (residual-clique check; Delete carries
    no reachability test) and scores each with one scalar ``("remove", x)`` off ``base = sorted(
    residual u Pa u {x})`` -- the PURE Schur remove delta. Improving iff the raw delta is below
    ``-_EPS_IMPROVE``; the argmin is over the quantized key. ``kept``, when given, holds each
    ``(y, base, x)``'s delta across passes: a scalar remove depends on nothing else.

    With ``constraints``, a required pair is never deleted, and ``H`` holds no ``h`` whose
    orientation ``y -> h`` or ``x -> h`` is forbidden: such an ``h`` stays in the residual, which
    must still be a clique. A move in ``excluded`` is never selected.
    """
    adj_sets, par_of, nbr = state
    best: BackwardResult = None
    best_key: tuple | None = None
    for y in targets:
        nbrs_y = nbr[y]
        paj = par_of[y]
        pa_set = set(int(p) for p in paj)
        for x in sorted(adj_sets[y]):
            directed = adj[y, x] == ops.ARROW  # x -> y
            undirected = adj[y, x] == ops.TAIL and adj[x, y] == ops.TAIL  # x -- y
            if not (directed or undirected):
                continue
            h0 = sorted(nbrs_y & adj_sets[x])
            h0_set = set(h0)
            candidates = ops.clique_residuals(adj, h0)
            if constraints is not None:
                if constraints.required[x, y] or constraints.required[y, x]:
                    continue
                f = constraints.forbidden
                pinned = {h for h in h0 if f[y, h] or f[x, h]}
                if pinned:
                    candidates = ((i, H) for i, H in candidates if pinned.isdisjoint(H))
            for _h_index, H in candidates:
                residual = h0_set - set(H)
                base = tuple(sorted(residual | pa_set | {int(x)}))
                delta = None if kept is None else kept.get((y, base, x))
                if delta is None:
                    delta = delta_engine.delta_many_with_base(y, base, [("remove", int(x))])[0]
                    if kept is not None:
                        kept[(y, base, x)] = delta
                if delta < -_EPS_IMPROVE:
                    key = move_key(delta, n, x, y, H, ranks)
                    if best_key is None or key < best_key:
                        if excluded and key[1:] in excluded:
                            continue
                        best_key = key
                        best = (key, x, y, H)
    return best


# ---- the incremental forward scan ---------------------------------------------------------------
#
# A target's Insert candidates (each source, subset ``T``, and the base it is scored off) depend on
# the target's own marks, the adjacencies among its undirected neighbors, and which of those
# neighbors each source is adjacent to. A move can change them only when it changes a mark of the
# target or of one of the target's undirected neighbors in the new graph, so the candidates of every
# other target are kept. The reachability condition reads the whole graph, so it is checked when a
# candidate is considered for selection.
#
# The selected move is the one forward_chunk_best selects, whose deltas come from one batch per
# base of the candidates that pass the reachability check. A kept candidate carries a screening
# delta from one batch of every candidate sharing its base. The two batches round differently, by
# a few units in the last place on well-conditioned data and by any amount when the rank-one
# update cancels, so the delta engine bounds the difference per candidate. A candidate is kept
# under the lowest delta that bound allows, or minus infinity when another batch can take the
# other side of a guard. The scan visits candidates in that order and recomputes each visited
# base group with forward_chunk_best's batch, until no lower bound can beat the best exact key.
#
# A batch's deltas depend only on the target, the base, and the batch's sources, and its bounds
# hold against any other batch over the same base. So a target enumerated again keeps a base's
# last batch while that batch covers the base's sources, and a delta whose bound is 0 is exact in
# every batch. The nodes a target reaches around a clique depend only on those nodes' successors,
# so a reach set is kept while no successor change among its nodes can cut a path inside it or
# open one out of it (:func:`_kept_reach`).

# The scan the forward phase uses in-process: ``True`` runs forward_chunk_best over every target at
# every pass; ``False`` keeps each target's candidates until a move touches it. Tests flip it to
# compare the two. Pooled workers always keep their candidates.
_FULL_RESCAN = False

# A candidate kept for selection: ``(lowest exact delta, x, t_index, T, base)``.
_Screened = tuple[float, int, int, tuple[int, ...], tuple[int, ...]]

# A base's group at one target: ``(clique, sources, batch, deltas, bounds)``. ``clique`` and
# ``sources`` are bitmasks of NA u T and of the group's sources; ``deltas`` and ``bounds`` map each
# source of ``batch``, a set of sources scored in one call that covers the group's, to its
# screening delta and bound.
_Group = tuple[int, int, frozenset[int], dict[int, float], dict[int, float]]


def _touched(before: np.ndarray | None, after: np.ndarray) -> np.ndarray:
    """A mask of the targets whose Insert candidates can differ between ``before`` and ``after``.

    Those are the nodes with a changed mark and their undirected neighbors in ``after``; every node
    when there is no ``before``.
    """
    if before is None:
        return np.ones(after.shape[0], dtype=bool)
    diff = before != after
    changed = diff.any(axis=0) | diff.any(axis=1)
    lines = (after == ops.TAIL) & (after.T == ops.TAIL)
    return changed | lines[changed].any(axis=0)


def _kept_reach(
    reach: dict[tuple[int, int], int], before: list[int], after: list[int]
) -> dict[tuple[int, int], int]:
    """The entries of ``reach`` whose reach set is the same under the successor masks ``after``.

    ``reach`` maps ``(start, clique)`` to :func:`~andrey.core._bitset.semidirected_reach_set` of
    ``before`` around that clique. A reach set ``R`` stays when each of its nodes whose successors
    changed loses no successor in ``R`` and gains none outside ``R`` and the clique: every path
    inside ``R`` remains, and no path leaves it. Every other entry is dropped.
    """
    if not reach:
        return reach
    changed = 0
    for u, (old, new) in enumerate(zip(before, after, strict=True)):
        if old != new:
            changed |= 1 << u
    if not changed:
        return reach
    kept = {}
    for key, seen in reach.items():
        hit = seen & changed
        if hit:
            outside = ~(seen | key[1])
            if any(before[u] & ~after[u] & seen or after[u] & outside for u in _bitset.nodes(hit)):
                continue
        kept[key] = seen
    return kept


class _ForwardScan:
    """The forward pass over fixed targets, keeping each target's candidates between passes.

    :meth:`best` returns what :func:`forward_chunk_best` returns for the same targets and graph.
    ``forbidden`` (``[i, j]``: no edge ``i -> j``) filters the candidates as there.
    """

    def __init__(
        self,
        targets: Sequence[int],
        delta_engine: DeltaEngine,
        n: int,
        max_parents: float | None,
        ranks: Sequence[int],
        forbidden: np.ndarray | None = None,
    ) -> None:
        self.targets = list(targets)
        self.delta_engine = delta_engine
        self.n = n
        self.max_parents = max_parents
        self.ranks = ranks
        self.everyone = (1 << n) - 1
        self.forbidden = forbidden
        # Per target, the nodes forbidden into it, as bitmasks; None without knowledge.
        self.into = None if forbidden is None else _bitset.row_masks(forbidden.T)
        self.adj: np.ndarray | None = None  # the graph the kept candidates were enumerated on
        # Per target: each base's group; the candidates that can improve, sorted by lowest exact
        # delta; and each base's last exact deltas with a mask of the passing sources they were
        # scored for.
        self.groups: dict[int, dict[tuple[int, ...], _Group]] = {}
        self.screened: dict[int, list[_Screened]] = {}
        self.exact: dict[int, dict[tuple[int, ...], tuple[int, dict[int, float]]]] = {}
        # Each (target, clique mask)'s semi-directed reach set on the kept graph, and the graph's
        # semi-directed successor masks, those of the empty graph before the first pass.
        self.reach: dict[tuple[int, int], int] = {}
        self.succ: list[int] = [0] * n

    def best(self, adj: np.ndarray, excluded: Excluded = None) -> ForwardResult:
        """The best-improving Insert into the targets not in ``excluded``, by :func:`move_key`."""
        if _FULL_RESCAN:
            state = neighborhood_state(adj)
            reachable = _reachable(adj)
            return forward_chunk_best(
                self.targets,
                adj,
                self.delta_engine,
                self.n,
                self.max_parents,
                state,
                reachable,
                self.ranks,
                self.forbidden,
                excluded,
            )
        self._refresh(adj)
        heap = [(moves[0][0], y, 0) for y, moves in self.screened.items() if moves]
        heapq.heapify(heap)
        scored: dict[tuple[int, tuple[int, ...]], tuple[dict[int, float], int]] = {}
        best: ForwardResult = None
        limit = -_EPS_IMPROVE  # no candidate whose exact delta exceeds this can be selected
        while heap and heap[0][0] <= limit:
            _low, y, i = heapq.heappop(heap)
            moves = self.screened[y]
            if i + 1 < len(moves):
                heapq.heappush(heap, (moves[i + 1][0], y, i + 1))
            _low, x, _t_index, T, base = moves[i]
            exact = scored.get((y, base))  # this pass's exact deltas and passing sources
            if exact is None:
                exact = scored[(y, base)] = self._exact(y, base)
            deltas, passing = exact
            if not (passing >> x) & 1 or not deltas[x] < -_EPS_IMPROVE:
                continue  # a y ~> x path misses the clique, or the Insert does not improve
            delta = deltas[x]
            key = move_key(delta, self.n, x, y, T, self.ranks)
            if best is None or key < best[0]:
                if excluded and key[1:] in excluded:
                    continue
                best = (key, x, y, T)
                # A delta that can still win rounds to this key or below, so it lies within half a
                # quantization step of it; the limit allows two steps as slack against rounding.
                # At a key of minus infinity, only a delta of minus infinity can tie.
                limit = min(-_EPS_IMPROVE, self.n * (key[0] + 2 * 10.0**-_KEY_DP))
        return best

    def _refresh(self, adj: np.ndarray) -> None:
        """Enumerate again the targets whose candidates differ from the kept graph's."""
        succ = _bitset.row_masks(adj == ops.TAIL)
        self.reach = _kept_reach(self.reach, self.succ, succ)
        touched = _touched(self.adj, adj)
        redo = [y for y in self.targets if touched[y]]
        if redo:
            adjacent = _bitset.row_masks(adj != ops.NULL)
            lines = _bitset.row_masks((adj == ops.TAIL) & (adj.T == ops.TAIL))
            for y in redo:
                self._enumerate(y, adj, adjacent, lines)
        self.adj = adj.copy()
        self.succ = succ

    def _enumerate(self, y: int, adj: np.ndarray, adjacent: list[int], lines: list[int]) -> None:
        """Group ``y``'s clique-valid candidates by base and score each group in one batch.

        ``adjacent`` and ``lines`` hold each node's adjacent nodes and undirected neighbors as
        bitmasks. The candidates are :func:`_insert_candidates`'s. Sources adjacent to the same
        undirected neighbors of ``y`` share ``NA``, ``t0``, and every subset ``T``, so the subsets
        are enumerated once per such group. A candidate is kept when the lowest exact delta its
        screening delta and bound allow can improve.
        """
        previous = self.groups.get(y, {})
        groups: dict[tuple[int, ...], _Group] = {}
        screened: list[_Screened] = []
        self.groups[y], self.screened[y] = groups, screened
        exact = self.exact.get(y, {})
        self.exact[y] = {}
        parents = np.flatnonzero(adj[y] == ops.ARROW).tolist()
        if self.max_parents is not None and len(parents) > self.max_parents:
            return  # y already carries the maximum admissible parent count.
        pa_set = set(parents)
        nbrs = lines[y]
        nonadjacent = self.everyone & ~adjacent[y] & ~(1 << y)
        free = nbrs  # the undirected neighbors T may hold
        if self.into is not None:
            nonadjacent &= ~self.into[y]
            free &= ~self.into[y]
            if not nonadjacent:
                return  # every source is forbidden into y or already adjacent to it.
        by_na: dict[int, tuple[list[int], int]] = {0: (_bitset.nodes(nonadjacent), nonadjacent)}
        if nbrs:
            lists: dict[int, list[int]] = {}
            for x in by_na[0][0]:
                lists.setdefault(adjacent[x] & nbrs, []).append(x)
            by_na = {na: (xs, _bitset.bitmask(xs)) for na, xs in lists.items()}
        members: dict[tuple[int, ...], tuple[list[tuple[int, int, tuple[int, ...]]], int]] = {}
        for na_mask, (xs, xs_mask) in by_na.items():
            na = _bitset.nodes(na_mask)
            for t_index, T in ops.clique_extensions(adj, na, _bitset.nodes(free & ~na_mask)):
                base = tuple(sorted(pa_set.union(na, T)))
                cands, mask = members.get(base, ([], 0))
                cands.extend((x, t_index, T) for x in xs)
                members[base] = (cands, mask | xs_mask)
        for base, (cands, mask) in members.items():
            cands.sort()  # ascending sources, as _insert_candidates yields them
            sources = [x for x, _, _ in cands]
            kept = previous.get(base)
            if kept is not None and kept[2].issuperset(sources):
                batch, deltas, bounds = kept[2:]
            else:
                bound_list: list[float] = []
                delta_list = self.delta_engine.delta_many_with_base(
                    y, base, [("add", x) for x in sources], bounds=bound_list
                )
                batch = frozenset(sources)
                deltas = dict(zip(sources, delta_list, strict=True))
                bounds = dict(zip(sources, bound_list, strict=True))
            groups[base] = (_bitset.bitmask(set(base) - pa_set), mask, batch, deltas, bounds)
            if base in exact:
                self.exact[y][base] = exact[base]
            for x, t_index, T in cands:
                bound = bounds[x]
                low = -math.inf if bound == math.inf else deltas[x] - bound
                if low < -_EPS_IMPROVE:
                    screened.append((low, x, t_index, T, base))
        screened.sort(key=lambda move: move[:3])

    def _exact(self, y: int, base: tuple[int, ...]) -> tuple[dict[int, float], int]:
        """Exact deltas of ``base``'s group at ``y`` by source, and a mask of its passing sources.

        A source passes when no semi-directed ``y ~> x`` path avoids the group's clique. The
        passing sources are scored in one batch, as :func:`forward_chunk_best` scores them. The
        group's batch serves when it holds exactly the passing sources, or when every passing
        source's bound is 0; otherwise the last exact batch serves while the same sources pass.
        """
        clique, sources, batch, deltas, bounds = self.groups[y][base]
        reach = self.reach.get((y, clique))
        if reach is None:
            reach = self.reach[(y, clique)] = _bitset.semidirected_reach_set(self.succ, y, clique)
        passing = sources & ~reach
        if passing.bit_count() == len(batch):
            return deltas, passing
        last = self.exact[y].get(base)
        if last is not None and last[0] == passing:
            return last[1], passing
        valid = _bitset.nodes(passing)
        if all(bounds[x] == 0.0 for x in valid):
            self.exact[y][base] = (passing, deltas)
            return deltas, passing
        moves = [("add", x) for x in valid]
        exact = dict(
            zip(valid, self.delta_engine.delta_many_with_base(y, base, moves), strict=True)
        )
        self.exact[y][base] = (passing, exact)
        return exact, passing


# ---- GES worker glue (module-level so forkserver/spawn can pickle it by name) --------------------
#
# Each worker is seeded with the read-only ``cov`` once (via the initializer) and owns a fixed
# target partition; a pass then sends only the current ``adj`` snapshot (int8, d*d bytes). Workers
# score through a private PURE Schur delta engine over the shared cov, so every process feeds
# identical bytes into ``local_score_bic`` -- bit-identical to serial under one BLAS thread.
# The task functions take their targets positionally, so ``StableWorkerSet.dispatch`` drives them
# directly. A worker always receives the same targets, so it keeps their forward candidates in
# one :class:`_ForwardScan` across the passes it runs. Background knowledge and the moves to skip
# travel with each pass.

# Per-worker state installed once by ``_init_worker`` and reused across every task and pass, so the
# delta engine's pure inv-cov cache persists across passes -- the cross-pass reuse lever.
_WORKER_DELTA: DeltaBICScore | None = None
_WORKER_N: int = 0
_WORKER_TOKEN: object = None
_WORKER_RANKS: list[int] = []  # each column's tie rank, from the parent
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)
_WORKER_SCAN: _ForwardScan | None = None  # this worker's forward scan, built on its first task


def _init_worker(
    cov: np.ndarray,
    n_samples: int,
    n_vars: int,
    lambda_value: float,
    run_token: object,
    ranks: list[int],
) -> None:
    """Seed a worker with the shared covariance and pin its numeric environment.

    Runs once per worker process. Caps BLAS to one thread (a 128-worker pool would otherwise
    oversubscribe, and multi-thread BLAS reduction order is not bit-stable) and builds the PURE
    Schur delta engine over the parent's exact ``cov`` bytes. Its inv-cov cache is a pure function
    of the parent set, so it persists across tasks and passes for free. ``n_samples`` is the
    observation count the BIC penalty uses; ``n_vars`` is the graph's node count; ``ranks`` holds
    each column's tie rank.
    """
    global _WORKER_DELTA, _WORKER_N, _WORKER_TOKEN, _WORKER_BLAS, _WORKER_RANKS
    from threadpoolctl import threadpool_limits

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _WORKER_DELTA = DeltaBICScore(BICScore.from_cov(cov, n_samples, lambda_value=lambda_value))
    _WORKER_N = n_vars
    _WORKER_TOKEN = run_token
    _WORKER_RANKS = ranks


def _forward_task(
    targets: Sequence[int],
    adj: np.ndarray,
    max_parents: float | None,
    run_token: object,
    constraints: Constraints | None = None,
    excluded: Excluded = None,
) -> ForwardResult:
    """Worker side of one forward chunk: refresh the kept candidates, scan, return the best."""
    global _WORKER_SCAN
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    if _WORKER_SCAN is None or _WORKER_SCAN.targets != list(targets):
        forbidden = None if constraints is None else constraints.forbidden
        _WORKER_SCAN = _ForwardScan(
            targets, _WORKER_DELTA, n, max_parents, _WORKER_RANKS, forbidden
        )
    return _WORKER_SCAN.best(adj, excluded)


def _backward_task(
    targets: Sequence[int],
    adj: np.ndarray,
    run_token: object,
    constraints: Constraints | None = None,
    excluded: Excluded = None,
) -> BackwardResult:
    """Worker side of one backward chunk."""
    global _WORKER_SCAN
    _WORKER_SCAN = None  # the forward phase has ended
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    state = neighborhood_state(adj)
    return backward_chunk_best(
        targets,
        adj,
        _WORKER_DELTA,
        n,
        state,
        _WORKER_RANKS,
        constraints=constraints,
        excluded=excluded,
    )


# ---- the density-aware crossover gate -----------------------------------------------------------
#
# A pass is worth fanning out only when its scan outweighs the worker round-trip (dispatch + pickle
# + collect). Since the inner cost is exponential in undirected degree (``2^|t0|``), the gate tracks
# *estimated candidate evaluations*, not a bare ``n`` -- so it is density-aware. A forward pass
# counts the targets the in-process scan would enumerate again: those touched since the graph it
# kept, which is every target on the first pass and after the workers ran the passes in between. A
# backward pass counts every target. It is checked per pass, and the worker set is created lazily
# on the first pass that crosses. The default is the fixed ``_DEFAULT_MIN_WORK`` below;
# ``ANDREY_GES_PARALLEL_MIN_WORK`` overrides it (0 sends every pass to the workers), and a fit with
# workers warns once on the default when a pass nears it.

# Estimated candidate evaluations below which a pass runs in-process. The default is a starting
# point, not tuned for any machine; `andrey calibrate` measures the value for this one.
_DEFAULT_MIN_WORK = 10_000

# The largest estimated work of a pass in the last fit with more than one worker; `andrey calibrate`
# reads it to state a measured crossover in the gate's units.
largest_pass_work = 0

_DEGREE_CAP = (
    20  # cap 2^deg so the estimate never overflows; above it a pass is parallel regardless
)


def _min_work() -> int:
    """The crossover gate in estimated candidate evals (env override, fixed default)."""
    return par.min_work(env.GES_PARALLEL_MIN_WORK, _DEFAULT_MIN_WORK)


def forward_work_per_target(n: int, state: NeighborState, max_parents: float | None) -> list[int]:
    """Estimated candidate evaluations for each forward target ``y`` (0 for a parent-capped target).

    ``n_nonadj(y) * 2^min(deg(y), CAP)``: ``n_nonadj(y)`` non-adjacent sources enumerate at most
    ``2^deg(y)`` subsets (``t0(x, y)`` sits in ``y``'s undirected neighbors). Drives both the gate
    (its sum) and the degree-aware chunker (its per-target profile).
    """
    adj_sets, par_of, nbr = state
    work = []
    for y in range(n):
        n_nonadj = n - 1 - len(adj_sets[y])
        capped = max_parents is not None and len(par_of[y]) > max_parents
        if n_nonadj <= 0 or capped:
            work.append(0)
        else:
            work.append(n_nonadj * (1 << min(len(nbr[y]), _DEGREE_CAP)))
    return work


def backward_work_per_target(n: int, state: NeighborState) -> list[int]:
    """Estimated candidate evals per backward target ``y``: ``n_adj(y) * 2^min(deg(y), CAP)``."""
    adj_sets, _par_of, nbr = state
    return [len(adj_sets[y]) * (1 << min(len(nbr[y]), _DEGREE_CAP)) for y in range(n)]


def _create_pool(
    cov: np.ndarray,
    n_samples: int,
    n_vars: int,
    lambda_value: float,
    run_token: object,
    workers: int,
    ranks: list[int],
) -> par.StableWorkerSet:
    """Build the persistent stable-assignment worker set -- the seam tests spy on (created once)."""
    return par.StableWorkerSet(
        n_vars,
        workers,
        _init_worker,
        (cov, n_samples, n_vars, lambda_value, run_token, ranks),
        context=_pick_context(),
    )


class _LazyWorkerSet:
    """The stable-assignment worker set, created on first use and reused across every pass.

    Never constructed if no pass crosses the work gate -- the property the no-pool-below-gate spy
    asserts by patching :func:`_create_pool`. Reused across passes so each worker's fixed-target
    inv-cov cache persists (the cross-pass locality lever). Holds the ``token`` (the cov identity a
    task carries so a worker can reject a mismatched run).
    """

    def __init__(self, score: BICScore, n_vars: int, workers: int, ranks: list[int]) -> None:
        self._score = score
        self._n_vars = n_vars
        self._workers = workers
        self._ranks = ranks
        self._wset: par.StableWorkerSet | None = None
        self.token: object = None

    def get(self) -> par.StableWorkerSet:
        if self._wset is None:
            self.token = hash(self._score.cov.tobytes())
            self._wset = _create_pool(
                self._score.cov,
                self._score.n,
                self._n_vars,
                self._score.lambda_value,
                self.token,
                self._workers,
                self._ranks,
            )
        return self._wset

    def close(self) -> None:
        if self._wset is not None:
            self._wset.close()
            self._wset = None


def _admitted(
    after: np.ndarray,
    before: np.ndarray,
    constraints: Constraints,
    excluded: set[tuple[int, int, tuple[int, ...]]],
    key: tuple,
) -> np.ndarray:
    """``after``, the CPDAG a move with ``key`` reaches from ``before``, when its class holds a DAG
    that satisfies ``constraints``; otherwise ``before``.

    An accepted move clears ``excluded``; a rejected one joins it, so the next scan of ``before``
    selects the next best move.
    """
    if admits(after, constraints):
        excluded.clear()
        return after
    excluded.add(key[1:])
    return before


def run_ges(
    adj: np.ndarray,
    score: LocalScore,
    n: int,
    max_parents: float | None,
    *,
    workers: int,
    ranks: list[int],
    full_scores: bool = False,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Both GES phases with per-pass parallelism: a pass fans out only above the fixed work gate.

    The serial and parallel searches are the same code path: below the gate a pass runs in-process
    over the whole target range (``workers=1`` never crosses it, so serial always takes it); above
    it the pass fans out to one lazily-created worker set that spans all remaining passes -- each
    worker owns a FIXED contiguous target partition, so its inv-cov cache persists across passes.
    Either way the reduce keys on :func:`move_key`, so the selected CPDAG is partition-invariant;
    ``ranks`` holds each column's :func:`~andrey.search.hc.tie_ranks`. Returns only the CPDAG;
    ``ges()`` recomputes the objective from it. ``full_scores`` swaps the in-process PURE Schur
    engine for the two-full-score reference (the quality gate's ``_full_scores`` arm); pooled
    workers only ever run Schur on BIC, so the reference arm, like any score other than
    :class:`~andrey.core.score.BICScore` (BDeu and the mixed-data scores), is held in-process and
    never fans out.

    ``constraints`` filters the candidates (:func:`forward_chunk_best`, :func:`backward_chunk_best`)
    and rejects a selected move whose class holds no DAG that satisfies them; the pass then runs
    again on the same graph without that move. Serial and pooled passes see the same rejections.
    """
    global largest_pass_work
    workers = max(1, min(workers, n))
    w_min = _min_work()
    engine: DeltaEngine
    lazy: _LazyWorkerSet | None
    if isinstance(score, BICScore) and not full_scores:
        engine = DeltaBICScore(score)
        lazy = _LazyWorkerSet(score, n, workers, ranks)
    else:
        engine = FullScoreDelta(score)
        lazy = None
    largest = 0
    # Warn once a pass comes near the default cutoff (only a pass that can fan out).
    warn = workers > 1 and lazy is not None and calibrate.cutoff_is_default("ges")
    forbidden = None if constraints is None else constraints.forbidden
    scan = _ForwardScan(range(n), engine, n, max_parents, ranks, forbidden)
    kept: dict[tuple[int, tuple[int, ...], int], float] = {}  # the in-process backward deltas
    excluded: set[tuple[int, int, tuple[int, ...]]] = (
        set()
    )  # the moves rejected on the current graph
    try:
        while True:  # forward phase
            work = [0]
            if workers > 1:
                # The in-process scan enumerates again every target touched since the graph it
                # kept, which after pooled passes is an older one (none before its first pass).
                touched = _touched(scan.adj, adj)
                per_target = forward_work_per_target(n, neighborhood_state(adj), max_parents)
                work = [w for w, redo in zip(per_target, touched, strict=True) if redo]
            largest = max(largest, sum(work))
            if warn and sum(work) >= calibrate.WARN_FRACTION * w_min:
                calibrate.warn_default_cutoff("ges", w_min)
                warn = False
            # The full-score engine never fans out: pooled workers always run the Schur engine on
            # BIC, so routing it to the workers would silently swap the two-full-score reference
            # for Schur. It stays in-process, keeping the quality gate's reference pure.
            if lazy is not None and workers > 1 and sum(work) > w_min:
                # Each worker scans its own fixed target partition (stable assignment); reduce keys
                # on the global order, so this is bit-identical to the whole-range serial scan.
                wset = lazy.get()
                best = reduce_best(
                    wset.dispatch(
                        _forward_task, adj, max_parents, lazy.token, constraints, excluded
                    )
                )
            else:
                best = scan.best(adj, excluded)
            if best is None:
                break
            key, x, y, T = best
            if constraints is None:
                adj = ops.insert_cpdag(adj, x, y, T)
            else:
                moved = ops.insert_cpdag(adj.copy(), x, y, T)
                adj = _admitted(moved, adj, constraints, excluded, key)
        excluded.clear()  # forward and backward moves share the key shape
        while True:  # backward phase
            state = neighborhood_state(adj)
            work = backward_work_per_target(n, state) if workers > 1 else [0]
            largest = max(largest, sum(work))
            if warn and sum(work) >= calibrate.WARN_FRACTION * w_min:
                calibrate.warn_default_cutoff("ges", w_min)
                warn = False
            if lazy is not None and workers > 1 and sum(work) > w_min:  # full scores stay serial
                wset = lazy.get()
                best = reduce_best(
                    wset.dispatch(_backward_task, adj, lazy.token, constraints, excluded)
                )
            else:
                best = backward_chunk_best(
                    range(n), adj, engine, n, state, ranks, kept, constraints, excluded
                )
            if best is None:
                break
            key, x, y, H = best
            if constraints is None:
                adj = ops.delete_cpdag(adj, x, y, H)
            else:
                moved = ops.delete_cpdag(adj.copy(), x, y, H)
                adj = _admitted(moved, adj, constraints, excluded, key)
    finally:
        if lazy is not None:
            lazy.close()
    if workers > 1:
        largest_pass_work = largest
    return adj
