"""Parallel GES candidate scans using :mod:`andrey.search._parallel`.

Each greedy pass scans Insert or Delete candidates and applies the best improving move. For each
target, candidates sharing ``base = sorted(NA u T u Pa)`` are scored in one batched Schur
``delta_many_with_base`` call. Fixed target ownership preserves arithmetic and reuses worker
caches across passes.

The move key is ``(stable_key(delta, n), x, y, index)``. ``stable_key`` quantizes ``delta / n``;
``index`` is the subset-enumeration position of ``T``, preserving the serial tie order. The CPDAG
is byte-identical across partitions and worker assignments.

The forward phase keeps each target's Insert candidates between passes and enumerates them again
only for the targets a move touches (:class:`_ForwardScan`); it selects the move a full scan
(:func:`forward_chunk_best`) selects.

``ges()`` uses :func:`run_ges`; ``workers=1`` scans all targets in-process. The objective is
recomputed from the final CPDAG by :func:`andrey.search.ges._recompute_objective`.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import numpy as np

from andrey.core import _bitset, env
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, _delta_two_scores
from andrey.search import _parallel as par
from andrey.search import calibrate
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP, stable_key
from andrey.search.ges import _reach_factory, _reform_cpdag

# Shared helpers re-exported for GES callers and tests.
even_chunks = par.even_chunks
balanced_chunks = par.balanced_chunks
_collect = par.collect
_fork_safe = par.fork_safe
_pick_context = par.pick_context
_StableWorkerSet = par.StableWorkerSet

# A chunk best, or ``None`` when the chunk has no improving operator. ``key`` is the total-order
# reduce key ``(stable_key(delta, n), x, y, index)``; ``T`` / ``H`` carry the move to apply.
ForwardResult = tuple[tuple, int, int, tuple[int, ...]] | None  # (key, x, y, T)
BackwardResult = tuple[tuple, int, int, tuple[int, ...]] | None  # (key, x, y, H)


def reduce_best(
    results: Sequence[ForwardResult | BackwardResult],
) -> ForwardResult | BackwardResult:
    """Select the best chunk by its ``(stable_key, x, y, index)`` key in ``r[0]``.

    Uses :func:`andrey.search._parallel.reduce_best`. Each ``(x, y, index)`` belongs to one target
    partition; unique keys make strict-``<`` reduction independent of completion order.
    """
    return par.reduce_best(results, key=lambda r: r[0])


class DeltaEngine(Protocol):
    """A move scorer: raw local-score deltas for single add/remove moves off a shared base.

    Satisfied by the PURE Schur :class:`~andrey.core.score_delta.DeltaBICScore` (production) and by
    :class:`_FullScoreDelta` (the two-full-score reference the quality gate pairs against).
    """

    score_obj: BICScore

    def delta_many_with_base(
        self, i: int, base_parents: Sequence[int], deltas: Sequence[tuple[str, int]]
    ) -> list[float]:
        """Return ``[score(i, base +/- u_k) - score(i, base)]`` for the requested moves."""
        ...


# Per-graph-state neighborhood precompute shared by every chunk of a pass.
NeighborState = tuple[list[set[int]], list[np.ndarray], list[set[int]]]


class _FullScoreDelta:
    """Reference delta engine: every delta from two full ``BICScore.score`` calls (the GES oracle).

    Exposes the same ``delta_many_with_base`` surface as :class:`DeltaBICScore` but never takes the
    Schur fast path, reproducing GES's exact two-full-score differencing. It is the
    ``_full_scores=True`` arm the quality gate pairs the Schur path against.
    """

    def __init__(self, score: BICScore) -> None:
        self.score_obj = score

    def delta_many_with_base(
        self, i: int, base_parents: Sequence[int], deltas: Sequence[tuple[str, int]]
    ) -> list[float]:
        base = tuple(sorted(int(p) for p in base_parents))
        return [_delta_two_scores(self.score_obj, int(i), base, u, op) for op, u in deltas]


def neighborhood_state(adj: np.ndarray, n: int) -> NeighborState:
    """The per-node adjacency sets, directed parents, and undirected neighbors for a graph state.

    Computed once per graph state (as the serial passes do) and shared read-only across all chunks;
    a chunk owning targets ``[y0, y1)`` iterates only its ``y`` slice but reads every node's arrays.
    Node ``v``'s entries equal ``ops.adjacent``, ``ops.parents``, and ``ops.undirected_neighbors``
    of ``v``; ``n`` is the node count of ``adj``.
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
    y: int, adj: np.ndarray, state: NeighborState
) -> Iterator[tuple[int, int, tuple[int, ...], set[int]]]:
    """Yield ``(x, t_index, T, clique)`` for every Insert into ``y`` whose ``NA u T`` is a clique.

    ``clique = NA u T`` is the node set every semi-directed ``y ~> x`` path must meet for the
    Insert to be valid; the caller checks that. Sources come in ascending ``x``, and each source's
    subsets in :func:`~andrey.search.operators.clique_extensions` order.
    """
    adj_sets, _par_of, nbr = state
    nbrs_y = nbr[y]
    # Insert requires a non-adjacent pair.
    sources = np.flatnonzero((adj[:, y] == ops.NULL) & (adj[y] == ops.NULL))
    for x in sources.tolist():
        if x == y:
            continue
        adj_x = adj_sets[x]
        na = nbrs_y & adj_x
        t0 = sorted(nbrs_y - adj_x)
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
) -> ForwardResult:
    """The best-improving Insert whose target is in ``targets``, by ``(stable_key, x, y, t_idx)``.

    Enumerates the chunk's valid Insert candidates score-free (clique + reachability), groups them
    per target by ``base = sorted(NA u T u Pa)``, and scores each group with one batched
    ``delta_many_with_base`` call over its ``("add", x)`` moves -- so one shared ``inv(cov[base,
    base])`` amortizes across all sources sharing a base. A candidate is improving iff its raw delta
    is below ``-_EPS_IMPROVE``; the argmin is over the quantized key, so visitation order and add
    batching cannot change the result. Grouping is per target, and a target lives wholly in one
    chunk, so the batch shape (hence each delta) is identical however the targets are partitioned.
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
        groups: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
        for x, t_index, T, clique in _insert_candidates(y, adj, state):
            if reachable(y, x, clique):
                continue  # some semi-directed y ~> x path misses the clique: invalid Insert.
            base = tuple(sorted(pa_set | clique))
            groups.setdefault(base, []).append((x, t_index, T))
        for base, cands in groups.items():
            moves = [("add", x) for x, _, _ in cands]
            for (x, t_index, T), delta in zip(
                cands, delta_engine.delta_many_with_base(y, base, moves), strict=True
            ):
                if delta < -_EPS_IMPROVE:
                    key = (stable_key(delta, n), x, y, t_index)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (key, x, y, T)
    return best


def backward_chunk_best(
    targets: Sequence[int],
    adj: np.ndarray,
    delta_engine: DeltaEngine,
    n: int,
    state: NeighborState,
) -> BackwardResult:
    """The best-improving Delete whose target is in ``targets``, by ``(stable_key, x, y, h_idx)``.

    Enumerates the chunk's valid Delete candidates score-free (residual-clique check; Delete carries
    no reachability test) and scores each with one scalar ``("remove", x)`` off ``base = sorted(
    residual u Pa u {x})`` -- the PURE Schur remove delta. Improving iff the raw delta is below
    ``-_EPS_IMPROVE``; the argmin is over the quantized key.
    """
    adj_sets, par_of, nbr = state
    best: BackwardResult = None
    best_key: tuple | None = None
    for y in targets:
        nbrs_y = nbr[y]
        paj = par_of[y]
        pa_set = set(int(p) for p in paj)
        for x in range(n):
            directed = adj[y, x] == ops.ARROW  # x -> y
            undirected = adj[y, x] == ops.TAIL and adj[x, y] == ops.TAIL  # x -- y
            if not (directed or undirected):
                continue
            h0 = sorted(nbrs_y & adj_sets[x])
            h0_set = set(h0)
            for h_index, H in ops.clique_residuals(adj, h0):
                residual = h0_set - set(H)
                base = tuple(sorted(residual | pa_set | {int(x)}))
                delta = delta_engine.delta_many_with_base(y, base, [("remove", int(x))])[0]
                if delta < -_EPS_IMPROVE:
                    key = (stable_key(delta, n), x, y, h_index)
                    if best_key is None or key < best_key:
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
# delta from one batch of every candidate sharing its base, which can differ in the last bits. The
# scan visits candidates in screening order and recomputes each visited base group with
# forward_chunk_best's batch, until no screening delta can come within the bound below of beating
# the best exact key. The bound assumes two batches over one base give deltas that differ by less
# than ``_SCREEN_TOL * n_samples``; the conditioning gate keeps about six digits of each residual,
# and measured differences stay below 1e-15 * n_samples.

# The scan the forward phase uses in-process: ``True`` runs forward_chunk_best over every target at
# every pass; ``False`` keeps each target's candidates until a move touches it. Tests flip it to
# compare the two. Pooled workers always keep their candidates.
_FULL_RESCAN = False

# Bound on the difference between a screening delta and the exact delta, per sample.
_SCREEN_TOL = 1e-6

# A candidate kept for selection: ``(screening delta, x, t_index, T, base)``.
_Screened = tuple[float, int, int, tuple[int, ...], tuple[int, ...]]


class _ForwardScan:
    """The forward pass over fixed targets, keeping each target's candidates between passes.

    :meth:`best` returns what :func:`forward_chunk_best` returns for the same targets and graph.
    """

    def __init__(
        self,
        targets: Sequence[int],
        delta_engine: DeltaEngine,
        n: int,
        max_parents: float | None,
    ) -> None:
        self.targets = list(targets)
        self.delta_engine = delta_engine
        self.n = n
        self.max_parents = max_parents
        self.tol = _SCREEN_TOL * delta_engine.score_obj.n
        self.build_reach = _reach_factory(n)
        self.adj: np.ndarray | None = None  # the graph the kept candidates were enumerated on
        # Per target: each base's clique mask and screening deltas by source; the candidates that
        # can improve, sorted by screening delta; and each base's last exact deltas with the
        # passing sources they were scored for.
        self.groups: dict[int, dict[tuple[int, ...], tuple[int, dict[int, float]]]] = {}
        self.screened: dict[int, list[_Screened]] = {}
        self.exact: dict[int, dict[tuple[int, ...], tuple[list[int], dict[int, float]]]] = {}

    def best(self, adj: np.ndarray, state: NeighborState) -> ForwardResult:
        """The best-improving Insert into the targets, by ``(stable_key, x, y, t_idx)``.

        ``state`` is :func:`neighborhood_state` of ``adj``.
        """
        if _FULL_RESCAN:
            reachable = self.build_reach(adj)
            return forward_chunk_best(
                self.targets, adj, self.delta_engine, self.n, self.max_parents, state, reachable
            )
        self._refresh(adj, state)
        heap = [(moves[0][0], y, 0) for y, moves in self.screened.items() if moves]
        heapq.heapify(heap)
        scored: dict[tuple[int, tuple[int, ...]], dict[int, float]] = {}  # this pass's groups
        succ: list[int] | None = None
        best: ForwardResult = None
        limit = -_EPS_IMPROVE  # no candidate whose exact delta reaches this can be selected
        while heap and heap[0][0] - self.tol < limit:
            _delta, y, i = heapq.heappop(heap)
            moves = self.screened[y]
            if i + 1 < len(moves):
                heapq.heappush(heap, (moves[i + 1][0], y, i + 1))
            _delta, x, t_index, T, base = moves[i]
            deltas = scored.get((y, base))
            if deltas is None:
                if succ is None:
                    succ = _bitset.tail_successor_masks_wide(adj)
                deltas = scored[(y, base)] = self._exact(y, base, succ)
            delta = deltas.get(x)
            if delta is None or not delta < -_EPS_IMPROVE:
                continue  # a y ~> x path misses the clique, or the Insert does not improve
            key = (stable_key(delta, self.n), x, y, t_index)
            if best is None or key < best[0]:
                best = (key, x, y, T)
                # Only a delta within one quantization step of this key can still win.
                limit = min(-_EPS_IMPROVE, self.n * (key[0] + 2 * 10.0**-_KEY_DP))
        return best

    def _refresh(self, adj: np.ndarray, state: NeighborState) -> None:
        """Enumerate again the targets whose candidates differ from the kept graph's."""
        if self.adj is None:
            stale = self.targets
        else:
            diff = self.adj != adj
            changed = diff.any(axis=0) | diff.any(axis=1)
            lines = (adj == ops.TAIL) & (adj.T == ops.TAIL)
            touched = changed | lines[changed].any(axis=0)
            stale = [y for y in self.targets if touched[y]]
        for y in stale:
            self._enumerate(y, adj, state)
        self.adj = adj.copy()

    def _enumerate(self, y: int, adj: np.ndarray, state: NeighborState) -> None:
        """Group ``y``'s clique-valid candidates by base and score each group in one batch."""
        groups: dict[tuple[int, ...], tuple[int, dict[int, float]]] = {}
        screened: list[_Screened] = []
        self.groups[y], self.screened[y], self.exact[y] = groups, screened, {}
        paj = state[1][y]
        if self.max_parents is not None and len(paj) > self.max_parents:
            return  # y already carries the maximum admissible parent count.
        pa_set = set(int(p) for p in paj)
        members: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
        for x, t_index, T, clique in _insert_candidates(y, adj, state):
            members.setdefault(tuple(sorted(pa_set | clique)), []).append((x, t_index, T))
        for base, cands in members.items():
            moves = [("add", x) for x, _, _ in cands]
            deltas = self.delta_engine.delta_many_with_base(y, base, moves)
            clique_mask = _bitset.bitmask(set(base) - pa_set)
            groups[base] = (clique_mask, {x: d for (x, _, _), d in zip(cands, deltas, strict=True)})
            for (x, t_index, T), d in zip(cands, deltas, strict=True):
                if d - self.tol < -_EPS_IMPROVE:
                    screened.append((d, x, t_index, T, base))
        screened.sort(key=lambda move: move[:3])

    def _exact(self, y: int, base: tuple[int, ...], succ: list[int]) -> dict[int, float]:
        """Exact deltas of ``base``'s group at ``y``, by source, for the sources that pass.

        A source passes when no semi-directed ``y ~> x`` path avoids the group's clique. The
        passing sources are scored in one batch, as :func:`forward_chunk_best` scores them. A batch
        of the same sources gives the same deltas, so the last batch is reused while they pass.
        """
        clique_mask, screening = self.groups[y][base]
        reach = _bitset.semidirected_reach_set(succ, y, clique_mask)
        valid = [x for x in screening if not (reach >> x) & 1]
        if len(valid) == len(screening):
            return screening
        last = self.exact[y].get(base)
        if last is not None and last[0] == valid:
            return last[1]
        moves = [("add", x) for x in valid]
        deltas = self.delta_engine.delta_many_with_base(y, base, moves)
        exact = dict(zip(valid, deltas, strict=True))
        self.exact[y][base] = (valid, exact)
        return exact


def _forward_inprocess(
    adj: np.ndarray, delta_engine: DeltaEngine, n: int, max_parents: float | None, n_chunks: int
) -> np.ndarray:
    """Forward phase over an in-process N-chunk decomposition (no pool; the chunk-invariance guard).

    Returns only the CPDAG; the objective is recomputed once from the final structure by ``ges()``,
    so no per-move total is threaded here.
    """
    scans = [
        _ForwardScan(chunk, delta_engine, n, max_parents) for chunk in even_chunks(n, n_chunks)
    ]
    while True:
        state = neighborhood_state(adj, n)
        best = reduce_best([scan.best(adj, state) for scan in scans])
        if best is None:
            return adj
        _key, x, y, T = best
        ops.apply_insert(adj, x, y, T)
        adj = _reform_cpdag(adj)


def _backward_inprocess(
    adj: np.ndarray, delta_engine: DeltaEngine, n: int, n_chunks: int
) -> np.ndarray:
    """Backward phase over an in-process N-chunk decomposition (no pool; chunk-invariance guard)."""
    while True:
        state = neighborhood_state(adj, n)
        results = [
            backward_chunk_best(chunk, adj, delta_engine, n, state)
            for chunk in even_chunks(n, n_chunks)
        ]
        best = reduce_best(results)
        if best is None:
            return adj
        _key, x, y, H = best
        ops.apply_delete(adj, x, y, H)
        adj = _reform_cpdag(adj)


# ---- GES worker glue (module-level so forkserver/spawn can pickle it by name) --------------------
#
# Each worker is seeded with the read-only ``cov`` once (via the initializer) and owns a fixed
# target partition; a pass then sends only the current ``adj`` snapshot (int8, d*d bytes). Workers
# score through a private PURE Schur delta engine over the shared cov, so every process feeds
# identical bytes into ``local_score_bic`` -- bit-identical to serial under one BLAS thread.
# Reachability is backend-invariant, so workers pin the numpy/set-BFS path (no per-worker numba
# warmup). The task functions take their targets positionally, so ``StableWorkerSet.dispatch``
# drives them directly. A worker always receives the same targets, so it keeps their forward
# candidates in one :class:`_ForwardScan` across the passes it runs.

# Per-worker state installed once by ``_init_worker`` and reused across every task and pass, so the
# delta engine's pure inv-cov cache persists across passes -- the cross-pass reuse lever.
_WORKER_DELTA: DeltaBICScore | None = None
_WORKER_N: int = 0
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)
_WORKER_SCAN: _ForwardScan | None = None  # this worker's forward scan, built on its first task


def _init_worker(
    cov: np.ndarray, n_samples: int, n_vars: int, lambda_value: float, run_token: object
) -> None:
    """Seed a worker with the shared covariance and pin its numeric environment.

    Runs once per worker process. Caps BLAS to one thread (a 128-worker pool would otherwise
    oversubscribe, and multi-thread BLAS reduction order is not bit-stable), pins the numpy backend
    (so an inherited ``ANDREY_BACKEND`` cannot make a worker resolve reachability differently), and
    builds the PURE Schur delta engine over the parent's exact ``cov`` bytes. Its inv-cov cache is a
    pure function of the parent set, so it persists across tasks and passes for free. ``n_samples``
    is the observation count the BIC penalty uses; ``n_vars`` is the graph's node count.
    """
    global _WORKER_DELTA, _WORKER_N, _WORKER_TOKEN, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend as _backend

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _backend.config.backend = "numpy"
    _WORKER_DELTA = DeltaBICScore(BICScore.from_cov(cov, n_samples, lambda_value=lambda_value))
    _WORKER_N = n_vars
    _WORKER_TOKEN = run_token


def _forward_task(
    targets: Sequence[int], adj: np.ndarray, max_parents: float | None, run_token: object
) -> ForwardResult:
    """Worker side of one forward chunk: refresh the kept candidates, scan, return the best."""
    global _WORKER_SCAN
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    if _WORKER_SCAN is None or _WORKER_SCAN.targets != list(targets):
        _WORKER_SCAN = _ForwardScan(targets, _WORKER_DELTA, n, max_parents)
    return _WORKER_SCAN.best(adj, neighborhood_state(adj, n))


def _backward_task(targets: Sequence[int], adj: np.ndarray, run_token: object) -> BackwardResult:
    """Worker side of one backward chunk."""
    global _WORKER_SCAN
    _WORKER_SCAN = None  # the forward phase has ended
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    state = neighborhood_state(adj, n)
    return backward_chunk_best(targets, adj, _WORKER_DELTA, n, state)


# ---- the density-aware crossover gate -----------------------------------------------------------
#
# A pass is worth fanning out only when its scan outweighs the worker round-trip (dispatch + pickle
# + collect). Since the inner cost is exponential in undirected degree (``2^|t0|``), the gate tracks
# *estimated candidate evaluations*, not a bare ``n`` -- so it is density-aware. It is checked per
# pass (GES starts from the empty graph: pass 1 is cheap, later passes densen), and the worker set
# is created lazily on the first pass that crosses. The default is the fixed ``_DEFAULT_MIN_WORK``
# below; ``ANDREY_GES_PARALLEL_MIN_WORK`` overrides it (0 sends every pass to the workers), and a
# fit with workers warns once on the default when a pass nears it.

# Estimated candidate evaluations below which a pass runs in-process. The default allows for the
# first pool start of a single fit in a fresh process; repeated fits in one process pay off from a
# lower value, which `andrey calibrate` measures.
_DEFAULT_MIN_WORK = 10_000

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


def estimate_forward_work(n: int, state: NeighborState, max_parents: float | None) -> int:
    """An O(n) upper bound on the forward pass's candidate-eval count (the gate's predictor)."""
    return sum(forward_work_per_target(n, state, max_parents))


def estimate_backward_work(n: int, state: NeighborState) -> int:
    """An O(n) upper bound on the backward pass's candidate-eval count (the gate's predictor)."""
    return sum(backward_work_per_target(n, state))


def _create_pool(
    cov: np.ndarray,
    n_samples: int,
    n_vars: int,
    lambda_value: float,
    run_token: object,
    workers: int,
) -> par.StableWorkerSet:
    """Build the persistent stable-assignment worker set -- the seam tests spy on (created once)."""
    return par.StableWorkerSet(
        n_vars,
        workers,
        _init_worker,
        (cov, n_samples, n_vars, lambda_value, run_token),
        context=_pick_context(),
    )


class _LazyWorkerSet:
    """The stable-assignment worker set, created on first use and reused across every pass.

    Never constructed if no pass crosses the work gate -- the property the no-pool-below-gate spy
    asserts by patching :func:`_create_pool`. Reused across passes so each worker's fixed-target
    inv-cov cache persists (the cross-pass locality lever). Holds the ``token`` (the cov identity a
    task carries so a worker can reject a mismatched run).
    """

    def __init__(self, score: BICScore, n_vars: int, workers: int) -> None:
        self._score = score
        self._n_vars = n_vars
        self._workers = workers
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
            )
        return self._wset

    def close(self) -> None:
        if self._wset is not None:
            self._wset.close()
            self._wset = None


def run_ges(
    adj: np.ndarray,
    score: BICScore,
    n: int,
    max_parents: float | None,
    *,
    workers: int,
    full_scores: bool = False,
) -> np.ndarray:
    """Both GES phases with per-pass parallelism: a pass fans out only above the fixed work gate.

    The serial and parallel searches are the same code path: below the gate a pass runs in-process
    over the whole target range (``workers=1`` never crosses it, so serial always takes it); above
    it the pass fans out to one lazily-created worker set that spans all remaining passes -- each
    worker owns a FIXED contiguous target partition, so its inv-cov cache persists across passes.
    Either way the reduce keys on ``(stable_key(delta, n), x, y, index)``, so the selected CPDAG is
    partition-invariant. Returns only the CPDAG; ``ges()`` recomputes the objective from it.
    ``full_scores`` swaps the in-process PURE Schur engine for the two-full-score reference (the
    quality gate's ``_full_scores`` arm); pooled workers only ever run Schur, so the reference arm
    is held in-process and never fans out.
    """
    workers = max(1, min(workers, n))
    w_min = _min_work()
    # Warn once a pass comes near the default cutoff (the pool never runs the full-score arm).
    warn = workers > 1 and not full_scores and calibrate.cutoff_is_default("ges")
    engine: DeltaEngine = _FullScoreDelta(score) if full_scores else DeltaBICScore(score)
    scan = _ForwardScan(range(n), engine, n, max_parents)
    lazy = _LazyWorkerSet(score, n, workers)
    try:
        while True:  # forward phase
            state = neighborhood_state(adj, n)
            work = forward_work_per_target(n, state, max_parents) if workers > 1 else [0]
            if warn and sum(work) >= calibrate.WARN_FRACTION * w_min:
                calibrate.warn_default_cutoff("ges", w_min)
                warn = False
            # The full-score reference arm never fans out: pooled workers always run the Schur
            # engine, so routing it to the workers would silently swap the two-full-score reference
            # for Schur. It stays in-process, keeping the quality gate's reference pure.
            if workers > 1 and sum(work) > w_min and not full_scores:
                # Each worker scans its own fixed target partition (stable assignment); reduce keys
                # on the global order, so this is bit-identical to the whole-range serial scan.
                wset = lazy.get()
                best = reduce_best(wset.dispatch(_forward_task, adj, max_parents, lazy.token))
            else:
                best = scan.best(adj, state)
            if best is None:
                break
            _key, x, y, T = best
            ops.apply_insert(adj, x, y, T)
            adj = _reform_cpdag(adj)
        while True:  # backward phase
            state = neighborhood_state(adj, n)
            work = backward_work_per_target(n, state) if workers > 1 else [0]
            if warn and sum(work) >= calibrate.WARN_FRACTION * w_min:
                calibrate.warn_default_cutoff("ges", w_min)
                warn = False
            if workers > 1 and sum(work) > w_min and not full_scores:  # reference arm stays serial
                wset = lazy.get()
                best = reduce_best(wset.dispatch(_backward_task, adj, lazy.token))
            else:
                best = backward_chunk_best(range(n), adj, engine, n, state)
            if best is None:
                break
            _key, x, y, H = best
            ops.apply_delete(adj, x, y, H)
            adj = _reform_cpdag(adj)
    finally:
        lazy.close()
    return adj
