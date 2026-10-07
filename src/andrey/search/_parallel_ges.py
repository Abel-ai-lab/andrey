"""Parallel GES candidate scans using :mod:`andrey.search._parallel`.

Each greedy pass scans Insert or Delete candidates and applies the best improving move. For each
target, candidates sharing ``base = sorted(NA u T u Pa)`` are scored in one batched Schur
``delta_many_with_base`` call. Fixed target ownership preserves arithmetic and reuses worker
caches across passes.

The move key is ``(stable_key(delta, n), x, y, index)``. ``stable_key`` quantizes ``delta / n``;
``index`` is the subset-enumeration position of ``T``, preserving the serial tie order. The CPDAG
is byte-identical across partitions and worker assignments.

``ges()`` uses :func:`run_ges`; ``workers=1`` scans all targets in-process. The objective is
recomputed from the final CPDAG by :func:`andrey.search.ges._recompute_objective`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from andrey.core import env
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, _delta_two_scores
from andrey.search import _parallel as par
from andrey.search import calibrate
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE, stable_key
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

    Uses :func:`andrey.search._parallel.reduce_best`; HC uses its default ``r[:4]`` key. Each
    ``(x, y, index)`` belongs to one target partition, so keys are globally unique and strict-``<``
    reduction is independent of completion order.
    """
    return par.reduce_best(results, key=lambda r: r[0])


class DeltaEngine(Protocol):
    """A move scorer: raw local-score deltas for single add/remove moves off a shared base.

    Satisfied by the PURE Schur :class:`~andrey.core.score_delta.DeltaBICScore` (production) and by
    :class:`_FullScoreDelta` (the two-full-score reference the quality gate pairs against).
    """

    def delta_many_with_base(
        self, i: int, base_parents: Sequence[int], deltas: Sequence[tuple[str, int]]
    ) -> list[float]:
        """Return ``[score(i, base +/- u_k) - score(i, base)]`` for the requested moves."""
        ...


# Per-graph-state neighborhood precompute shared by every chunk of a pass.
NeighbourState = tuple[list[set[int]], list[np.ndarray], list[set[int]]]


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


def neighbourhood_state(adj: np.ndarray, n: int) -> NeighbourState:
    """The per-node adjacency sets, directed parents, and undirected neighbors for a graph state.

    Computed once per graph state (as the serial passes do) and shared read-only across all chunks;
    a chunk owning targets ``[y0, y1)`` iterates only its ``y`` slice but reads every node's arrays.
    """
    adj_sets = [set(ops.adjacent(adj, v).tolist()) for v in range(n)]
    par_of = [ops.parents(adj, v) for v in range(n)]
    nbr = [set(ops.undirected_neighbors(adj, v).tolist()) for v in range(n)]
    return adj_sets, par_of, nbr


def forward_chunk_best(
    targets: Sequence[int],
    adj: np.ndarray,
    delta_engine: DeltaEngine,
    n: int,
    max_parents: float | None,
    state: NeighbourState,
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
    adj_sets, par_of, nbr = state
    best: ForwardResult = None
    best_key: tuple | None = None
    for y in targets:
        paj = par_of[y]
        if max_parents is not None and len(paj) > max_parents:
            continue  # y already carries the maximum admissible parent count.
        nbrs_y = nbr[y]
        pa_set = set(int(p) for p in paj)
        # Group valid candidates by their base parent set; each base scores its adds in one batch.
        groups: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
        for x in range(n):
            if x == y or adj[x, y] != ops.NULL or adj[y, x] != ops.NULL:
                continue  # Insert requires a non-adjacent pair.
            adj_x = adj_sets[x]
            na = nbrs_y & adj_x
            t0 = sorted(nbrs_y - adj_x)
            for t_index, T in ops.clique_extensions(adj, na, t0):
                clique = na | set(T)
                if reachable(y, x, clique):
                    continue  # some semi-directed y ~> x path misses the clique: invalid Insert.
                base = tuple(sorted(pa_set | na | set(T)))
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
    state: NeighbourState,
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


def _forward_inprocess(
    adj: np.ndarray, delta_engine: DeltaEngine, n: int, max_parents: float | None, n_chunks: int
) -> np.ndarray:
    """Forward phase over an in-process N-chunk decomposition (no pool; the chunk-invariance guard).

    Returns only the CPDAG; the objective is recomputed once from the final structure by ``ges()``,
    so no per-move total is threaded here.
    """
    build_state = _reach_factory(n)
    while True:
        state = neighbourhood_state(adj, n)
        reachable = build_state(adj)
        results = [
            forward_chunk_best(chunk, adj, delta_engine, n, max_parents, state, reachable)
            for chunk in even_chunks(n, n_chunks)
        ]
        best = reduce_best(results)
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
        state = neighbourhood_state(adj, n)
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
# drives them directly.

# Per-worker state installed once by ``_init_worker`` and reused across every task and pass, so the
# delta engine's pure inv-cov cache persists across passes -- the cross-pass reuse lever.
_WORKER_DELTA: DeltaBICScore | None = None
_WORKER_N: int = 0
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)


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
    """Worker side of one forward chunk: rebuild the per-state structure, scan, return the best."""
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    state = neighbourhood_state(adj, n)
    reachable = _reach_factory(n)(adj)
    return forward_chunk_best(targets, adj, _WORKER_DELTA, n, max_parents, state, reachable)


def _backward_task(targets: Sequence[int], adj: np.ndarray, run_token: object) -> BackwardResult:
    """Worker side of one backward chunk."""
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("GES worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_DELTA is not None  # _init_worker runs before any task
    n = _WORKER_N
    state = neighbourhood_state(adj, n)
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


def forward_work_per_target(n: int, state: NeighbourState, max_parents: float | None) -> list[int]:
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


def backward_work_per_target(n: int, state: NeighbourState) -> list[int]:
    """Estimated candidate evals per backward target ``y``: ``n_adj(y) * 2^min(deg(y), CAP)``."""
    adj_sets, _par_of, nbr = state
    return [len(adj_sets[y]) * (1 << min(len(nbr[y]), _DEGREE_CAP)) for y in range(n)]


def estimate_forward_work(n: int, state: NeighbourState, max_parents: float | None) -> int:
    """An O(n) upper bound on the forward pass's candidate-eval count (the gate's predictor)."""
    return sum(forward_work_per_target(n, state, max_parents))


def estimate_backward_work(n: int, state: NeighbourState) -> int:
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
    build_state = _reach_factory(n)
    engine: DeltaEngine = _FullScoreDelta(score) if full_scores else DeltaBICScore(score)
    lazy = _LazyWorkerSet(score, n, workers)
    try:
        while True:  # forward phase
            state = neighbourhood_state(adj, n)
            work = forward_work_per_target(n, state, max_parents)
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
                # Only the in-process scan needs reachability; pooled workers rebuild it themselves.
                reachable = build_state(adj)
                best = forward_chunk_best(range(n), adj, engine, n, max_parents, state, reachable)
            if best is None:
                break
            _key, x, y, T = best
            ops.apply_insert(adj, x, y, T)
            adj = _reform_cpdag(adj)
        while True:  # backward phase
            state = neighbourhood_state(adj, n)
            work = backward_work_per_target(n, state)
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
