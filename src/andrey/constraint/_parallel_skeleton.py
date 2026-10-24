"""Process-parallel decomposition of one PC/FCI skeleton conditioning-size pass.

At a fixed conditioning size the still-adjacent pairs are scanned independently: each pair's
separating set is a read-only function of the start-of-size neighborhood snapshot and the
correlation matrix, and removals are applied only after the whole size completes. So the pass is
embarrassingly parallel across pairs. The pairs are cut into contiguous chunks of about equal test
counts, and the reduce concatenates the per-chunk removals in chunk order, so the removals arrive
in the serial scan's order with the same separating sets and largest separating p-values. A pass
that fails raises the exception of its first failing chunk in that order, the exception the serial
scan meets first.

Only the FisherZ and StudentT tests run on the pool
(:func:`~andrey.constraint._skeleton.canonical_capable`), and only a pass of at least
:data:`MIN_POOL_QUERIES` tests, or the ``ANDREY_SKELETON_PARALLEL_MIN_WORK`` that overrides it; any
other pass scans in the caller's process. Workers are seeded once with the parent's exact
correlation-matrix bytes (via :meth:`~andrey.core.ci.PartialCorrelationTest.from_corr` of the
parent's test class) and pin BLAS to one thread, so every process feeds identical float64 into the
same closed forms and the skeleton is bit-identical to the serial oracle. A worker rebuilds the
start-of-size neighborhood from the adjacency snapshot the task carries and delegates to the shared
:func:`~andrey.constraint._skeleton.scan_pairs`.

An RCoT test runs on a pool of its own (:class:`RCoTWorkerPool`), with its own cutoff in RCoT tests
times rows, :data:`MIN_POOL_RCOT_WORK` or the ``ANDREY_RCOT_PARALLEL_MIN_WORK`` that overrides it,
since a test's cost grows with the rows and is about a thousand times a Fisher-z test's. The
caller's process builds the pass's stacks as the in-process scan does
(:func:`~andrey.constraint._skeleton.scan_pairs`), and the pool evaluates each stack's queries, each
conditioning set's in one worker. A worker holds the parent test's ranked, standardized data, ranks,
feature counts, and resolved seed, and a p-value depends only on these and its query, so it is the
in-process p-value bit for bit, whichever worker computes it.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, cast

import numpy as np

from andrey.constraint._skeleton import (
    Removal,
    _stacked_scan,
    canonical_capable,
    is_rcot,
    pair_queries,
    scan_pairs,
)
from andrey.core import env
from andrey.core.ci import CITest, PartialCorrelationTest
from andrey.search import _parallel as par
from andrey.search import calibrate
from andrey.search._parallel import balanced_chunks as _balanced_chunks
from andrey.search._parallel import collect as _collect
from andrey.search._parallel import pick_context as _pick_context

if TYPE_CHECKING:
    from andrey.core.ci_kernel import RCoT

#: The default for the fewest tests a conditioning-size pass needs to run on the worker pool; a
#: smaller pass scans in the caller's process. ``ANDREY_SKELETON_PARALLEL_MIN_WORK`` overrides it
#: (0 sends every pass to the pool), and a fit with workers warns once on the default when a pass
#: nears it; ``andrey calibrate --method skeleton`` measures the value for a machine.
MIN_POOL_QUERIES: int = 500_000

# Per-worker state installed once by _init_worker and reused across every task and size pass.
_WORKER_TEST: PartialCorrelationTest | None = None
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)


def _init_worker(
    test_class: type[PartialCorrelationTest], corr: np.ndarray, n_samples: int, run_token: object
) -> None:
    """Seed a worker with the shared correlation matrix and pin its numeric environment.

    Runs once per worker process: caps BLAS to one thread (a large pool would otherwise
    oversubscribe, and multi-thread BLAS reduction order is not bit-stable on the larger
    conditioning submatrices), pins the numpy backend, and rebuilds the parent's test class over
    the parent's exact correlation bytes.
    """
    global _WORKER_TEST, _WORKER_TOKEN, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend as _backend

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _backend.config.backend = "numpy"
    _WORKER_TEST = test_class.from_corr(corr, n_samples)
    _WORKER_TOKEN = run_token


def _scan_task(payload: tuple) -> list[Removal]:
    """Worker side of one chunk: rebuild the snapshot neighborhood, scan the chunk's pairs."""
    pairs, adj_mask, size, alpha, record_pvalues, run_token = payload
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("skeleton worker pool reused across runs (corr/token mismatch)")
    assert _WORKER_TEST is not None  # _init_worker runs before any task
    d = adj_mask.shape[0]
    neighbors = [np.flatnonzero(adj_mask[i]) for i in range(d)]
    return scan_pairs(_WORKER_TEST, pairs, neighbors, size, alpha, record_pvalues=record_pvalues)


class SkeletonWorkerPool:
    """Persistent worker pool for the skeleton size passes; created lazily, reused across sizes.

    The pool starts on the first pass that reaches the cutoff, :data:`MIN_POOL_QUERIES` tests or the
    ``ANDREY_SKELETON_PARALLEL_MIN_WORK`` that overrides it, seeded once with the test's
    correlation matrix; each such pass submits its weighted chunks plus the adjacency snapshot and
    concatenates the per-chunk removals. The correlation bytes tag a run token every task
    re-checks, so a stale pool can never mix a prior run's numbers in. The caller passes a
    :func:`~andrey.constraint._skeleton.canonical_capable` test. A bad override raises when the pool
    is created, before any pass.
    """

    def __init__(self, test: CITest, workers: int) -> None:
        corr, n_samples = cast("PartialCorrelationTest", test).to_seed()
        self._test_class = cast("type[PartialCorrelationTest]", type(test))
        self._corr = np.ascontiguousarray(corr, dtype=np.float64)
        self._n = int(n_samples)
        self._workers = max(1, workers)
        self._min_queries = par.min_work(env.SKELETON_PARALLEL_MIN_WORK, MIN_POOL_QUERIES)
        # Warn once a pass comes near the default cutoff (only a fit that can fan out).
        self._warn = self._workers > 1 and calibrate.cutoff_is_default("skeleton")
        self._pool: ProcessPoolExecutor | None = None
        self._token: object = None

    def _ensure(self) -> tuple[ProcessPoolExecutor, object]:
        if self._pool is None:
            self._token = hash(self._corr.tobytes())
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=_pick_context(),
                initializer=_init_worker,
                initargs=(self._test_class, self._corr, self._n, self._token),
            )
        return self._pool, self._token

    def scan(
        self,
        pairs: list[tuple[int, int]],
        neighbors: list[np.ndarray],
        adj_mask: np.ndarray,
        size: int,
        alpha: float,
        record_pvalues: bool = True,
        weights: list[int] | None = None,
    ) -> list[Removal] | None:
        """Scan ``pairs`` at conditioning ``size`` across the worker pool; return all removals.

        ``neighbors`` is the start-of-size snapshot and ``adj_mask`` the boolean adjacency it was
        read from, which the workers rebuild it from. ``weights`` holds each pair's test count
        (:func:`~andrey.constraint._skeleton.pair_queries`) when the caller has taken it for the
        pass, and is counted here otherwise. The removals come in the order
        :func:`~andrey.constraint._skeleton.scan_pairs` gives them, each with its pair's largest
        separating p-value, or ``None`` when ``record_pvalues`` is false. Returns ``None``, running
        nothing, when the pass holds fewer tests than the cutoff; the caller then scans it
        in-process.
        """
        if weights is None:
            weights = [pair_queries(neighbors, x, y, size) for x, y in pairs]
        queries = sum(weights)
        if self._warn and queries >= calibrate.WARN_FRACTION * self._min_queries:
            calibrate.warn_default_cutoff("skeleton", self._min_queries)
            self._warn = False
        if queries < self._min_queries:
            return None
        pool, token = self._ensure()
        # Contiguous chunks collected in submission order keep the serial scan's pair order.
        futures = [
            pool.submit(
                _scan_task,
                ([pairs[i] for i in chunk], adj_mask, size, alpha, record_pvalues, token),
            )
            for chunk in _balanced_chunks(weights, self._workers)
        ]
        return [removal for chunk in _collect(futures, ordered=True) for removal in chunk]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


#: The default for the least work, RCoT tests times rows, an RCoT conditioning-size pass needs to
#: run on the worker pool; a smaller pass scans in the caller's process.
#: ``ANDREY_RCOT_PARALLEL_MIN_WORK`` overrides it (0 sends every pass to the pool), and a fit with
#: workers warns once on the default when a pass nears it; ``andrey calibrate --method rcot``
#: measures the value for a machine. On an Intel Core i7-7820X with 4 workers it measured 270,000,
#: for many fits per process; the default also repays the 0.8 s the first pool in a process takes
#: to start, which the calibration's passes put near a million.
MIN_POOL_RCOT_WORK: int = 1_000_000

# How many queries a conditioning set's own features and factor cost: at 10,000 rows a set took 32
# ms and a query 1.3 ms. It only balances the workers' shares.
_SET_COST = 25

# The RCoT test a worker installs once, and the run it belongs to.
_WORKER_RCOT: RCoT | None = None


def _init_rcot_worker(state: tuple, run_token: object) -> None:
    """Rebuild the parent's RCoT test from its state and pin the worker to one BLAS thread."""
    global _WORKER_RCOT, _WORKER_TOKEN, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend as _backend
    from andrey.core.ci_kernel import RCoT

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _backend.config.backend = "numpy"
    _WORKER_RCOT = RCoT._from_state(*state)
    _WORKER_TOKEN = run_token


def _rcot_task(payload: tuple) -> np.ndarray:
    """Worker side: the p-values of one share of a stack's queries, in its order."""
    xs, ys, conditions, run_token = payload
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("RCoT worker pool reused across runs (data/token mismatch)")
    assert _WORKER_RCOT is not None  # _init_rcot_worker runs before any task
    try:
        return _WORKER_RCOT._canonical_pvalues(xs, ys, conditions)
    finally:
        # A share holds whole conditioning sets, so the next share needs none of these features.
        _WORKER_RCOT._residual.cache_clear()
        _WORKER_RCOT._cond_factor.cache_clear()


class RCoTWorkerPool:
    """Worker pool for the skeleton passes of an RCoT test; created lazily, reused across sizes.

    The pool starts on the first pass whose work, its RCoT tests times the rows, reaches
    :data:`MIN_POOL_RCOT_WORK` or the ``ANDREY_RCOT_PARALLEL_MIN_WORK`` that overrides it. Such a
    pass is scanned by :func:`~andrey.constraint._skeleton.scan_pairs`'s stacked scan, with each
    stack's queries grouped by conditioning set and the sets split across the workers in shares of
    about equal cost; the p-values come back in the stack's order. A share that raises makes the
    caller's test evaluate the stack, so the error is the one the in-process scan raises. A bad
    override raises when the pool is created, before any pass.
    """

    def __init__(self, test: RCoT, workers: int) -> None:
        self._test = test
        self._workers = max(1, workers)
        self._min_work = par.min_work(env.RCOT_PARALLEL_MIN_WORK, MIN_POOL_RCOT_WORK)
        self._warn = self._workers > 1 and calibrate.cutoff_is_default("rcot")
        self._pool: ProcessPoolExecutor | None = None
        self._token: object = None

    def _ensure(self) -> tuple[ProcessPoolExecutor, object]:
        if self._pool is None:
            self._token = hash(self._test._data.tobytes())
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=_pick_context(),
                initializer=_init_rcot_worker,
                initargs=(self._test._state(), self._token),
            )
        return self._pool, self._token

    def scan(
        self,
        pairs: list[tuple[int, int]],
        neighbors: list[np.ndarray],
        adj_mask: np.ndarray,
        size: int,
        alpha: float,
        record_pvalues: bool = True,
        weights: list[int] | None = None,
    ) -> list[Removal] | None:
        """Scan ``pairs`` at conditioning ``size`` on the pool; return the removals in-process
        :func:`~andrey.constraint._skeleton.scan_pairs` returns, or ``None``, running nothing, when
        the pass's work is below the cutoff. The arguments are :meth:`SkeletonWorkerPool.scan`'s;
        ``adj_mask`` is unused, since the stacks are built here."""
        if weights is None:
            weights = [pair_queries(neighbors, x, y, size) for x, y in pairs]
        work = sum(weights) * self._test._n
        if self._warn and work >= calibrate.WARN_FRACTION * self._min_work:
            calibrate.warn_default_cutoff("rcot", self._min_work)
            self._warn = False
        if work < self._min_work:
            return None
        if size:
            self._test._warn_few_rows()  # a worker's warning would not reach the caller
        return _stacked_scan(self._pvalues, pairs, neighbors, size, alpha, None, record_pvalues)

    def _pvalues(self, xs: np.ndarray, ys: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        """The p-values of a stack's queries, in order, each conditioning set's in one worker."""
        test = self._test
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, cond in enumerate(conditions.tolist()):
            groups.setdefault(test._condition_key(cond), []).append(i)
        sets = [groups[key] for key in sorted(groups)]
        shares = [
            np.array([i for k in share for i in sets[k]], dtype=np.intp)
            for share in _balanced_chunks([_SET_COST + len(s) for s in sets], self._workers)
        ]
        pool, token = self._ensure()
        futures = [
            pool.submit(_rcot_task, (xs[rows], ys[rows], conditions[rows], token))
            for rows in shares
        ]
        try:
            parts = _collect(futures, ordered=True)
        except Exception as exc:
            if isinstance(exc.__cause__, BrokenProcessPool):
                raise
            return test._canonical_pvalues(xs, ys, conditions)
        pvalues = np.empty(len(xs))
        for rows, part in zip(shares, parts):
            pvalues[rows] = part
        return pvalues

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


def skeleton_pool(test: CITest, workers: int) -> SkeletonWorkerPool | RCoTWorkerPool | None:
    """The worker pool for ``test``'s skeleton passes with ``workers`` processes: a FisherZ or
    StudentT test's (:class:`SkeletonWorkerPool`), an RCoT test's (:class:`RCoTWorkerPool`), or
    ``None`` with one worker or another test, whose passes scan in the caller's process."""
    if workers <= 1:
        return None
    if canonical_capable(test):
        return SkeletonWorkerPool(test, workers)
    if is_rcot(test):
        return RCoTWorkerPool(cast("RCoT", test), workers)
    return None
