"""Process-parallel decomposition of one PC/FCI skeleton conditioning-size pass.

At a fixed conditioning size the still-adjacent pairs are scanned independently: each pair's
separating set is a read-only function of the start-of-size neighborhood snapshot and the
correlation matrix, and removals are applied only after the whole size completes. So the pass is
embarrassingly parallel across pairs. The pairs are cut into contiguous chunks of about equal test
counts, and the reduce concatenates the per-chunk removals in chunk order, so the removals arrive
in the serial scan's order with the same separating sets and largest separating p-values. A pass
that fails raises the exception of its first failing chunk in that order, the exception the serial
scan meets first.

Only the FisherZ and StudentT tests run on the pool (:func:`pool_capable`), and only a pass of at
least :data:`MIN_POOL_QUERIES` tests; any other pass scans in the caller's process. Workers are
seeded once with the parent's exact correlation-matrix bytes (via
:meth:`~andrey.core.ci.PartialCorrelationTest.from_corr` of the parent's test class) and pin BLAS
to one thread, so every process feeds identical float64 into the same closed forms and the
skeleton is bit-identical to the serial oracle. A worker rebuilds the start-of-size neighborhood
from the adjacency snapshot the task carries and delegates to the shared
:func:`~andrey.constraint._skeleton.scan_pairs`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import cast

import numpy as np

from andrey.constraint._skeleton import Removal, pair_queries, scan_pairs
from andrey.core.ci import CITest, FisherZ, PartialCorrelationTest, StudentT

# The start-method policy, the work-balanced chunking, and the fail-loud collect are the same
# primitives the parallel greedy search uses, so the skeleton reuses those helpers rather than
# redefining them; only the correlation-seeded worker glue below is skeleton-specific.
from andrey.search._parallel import balanced_chunks as _balanced_chunks
from andrey.search._parallel import collect as _collect
from andrey.search._parallel import pick_context as _pick_context

#: The fewest tests a conditioning-size pass needs to run on the worker pool; a smaller pass scans
#: in the caller's process. Starting the pool costs about half a second per fit, which a pass of
#: this size repays at 4 to 8 workers; a smaller pass finishes sooner in the caller's process.
MIN_POOL_QUERIES: int = 500_000

# Per-worker state installed once by _init_worker and reused across every task and size pass.
_WORKER_TEST: PartialCorrelationTest | None = None
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)


def pool_capable(test: CITest) -> bool:
    """Whether ``test`` runs on the worker pool: a FisherZ or StudentT test, which a worker rebuilds
    from the correlation matrix. A subclass or a wrapper of either scans in the caller's process."""
    return type(test) in (FisherZ, StudentT)


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

    The pool starts on the first pass that reaches :data:`MIN_POOL_QUERIES` tests, seeded once with
    the test's correlation matrix; each such pass submits its weighted chunks plus the adjacency
    snapshot and concatenates the per-chunk removals. The correlation bytes tag a run token every
    task re-checks, so a stale pool can never mix a prior run's numbers in. The caller passes a
    :func:`pool_capable` test.
    """

    def __init__(self, test: CITest, workers: int) -> None:
        corr, n_samples = cast("PartialCorrelationTest", test).to_seed()
        self._test_class = cast("type[PartialCorrelationTest]", type(test))
        self._corr = np.ascontiguousarray(corr, dtype=np.float64)
        self._n = int(n_samples)
        self._workers = max(1, workers)
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
    ) -> list[Removal] | None:
        """Scan ``pairs`` at conditioning ``size`` across the worker pool; return all removals.

        ``neighbors`` is the start-of-size snapshot and ``adj_mask`` the boolean adjacency it was
        read from, which the workers rebuild it from. The removals come in the order
        :func:`~andrey.constraint._skeleton.scan_pairs` gives them, each with its pair's largest
        separating p-value, or ``None`` when ``record_pvalues`` is false. Returns ``None``, running
        nothing, when the pass holds fewer than :data:`MIN_POOL_QUERIES` tests; the caller then
        scans it in-process.
        """
        weights = [pair_queries(neighbors, x, y, size) for x, y in pairs]
        if sum(weights) < MIN_POOL_QUERIES:
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
