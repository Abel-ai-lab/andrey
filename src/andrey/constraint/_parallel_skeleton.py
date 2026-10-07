"""Process-parallel decomposition of one PC/FCI skeleton conditioning-size pass.

At a fixed conditioning size the still-adjacent pairs are scanned independently: each pair's
separating set is a read-only function of the start-of-size neighborhood snapshot and the
correlation matrix, and removals are applied only after the whole size completes. So the pass is
embarrassingly parallel across pairs, and the reduce is a plain concatenation of per-chunk removals
with no tie-break -- the recovered adjacency and separating sets are identical to the serial scan
regardless of how the pairs are partitioned or the order chunks complete.

Workers are seeded once with the parent's exact correlation-matrix bytes (via
:meth:`~andrey.core.ci.FisherZ.from_corr`) and pin BLAS to one thread, so every process feeds
identical float64 into the Fisher-Z closed forms and the skeleton is bit-identical to the serial
oracle. A worker rebuilds the start-of-size neighborhood from the adjacency snapshot the task
carries and delegates to the shared :func:`~andrey.constraint._skeleton.scan_pairs`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from andrey.constraint._skeleton import Removal, scan_pairs
from andrey.core.ci import CITest, FisherZ

# The start-method policy and fail-loud collect are the same primitives the parallel greedy search
# uses, so the skeleton reuses those helpers rather than redefining them; only the pair-axis
# chunking and the correlation-seeded worker glue below are skeleton-specific.
from andrey.search._parallel import collect as _collect
from andrey.search._parallel import pick_context as _pick_context

# Per-worker state installed once by _init_worker and reused across every task and size pass.
_WORKER_TEST: FisherZ | None = None
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = (
    None  # a held threadpoolctl controller capping this worker's BLAS to 1 thread
)


def even_pair_chunks(pairs: list[tuple[int, int]], n_chunks: int) -> list[list[tuple[int, int]]]:
    """Partition ``pairs`` into up to ``n_chunks`` contiguous chunks of ~equal count."""
    if not pairs:
        return []
    n = len(pairs)
    parts = np.array_split(np.arange(n), max(1, min(n_chunks, n)))
    return [[pairs[i] for i in part.tolist()] for part in parts if len(part)]


def _init_worker(corr: np.ndarray, n_samples: int, run_token: object) -> None:
    """Seed a worker with the shared correlation matrix and pin its numeric environment.

    Runs once per worker process: caps BLAS to one thread (a large pool would otherwise
    oversubscribe, and multi-thread BLAS reduction order is not bit-stable on the larger
    conditioning submatrices), pins the numpy backend, and rebuilds the Fisher-Z test over the
    parent's exact correlation bytes.
    """
    global _WORKER_TEST, _WORKER_TOKEN, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend as _backend

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _backend.config.backend = "numpy"
    _WORKER_TEST = FisherZ.from_corr(corr, n_samples)
    _WORKER_TOKEN = run_token


def _scan_task(payload: tuple) -> list[Removal]:
    """Worker side of one chunk: rebuild the snapshot neighborhood, scan the chunk's pairs."""
    pairs, adj_mask, size, alpha, run_token = payload
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("skeleton worker pool reused across runs (corr/token mismatch)")
    assert _WORKER_TEST is not None  # _init_worker runs before any task
    d = adj_mask.shape[0]
    neighbours = [np.flatnonzero(adj_mask[i]) for i in range(d)]
    return scan_pairs(_WORKER_TEST, pairs, neighbours, size, alpha)


class SkeletonWorkerPool:
    """Persistent worker pool for the skeleton size passes; created lazily, reused across sizes.

    The pool is seeded once with the test's correlation matrix; each size pass submits the chunked
    pair list plus the adjacency snapshot and reduces the per-chunk removals. The correlation bytes
    tag a run token every task re-checks, so a stale pool can never mix a prior run's numbers in.
    """

    def __init__(self, test: CITest, workers: int) -> None:
        corr, n_samples = test.to_seed()  # ty: ignore[unresolved-attribute]  # gated by caller
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
                initargs=(self._corr, self._n, self._token),
            )
        return self._pool, self._token

    def scan(
        self,
        pairs: list[tuple[int, int]],
        adj_mask: np.ndarray,
        size: int,
        alpha: float,
    ) -> list[Removal]:
        """Scan ``pairs`` at conditioning ``size`` across the worker pool; return all removals."""
        pool, token = self._ensure()
        payloads = [
            (chunk, adj_mask, size, alpha, token)
            for chunk in even_pair_chunks(pairs, self._workers)
        ]
        futures = [pool.submit(_scan_task, payload) for payload in payloads]
        removals: list[Removal] = []
        for chunk_removals in _collect(futures):
            removals.extend(chunk_removals)
        return removals

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
