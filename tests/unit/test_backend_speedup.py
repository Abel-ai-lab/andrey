"""Perf gate: the numba reachability kernel is faster than the numpy reference (same result).

Timing is hardware-noisy, so this asserts only the direction (numba < numpy) on a workload where the
kernel's ~4x margin dwarfs the noise, plus the two backends agreeing. numba-gated, so the default
(no-accel) lane skips it.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from andrey.core import _bitset, backend

pytestmark = pytest.mark.skipif(
    backend.numba() is None, reason="numba (the [numba] extra) is not installed"
)


def _bench(fn, *, repeats: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):  # discard: JIT compile + cache warm-up
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _workload(d: int = 48, n_queries: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    adj = rng.random((d, d)) < 0.25
    np.fill_diagonal(adj, False)
    weights = np.uint64(1) << np.arange(d, dtype=np.uint64)
    succ = (adj.astype(np.uint64) * weights).sum(axis=1)
    qs = [
        (int(rng.integers(0, d)), int(rng.integers(0, d)), int(rng.integers(0, 1 << d)))
        for _ in range(n_queries)
    ]
    return succ, qs


def test_numba_reachability_faster_than_numpy():
    succ, qs = _workload()
    npy = _bitset._semidirected_reaches_numpy
    nba = _bitset._numba_reaches()
    run_np = lambda: [npy(succ, s, t, b) for s, t, b in qs]  # noqa: E731
    run_nb = lambda: [bool(nba(succ, s, t, np.uint64(b))) for s, t, b in qs]  # noqa: E731
    assert run_np() == run_nb()  # same booleans, both backends
    t_np, t_nb = _bench(run_np), _bench(run_nb)
    assert t_nb < t_np, f"numba ({t_nb:.4f}s) is not faster than numpy ({t_np:.4f}s)"
