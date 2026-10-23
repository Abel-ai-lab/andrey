"""GES forward and backward phases over an in-process decomposition into N target chunks.

The phases run the pool's per-chunk scans and its ``move_key`` reduce in one process, so a test can
compare any chunk count with serial GES without starting workers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from andrey.search import _parallel as par
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import _reform_cpdag


def forward_inprocess(
    adj: np.ndarray,
    delta_engine: pg.DeltaEngine,
    n: int,
    max_parents: float | None,
    n_chunks: int,
    ranks: Sequence[int],
    cap: float | None = None,
) -> np.ndarray:
    """The forward phase over ``n_chunks`` target chunks, under ``max_parents`` as the forward gate
    and ``cap`` as the parent cap; returns the CPDAG."""
    scans = [
        pg._ForwardScan(chunk, delta_engine, n, max_parents, ranks, None, cap)
        for chunk in par.even_chunks(n, n_chunks)
    ]
    while True:
        best = pg.reduce_best([scan.best(adj) for scan in scans])
        if best is None:
            return adj
        _key, x, y, T = best
        ops.apply_insert(adj, x, y, T)
        adj = _reform_cpdag(adj)


def backward_inprocess(
    adj: np.ndarray,
    delta_engine: pg.DeltaEngine,
    n: int,
    n_chunks: int,
    ranks: Sequence[int],
    cap: float | None = None,
) -> np.ndarray:
    """The backward phase over ``n_chunks`` target chunks, under ``cap``; returns the CPDAG."""
    while True:
        state = pg.neighborhood_state(adj)
        results = [
            pg.backward_chunk_best(chunk, adj, delta_engine, n, state, ranks, cap=cap)
            for chunk in par.even_chunks(n, n_chunks)
        ]
        best = pg.reduce_best(results)
        if best is None:
            return adj
        _key, x, y, H = best
        ops.apply_delete(adj, x, y, H)
        adj = _reform_cpdag(adj)
