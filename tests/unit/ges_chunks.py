"""GES forward and backward phases over an in-process decomposition into N target chunks.

The phases run the pool's per-chunk scans and its ``move_key`` reduce in one process, so a test can
compare any chunk count with serial GES without starting workers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.core.orient import to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import _parallel as par
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import _recompute_objective, _reform_cpdag
from andrey.search.hc import tie_ranks


def forward_inprocess(
    adj: np.ndarray,
    delta_engine: pg.DeltaEngine,
    n: int,
    max_parents: float | None,
    n_chunks: int,
    ranks: Sequence[int],
) -> np.ndarray:
    """The forward phase over ``n_chunks`` target chunks; returns the CPDAG."""
    scans = [
        pg._ForwardScan(chunk, delta_engine, n, max_parents, ranks)
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
    adj: np.ndarray, delta_engine: pg.DeltaEngine, n: int, n_chunks: int, ranks: Sequence[int]
) -> np.ndarray:
    """The backward phase over ``n_chunks`` target chunks; returns the CPDAG."""
    while True:
        state = pg.neighborhood_state(adj)
        results = [
            pg.backward_chunk_best(chunk, adj, delta_engine, n, state, ranks)
            for chunk in par.even_chunks(n, n_chunks)
        ]
        best = pg.reduce_best(results)
        if best is None:
            return adj
        _key, x, y, H = best
        ops.apply_delete(adj, x, y, H)
        adj = _reform_cpdag(adj)


def ges_inprocess(
    x: np.ndarray,
    n_chunks: int,
    max_p: float | None = None,
    engine: Callable[[BICScore], pg.DeltaEngine] = DeltaBICScore,
) -> tuple[np.ndarray, float]:
    """Full GES over ``n_chunks`` target chunks, set up as ``ges()`` sets up a serial run.

    ``engine`` builds the delta engine from the BIC score: the Schur ``DeltaBICScore``, or
    ``FullScoreDelta`` for two full scores per move. Returns the CPDAG's marks and the objective
    recomputed from it, as ``ges()`` does.
    """
    X = np.asarray(x, dtype=np.float64)
    n = X.shape[1]
    max_parents = n / 2 if max_p is None else max_p
    score = BICScore(X, lambda_value=1.0)
    delta_engine = engine(score)
    adj = np.zeros((n, n), dtype=np.int8)
    with threadpool_limits(limits=1):
        ranks = tie_ranks(X).tolist()
        adj = forward_inprocess(adj, delta_engine, n, max_parents, n_chunks, ranks)
        adj = backward_inprocess(adj, delta_engine, n, n_chunks, ranks)
        total = _recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag").to_numpy(), total
