"""Greedy Equivalence Search over the linear-Gaussian BIC, the engine behind :func:`andrey.ges`.

Each candidate is scored by differencing two full local scores; the running total is the objective
returned alongside the CPDAG. The algorithm, its settings, and its reference are described on
:func:`andrey.ges`.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.core import GraphStructure, _bitset, backend
from andrey.core.orient import dag2cpdag, pdag2dag, to_structure
from andrey.core.score import BICScore, Score
from andrey.search import operators as ops


def _reform_cpdag(adj: np.ndarray) -> np.ndarray:
    """Canonicalize a search PDAG to its CPDAG via a consistent DAG extension."""
    return dag2cpdag(pdag2dag(adj))


# Above this variable count the uint64 successor packing overflows, so the semi-directed-path check
# switches to a width-unlimited traversal. Tests monkeypatch it to force that path at small d.
_BITSET_MAX_D = 64

# How the d > 64 semi-directed reachability check traverses. Both the arbitrary-width Python-int
# bitmask BFS (``wide``) and the per-candidate set BFS (``set``) compute the same boolean predicate;
# they differ only in speed, and which wins depends on graph density. A micro-benchmark puts the
# crossover near a mean semi-directed out-degree of ``_WIDE_MEAN_DEG``: the bitmask BFS batches many
# nodes per machine-word op (winning on denser graphs, ~1.1-1.5x) but pays Python big-int overhead
# the set BFS avoids on very sparse ones. ``auto`` picks the cheaper traversal per graph state;
# ``wide`` / ``set`` force one path (the differential test forces each to prove bit-identity).
_REACH_MODE = "auto"  # "auto" | "wide" | "set"
_WIDE_MEAN_DEG = 5.0


def _reach_factory(n: int):
    """Return ``build_state(adj) -> reachable(y, x, clique)`` for a graph on ``n`` variables.

    ``reachable(y, x, clique)`` is ``True`` when ``x`` is semi-directed reachable from ``y`` once
    the ``clique`` nodes are removed -- the condition that invalidates ``Insert(x, y, T)``. For
    ``n <= _BITSET_MAX_D`` it packs per-node ``uint64`` successor masks and runs the
    (numba-eligible) bitmask BFS. Above the cap the uint64 packing overflows, so per graph state it
    picks the faster of two width-unlimited, boolean-identical traversals: the arbitrary-width
    Python-int bitmask BFS (:func:`andrey.core._bitset.semidirected_reaches_wide`) for denser
    states, the set BFS for very sparse ones (see ``_REACH_MODE``). The backend is resolved
    once per run, but the successor structure is rebuilt per graph state (``build_state``) because
    the CPDAG is re-formed after every accepted move.
    """
    if n <= _BITSET_MAX_D:
        reaches = _bitset.reaches_impl()

        def build_state(adj: np.ndarray):
            succ = _bitset.tail_successor_masks(adj)
            return lambda y, x, clique: reaches(succ, y, x, _bitset.bitmask(clique))

        return build_state

    def build_wide(adj: np.ndarray):
        succ = _bitset.tail_successor_masks_wide(adj)
        return lambda y, x, clique: _bitset.semidirected_reaches_wide(
            succ, y, x, _bitset.bitmask(clique)
        )

    def build_set(adj: np.ndarray):
        succ = ops.tail_successor_lists(adj)
        return lambda y, x, clique: not ops.blocks_semi_directed_paths(succ, y, x, clique)

    if _REACH_MODE == "wide":
        return build_wide
    if _REACH_MODE == "set":
        return build_set

    def build_state(adj: np.ndarray):
        # Mean semi-directed out-degree of this state decides the cheaper traversal (bit-identical).
        mean_deg = int(np.count_nonzero(adj == ops.TAIL)) / adj.shape[0]
        return (build_wide if mean_deg >= _WIDE_MEAN_DEG else build_set)(adj)

    return build_state


def _recompute_objective(adj: np.ndarray, score: Score, n: int) -> float:
    """Total BIC of the discovered CPDAG: ``sum_v score(v, Pa_v)`` over a consistent DAG extension.

    The objective is recomputed from the final structure rather than accumulated from the search's
    Schur deltas, so it is independent of the (non-bit-pure) delta arithmetic and identical for
    serial and parallel runs on the same CPDAG. Any consistent extension gives the same total (BIC
    is score-equivalent), so ``pdag2dag``'s choice of representative is immaterial.
    """
    dag = pdag2dag(adj)
    return float(sum(score.score(v, ops.parents(dag, v).tolist()) for v in range(n)))


def ges(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    maxP: float | None = None,
    _full_scores: bool = False,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Greedy Equivalence Search over the linear-Gaussian BIC score.

    Runs the forward (Insert) then backward (Delete) greedy phases from the empty graph, re-forming
    the CPDAG after every accepted move, then returns the discovered CPDAG and its total BIC
    objective (lower is better). Each candidate move is scored by the Schur delta-BIC; the
    returned objective is recomputed from the final CPDAG (``sum_v score(v, Pa_v)``), so it is
    independent of the delta arithmetic and identical for serial and parallel runs. Serial and
    parallel are the same decomposition: ``ges()`` always routes through
    :func:`~andrey.search._parallel_ges.run_ges`, which fans out only above the fixed work
    threshold.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score; ``"local_score_BIC"`` (linear-Gaussian BIC) is supported.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.
    maxP : float or None
        Maximum directed-parent count a node may already hold to remain an eligible Insert target in
        the forward phase; ``None`` defaults to ``n_features / 2``. The backward phase is uncapped.
    _full_scores : bool
        Private test lever: score every candidate by two full local scores (the pre-Schur reference)
        instead of the Schur delta. Tests compare the two scoring paths; this option is not part of
        the public contract.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total BIC score.
    """
    if score_func != "local_score_BIC":
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use 'local_score_BIC'")
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    max_parents = n / 2 if maxP is None else maxP
    score = BICScore(X, lambda_value=lambda_value)

    adj = np.zeros((n, n), dtype=np.int8)
    # Local import breaks the ges <-> _parallel_ges cycle.
    from andrey.search import _parallel_ges

    # Pin BLAS to one thread for the score-diff loops (the covariance above is already computed):
    # multi-thread BLAS reduction order is not bit-stable, so pinning makes the score host- and
    # thread-count-reproducible and equal to the (1-thread) parallel workers.
    with threadpool_limits(limits=1):
        workers = backend.worker_count() if n > 1 else 1
        adj = _parallel_ges.run_ges(
            adj, score, n, max_parents, workers=workers, full_scores=_full_scores
        )
        total = _recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag"), total
