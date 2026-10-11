"""Tabu search over DAGs, the engine behind :func:`andrey.tabu`.

The algorithm, its settings, and its references are described on :func:`andrey.tabu`; ``_search``
runs it. Moves are scanned by :class:`andrey.search.hc.MoveScanner` and, above hill climbing's work
gate, by its worker set (:mod:`andrey.search._parallel_hc`), so tabu search breaks ties as hill
climbing does and gives the same result with any worker count. The best DAG visited is converted
to its CPDAG and reported with its total BIC (lower is better).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.core.warning_policy import SearchLimitWarning, warn_once
from andrey.search import _parallel_hc as phc
from andrey.search import calibrate
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.hc import MoveScanner, ScannedMove, StructureKey, _apply_move

__all__ = ["tabu"]


class _Scanner:
    """The best admitted move of a pass: in-process below the work gate, on workers above it.

    Tabu search shares hill climbing's gate (``ANDREY_HC_PARALLEL_MIN_WORK``) and worker set: one
    pass scans the same moves. The worker set is created on first use and kept for every pass, so
    the workers' deltas persist.
    """

    def __init__(self, delta: DeltaBICScore, d: int, workers: int) -> None:
        self._scanner = MoveScanner(delta, d)  # the in-process scan; workers hold their own
        workers = max(1, min(workers, d))
        w_min = phc._min_work()
        work = phc._estimate_work(d)
        if workers > 1 and work >= calibrate.WARN_FRACTION * w_min:
            if calibrate.cutoff_is_default("hc"):
                calibrate.warn_default_cutoff("hc", w_min)
        self._fan_out = workers > 1 and work > w_min
        self._lazy = phc._LazyWorkerSet(delta.score_obj, d, workers)

    def best(
        self,
        adj_int: list[int],
        parents_of: dict[int, list[int]],
        floor: float,
        tabu: frozenset[StructureKey],
    ) -> ScannedMove | None:
        """The best legal move with delta below ``floor`` that does not lead into ``tabu``."""
        if self._fan_out:
            wset = self._lazy.get()
            results = wset.dispatch(
                phc._scan_task, adj_int, parents_of, self._lazy.token, floor, tabu
            )
            return phc.reduce_best(results)
        return self._scanner.best(adj_int, parents_of, floor=floor, tabu=tabu)

    def close(self) -> None:
        self._lazy.close()


def _search(
    score: BICScore, d: int, tabu_length: int, max_tabu: int, max_iter: int | None, workers: int
) -> tuple[StructureKey, int]:
    """Run tabu search from the empty graph; return the best DAG's structure key and moves taken.

    Each step puts the current DAG in the tabu list, then takes the best improving move that does
    not lead into the list. Without one, the search stops if it has taken ``max_tabu``
    non-improving moves since the last new best; otherwise it counts one more and takes the best
    move that does not lead into the list, which leaves the score unchanged or worse. A new best
    score, lower by more than ``_EPS_IMPROVE``, resets the count. The search also stops when every
    legal move leads into the list, or after ``max_iter`` moves unless it is ``None``.
    """
    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    local = [score.score(i, []) for i in range(d)]
    best_key: StructureKey = tuple(adj_int)
    best_total = math.fsum(local)
    recent: deque[StructureKey] = deque(maxlen=tabu_length)
    losses = 0
    moves = 0
    scanner = _Scanner(DeltaBICScore(score), d, workers)
    try:
        while max_iter is None or moves < max_iter:
            recent.append(tuple(adj_int))
            tabu = frozenset(recent)
            found = scanner.best(adj_int, parents_of, -_EPS_IMPROVE, tabu)
            if found is None:
                if losses >= max_tabu:
                    break
                losses += 1
                found = scanner.best(adj_int, parents_of, math.inf, tabu)
                if found is None:  # every legal move leads into the tabu list
                    break
            move = found[4]
            _apply_move(adj_int, parents_of, move)
            moves += 1
            op, u, v = move
            for node in (u, v) if op == "reverse" else (v,):
                local[node] = score.score(node, parents_of[node])
            total = math.fsum(local)
            if total < best_total - _EPS_IMPROVE:
                best_key, best_total, losses = tuple(adj_int), total, 0
    finally:
        scanner.close()
    return best_key, moves


def tabu(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    max_iter: int | None = None,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by tabu search over the linear-Gaussian BIC score.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score; ``"local_score_BIC_from_cov"`` (linear-Gaussian BIC) is supported.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.
    tabu_length : int
        Number of recently visited DAGs a move may not return to.
    max_tabu : int or None
        Non-improving moves allowed without a new best score before the search stops, at most
        ``tabu_length``; ``None`` means ``tabu_length``.
    max_iter : int or None
        Maximum number of moves, improving or not; ``None`` sets no limit. Taking ``max_iter`` moves
        warns with ``SearchLimitWarning``.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) of the best DAG visited and that DAG's total BIC score.
    """
    if score_func != "local_score_BIC_from_cov":
        raise NotImplementedError(
            f"unsupported score_func {score_func!r}; use 'local_score_BIC_from_cov'"
        )
    if max_tabu is None:
        max_tabu = tabu_length
    if not 1 <= max_tabu <= tabu_length:
        raise ValueError(f"max_tabu must lie in [1, tabu_length={tabu_length}], got {max_tabu}")
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score = BICScore(X, lambda_value=lambda_value)

    # Pin BLAS to one thread so the Schur updates are thread-count-reproducible and equal to the
    # (1-thread) workers, as hill climbing does.
    with backend.one_blas_thread():
        best_key, moves = _search(
            score, d, tabu_length, max_tabu, max_iter, workers=backend.worker_count()
        )
    if max_iter is not None and moves == max_iter:
        warn_once(
            f"tabu stopped at its limit of max_iter={max_iter} moves; a longer search may find a "
            "better graph, so raise max_iter",
            SearchLimitWarning,
        )

    # The best DAG as unsigned endpoint marks (u -> v: TAIL at u, ARROW at v), reduced to its CPDAG.
    parents = [[u for u in range(d) if best_key[u] >> v & 1] for v in range(d)]
    dag = np.full((d, d), NULL, dtype=np.int8)
    for v in range(d):
        for p in parents[v]:
            dag[p, v] = TAIL
            dag[v, p] = ARROW
    total = float(sum(score.score(i, parents[i]) for i in range(d)))
    return to_structure(dag2cpdag(dag), kind="cpdag"), total
