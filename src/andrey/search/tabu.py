"""Tabu search over DAGs, the engine behind :func:`andrey.tabu`.

``_search`` uses :class:`andrey.search.hc.MoveScanner`, sharing hill climbing's tie rules. Output is
the best DAG's CPDAG and total BIC; lower is better.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score_delta import DeltaBICScore, ScoreDifference, make_score_engine
from andrey.core.warning_policy import SearchLimitWarning, warn_once
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.hc import MoveScanner, StructureKey, _apply_move

__all__ = ["tabu"]


def _search(
    delta: DeltaBICScore | ScoreDifference,
    d: int,
    tabu_length: int,
    max_tabu: int,
    max_iter: int | None,
) -> tuple[StructureKey, int]:
    """Run tabu search from the empty graph; return the best DAG's structure key and moves taken.

    Each step adds the current DAG to the tabu list and takes the best improving move outside it.
    Without one, the search stops after ``max_tabu`` non-improving moves since the last new best;
    otherwise it counts one more and takes the best non-improving move outside the list, leaving the
    score unchanged or worse. A new best score lower by more than ``_EPS_IMPROVE`` resets the count.
    The search also stops when every legal move leads into the list, or after ``max_iter`` moves
    unless it is ``None``.
    """
    score = delta.score_obj
    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    local = [score.score(i, []) for i in range(d)]
    best_key: StructureKey = tuple(adj_int)
    best_total = math.fsum(local)
    recent: deque[StructureKey] = deque(maxlen=tabu_length)
    losses = 0
    moves = 0
    scanner = MoveScanner(delta, d)
    while max_iter is None or moves < max_iter:
        recent.append(tuple(adj_int))
        tabu = frozenset(recent)
        found = scanner.best(adj_int, parents_of, floor=-_EPS_IMPROVE, tabu=tabu)
        if found is None:
            if losses >= max_tabu:
                break
            losses += 1
            found = scanner.best(adj_int, parents_of, floor=math.inf, tabu=tabu)
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
        Non-improving moves allowed without a new best score before stopping, at most
        ``tabu_length``; ``None`` means ``tabu_length``.
    max_iter : int or None
        Maximum moves, improving or not; ``None`` sets no limit. Taking ``max_iter`` moves warns
        with ``SearchLimitWarning``.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) of the best DAG visited and that DAG's total BIC score.
    """
    if max_tabu is None:
        max_tabu = tabu_length
    if not 1 <= max_tabu <= tabu_length:
        raise ValueError(f"max_tabu must lie in [1, tabu_length={tabu_length}], got {max_tabu}")
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score, delta = make_score_engine(
        score_func, X, lambda_value=lambda_value, bic_name="local_score_BIC_from_cov"
    )

    # One BLAS thread keeps the Schur updates reproducible across thread counts.
    with backend.one_blas_thread():
        best_key, moves = _search(delta, d, tabu_length, max_tabu, max_iter)
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
