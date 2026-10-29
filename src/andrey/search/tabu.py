"""Tabu search over DAGs, the engine behind :func:`andrey.tabu`.

``_search`` uses :class:`andrey.search.hc.MoveScanner`, sharing hill climbing's tie rules. Output is
the best DAG's CPDAG and total score; lower is better.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.score import (
    DISCRETE_SCORES,
    MIXED_SCORES,
    SCORES,
    BICScore,
    LocalScore,
    make_score,
)
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta
from andrey.core.warning_policy import warn_search_limit
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.hc import MoveScanner, StructureKey, _apply_move, start_dag, tie_ranks

__all__ = ["tabu"]

#: The local scores :func:`tabu` accepts.
TABU_SCORES: tuple[str, ...] = SCORES


def _search(
    score: LocalScore,
    ranks: np.ndarray,
    tabu_length: int,
    max_tabu: int,
    max_iter: int | None,
    constraints: Constraints | None = None,
    forbidden: np.ndarray | None = None,
    max_parents: int | None = None,
) -> tuple[StructureKey, int]:
    """Run tabu search from the start DAG; return the best DAG's structure key and moves taken.

    Each step adds the current DAG to the tabu list and takes the best improving move outside it.
    Without one, the search stops after ``max_tabu`` non-improving moves since the last new best;
    otherwise it counts one more and takes the best non-improving move outside the list, leaving the
    score unchanged or worse. A new best score lower by more than ``_EPS_IMPROVE`` resets the count.
    The search also stops when every legal move leads into the list, or after ``max_iter`` moves
    unless it is ``None``. ``ranks`` orders tied moves, :func:`~andrey.search.hc.tie_ranks` of the
    data. The start DAG is empty, or with ``constraints`` the required edges; no move then removes
    or reverses one or adds a forbidden edge. ``forbidden[v, u]`` bars the edge ``u -> v`` beside
    the knowledge: a mixed-data score's edges from a continuous into a categorical column. With
    ``max_parents``, no move gives a node more parents.
    """
    d = len(ranks)
    adj_int, parents_of = start_dag(d, constraints, max_parents)
    local = [score.score(i, parents_of[i]) for i in range(d)]
    best_key: StructureKey = tuple(adj_int)
    best_total = math.fsum(local)
    recent: deque[StructureKey] = deque(maxlen=tabu_length)
    losses = 0
    moves = 0
    delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
    required = None
    if constraints is not None:
        knowledge_forbidden = constraints.forbidden.T  # [v, u]: no u -> v
        forbidden = knowledge_forbidden if forbidden is None else forbidden | knowledge_forbidden
        required = constraints.required.T
    scanner = MoveScanner(
        delta, d, ranks=ranks, forbidden=forbidden, required=required, max_parents=max_parents
    )
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
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    max_iter: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
    max_parents: int | None = None,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by tabu search over a linear-Gaussian, categorical, or mixed-data score.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score, one of :data:`TABU_SCORES`: ``"bic"`` (linear-Gaussian BIC), ``"bdeu"``
        (:class:`~andrey.core.score.BDeuScore` with its default structure prior) or
        ``"discrete_bic"`` (:class:`~andrey.core.score.CategoricalBICScore`), both on category
        codes, or
        ``"degen_gauss"`` or ``"cond_gauss"``, the mixed-data scores over ``categorical``.
    penalty_discount : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    equivalent_sample_size : float
        Prior weight of ``"bdeu"``, the equivalent sample size of its uniform Dirichlet prior
        (:class:`~andrey.core.score.BDeuScore`); positive. The other scores ignore it.
    categorical : ndarray of bool or None
        The categorical columns, for a mixed-data score; such a score allows no edge from a
        continuous into a categorical column.
    tabu_length : int
        Number of recently visited DAGs a move may not return to.
    max_tabu : int or None
        Non-improving moves allowed without a new best score before stopping, at most
        ``tabu_length``; ``None`` means ``tabu_length``.
    max_iter : int or None
        Maximum moves, improving or not; ``None`` sets no limit. Taking ``max_iter`` moves warns
        with ``SearchLimitWarning``.
    constraints : Constraints or None
        Background knowledge: the search starts from the required edges, never removes or reverses
        one, and never adds a forbidden edge. The CPDAG is then narrowed by the knowledge
        (:func:`~andrey.core.knowledge.mpdag`).
    max_parents : int or None
        Most parents of any node; no move gives a node more. ``None`` sets no cap.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) of the best DAG visited and that DAG's total score.
    """
    if score_func not in TABU_SCORES:
        raise NotImplementedError(
            f"unsupported score_func {score_func!r}; use one of {TABU_SCORES}"
        )
    if max_tabu is None:
        max_tabu = tabu_length
    if not 1 <= max_tabu <= tabu_length:
        raise ValueError(f"max_tabu must lie in [1, tabu_length={tabu_length}], got {max_tabu}")
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score = make_score(
        score_func,
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        categorical=categorical,
    )
    mask = np.full(d, score_func in DISCRETE_SCORES)
    forbidden = None
    if score_func in MIXED_SCORES:  # [v, u]: categorical v, continuous u
        mask = column_mask(categorical, d)
        forbidden = mask[:, None] & ~mask[None, :]
        if constraints is not None and (constraints.required & forbidden.T).any():
            raise ValueError(
                "knowledge requires an edge from a continuous into a categorical column, which a "
                "mixed-data score does not allow"
            )

    ranks = tie_ranks(X, mask)
    # One BLAS thread keeps the Schur updates reproducible across thread counts.
    with backend.one_blas_thread():
        best_key, moves = _search(
            score, ranks, tabu_length, max_tabu, max_iter, constraints, forbidden, max_parents
        )
    if max_iter is not None and moves == max_iter:
        warn_search_limit("tabu", "max_iter", max_iter, "moves")

    # The best DAG as unsigned endpoint marks (u -> v: TAIL at u, ARROW at v), reduced to its CPDAG.
    parents = [[u for u in range(d) if best_key[u] >> v & 1] for v in range(d)]
    dag = np.full((d, d), NULL, dtype=np.int8)
    for v in range(d):
        for p in parents[v]:
            dag[p, v] = TAIL
            dag[v, p] = ARROW
    total = float(sum(score.score(i, parents[i]) for i in range(d)))
    cpdag = dag2cpdag(dag)
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    return to_structure(cpdag, kind="cpdag"), total
