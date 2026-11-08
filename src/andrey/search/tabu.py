"""Tabu search over DAGs, the engine behind :func:`andrey.tabu`.

``_search`` takes the moves of :class:`andrey.search.hc.MoveScanner`, and
:func:`andrey.search.hc.greedy_search` runs its starts, so tabu search shares hill climbing's tie
rules, knowledge, and restarts. Output is the best DAG's CPDAG and total score; lower is better.
"""

from __future__ import annotations

import math
from collections import deque
from functools import partial

import numpy as np

from andrey.core import GraphStructure
from andrey.core.knowledge import Constraints
from andrey.core.ties import EPS_IMPROVE
from andrey.search.hc import MoveScanner, StructureKey, apply_move, greedy_search

__all__ = ["tabu"]


def _search(
    scanner: MoveScanner,
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    max_iter: int | None,
    *,
    tabu_length: int,
    max_tabu: int,
) -> int:
    """Run tabu search from the DAG ``adj_int``; end on the best DAG visited and count the moves.

    Each step adds the current DAG to the tabu list and takes the best improving move outside it.
    Without one, the search stops after ``max_tabu`` non-improving moves since the last new best;
    otherwise it counts one more and takes the best non-improving move outside the list, leaving the
    score unchanged or worse. A new best score lower by more than ``EPS_IMPROVE`` resets the count.
    The search also stops when every legal move leads into the list, or after ``max_iter`` moves
    unless it is ``None``. The moves keep the rules of ``scanner``, and the totals add up its
    ``score``. ``adj_int`` and ``parents_of``, the bitset rows and sorted parent sets of the start
    DAG, become those of the best DAG visited.
    """
    score = scanner.score
    d = len(adj_int)
    local = [score.score(i, parents_of[i]) for i in range(d)]
    best_key: StructureKey = tuple(adj_int)
    best_total = math.fsum(local)
    recent: deque[StructureKey] = deque(maxlen=tabu_length)
    losses = 0
    moves = 0
    while max_iter is None or moves < max_iter:
        recent.append(tuple(adj_int))
        tabu = frozenset(recent)
        found = scanner.best(adj_int, parents_of, floor=-EPS_IMPROVE, tabu=tabu)
        if found is None:
            if losses >= max_tabu:
                break
            losses += 1
            found = scanner.best(adj_int, parents_of, floor=math.inf, tabu=tabu)
            if found is None:  # every legal move leads into the tabu list
                break
        move = found[4]
        apply_move(adj_int, parents_of, move)
        moves += 1
        op, u, v = move
        for node in (u, v) if op == "reverse" else (v,):
            local[node] = score.score(node, parents_of[node])
        total = math.fsum(local)
        if total < best_total - EPS_IMPROVE:
            best_key, best_total, losses = tuple(adj_int), total, 0
    adj_int[:] = best_key
    for v in range(d):
        parents_of[v] = [u for u in range(d) if best_key[u] >> v & 1]
    return moves


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
    n_starts: int = 1,
    n_perturbations: int = 1,
    seed: int = 0,
    sources: tuple[int, ...] = (),
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by tabu search over a linear-Gaussian, categorical, or mixed-data score.

    Each start after the first searches again from the best DAG so far after
    :func:`~andrey.search.hc.perturb` makes ``n_perturbations`` random changes to it, which keep
    the moves' rules, with a new tabu list, and replaces it only when its best DAG scores lower by
    more than :data:`~andrey.search.hc.START_EPS`.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score, one of :data:`~andrey.core.score.SCORES`: ``"bic"`` (linear-Gaussian BIC),
        ``"bdeu"`` (:class:`~andrey.core.score.BDeuScore` with its default structure prior) or
        ``"discrete_bic"`` (:class:`~andrey.core.score.CategoricalBICScore`), both on category
        codes, or ``"degen_gauss"`` or ``"cond_gauss"``, the mixed-data scores over
        ``categorical``.
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
        Maximum moves of each start, improving or not; ``None`` sets no limit. A start that takes
        ``max_iter`` moves warns with ``SearchLimitWarning``, once per call.
    constraints : Constraints or None
        Background knowledge: the search starts from the required edges, never removes or reverses
        one, and never adds a forbidden edge. The CPDAG is then narrowed by the knowledge
        (:func:`~andrey.core.knowledge.mpdag`).
    max_parents : int or None
        Most parents of any node; no move gives a node more. ``None`` sets no cap.
    n_starts : int
        Number of searches, at least ``1``.
    n_perturbations : int
        Random changes to the best DAG so far before each start after the first, at least ``1``.
    seed : int
        Seed of the changes; start ``k`` draws from :func:`~andrey.core.seeding.start_rng`.
    sources : tuple of int
        Columns no other column may cause, such as an environment index, as for
        :func:`~andrey.search.hc.hc`.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) of the best DAG visited and that DAG's total score; with
        ``sources``, the partially directed graph that also directs their edges.
    """
    if max_tabu is None:
        max_tabu = tabu_length
    if not 1 <= max_tabu <= tabu_length:
        raise ValueError(f"max_tabu must lie in [1, tabu_length={tabu_length}], got {max_tabu}")
    return greedy_search(
        data,
        partial(_search, tabu_length=tabu_length, max_tabu=max_tabu),
        method="tabu",
        score_func=score_func,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        max_iter=max_iter,
        categorical=categorical,
        constraints=constraints,
        max_parents=max_parents,
        n_starts=n_starts,
        n_perturbations=n_perturbations,
        seed=seed,
        sources=sources,
    )
