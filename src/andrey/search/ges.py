"""Greedy Equivalence Search over a decomposable score, the engine behind :func:`andrey.ges`.

The linear-Gaussian BIC scores each candidate by its Schur delta; the categorical BDeu and BIC and
the mixed-data scores difference two full local scores. The objective returned alongside the CPDAG
is recomputed from it. The algorithm, its settings, and its reference are described on
:func:`andrey.ges`.
"""

from __future__ import annotations

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, backend
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import _cpdag_of_pdag, pdag2dag, to_structure
from andrey.core.score import DISCRETE_SCORES, MIXED_SCORES, SCORES, Score, make_score
from andrey.search import operators as ops
from andrey.search.hc import tie_ranks


def _reform_cpdag(adj: np.ndarray) -> np.ndarray:
    """Canonicalize a search PDAG to its CPDAG via a consistent DAG extension."""
    return _cpdag_of_pdag(adj)


def required_dag(constraints: Constraints) -> np.ndarray:
    """The DAG made of the required edges, as a mark matrix: where a search under knowledge starts,
    once taken to its equivalence class."""
    dag = np.zeros(constraints.required.shape, dtype=np.int8)
    dag[constraints.required] = TAIL
    dag[constraints.required.T] = ARROW
    return dag


def _reachable(adj: np.ndarray):
    """Return ``reachable(y, x, clique)`` for the CPDAG ``adj``, the full-rescan references' check.

    ``reachable(y, x, clique)`` is ``True`` when ``x`` is semi-directed reachable from ``y`` once
    the ``clique`` nodes are removed: the condition that invalidates ``Insert(x, y, T)``.
    """
    succ = ops.tail_successor_lists(adj)
    return lambda y, x, clique: not ops.blocks_semi_directed_paths(succ, y, x, clique)


def _recompute_objective(adj: np.ndarray, score: Score, n: int) -> float:
    """Total score of the CPDAG found: ``sum_v score(v, Pa_v)`` over a consistent DAG extension.

    The objective is recomputed from the final structure rather than accumulated from the search's
    Schur deltas, so it is independent of the (non-bit-pure) delta arithmetic and identical for
    serial and parallel runs on the same CPDAG. Any consistent extension gives the same total (BIC,
    the categorical scores, and the mixed-data scores are score-equivalent), so ``pdag2dag``'s
    choice of representative is immaterial.
    """
    dag = pdag2dag(adj)
    return float(sum(score.score(v, ops.parents(dag, v).tolist()) for v in range(n)))


#: How :func:`ges` applies ``max_parents``: ``"cap"`` on the most parents any DAG of the class gives
#: a node, in every CPDAG of the search, ``"gate"`` on its directed parents in the forward phase.
PARENT_RULES: tuple[str, ...] = ("cap", "gate")


def _parent_limits(
    n: int, max_parents: float | None, rule: str
) -> tuple[float | None, float | None]:
    """The forward phase's gate and the cap that ``max_parents`` sets under ``rule``.

    No ``max_parents`` leaves the gate of ``n / 2`` and no cap, under either rule.
    """
    if rule not in PARENT_RULES:
        raise ValueError(f"unsupported _parent_rule {rule!r}; use one of {PARENT_RULES}")
    if max_parents is None:
        return n / 2, None
    return (max_parents, None) if rule == "gate" else (None, max_parents)


#: The local scores :func:`ges` accepts.
GES_SCORES: tuple[str, ...] = SCORES


def ges(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: float | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
    workers: int | None = None,
    _parent_rule: str = "cap",
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Greedy Equivalence Search over a BIC, categorical, or mixed-data score.

    Runs the forward (Insert) then backward (Delete) greedy phases from the empty graph, re-forming
    the CPDAG after every accepted move, then returns the discovered CPDAG and its total objective
    (lower is better). Under BIC each candidate move is scored by the Schur delta-BIC; under the
    other scores by differencing two full local scores, in this process. The returned objective is
    recomputed from the final CPDAG (``sum_v score(v, Pa_v)``), so it is independent of the delta
    arithmetic and identical for serial and parallel runs. Serial and parallel are the same
    decomposition: ``ges()`` always routes through :func:`~andrey.search._parallel_ges.run_ges`,
    which fans out BIC passes only above the fixed work threshold.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score: ``"local_score_BIC"`` or ``"local_score_BIC_from_cov"`` (linear-Gaussian BIC),
        ``"local_score_BDeu"`` (BDeu on category codes,
        :class:`~andrey.core.score.BDeuScore` with its default priors),
        ``"local_score_BIC_categorical"`` (the multinomial BIC on category codes,
        :class:`~andrey.core.score.CategoricalBICScore`), or a mixed-data BIC,
        ``"local_score_degen_gauss"`` (:class:`~andrey.core.score.DGScore`) or
        ``"local_score_cond_gauss"`` (:class:`~andrey.core.score.CGScore`).
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    max_parents : float or None
        A cap on the most parents any DAG of the class gives a node, in the returned CPDAG and in
        the CPDAG after each move (:func:`~andrey.search.operators.max_parents_in_class`): an Insert
        or Delete whose CPDAG breaks it is skipped for the next best move. ``None`` sets no cap and
        keeps the forward phase's gate on a node that already holds more than ``n_features / 2``
        directed parents, which the backward phase lacks. With a cap, ``constraints`` may not
        require edges that already break it.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.
    constraints : Constraints or None
        Background knowledge. An Insert never adds a forbidden edge, a Delete never removes a
        required one, the search starts from the class of the required edges, and a move whose
        class holds no DAG that satisfies the knowledge is skipped for the next best one. The
        result is that class narrowed by the knowledge (:func:`~andrey.core.knowledge.mpdag`).
    workers : int or None
        Worker processes for the BIC passes above the work threshold; ``None`` reads
        ``andrey.config.num_workers``.
    _parent_rule : {"cap", "gate"}
        Private lever for the benchmark harness: ``"gate"`` reads ``max_parents`` as the forward
        phase's gate, the rule of the default of ``n_features / 2``: a node that holds more than
        ``max_parents`` directed parents gains no new ones, and a move can still leave a node above
        it. Not part of the public contract.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total score.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    gate, cap = _parent_limits(n, max_parents, _parent_rule)
    if score_func not in GES_SCORES:
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use one of {GES_SCORES}")
    score = make_score(
        score_func, X, lambda_value=lambda_value, ebic_gamma=ebic_gamma, categorical=categorical
    )

    adj = np.zeros((n, n), dtype=np.int8)
    if constraints is not None and constraints.required.any():
        adj = _cpdag_of_pdag(required_dag(constraints))
    if cap is not None and ops.exceeds_parent_cap(adj, cap):
        raise ValueError(
            f"the required edges give a node more than max_parents={max_parents} parents "
            "in some graph of their class"
        )
    # Local import breaks the ges <-> _parallel_ges cycle.
    from andrey.search import _parallel_ges

    # Pin BLAS to one thread for the score-diff loops (the covariance above is already computed):
    # multi-thread BLAS reduction order is not bit-stable, so pinning makes the score host- and
    # thread-count-reproducible and equal to the (1-thread) parallel workers.
    mask = np.full(n, score_func in DISCRETE_SCORES)
    if score_func in MIXED_SCORES:
        mask = column_mask(categorical, n)
    ranks = tie_ranks(X, mask).tolist()
    with backend.one_blas_thread():
        workers = backend.worker_count(workers) if n > 1 else 1
        adj = _parallel_ges.run_ges(
            adj,
            score,
            n,
            gate,
            workers=workers,
            ranks=ranks,
            constraints=constraints,
            cap=cap,
        )
        total = _recompute_objective(adj, score, n)
    if constraints is not None:
        adj = mpdag(adj, constraints)
    return to_structure(adj, kind="cpdag"), total
