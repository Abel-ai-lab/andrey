"""Restrict-maximize searches, the engine behind :func:`andrey.mmhc` and :func:`andrey.rsmax2`.

The algorithm, its settings, and its references are described on :func:`andrey.rsmax2`. The
restrict phase learns an undirected candidate graph with conditional-independence tests; the
maximize phase runs hill climbing or tabu search with every pair outside it forbidden in both
directions, through the same :class:`~andrey.core.knowledge.Constraints` that carry background
knowledge.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from andrey.constraint.blanket import blanket_graph, mmpc_skeleton
from andrey.constraint.pc import _discover_skeleton
from andrey.core import NULL, GraphStructure, backend
from andrey.core.ci import DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS, make_indep_test, pins_blas
from andrey.core.knowledge import Constraints
from andrey.core.mixed import column_mask
from andrey.search.hc import hc, tie_ranks
from andrey.search.tabu import tabu

#: The restrict phases: MMPC's skeleton, the moral graph of IAMB's or Inter-IAMB's Markov
#: blankets, or PC's skeleton.
RESTRICTS: tuple[str, ...] = ("mmpc", "iamb", "inter_iamb", "pc")
#: The maximize phases: hill climbing or tabu search.
MAXIMIZES: tuple[str, ...] = ("hc", "tabu")


def candidate_graph(
    X: np.ndarray,
    restrict: str,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
) -> np.ndarray:
    """The restrict phase: the pairs the maximize phase may join, a symmetric boolean matrix.

    ``"mmpc"`` keeps a pair when each variable is in the other's MMPC set and ``"iamb"`` or
    ``"inter_iamb"`` when each is in the other's Markov blanket
    (:mod:`andrey.constraint.blanket`); ``"pc"`` keeps PC's skeleton
    (:func:`~andrey.constraint.pc._discover_skeleton`). Each runs the ``indep_test`` test at level
    ``alpha``; ``categorical`` marks the categorical columns for a mixed-data test.
    """
    d = X.shape[1]
    if restrict == "pc":
        adj, _, _ = _discover_skeleton(
            X,
            alpha,
            indep_test,
            categorical=categorical,
            record_pvalues=False,
            workers=backend.worker_count(None),
        )
        return adj != NULL
    test = make_indep_test(indep_test, X, categorical=categorical)
    if indep_test in DISCRETE_INDEP_TESTS:
        mask = np.ones(d, dtype=bool)
    elif indep_test in MIXED_INDEP_TESTS:
        mask = column_mask(categorical, d)
    else:
        mask = None
    ranks = tie_ranks(X, mask).tolist()
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        if restrict == "mmpc":
            return mmpc_skeleton(test, d, alpha, ranks)
        return blanket_graph(test, d, alpha, ranks, interleaved=restrict == "inter_iamb")


def restricted(constraints: Constraints | None, candidates: np.ndarray) -> Constraints:
    """``constraints`` with every pair outside ``candidates`` forbidden in both directions.

    A pair a required edge joins stays allowed; a pair the knowledge makes non-adjacent stays
    forbidden. ``None`` constrains nothing else.
    """
    d = candidates.shape[0]
    if constraints is None:
        none = np.zeros((d, d), dtype=bool)
        constraints = Constraints(none, none.copy(), np.full(d, -1, dtype=np.intp))
    required = constraints.required
    forbidden = constraints.forbidden | ~(candidates | required | required.T)
    np.fill_diagonal(forbidden, False)
    return Constraints(forbidden, required, constraints.tier)


def rsmax2(
    X: np.ndarray,
    *,
    restrict: str = "mmpc",
    maximize: str = "hc",
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    max_iter: int | None = None,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float, np.ndarray]:
    """Run the ``restrict`` phase, then the ``maximize`` score search inside its candidate graph.

    Returns the CPDAG, the total score of the DAG found (lower is better), and the candidate graph.
    ``constraints`` reach the maximize phase whole, and the candidate graph as its allowed pairs
    together with the pairs of the required edges (:func:`restricted`).
    """
    candidates = candidate_graph(
        X, restrict, alpha=alpha, indep_test=indep_test, categorical=categorical
    )
    limits = restricted(constraints, candidates)
    if maximize == "hc":
        cpdag, score = hc(
            X,
            score_func=score_func,
            lambda_value=lambda_value,
            max_iter=max_iter,
            categorical=categorical,
            constraints=limits,
        )
    else:
        cpdag, score = tabu(
            X,
            score_func=score_func,
            lambda_value=lambda_value,
            tabu_length=tabu_length,
            max_tabu=max_tabu,
            max_iter=max_iter,
            constraints=limits,
        )
    return cpdag, score, candidates
