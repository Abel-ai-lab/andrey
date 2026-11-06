"""Restrict-maximize searches, the engine behind :func:`andrey.mmhc` and :func:`andrey.rsmax2`.

The algorithm, its settings, and its references are described on :func:`andrey.rsmax2`. The
restrict phase learns an undirected candidate graph with conditional-independence tests; the
maximize phase runs hill climbing or tabu search with every pair outside it forbidden in both
directions, through the same :class:`~andrey.core.knowledge.Constraints` that carry background
knowledge.
"""

from __future__ import annotations

import numpy as np

from andrey.constraint.blanket import local_structure
from andrey.constraint.pc import _discover_skeleton
from andrey.core import NULL, GraphStructure, backend
from andrey.core.knowledge import Constraints
from andrey.search.hc import hc
from andrey.search.tabu import tabu

#: The restrict phases: the skeleton of MMPC, of IAMB, or of Inter-IAMB, or PC's skeleton.
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
    seed: int | None = None,
    max_depth: int | None = None,
) -> np.ndarray:
    """The restrict phase: the pairs the maximize phase may join, a symmetric boolean matrix.

    ``"mmpc"``, ``"iamb"``, and ``"inter_iamb"`` keep the skeleton
    :func:`~andrey.constraint.blanket.local_structure` returns for them: the pairs each in the
    other's MMPC set, or each in the other's Markov blanket and not separated by a subset of the
    smaller blanket (:mod:`andrey.constraint.blanket`). ``"pc"``
    keeps PC's skeleton (:func:`~andrey.constraint.pc._discover_skeleton`). Each runs the
    ``indep_test`` test at level ``alpha``; ``categorical`` marks the categorical columns for a
    mixed-data test; ``seed`` fixes the draws of a stochastic test (``None`` uses
    ``ANDREY_SEED``); ``max_depth`` caps every test's conditioning set.
    """
    if restrict == "pc":
        adj, _, _ = _discover_skeleton(
            X,
            alpha,
            indep_test,
            categorical=categorical,
            record_pvalues=False,
            workers=backend.worker_count(None),
            max_depth=max_depth,
            warn_expensive=True,
            seed=seed,
        )
        return adj != NULL
    marks = local_structure(
        X,
        restrict,
        "skeleton",
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
    )
    return marks != NULL


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
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_iter: int | None = None,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    max_parents: int | None = None,
) -> tuple[GraphStructure, float, np.ndarray]:
    """Run the ``restrict`` phase, then the ``maximize`` score search inside its candidate graph.

    Returns the CPDAG, the total score of the DAG found (lower is better), and the candidate graph.
    ``constraints`` reach the maximize phase whole, and the candidate graph as its allowed pairs
    together with the pairs of the required edges (:func:`restricted`). ``seed`` fixes the draws of
    a stochastic ``indep_test`` in the restrict phase (``None`` uses ``ANDREY_SEED``). ``max_depth``
    caps the restrict phase's conditioning sets and ``max_parents`` each node's parents in the
    maximize phase.
    """
    candidates = candidate_graph(
        X,
        restrict,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
    )
    limits = restricted(constraints, candidates)
    if maximize == "hc":
        cpdag, score = hc(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            max_iter=max_iter,
            categorical=categorical,
            constraints=limits,
            max_parents=max_parents,
        )
    else:
        cpdag, score = tabu(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            tabu_length=tabu_length,
            max_tabu=max_tabu,
            max_iter=max_iter,
            categorical=categorical,
            constraints=limits,
            max_parents=max_parents,
        )
    return cpdag, score, candidates
