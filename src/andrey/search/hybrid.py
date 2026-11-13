"""Restrict-maximize searches, the engine behind :func:`andrey.mmhc` and :func:`andrey.rsmax2`.

The algorithm, its settings, and its references are described on :func:`andrey.rsmax2`. The
restrict phase learns an undirected candidate graph with conditional-independence tests; the
maximize phase runs hill climbing or tabu search with every pair outside it forbidden in both
directions, through the same :class:`~andrey.core.knowledge.Constraints` that carry background
knowledge. The MMPC and PC restrict phases read the knowledge as those learners do on their own,
so the two ends of a required edge condition the tests of the other candidates; the IAMB and
Inter-IAMB restrict phases read the data alone (:func:`candidate_graph`).
"""

from __future__ import annotations

import numpy as np

from andrey.constraint.blanket import local_structure
from andrey.constraint.pc import adjacency_search, knowledge_masks
from andrey.core import NULL, GraphStructure, backend
from andrey.core.ci import make_indep_test
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
    degrees_of_freedom: str = "mean",
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """The restrict phase: the pairs the maximize phase may join, a symmetric boolean matrix.

    ``"mmpc"``, ``"iamb"``, and ``"inter_iamb"`` keep the skeleton
    :func:`~andrey.constraint.blanket.local_structure` returns for them: the pairs each in the
    other's MMPC set, or each in the other's Markov blanket and not separated by a subset of the
    smaller blanket (:mod:`andrey.constraint.blanket`). ``"pc"``
    keeps PC's skeleton (:func:`~andrey.constraint.pc.adjacency_search`). Each runs the
    ``indep_test`` test at level ``alpha``; ``categorical`` marks the categorical columns for a
    mixed-data test; ``seed`` fixes the draws of a stochastic test (``None`` uses
    ``ANDREY_SEED``); ``degrees_of_freedom`` reaches a categorical test; ``max_depth`` caps every
    test's conditioning set.

    ``constraints`` reach ``"mmpc"`` and ``"pc"`` as they do when the learner runs on its own: the
    pair of a required edge is kept without a test and conditions the tests of the other pairs
    (each end is a member of the other's set from the start), and a pair the knowledge makes
    non-adjacent is never a candidate. A neighbor the knowledge names can separate a pair that the
    neighbors found from the data leave together. ``"iamb"`` and ``"inter_iamb"`` do not read
    them: required edges enlarge the blankets, and the step that drops a pair some subset of the
    smaller blanket separates then drops more true pairs, while their skeleton holds almost no
    false pair for the knowledge to remove (:func:`andrey.rsmax2` gives the measured cost).
    :func:`restricted` adds the pairs of the required edges for every ``restrict``.
    """
    if restrict == "pc":
        allowed, keep, _ = knowledge_masks(constraints)
        skeleton, _, _ = adjacency_search(
            make_indep_test(
                indep_test,
                X,
                categorical=categorical,
                seed=seed,
                degrees_of_freedom=degrees_of_freedom,
            ),
            X.shape[1],
            alpha,
            support=allowed,
            keep=keep,
            max_depth=max_depth,
            workers=backend.worker_count(None),
            warn_expensive=True,
        )
        return skeleton != NULL
    marks = local_structure(
        X,
        restrict,
        "skeleton",
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints if restrict == "mmpc" else None,
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
    degrees_of_freedom: str = "mean",
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
    ``constraints`` reach the restrict phase as :func:`candidate_graph` describes, for ``"mmpc"``
    and ``"pc"`` only, and the maximize phase whole, with the candidate graph as its allowed pairs
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
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints,
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
