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

from andrey.constraint.blanket import (
    blanket_cpdag,
    blanket_graph,
    markov_blanket,
    mmpc_sets,
    mmpc_skeleton,
    parents_children,
    prune_spouses,
    reach,
)
from andrey.constraint.pc import _discover_skeleton
from andrey.core import NULL, TAIL, GraphStructure, backend
from andrey.core.ci import (
    DISCRETE_INDEP_TESTS,
    MIXED_INDEP_TESTS,
    CITest,
    make_indep_test,
    pins_blas,
)
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
    seed: int | None = None,
    max_depth: int | None = None,
) -> np.ndarray:
    """The restrict phase: the pairs the maximize phase may join, a symmetric boolean matrix.

    ``"mmpc"`` keeps a pair when each variable is in the other's MMPC set and ``"iamb"`` or
    ``"inter_iamb"`` when each is in the other's Markov blanket
    (:mod:`andrey.constraint.blanket`); ``"pc"`` keeps PC's skeleton
    (:func:`~andrey.constraint.pc._discover_skeleton`). Each runs the ``indep_test`` test at level
    ``alpha``; ``categorical`` marks the categorical columns for a mixed-data test; ``seed`` fixes
    the draws of a stochastic test (``None`` uses ``ANDREY_SEED``); ``max_depth`` caps every
    test's conditioning set.
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
    return local_graph(
        X,
        restrict,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
    )


def _ranked_test(
    X: np.ndarray, indep_test: str, categorical: np.ndarray | None, seed: int | None
) -> tuple[CITest, list[int]]:
    """The ``indep_test`` test, seeded by ``seed``, over ``X`` and each column's
    :func:`~andrey.search.hc.tie_ranks`, its categorical columns read as categories."""
    d = X.shape[1]
    test = make_indep_test(indep_test, X, categorical=categorical, seed=seed)
    if indep_test in DISCRETE_INDEP_TESTS:
        mask = np.ones(d, dtype=bool)
    elif indep_test in MIXED_INDEP_TESTS:
        mask = column_mask(categorical, d)
    else:
        mask = None
    return test, tie_ranks(X, mask).tolist()


def local_graph(
    X: np.ndarray,
    method: str,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Every variable's MMPC (``"mmpc"``), IAMB (``"iamb"``), or Inter-IAMB (``"inter_iamb"``) set
    under the AND rule: MMPC's skeleton or the moral graph, a symmetric boolean matrix. ``seed``
    fixes the draws of a stochastic test (``None`` uses ``ANDREY_SEED``); ``max_depth`` caps every
    test's conditioning set; ``constraints`` set the members as :mod:`andrey.constraint.blanket`
    describes."""
    test, ranks = _ranked_test(X, indep_test, categorical, seed)
    d = X.shape[1]
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        if method == "mmpc":
            return mmpc_skeleton(test, d, alpha, ranks, max_depth, constraints)
        return blanket_graph(
            test,
            d,
            alpha,
            ranks,
            interleaved=method == "inter_iamb",
            max_depth=max_depth,
            constraints=constraints,
        )


def local_structure(
    X: np.ndarray,
    method: str,
    structure: str,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Every variable's MMPC, IAMB, or Inter-IAMB set joined into the graph ``structure`` names, as
    an endpoint-mark matrix.

    ``"moral"`` (IAMB, Inter-IAMB) is :func:`local_graph`'s moral graph, undirected. ``"skeleton"``
    is MMPC's skeleton, or the moral graph less the pairs :func:`prune_spouses` separates.
    ``"cpdag"`` orients that skeleton (:func:`~andrey.constraint.blanket.blanket_cpdag`) against
    the blankets: the moral graph, or for MMPC each variable's set together with its members' sets
    before the symmetry check (:func:`~andrey.constraint.blanket.reach`). The other settings are
    :func:`local_graph`'s.
    """
    test, ranks = _ranked_test(X, indep_test, categorical, seed)
    d = X.shape[1]
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        if method == "mmpc":
            sets = mmpc_sets(test, d, alpha, ranks, max_depth, constraints)
            skeleton, blankets = sets & sets.T, reach(sets)
        else:
            blankets = blanket_graph(
                test,
                d,
                alpha,
                ranks,
                interleaved=method == "inter_iamb",
                max_depth=max_depth,
                constraints=constraints,
            )
            skeleton = (
                blankets
                if structure == "moral"
                else prune_spouses(test, blankets, alpha, max_depth, constraints)
            )
        if structure != "cpdag":
            return np.where(skeleton, TAIL, NULL).astype(np.int8)
        return blanket_cpdag(test, skeleton, blankets, alpha, max_depth, constraints)


def local_set(
    X: np.ndarray,
    method: str,
    target: int,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> list[int]:
    """``target``'s set in the order its members joined, by MMPC (``"mmpc"``), IAMB (``"iamb"``),
    or Inter-IAMB (``"inter_iamb"``).

    MMPC keeps the members of ``target``'s parents and children whose own set holds ``target``
    (the symmetry check); IAMB and Inter-IAMB return the Markov blanket. ``seed`` fixes the draws of
    a stochastic test (``None`` uses ``ANDREY_SEED``); ``max_depth`` caps every test's conditioning
    set; ``constraints`` set the members as :mod:`andrey.constraint.blanket` describes.
    """
    test, ranks = _ranked_test(X, indep_test, categorical, seed)
    d = X.shape[1]
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        if method == "mmpc":
            found = parents_children(test, target, d, alpha, ranks, max_depth, constraints)
            return [
                x
                for x in found
                if target in parents_children(test, x, d, alpha, ranks, max_depth, constraints)
            ]
        return markov_blanket(
            test,
            target,
            d,
            alpha,
            ranks,
            interleaved=method == "inter_iamb",
            max_depth=max_depth,
            constraints=constraints,
        )


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
