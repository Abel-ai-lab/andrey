"""GFCI, BOSS-FCI, and GRaSP-FCI: the engines behind :func:`andrey.gfci`, :func:`andrey.boss_fci`,
and :func:`andrey.grasp_fci`.

The algorithms, their settings, and their references are described on those functions.

The steps follow Algorithm 1 of Ogarrio, Spirtes, and Ramsey (2016). GES, BOSS, or GRaSP returns a
CPDAG, the pattern, and :func:`fci_from_pattern` runs the rest. FCI's adjacency search, started from
the pattern's adjacencies instead of a complete graph, removes the pairs a subset of either
endpoint's neighbors separates. An unshielded triple of the result is a collider when it is one in
the pattern, or, when the pattern shields it, when the separating set of its endpoints leaves out
its middle node. The possible-d-separation and R0-R10 passes of :mod:`andrey.constraint.fci`
follow, with its rule implementations, conditional-independence tests, and mark convention.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import (
    check_collider_rule,
    marginal_separated,
    separating_subsets,
    vote_triples,
)
from andrey.constraint.fci import (
    _adjacent,
    _largest_size,
    circle_pag,
    fci_orient,
    knowledge_marks,
    orient_colliders,
    possible_dsep_step,
    validate_possible_dsep,
)
from andrey.constraint.fci_jci import mark_index
from andrey.constraint.pc import adjacency_search, knowledge_masks
from andrey.core import NULL, GraphStructure, backend
from andrey.core.ci import CITest, make_indep_test
from andrey.core.orient import from_structure
from andrey.search.boss import boss
from andrey.search.ges import ges
from andrey.search.grasp import grasp

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def gfci(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: np.ndarray | None = None,
    possible_dsep: str = "order_free",
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool = False,
    workers: int | None = None,
    seed: int | None = None,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> GraphStructure:
    """Discover a PAG with GES adjacency and FCI orientation.

    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data score and test.
    ``possible_dsep`` (``"order_free"`` or ``"fast"``) selects the order the possible-d-sep step
    visits the nodes in: the :func:`~andrey.core.ties.tie_ranks` order or the column order.
    ``in_column_order`` applies the orientation rules one firing at a time in column order instead
    of in order-free rounds (:func:`~andrey.constraint.fci.fci_orient`).
    ``max_parents`` reaches the GES phase as :func:`~andrey.search.ges.ges`'s cap.
    ``max_depth`` caps the size of every conditioning set a test uses, as in
    :func:`~andrey.constraint.fci.fci`, and of the separating sets :func:`_derive_sepsets` searches.
    ``constraints`` reach the GES phase, and the test phase honors them as FCI does: no step tests
    or removes a required pair, and their endpoints are set before the orientation rules. GES never
    joins a pair they make non-adjacent, so its colliders over that pair come from the pattern, and
    its separating set from :func:`_derive_sepsets`, as for any pair GES leaves non-adjacent.
    ``workers`` worker processes share the GES passes and the adjacency search's larger
    conditioning-size passes, each above its own work threshold; ``None`` reads
    ``andrey.config.n_jobs``. ``collider_rule`` ``"conservative"`` or ``"majority"`` decides
    the triples the pattern shields by the vote of
    :func:`~andrey.constraint._skeleton.vote_triples` after the possible-d-sep step. ``seed`` fixes
    the draws of a stochastic ``indep_test`` (:data:`~andrey.core.ci.STOCHASTIC_INDEP_TESTS`);
    ``None`` uses ``ANDREY_SEED``. The private ``_ambiguous`` sink receives the triples no
    orientation rule reads as a collider or a non-collider, keyed ``(a, b, c)`` with ``a < c``.
    """
    X = np.asarray(data, dtype=np.float64)
    workers = backend.worker_count(workers)

    check_collider_rule(collider_rule)
    validate_possible_dsep(possible_dsep)
    cpdag_structure, _score = ges(
        X,
        score_func=score_func,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        max_parents=max_parents,
        categorical=categorical,
        constraints=constraints,
        workers=workers,
    )
    return fci_from_pattern(
        X,
        cpdag_structure,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
        categorical=categorical,
        possible_dsep=possible_dsep,
        max_depth=max_depth,
        constraints=constraints,
        in_column_order=in_column_order,
        workers=workers,
        seed=seed,
        _ambiguous=_ambiguous,
    )


def boss_fci(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    n_starts: int = 1,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: np.ndarray | None = None,
    possible_dsep: str = "order_free",
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool = False,
    workers: int | None = None,
    seed: int | None = None,
    context: int | None = None,
    confounded_index: bool = True,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> GraphStructure:
    """Discover a PAG with BOSS adjacency and FCI orientation.

    The score settings, ``seed``, ``max_parents``, ``n_starts``, and ``constraints`` reach the BOSS
    search (:func:`~andrey.search.boss.boss`), and the rest act as for :func:`gfci`. ``context``,
    the column of an environment index, leads every BOSS order, and the test phase gives its edges
    the marks of Joint Causal Inference (:func:`fci_from_pattern`, with ``confounded_index``).
    """
    X = np.asarray(data, dtype=np.float64)
    check_collider_rule(collider_rule)
    validate_possible_dsep(possible_dsep)
    pattern, _score = boss(
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        seed=0 if seed is None else seed,
        n_starts=n_starts,
        score_func=score_func,
        max_parents=max_parents,
        categorical=categorical,
        constraints=constraints,
        sources=() if context is None else (context,),
    )
    return fci_from_pattern(
        X,
        pattern,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
        categorical=categorical,
        possible_dsep=possible_dsep,
        max_depth=max_depth,
        constraints=constraints,
        in_column_order=in_column_order,
        workers=backend.worker_count(workers),
        seed=seed,
        context=context,
        confounded_index=confounded_index,
        _ambiguous=_ambiguous,
    )


def grasp_fci(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    grasp_depth: int = 3,
    max_parents: int | None = None,
    n_starts: int = 1,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: np.ndarray | None = None,
    possible_dsep: str = "order_free",
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool = False,
    workers: int | None = None,
    seed: int | None = None,
    context: int | None = None,
    confounded_index: bool = True,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> GraphStructure:
    """Discover a PAG with GRaSP adjacency and FCI orientation.

    The score settings, ``grasp_depth``, ``seed``, ``max_parents``, ``n_starts``, and
    ``constraints`` reach the GRaSP search (:func:`~andrey.search.grasp.grasp`), and the rest act
    as for :func:`gfci`. ``context`` and ``confounded_index`` act as for :func:`boss_fci`.
    """
    X = np.asarray(data, dtype=np.float64)
    check_collider_rule(collider_rule)
    validate_possible_dsep(possible_dsep)
    pattern, _score = grasp(
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        depth=grasp_depth,
        seed=0 if seed is None else seed,
        n_starts=n_starts,
        score_func=score_func,
        max_parents=max_parents,
        categorical=categorical,
        constraints=constraints,
        sources=() if context is None else (context,),
    )
    return fci_from_pattern(
        X,
        pattern,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
        categorical=categorical,
        possible_dsep=possible_dsep,
        max_depth=max_depth,
        constraints=constraints,
        in_column_order=in_column_order,
        workers=backend.worker_count(workers),
        seed=seed,
        context=context,
        confounded_index=confounded_index,
        _ambiguous=_ambiguous,
    )


def fci_from_pattern(
    X: np.ndarray,
    cpdag: GraphStructure,
    *,
    alpha: float,
    indep_test: str,
    collider_rule: str,
    categorical: np.ndarray | None,
    possible_dsep: str,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool,
    workers: int = 1,
    seed: int | None = None,
    context: int | None = None,
    confounded_index: bool = True,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> GraphStructure:
    """Prune and orient the score search's ``cpdag`` with FCI's tests and rules: a PAG.

    FCI's adjacency search starts from the pattern's adjacencies; the pattern then decides the
    colliders on triples whose ends it leaves non-adjacent, and the separating sets the others,
    before the possible-d-separation step and the rules R1-R10. ``max_depth``, ``constraints``,
    ``workers`` (a worker count), ``seed``, and ``_ambiguous`` act on the test phase as for
    :func:`gfci`.

    ``context``, the column of an environment index, runs the test phase as
    :func:`~andrey.constraint.fci_jci.fci_jci` runs FCI's: before each collider pass, every edge
    of the index gets the marks of Joint Causal Inference
    (:func:`~andrey.constraint.fci_jci.mark_index`, with ``confounded_index``), and the rules leave
    out R5-R7, since JCI assumes no selection bias. The PAG still holds the index.
    """
    pattern = from_structure(cpdag)

    test = make_indep_test(indep_test, X, categorical=categorical, seed=seed)
    skeleton, sepsets, _ = adjacency_search(
        test,
        X.shape[1],
        alpha,
        support=pattern != NULL,
        keep=knowledge_masks(constraints)[1],
        max_depth=max_depth,
        workers=workers,
    )
    sepsets.update(
        _derive_sepsets(pattern, test, alpha, max_depth=max_depth, in_column_order=in_column_order)
    )

    marks = circle_pag(skeleton)
    knowledge_marks(marks, constraints)
    if context is not None:
        mark_index(marks, context, confounded_index)
    orient_colliders(marks, sepsets, pattern=pattern)
    marks = possible_dsep_step(
        marks,
        X,
        test,
        alpha,
        sepsets,
        possible_dsep=possible_dsep,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints,
    )
    if context is not None:
        mark_index(marks, context, confounded_index)
    ambiguous = (
        frozenset(
            triple
            for triple in vote_triples(
                marks, test, alpha, sepsets, collider_rule, max_depth=max_depth
            )
            if _adjacent(pattern, triple[0], triple[2])
        )
        if collider_rule != "sepsets"
        else frozenset()
    )
    if _ambiguous is not None:
        _ambiguous.update(ambiguous)
    orient_colliders(marks, sepsets, ambiguous, pattern=pattern)
    fci_orient(
        marks,
        test,
        alpha,
        sepsets,
        ambiguous,
        max_depth=max_depth,
        in_column_order=in_column_order,
        selection_bias=context is None,
    )

    return GraphStructure.from_numpy(marks, kind="pag")


def _derive_sepsets(
    cpdag: np.ndarray,
    test: CITest,
    alpha: float,
    *,
    max_depth: int | None = None,
    in_column_order: bool = False,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Separating sets for every non-adjacent CPDAG pair, from the pooled two-endpoint adjacency.

    For each non-adjacent ``(x, y)`` (``x < y``, row-major), searches subsets of
    ``adj(x) | adj(y) - {x, y}``, of at most ``max_depth`` members when one is given, by increasing
    size and records, both ways, the union of every subset of the first size that makes the pair
    independent, as the adjacency search records its sets; a pair with no such subset gets no
    entry. The union does not depend on the order of the tests. With ``in_column_order``, the first
    subset found in column order is recorded instead, and no subset after it is tested. The
    orientation rules that read a separating set use these for the pairs GES left non-adjacent.

    A FisherZ, StudentT, or degenerate Gaussian test answers the size-0 query of every pair in one
    canonical query array (:func:`~andrey.constraint._skeleton.marginal_separated`) and evaluates
    each nonempty size's subsets in chunks of them
    (:func:`~andrey.constraint._skeleton.separating_subsets`), with the p-values of the scalar
    calls. Every query of another test, or of ``in_column_order``, is a
    scalar call.
    """
    adjacent = cpdag != NULL
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    xs, ys = np.nonzero(np.triu(~adjacent, 1))
    marginal = None if in_column_order else marginal_separated(test, xs, ys, alpha)
    for k, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
        pool = [v for v in np.flatnonzero(adjacent[x] | adjacent[y]).tolist() if v != x and v != y]
        for size in range(_largest_size(len(pool), max_depth) + 1):
            subsets = combinations(pool, size)
            if in_column_order:
                first = next((cond for cond in subsets if test(x, y, cond) > alpha), None)
                found = [] if first is None else [first]
            elif size == 0 and marginal is not None:
                found = [()] if marginal[k] else []
            else:
                found = separating_subsets(test, x, y, size, subsets, alpha)
            if found:
                union = tuple(sorted(set().union(*found)))
                sepsets[(x, y)] = union
                sepsets[(y, x)] = union
                break
    return sepsets
