"""FCI-JCI, the engine behind :func:`andrey.fci_jci`.

The algorithm, its settings, and its references are described on :func:`andrey.fci_jci`.

The domain index joins the data as a last variable, the context variable of Joint Causal Inference
(Mooij, Magliacane, and Claassen 2020). FCI's steps run over the augmented data with the rule
implementations, tests, and mark convention of :mod:`andrey.constraint.fci`. Before each collider
pass, every edge the index keeps gets an arrowhead at its data variable, since no data variable
causes the index (JCI Assumption 1); with ``confounded_index=False`` the edge also gets a tail at
the index, since no hidden variable causes both (Assumption 2). The marks go on before the collider
pass, as in the authors' reference implementation. The orientation rules change only circles, so
these marks persist. JCI assumes no selection bias, so the rules are R1-R4 and
R8-R10, without R5-R7. The index node is dropped from the returned PAG, and its edges are returned
beside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import check_collider_rule, unseparated_triples
from andrey.constraint.fci import (
    _circle_pag,
    _fast_adjacency_search,
    _fci_orient,
    _knowledge_marks,
    _orient_colliders,
    _remove_by_possible_dsep,
    _test_in_rank_order,
    _validate_possible_dsep,
    _vote_sepsets,
    fci,
)
from andrey.constraint.pc import check_knowledge_cycle
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, backend
from andrey.core.ci import make_indep_test

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

#: The kind of an index edge, by the mark at the index; the mark at the data variable is an
#: arrowhead.
_INDEX_EDGE = {TAIL: "directed", CIRCLE: "partially_directed", ARROW: "bidirected"}


def fci_jci(
    data: np.ndarray,
    domain_index: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: np.ndarray | None = None,
    possible_dsep: str = "order_free",
    confounded_index: bool = True,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool = False,
    workers: int | None = None,
    seed: int | None = None,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> tuple[GraphStructure, list[tuple[int, str]]]:
    """Discover a PAG from multi-domain ``data`` with FCI and the domain index as a context node.

    ``data`` is an ``(n_samples, n_features)`` array; ``domain_index`` is the ``(n_samples,)`` or
    ``(n_samples, 1)`` domain/context index (one label per row). ``alpha``, ``indep_test``,
    ``collider_rule``, ``possible_dsep``, ``max_depth``, ``in_column_order``, ``workers``, ``seed``,
    and ``_ambiguous`` act as for :func:`~andrey.constraint.fci.fci`, on the data with the index
    appended as a last column; ``categorical``, a boolean mask of the data's categorical columns,
    reaches a mixed-data test, which also takes the index as categorical. ``constraints``, over the
    data's columns, act as for FCI, with the index unconstrained. ``confounded_index=False`` assumes
    no hidden variable causes both the index and a data variable. With a single domain the index is
    left out and the search is FCI on ``data`` alone.

    Returns the PAG over the ``n_features`` data variables as a ``GraphStructure`` of kind
    ``"pag"``, and the index's edges as ``(variable, kind)`` pairs in variable order: ``kind`` is
    ``"directed"`` (index -> variable), ``"partially_directed"`` (index o-> variable), or
    ``"bidirected"`` (index <-> variable).
    """
    X = np.asarray(data, dtype=np.float64)
    context = checked_index(domain_index, X.shape[0])
    check_collider_rule(collider_rule)
    _validate_possible_dsep(possible_dsep)
    if np.ptp(context) == 0:
        # One domain: no change to detect, and a constant has no correlation to test.
        pag = fci(
            X,
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
        return pag, []

    d = X.shape[1]
    augmented = np.concatenate([X, context], axis=1)
    mask = None if categorical is None else np.append(np.asarray(categorical, bool), True)
    padded = None if constraints is None else constraints.padded(d + 1)
    test = make_indep_test(indep_test, augmented, categorical=mask, seed=seed)
    allowed = keep = apart = None
    if padded is not None:
        allowed = padded.support()
        keep = padded.required | padded.required.T
        apart = ~allowed
    support, sepsets = _fast_adjacency_search(
        augmented,
        test,
        alpha,
        workers=backend.worker_count(workers),
        support=allowed,
        keep=keep,
        apart=apart,
        max_depth=max_depth,
    )

    marks = _circle_pag(support)
    _knowledge_marks(marks, padded)
    _mark_index(marks, d, confounded_index)
    _orient_colliders(marks, sepsets)
    if possible_dsep == "fast":
        _remove_by_possible_dsep(marks, test, alpha, sepsets, keep=keep, max_depth=max_depth)
    else:
        by_rank, ranked_test = _test_in_rank_order(augmented, indep_test, mask, seed)
        _remove_by_possible_dsep(
            marks, ranked_test, alpha, sepsets, by_rank, keep=keep, max_depth=max_depth
        )
    marks = _circle_pag(marks != NULL)
    _knowledge_marks(marks, padded)
    _mark_index(marks, d, confounded_index)
    if padded is not None:
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW), closing=(marks == ARROW).T)
    ambiguous = unseparated_triples(marks != NULL, sepsets, padded)
    if collider_rule != "sepsets":
        ambiguous |= _vote_sepsets(
            marks, test, alpha, sepsets, rule=collider_rule, max_depth=max_depth, skip=ambiguous
        )
    if _ambiguous is not None:
        _ambiguous.update(ambiguous)
    _orient_colliders(marks, sepsets, ambiguous)
    _fci_orient(
        marks,
        test,
        alpha,
        sepsets,
        ambiguous,
        max_depth=max_depth,
        in_column_order=in_column_order,
        selection_bias=False,
    )

    index_edges = [
        (int(j), _INDEX_EDGE[int(marks[d, j])]) for j in np.flatnonzero(marks[d, :d] != NULL)
    ]
    return GraphStructure.from_numpy(marks[:d, :d].copy(), kind="pag"), index_edges


def checked_index(domain_index: object, n_rows: int) -> np.ndarray:
    """Return ``domain_index`` as an ``(n_rows, 1)`` ``float64`` column.

    Raises ``ValueError`` unless it holds one finite value per row, as shape ``(n_rows,)`` or
    ``(n_rows, 1)``; a wider index reshaped to a column would pair rows with the wrong values.
    """
    context = np.asarray(domain_index, dtype=np.float64)
    if context.shape not in ((n_rows,), (n_rows, 1)):
        raise ValueError(
            f"domain_index must hold one value per row of data, as shape ({n_rows},) or "
            f"({n_rows}, 1); got shape {context.shape}"
        )
    if not np.isfinite(context).all():
        raise ValueError("domain_index contains NaN or inf")
    return context.reshape(-1, 1)


def _mark_index(marks: np.ndarray, index: int, confounded: bool) -> None:
    """Set the JCI marks on every edge of node ``index``, in place.

    Each edge gets an arrowhead at its other node, and, unless ``confounded``, a tail at ``index``.
    """
    neighbors = np.flatnonzero(marks[index] != NULL)
    marks[neighbors, index] = ARROW
    if not confounded:
        marks[index, neighbors] = TAIL
