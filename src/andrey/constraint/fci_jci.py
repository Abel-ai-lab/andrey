"""FCI-JCI, the engine behind :func:`andrey.fci_jci`.

The algorithm, its settings, and its references are described on :func:`andrey.fci_jci`.

The environment index joins the data as a last variable, the context variable of Joint Causal
Inference (Mooij, Magliacane, and Claassen 2020). FCI's steps run over the augmented data with the
rule implementations, tests, and mark convention of :mod:`andrey.constraint.fci`. Before each
collider pass, every edge the index keeps gets an arrowhead at its data variable, since no data
variable causes the index (JCI Assumption 1); with ``confounded_index=False`` the edge also gets a
tail at the index, since no hidden variable causes both (Assumption 2). The orientation rules
change only circles, so these marks persist. JCI assumes no selection bias, so the rules are R1-R4
and R8-R10, without R5-R7. The index node is dropped from the returned PAG, and its edges are
returned beside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import check_collider_rule, unseparated_triples, vote_triples
from andrey.constraint.fci import (
    circle_pag,
    fci,
    fci_orient,
    knowledge_marks,
    orient_colliders,
    possible_dsep_step,
    validate_possible_dsep,
)
from andrey.constraint.pc import adjacency_search, knowledge_masks
from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.ci import make_indep_test
from andrey.core.structure import EDGE_TYPES

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def fci_jci(
    data: np.ndarray,
    env: np.ndarray,
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
    """Discover a PAG from multi-domain ``data`` with FCI and the environment index as a context.

    ``data`` is an ``(n_samples, n_features)`` array; ``env`` is the ``(n_samples,)`` or
    ``(n_samples, 1)`` environment index (one number per row). ``alpha``, ``indep_test``,
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
    context = np.asarray(env, dtype=np.float64)
    # One value per row: a wider index reshaped to a column would pair rows with the wrong values.
    if context.shape not in ((X.shape[0],), (X.shape[0], 1)):
        raise ValueError(
            f"env must hold one value per row of data, as shape ({X.shape[0]},) or "
            f"({X.shape[0]}, 1); got shape {context.shape}"
        )
    if not np.isfinite(context).all():
        raise ValueError("env contains NaN or inf")
    context = context.reshape(-1, 1)
    check_collider_rule(collider_rule)
    validate_possible_dsep(possible_dsep)
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
    allowed, keep, apart = knowledge_masks(padded)
    workers = backend.worker_count(workers)
    skeleton, sepsets, _ = adjacency_search(
        test,
        d + 1,
        alpha,
        support=allowed,
        keep=keep,
        apart=apart,
        max_depth=max_depth,
        workers=workers,
    )

    marks = circle_pag(skeleton)
    knowledge_marks(marks, padded)
    mark_index(marks, d, confounded_index)
    orient_colliders(marks, sepsets)
    # Its knowledge check runs before the index marks: no mark at the index is an arrowhead, so
    # the index cannot lie on a cycle the check reads.
    marks = possible_dsep_step(
        marks,
        augmented,
        test,
        alpha,
        sepsets,
        possible_dsep=possible_dsep,
        indep_test=indep_test,
        categorical=mask,
        seed=seed,
        max_depth=max_depth,
        constraints=padded,
        workers=workers,
    )
    mark_index(marks, d, confounded_index)
    ambiguous = unseparated_triples(marks != NULL, sepsets, padded)
    if collider_rule != "sepsets":
        ambiguous |= vote_triples(
            marks, test, alpha, sepsets, collider_rule, max_depth=max_depth, skip=ambiguous
        )
    if _ambiguous is not None:
        _ambiguous.update(ambiguous)
    orient_colliders(marks, sepsets, ambiguous)
    fci_orient(
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
        (int(j), EDGE_TYPES[(int(marks[d, j]), int(marks[j, d]))][0])
        for j in np.flatnonzero(marks[d, :d] != NULL)
    ]
    return GraphStructure.from_numpy(marks[:d, :d].copy(), kind="pag"), index_edges


def mark_index(marks: np.ndarray, index: int, confounded: bool) -> None:
    """Set the JCI marks on every edge of node ``index``, in place.

    Each edge gets an arrowhead at its other node, and, unless ``confounded``, a tail at ``index``.
    """
    neighbors = np.flatnonzero(marks[index] != NULL)
    marks[neighbors, index] = ARROW
    if not confounded:
        marks[index, neighbors] = TAIL
