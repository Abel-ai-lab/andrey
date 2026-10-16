"""CDNOD, the engine behind :func:`andrey.cdnod`.

The algorithm, its settings, and its references are described on :func:`andrey.cdnod`.

The context index joins the data as a last variable. Every edge the context keeps is oriented out
of it, then colliders are oriented as PC orients them, before Meek's rules run. The context node is
dropped from the returned CPDAG.
"""

from __future__ import annotations

import numpy as np

from andrey.constraint.pc import _discover_skeleton, _orient_colliders, _undirect_conflicts
from andrey.core import ARROW, NULL, TAIL, GraphStructure
from andrey.core.orient import meek, to_structure


def cdnod(
    data: np.ndarray,
    c_indx: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: np.ndarray | None = None,
) -> GraphStructure:
    """Discover a CPDAG from nonstationary ``data`` with the CDNOD algorithm and the ``indep_test``.

    ``data`` is an ``(n_samples, n_features)`` array; ``c_indx`` is the ``(n_samples,)`` or
    ``(n_samples, 1)`` domain/context index (one label per row); ``alpha`` is the independence-test
    significance level in ``(0, 1)`` (a pair is separated when its p-value exceeds ``alpha``);
    ``indep_test`` selects the CI test, run over the augmented matrix. The context is appended as an
    extra variable and the result is projected back to the ``n_features`` data variables.
    ``categorical``, a boolean mask of the data's categorical columns, reaches a mixed-data test,
    which also takes the context as categorical. Returns the CPDAG as a ``GraphStructure`` of kind
    ``"cpdag"``.
    """
    X = np.asarray(data, dtype=np.float64)
    context = np.asarray(c_indx, dtype=np.float64)
    # One value per row: a wider index reshaped to a column would pair rows with the wrong values.
    if context.shape not in ((X.shape[0],), (X.shape[0], 1)):
        raise ValueError(
            f"c_indx must hold one value per row of data, as shape ({X.shape[0]},) or "
            f"({X.shape[0]}, 1); got shape {context.shape}"
        )
    if not np.isfinite(context).all():
        raise ValueError("c_indx contains NaN or inf")
    context = context.reshape(-1, 1)
    d = X.shape[1]
    if np.ptp(context) == 0:
        # One domain: no change to detect, and a constant has no correlation to test, so the
        # search runs on ``data`` alone and gives PC's graph.
        skeleton, sepsets, pvalues = _discover_skeleton(
            X, alpha, indep_test, categorical=categorical
        )
        pdag = _orient_with_context(skeleton, sepsets, pvalues, context=None)
    else:
        augmented = np.concatenate([X, context], axis=1)
        mask = None if categorical is None else np.append(np.asarray(categorical, bool), True)
        skeleton, sepsets, pvalues = _discover_skeleton(
            augmented, alpha, indep_test, categorical=mask
        )
        pdag = _orient_with_context(skeleton, sepsets, pvalues, context=d)
    completed = _undirect_conflicts(meek(pdag))
    return to_structure(completed[:d, :d], kind="cpdag")


def _orient_with_context(
    skeleton: np.ndarray,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    pvalues: dict[tuple[int, int], float],
    *,
    context: int | None,
) -> np.ndarray:
    """Orient the augmented skeleton: context edges outward, then unshielded colliders.

    Every edge the ``context`` node keeps becomes ``context -> neighbor`` first, so the context is
    a source in every triple it touches. Colliders are then oriented as PC orients them
    (:func:`~andrey.constraint.pc._orient_colliders`), which skips a collider when an edge it would
    orient already has an arrowhead at the other end; so none puts an arrowhead at the context.
    ``context=None`` orients colliders only, for a skeleton without a context node.
    """
    pdag = skeleton.copy()
    if context is not None:
        neighbors = np.flatnonzero(skeleton[context] != NULL)
        pdag[context, neighbors] = TAIL
        pdag[neighbors, context] = ARROW
    return _orient_colliders(pdag, sepsets, pvalues)
