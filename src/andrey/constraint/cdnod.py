"""CDNOD, the engine behind :func:`andrey.cdnod`.

The algorithm, its settings, and its references are described on :func:`andrey.cdnod`.

The context index joins the data as a last variable. Every edge it keeps is oriented out of it
before the collider and Meek passes, which follow PC's conflict rule and so never overwrite those
orientations. The context node is dropped from the returned CPDAG.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from andrey.constraint.pc import _discover_skeleton, _lookup_sepset, _points_to
from andrey.core import ARROW, NULL, TAIL, GraphStructure
from andrey.core.orient import meek, to_structure


def cdnod(
    data: np.ndarray, c_indx: np.ndarray, *, alpha: float = 0.05, indep_test: str = "fisherz"
) -> GraphStructure:
    """Discover a CPDAG from nonstationary ``data`` with the CDNOD algorithm and the ``indep_test``.

    ``data`` is an ``(n_samples, n_features)`` array; ``c_indx`` is the ``(n_samples,)`` or
    ``(n_samples, 1)`` domain/context index (one label per row); ``alpha`` is the independence-test
    significance level in ``(0, 1)`` (a pair is separated when its p-value exceeds ``alpha``);
    ``indep_test`` selects the CI test (``"fisherz"`` shipped, run over the augmented matrix). The
    context is appended as an extra variable and the result is projected back to the
    ``n_features`` data variables. Returns the CPDAG as a ``GraphStructure`` of kind ``"cpdag"``.
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
        skeleton, sepsets = _discover_skeleton(X, alpha, indep_test)
        pdag = _orient_with_context(skeleton, sepsets, context=None)
    else:
        augmented = np.concatenate([X, context], axis=1)
        skeleton, sepsets = _discover_skeleton(augmented, alpha, indep_test)
        pdag = _orient_with_context(skeleton, sepsets, context=d)
    completed = meek(pdag)
    return to_structure(completed[:d, :d], kind="cpdag")


def _orient_with_context(
    skeleton: np.ndarray,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    *,
    context: int | None,
) -> np.ndarray:
    """Orient the augmented skeleton: context edges outward, then unshielded colliders.

    Every edge the ``context`` node retains is oriented ``context -> neighbour`` up front, so the
    context is a source in every triple it touches. Unshielded colliders are then oriented with the
    prioritize-existing rule (``uc_priority=2``): a collider ``x -> y <- z`` is oriented only when
    ``y`` is absent from the separating set of ``x`` and ``z`` and neither ``y -> x`` nor ``y -> z``
    is already directed, so the context orientations survive. ``context=None`` orients colliders
    only, for a skeleton without a context node.
    """
    n = skeleton.shape[0]
    pdag = np.where(skeleton != NULL, np.int8(TAIL), np.int8(NULL))
    np.fill_diagonal(pdag, np.int8(NULL))
    neighbours = [] if context is None else np.nonzero(skeleton[context] != NULL)[0].tolist()
    for neighbour in neighbours:
        pdag[context, neighbour], pdag[neighbour, context] = TAIL, ARROW
    for y in range(n):
        nbrs = np.nonzero(skeleton[y] != NULL)[0].tolist()
        for a, b in combinations(range(len(nbrs)), 2):
            x, z = nbrs[a], nbrs[b]
            if skeleton[x, z] != NULL:
                continue
            if y in _lookup_sepset(sepsets, x, z):
                continue
            if _points_to(pdag, y, x) or _points_to(pdag, y, z):
                continue
            pdag[x, y], pdag[y, x] = TAIL, ARROW
            pdag[z, y], pdag[y, z] = TAIL, ARROW
    return pdag
