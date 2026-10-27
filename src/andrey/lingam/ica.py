"""ICA-based LiNGAM, the engine behind :func:`andrey.ica_lingam`.

The algorithm, its settings, and its reference are described on :func:`andrey.ica_lingam`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

from andrey.core.seeding import resolve_seed
from andrey.lingam.adjacency import estimate
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def _search_causal_order(
    matrix: np.ndarray, ranks: np.ndarray, precedence: np.ndarray | None = None
) -> list[int] | None:
    """Strict causal order of a matrix by repeatedly peeling off all-zero rows, or ``None``.

    When several rows are all zero, the one with the lowest ``ranks`` entry goes first. With
    ``precedence`` (``precedence[a, b]``: ``a`` comes before ``b``), a row is peeled only when no
    remaining row must come before it.
    """
    causal_order: list[int] = []
    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    while len(matrix) > 0:
        zero = np.sum(np.abs(matrix), axis=1) == 0
        if precedence is not None:
            zero &= ~precedence[np.ix_(original_index, original_index)].any(axis=0)
        row_index_list = np.where(zero)[0]
        if len(row_index_list) == 0:
            break
        target_index = row_index_list[np.argmin(ranks[original_index[row_index_list]])]
        causal_order.append(int(original_index[target_index]))
        keep = np.ones(len(matrix), dtype=bool)
        keep[target_index] = False
        original_index = original_index[keep]
        matrix = matrix[keep][:, keep]

    if len(causal_order) != row_num:
        return None
    return causal_order


def _estimate_causal_order(
    matrix: np.ndarray, ranks: np.ndarray, precedence: np.ndarray | None = None
) -> list[int] | None:
    """Approximate a lower-triangular order: zero smallest entries until a strict order appears.

    ``precedence`` reaches :func:`_search_causal_order`, so the order keeps it.
    """
    matrix = matrix.copy()
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)

    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    causal_order = None
    for i, j in pos_list[initial_zero_num:]:
        matrix[i, j] = 0
        causal_order = _search_causal_order(matrix, ranks, precedence)
        if causal_order is not None:
            break
    return causal_order


def ica_lingam(
    X: np.ndarray,
    *,
    random_state: int | None = None,
    max_iter: int = 1000,
    constraints: Constraints | None = None,
) -> tuple[list[int], np.ndarray]:
    """Fit ICA-LiNGAM: return ``(causal_order, B)`` with ``x_i = sum_j B[i, j] x_j``.

    FastICA gives an unmixing matrix ``W`` of the z-scored columns. Each entry is read as
    ``t_ij = |W_ij| / ||W_:j||``, its size against its estimation error: by the equivariance of
    ICA, the error of ``W_ij`` grows with the norm of column ``j``, so ``t_ij`` ranks the entries
    as the Wald statistics of Shimizu et al. (2006, Sec. 6.1) do when every independent component
    has the same distribution. A Hungarian assignment on ``1 / t`` permutes the rows so the
    diagonal holds no small entry. The causal order is the permutation that makes the
    off-diagonal ``t`` closest to lower-triangular, found by zeroing its smallest entries until a
    strict order appears. ``t`` does not change when a column is multiplied by a positive number,
    so neither does the order; the paper's ``1 / |W|`` and ``|B|`` do (Shimizu et al. 2011,
    Sec. 2.4). When several rows could come next, the one with the lowest
    :func:`~andrey.search.hc.tie_ranks` rank does, so the choice moves with its column when the
    columns are reordered. The weights are then re-estimated from that order with adaptive Lasso,
    as DirectLiNGAM does, so they depend only on the order and the data.

    ``random_state`` resolves through :func:`andrey.core.seeding.resolve_seed` (default seed ``0``)
    and is passed straight to FastICA. FastICA starts from a random rotation of the whitened data;
    z-scoring first keeps that start, and so the solution, independent of the columns' units.

    With background knowledge ``constraints``, the order search reads the strength of every edge
    they forbid as ``0`` (the reverse of each required edge, and every edge into an earlier tier,
    included) and peels a variable only after every variable that
    :meth:`~andrey.core.knowledge.Constraints.precedence` puts before it. The re-estimation drops
    forbidden parents and keeps required ones unpenalized, as DirectLiNGAM's does.
    """
    X = np.asarray(X, dtype=np.float64)
    seed = resolve_seed(random_state)
    if X.shape[1] == 1:
        return [0], np.zeros((1, 1))  # one variable: nothing to order, no edge to weigh

    ica = FastICA(max_iter=max_iter, random_state=seed)
    ica.fit((X - X.mean(axis=0)) / X.std(axis=0))
    w_ica = ica.components_
    t = np.abs(w_ica) / np.linalg.norm(w_ica, axis=0)

    _, col_index = linear_sum_assignment(1 / t)
    strength = np.zeros_like(t)
    strength[col_index] = t
    np.fill_diagonal(strength, 0.0)

    if constraints is None:
        causal_order = _estimate_causal_order(strength, tie_ranks(X))
    else:
        strength[constraints.forbidden.T] = 0.0  # strength[i, j] weighs j -> i
        causal_order = _estimate_causal_order(strength, tie_ranks(X), constraints.precedence())
    if causal_order is None:
        raise RuntimeError("ICA-LiNGAM failed to find a causal order (no nilpotent permutation)")

    if constraints is None:
        return causal_order, estimate(X, causal_order)
    B = estimate(X, causal_order, forbidden=constraints.forbidden, required=constraints.required)
    return causal_order, B
