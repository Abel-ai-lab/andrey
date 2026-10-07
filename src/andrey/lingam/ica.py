"""ICA-based LiNGAM, the engine behind :func:`andrey.ica_lingam`.

The algorithm, its settings, and its reference are described on :func:`andrey.ica_lingam`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

from andrey.core.seeding import resolve_seed
from andrey.lingam.adjacency import estimate


def _search_causal_order(matrix: np.ndarray) -> list[int] | None:
    """Strict causal order of a matrix by repeatedly peeling off all-zero rows, or ``None``."""
    causal_order: list[int] = []
    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    while len(matrix) > 0:
        row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
        if len(row_index_list) == 0:
            break
        target_index = row_index_list[0]
        causal_order.append(int(original_index[target_index]))
        original_index = np.delete(original_index, target_index, axis=0)
        mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
        matrix = matrix[mask][:, mask]

    if len(causal_order) != row_num:
        return None
    return causal_order


def _estimate_causal_order(matrix: np.ndarray) -> list[int] | None:
    """Approximate a lower-triangular order: zero smallest entries until a strict order appears."""
    matrix = matrix.copy()
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)

    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    causal_order = None
    for i, j in pos_list[initial_zero_num:]:
        matrix[i, j] = 0
        causal_order = _search_causal_order(matrix)
        if causal_order is not None:
            break
    return causal_order


def ica_lingam(
    X: np.ndarray, *, random_state: int | None = None, max_iter: int = 1000
) -> tuple[list[int], np.ndarray]:
    """Fit ICA-LiNGAM: return ``(causal_order, B)`` with ``x_i = sum_j B[i, j] x_j``.

    FastICA gives an unmixing matrix. A Hungarian assignment on ``1 / |W|`` permutes its rows so
    the diagonal is largest, and each row is divided by its diagonal. The causal order is the
    permutation that makes ``I - W`` closest to lower-triangular, found by zeroing its smallest
    entries until a strict order appears. The weights are then re-estimated from that order with
    adaptive Lasso, as DirectLiNGAM does, so they depend only on the order and the data.

    ``random_state`` resolves through :func:`andrey.core.seeding.resolve_seed` (default seed ``0``)
    and is passed straight to FastICA.
    """
    X = np.asarray(X, dtype=np.float64)
    seed = resolve_seed(random_state)

    ica = FastICA(max_iter=max_iter, random_state=seed)
    ica.fit(X)
    w_ica = ica.components_

    _, col_index = linear_sum_assignment(1 / np.abs(w_ica))
    pw_ica = np.zeros_like(w_ica)
    pw_ica[col_index] = w_ica

    diag = np.diag(pw_ica)[:, np.newaxis]
    w_estimate = pw_ica / diag
    b_estimate = np.eye(len(w_estimate)) - w_estimate

    causal_order = _estimate_causal_order(b_estimate)
    if causal_order is None:
        raise RuntimeError("ICA-LiNGAM failed to find a causal order (no nilpotent permutation)")

    B = estimate(X, causal_order)
    return causal_order, B
