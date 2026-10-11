"""Category codes for the discrete tests and scores.

:func:`encode` re-encodes each column of a data matrix to the codes ``0..k-1``, in sorted value
order, with ``k`` the column's number of distinct values. :func:`configurations` numbers the joint
categories of several columns that occur in the data, so a contingency table over them has one
layer per observed configuration however large the product of the cardinalities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def encode(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each column's category codes and the number of categories per column.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Finite values; each distinct value of a column is one category.

    Returns
    -------
    codes : ndarray of int64, shape (n_samples, n_features)
        Column ``j`` holds ``0..k_j-1``, numbering the column's distinct values in ascending order.
    cardinalities : ndarray of int64, shape (n_features,)
        ``k_j``, the number of distinct values in column ``j``.
    """
    arr = np.asarray(data)
    codes = np.empty(arr.shape, dtype=np.int64)
    for j in range(arr.shape[1]):
        codes[:, j] = np.unique(arr[:, j], return_inverse=True)[1].reshape(-1)
    cardinalities = codes.max(axis=0) + 1 if arr.shape[0] else np.zeros(arr.shape[1], np.int64)
    return codes, cardinalities.astype(np.int64)


def configurations(
    codes: np.ndarray, cardinalities: np.ndarray, columns: Sequence[int]
) -> tuple[np.ndarray, int]:
    """Number each row's joint category over ``columns`` by the configurations that occur.

    Configurations are ordered by their mixed-radix code (the first column most significant), and
    only those present in the data are numbered, so the count never exceeds the number of rows.
    With no columns every row is configuration ``0``.

    Parameters
    ----------
    codes : ndarray of int64, shape (n_samples, n_features)
        Category codes from :func:`encode`.
    cardinalities : ndarray of int64, shape (n_features,)
        Categories per column, from :func:`encode`.
    columns : sequence of int
        The columns whose joint category is numbered.

    Returns
    -------
    index : ndarray of int64, shape (n_samples,)
        Each row's configuration, ``0..m-1``.
    m : int
        The number of configurations that occur.
    """
    n = codes.shape[0]
    index = np.zeros(n, dtype=np.int64)
    size = 1
    for column in columns:
        k = int(cardinalities[column])
        index = index * k + codes[:, column]
        size *= k
        # Renumber before the mixed-radix code can outgrow the rows (and int64): at most n occur.
        if size > n:
            index, size = _compact(index, size, n)
    if size > 1:
        index, size = _compact(index, size, n)
    return index, size


def _compact(index: np.ndarray, size: int, n: int) -> tuple[np.ndarray, int]:
    """Renumber the codes in ``index`` that occur as ``0..m-1``, keeping their order."""
    if size <= 4 * n:
        present = np.bincount(index, minlength=size) > 0
        remap = np.cumsum(present) - 1
        return remap[index], int(present.sum())
    values, inverse = np.unique(index, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64), int(values.size)
