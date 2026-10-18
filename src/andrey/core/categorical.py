"""Category codes for the discrete tests and scores.

:func:`encode` re-encodes each column of a data matrix to the codes ``0..k-1``, in sorted value
order, with ``k`` the column's number of distinct values. :func:`configurations` numbers the joint
categories of several columns that occur in the data, so a contingency table over them has one
layer per observed configuration however large the product of the cardinalities.
:func:`table_counts` counts a column's categories in each occurring configuration of several
parent sets at once.
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


# A table with at most this many cells per row, or _DENSE_MIN cells, is counted over every possible
# configuration with one bincount and no renumbering; its empty configurations are dropped after.
_DENSE_ROWS = 4
_DENSE_MIN = 1 << 16
# The most row indices, and table cells, one batched bincount holds.
_BATCH_CELLS = 1 << 22


def table_counts(
    codes: np.ndarray, cardinalities: np.ndarray, node: int, parent_sets: Sequence[Sequence[int]]
) -> list[np.ndarray]:
    """Count ``node``'s categories in each configuration of each parent set that occurs.

    Each table equals the one :func:`configurations` gives: ``counts[j, k]`` is the rows in the
    ``j``-th occurring configuration of the parents, in mixed-radix order with the first parent the
    most significant, where ``node`` takes category ``k``. A parent set whose table fits within a
    few cells per row is counted over every possible configuration, several sets in one bincount,
    and its empty configurations dropped; a larger one is numbered by :func:`configurations`.

    Parameters
    ----------
    codes : ndarray of int64, shape (n_samples, n_features)
        Category codes from :func:`encode`.
    cardinalities : ndarray of int64, shape (n_features,)
        Categories per column, from :func:`encode`.
    node : int
        The column whose categories are counted.
    parent_sets : sequence of sequence of int
        The parent sets, each in the order of its mixed-radix digits.

    Returns
    -------
    list of ndarray of int64, each of shape (m, r)
        One table per parent set, in order: ``m`` occurring configurations by the node's ``r``
        categories.
    """
    n = codes.shape[0]
    r = int(cardinalities[node])
    target = codes[:, node]
    out: list[np.ndarray] = [np.empty((0, r), dtype=np.int64)] * len(parent_sets)
    dense: list[tuple[int, int]] = []
    limit = max(_DENSE_ROWS * n, _DENSE_MIN)
    for i, parents in enumerate(parent_sets):
        q = 1
        for p in parents:
            q *= int(cardinalities[p])
        if q * r <= limit:
            dense.append((i, q))
        else:
            config, m = configurations(codes, cardinalities, parents)
            out[i] = np.bincount(config * r + target, minlength=m * r).reshape(m, r)
    start = 0
    while start < len(dense):
        stop, cells = start, 0
        while stop < len(dense) and (stop == start or (stop - start + 1) * n <= _BATCH_CELLS):
            if stop > start and cells + dense[stop][1] * r > _BATCH_CELLS:
                break
            cells += dense[stop][1] * r
            stop += 1
        part = dense[start:stop]
        flat = np.empty((len(part), n), dtype=np.int64)
        offset = 0
        for b, (i, q) in enumerate(part):
            row = flat[b]
            row[:] = 0
            for p in parent_sets[i]:
                row *= int(cardinalities[p])
                row += codes[:, p]
            row *= r
            row += target + offset
            offset += q * r
        counts = np.bincount(flat.ravel(), minlength=offset)
        offset = 0
        for i, q in part:
            table = counts[offset : offset + q * r].reshape(q, r)
            out[i] = table[table.any(axis=1)]
            offset += q * r
        start = stop
    return out


def _compact(index: np.ndarray, size: int, n: int) -> tuple[np.ndarray, int]:
    """Renumber the codes in ``index`` that occur as ``0..m-1``, keeping their order."""
    if size <= 4 * n:
        present = np.bincount(index, minlength=size) > 0
        remap = np.cumsum(present) - 1
        return remap[index], int(present.sum())
    values, inverse = np.unique(index, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64), int(values.size)
