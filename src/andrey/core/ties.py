"""The tie-break rule every method follows.

Where two choices of a method score alike, the method takes the one whose columns rank first by
:func:`tie_ranks`, a rank computed from the data in any row order, unit, or sign, so reordering the
columns relabels the result and changes nothing else. :func:`tie_mask` marks the columns a score or
a CI test reads as categories, which :func:`tie_ranks` ranks by the rows they group. Under
``tie_break="position"`` (:data:`andrey.config`, ``ANDREY_TIE_BREAK``) each column ranks by its
position, and :func:`resolve_collider_order`, :func:`resolve_in_column_order`, and
:func:`resolve_possible_dsep` select the column-order variants of the constraint-based steps.

Scores and p-values that differ only in their last bits count as equal: a method compares them
rounded to :data:`KEY_DP` decimal places (:func:`tie_key`, :func:`stable_key`), and a greedy search
counts a move as improving only when it lowers the score by more than :data:`EPS_IMPROVE`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import cmp_to_key
from itertools import groupby

import numpy as np
import numpy.typing as npt

from . import backend
from .ci import DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS
from .mixed import column_mask
from .score import DISCRETE_SCORES, MIXED_SCORES

# A score difference, such as the Schur delta-BIC, differs in its last bits across evaluation paths
# (einsum batch shape, residual-cache history), so the searches compare moves by the two constants
# below.

#: A move counts as improving only when its raw delta is more than this below zero. It sits far
#: above float64 Schur-delta noise and below the smallest real score improvement, so the set of
#: accepted moves is deterministic.
EPS_IMPROVE = 1e-9

#: Decimal places kept when a score difference, a statistic, or a p-value enters a tie-break key.
#: A greedy search rounds ``delta / n``, where ``n`` is its scale factor. The BIC delta grows with
#: the sample count (``delta = n_samples * (log R_new - log R_old) +/- ...``), so its rounding noise
#: does too; hill climbing divides by that sample count, so ``delta / n`` tracks the O(1) signal and
#: a fixed grid neither collapses distinct moves nor breaks ties badly across scales. GES divides by
#: the variable count instead: its per-target batches already make each delta bit-identical in
#: serial and parallel runs, so the key only orders the remaining ties.
KEY_DP = 9

# Standardized values enter a column's tie-break key rounded to _Z_DP decimal places, so copies of
# one column in other units, which differ in their last bits, share it; two columns sharing it are
# ordered by the first row where they differ by more than _Z_TOL.
_Z_DP = 6
_Z_TOL = 1e-12


def stable_key(delta: float, n: int) -> float:
    """Scale-invariant, chunk-invariant reduce key component: ``round(delta / n, KEY_DP)``.

    ``n`` is the caller's scale factor (see :data:`KEY_DP`); GES and GnIES pass the variable
    count. Two deltas that differ only in their last bits quantize to one bucket, so the reduce
    breaks the resulting tie on the deterministic remainder of the caller's key rather than on
    unstable low-order bits. Serial and parallel runs therefore select byte-identical moves however
    the targets are chunked or the adds are batched into the einsum.
    """
    return round(delta / n, KEY_DP)


def tie_key(value: float) -> float:
    """``value``, such as a log p-value, rounded to :data:`KEY_DP` decimal places."""
    return round(value, KEY_DP)


def tie_mask(name: str, categorical: npt.ArrayLike | None, d: int) -> np.ndarray:
    """The columns that the local score or CI test ``name`` reads as categories, for
    :func:`tie_ranks`.

    A categorical score or test (``"bdeu"``, ``"discrete_bic"``, ``"chisq"``, ``"gsq"``) reads
    every column as categories, a mixed-data one (``"degen_gauss"``, ``"cond_gauss"``) the columns
    ``categorical`` marks, and any other none. Ranked this way, relabeling a column's categories
    leaves its rank unchanged.
    """
    if name in DISCRETE_SCORES or name in DISCRETE_INDEP_TESTS:
        return np.ones(d, dtype=bool)
    if name in MIXED_SCORES or name in MIXED_INDEP_TESTS:
        return column_mask(categorical, d)
    return np.zeros(d, dtype=bool)


def resolve_collider_order(collider_order: str | None) -> str:
    """Return ``collider_order``, or for ``None`` the order ``andrey.config.tie_break`` selects.

    ``"rank"`` selects ``"p_value"`` and ``"position"`` selects ``"column"``. An explicit
    ``collider_order`` is returned as given, without reading the setting.
    """
    if collider_order is not None:
        return collider_order
    return "column" if backend.config.tie_break == "position" else "p_value"


def resolve_in_column_order() -> bool:
    """Whether ``andrey.config.tie_break`` is ``"position"``, which fires the orientation rules.

    Under ``"position"``, Meek's rules and FCI's rules R1-R10 fire one at a time in column order
    (``in_column_order``); under ``"rank"``, they run in order-free rounds.
    """
    return backend.config.tie_break == "position"


def resolve_possible_dsep(possible_dsep: str | None) -> str:
    """Return ``possible_dsep``, or for ``None`` the order ``andrey.config.tie_break`` selects.

    ``"rank"`` selects ``"order_free"`` and ``"position"`` selects ``"fast"``. An explicit
    ``possible_dsep`` is returned as given, without reading the setting.
    """
    if possible_dsep is not None:
        return possible_dsep
    return "fast" if backend.config.tie_break == "position" else "order_free"


def tie_ranks(x: np.ndarray, categorical: np.ndarray | None = None) -> np.ndarray:
    """Each column's rank for choosing between moves with equal rounded score changes.

    A column's rank comes from what the data say about it in any row order, unit, or sign. A
    continuous column enters by its ranks within the column, doubled and centered so that they are
    integers summing to zero, and a categorical column by which rows share a category. Each pair of
    columns has one relation, an integer or a list of integers:

    - two continuous columns: the absolute sum of their centered ranks' products, the numerator of
      Spearman's correlation without its sign;
    - a categorical and a continuous column: each category's count and the absolute sum of the
      continuous column's centered ranks over its rows;
    - two categorical columns: the counts of their joint table.

    A column's key is the sorted list of its relations to the other columns. Where two columns'
    keys are equal, as they always are with two columns or fewer, each of them adds its marginal:
    a continuous column's absolute skewness and kurtosis rounded to 9 decimal places, or a
    categorical column's category counts. Where keys are still equal, each of them adds its pattern
    in row order: a continuous column's rise, fall, or tie from each row to the next, or a
    categorical column's categories numbered in order of first appearance; and where they are equal
    after that, a continuous column adds its standardized values, its values minus their mean over
    their root mean square, rounded to 6 decimal places. Columns are ordered by a hash of their
    keys, so the rank is arbitrary with respect to causal structure. Continuous columns whose keys
    stay equal are ordered by their standardized values at the first row where these differ by more
    than ``1e-12``, the lower first, and columns left equal are ordered by position: identical
    columns, copies of one column in other units, and categorical columns that group the rows
    alike. A missing value (NaN) takes the middle rank of its continuous column and forms a
    category of its own in a categorical column.

    The ranks stay the same when the rows are shuffled, when a column's units change to
    ``a * x + b`` with ``a > 0``, when a column's sign flips, and when a categorical column's
    categories are relabeled; a new column order reorders the ranks with the columns. The limits:

    - A change of units can round two distinct values to one, which changes the column's ranks.
    - The marginal is rounded, so in rare cases a change of units or a sign flip moves it across a
      rounding boundary.
    - Columns that no row order, unit, or sign tells apart rank by their pattern in row order, then
      by their standardized values in row order, so a row shuffle can change which of them ranks
      first, while a new column order still relabels them. These are copies of one column in the
      other sign, two columns whose rows also appear with the two swapped, and columns that share
      their ranks and whose marginals agree to 9 decimal places, such as a copy whose standardized
      values are nudged by more than ``1e-12``.
    - Identical columns, copies of one column in other units, and categorical columns that group
      the rows alike rank by position, so a new column order changes which of them ranks first. A
      copy whose offset is large next to its spread keeps fewer bits of the column, so in rare cases
      its standardized values differ by more than ``1e-12`` or round to a different sixth decimal,
      and it ranks by them instead.
    - The relations are exact integers and agree on every machine. The marginal and the
      standardized values come from floating-point sums, so in rare cases they differ between
      machines.

    Under ``tie_break="position"`` (:data:`andrey.config`, ``ANDREY_TIE_BREAK``), each column's
    rank is its position, so the methods that order ties by these ranks order them by column
    position.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    categorical : ndarray of bool, shape (n_features,), optional
        Marks the categorical columns, whose distinct values are their categories. ``None`` treats
        all columns as continuous.

    Returns
    -------
    ndarray of int, shape (n_features,)
        Each column's position in the tie-break order, from zero to ``n_features - 1``. Lower ranks
        take precedence when moves have equal rounded score changes.
    """
    if backend.config.tie_break == "position":
        return np.arange(x.shape[1])
    x = np.asarray(x, dtype=np.float64)
    mask = (
        np.zeros(x.shape[1], dtype=bool) if categorical is None else np.asarray(categorical, bool)
    )
    keys = _relation_keys(x, mask)
    for extend in (_marginal_key, _row_pattern, _values):  # only where the keys so far are equal
        shared = {key for key, count in Counter(keys).items() if count > 1}
        keys = [
            key + extend(col, is_categorical) if key in shared else key
            for key, col, is_categorical in zip(keys, x.T, mask.tolist(), strict=True)
        ]
    digests = [hashlib.blake2b(key, digest_size=8).digest() for key in keys]
    by_digest = sorted(range(len(digests)), key=digests.__getitem__)
    order = []
    for _, run in groupby(by_digest, digests.__getitem__):
        tied = list(run)
        if len(tied) > 1 and not mask[tied[0]]:  # equal digests: the standardized values decide
            z = {j: _standardized(x[:, j]) for j in tied}
            tied.sort(key=cmp_to_key(lambda i, j: _first_difference(z[i], z[j])))
        order.extend(tied)
    ranks = np.empty(len(order), dtype=np.intp)
    ranks[order] = np.arange(len(order))
    return ranks


def _relation_keys(x: np.ndarray, mask: np.ndarray) -> list[bytes]:
    """Each column's type and sorted relations to the other columns (:func:`tie_ranks`)."""
    continuous, categorical = np.flatnonzero(~mask), np.flatnonzero(mask)
    ranks = _centered_ranks(x[:, continuous])
    c = len(continuous)
    products = _abs_products(ranks)[~np.eye(c, dtype=bool)].reshape(c, max(c - 1, 0))
    sorted_products = dict(zip(continuous.tolist(), np.sort(products, axis=1), strict=True))
    relations = _categorical_relations(
        x[:, categorical], ranks, continuous, categorical, x.shape[1]
    )
    keys = []
    for j, is_categorical in enumerate(mask.tolist()):
        if is_categorical:  # its relations with every other column
            head, others = b"K", np.delete(relations[np.searchsorted(categorical, j)], j)
        else:  # its products with the other continuous columns, its relations with the categorical
            head, others = b"C" + sorted_products[j].astype("<u8").tobytes(), relations[:, j]
        keys.append(head + np.sort(others).tobytes())
    return keys


def _categorical_relations(
    x: np.ndarray, ranks: np.ndarray, continuous: np.ndarray, categorical: np.ndarray, d: int
) -> np.ndarray:
    """``out[a, k]``: the 16-byte hash of the relation of categorical column ``a`` and column ``k``:
    each category's count and absolute centered rank sum with a continuous column, or the sorted
    joint counts with another categorical column (:func:`tie_ranks`)."""
    n, q = x.shape
    out = np.zeros((q, d), dtype="S16")
    if not n:  # no rows: every relation is the empty list
        out[:, continuous] = _digests(b"m", np.zeros((1, 0)))
        out[:, categorical] = _digests(b"t", np.zeros((1, 0)))
        return out
    if not q:
        return out
    codes = np.column_stack([np.unique(col, return_inverse=True)[1].reshape(-1) for col in x.T])
    sizes = codes.max(axis=0) + 1
    start = np.concatenate([[0], np.cumsum(sizes)])
    # One 0/1 column per category of each categorical column. Their products and their sums of
    # centered ranks are whole numbers below 2**53, which float64 holds exactly.
    indicators = np.zeros((n, start[-1]))
    indicators[np.arange(n)[:, None], start[:-1] + codes] = 1.0
    counts = indicators.sum(axis=0).astype(np.int64)
    sums = np.abs(indicators.T @ ranks)  # each category's sum of each continuous column's ranks
    # With a continuous column: the (count, |sum|) pairs of the categories, sorted.
    for a in range(q):
        rows = slice(start[a], start[a + 1])
        held = np.broadcast_to(counts[rows, None], (sizes[a], len(continuous)))
        order = np.lexsort((sums[rows], held), axis=0)
        pairs = np.stack(
            [np.take_along_axis(held, order, 0), np.take_along_axis(sums[rows], order, 0)]
        )
        out[a, continuous] = _digests(
            b"m", pairs.transpose(2, 1, 0).reshape(len(continuous), 2 * sizes[a])
        )
    # The joint table of two categorical columns, from the products of the indicators of all but
    # each column's last category, which give all but the table's last row and column, and the
    # counts, which complete them.
    last = start[1:] - 1
    kept = np.setdiff1d(np.arange(start[-1]), last)
    inner = np.cumsum(np.concatenate([[0], sizes - 1]))
    products = indicators[:, kept].T @ indicators[:, kept]
    first, second = np.triu_indices(q, 1)
    for ka, kb in set(zip(sizes[first].tolist(), sizes[second].tolist(), strict=True)):
        pick = (sizes[first] == ka) & (sizes[second] == kb)
        a, b = first[pick], second[pick]
        block = products[
            (inner[a][:, None] + np.arange(ka - 1))[:, :, None],
            (inner[b][:, None] + np.arange(kb - 1))[:, None, :],
        ]
        ca = counts[start[a][:, None] + np.arange(ka)]
        cb = counts[start[b][:, None] + np.arange(kb)]
        table = np.empty((len(a), ka, kb))
        table[:, :-1, :-1] = block
        table[:, :-1, -1] = ca[:, :-1] - block.sum(axis=2)
        table[:, -1, :-1] = cb[:, :-1] - block.sum(axis=1)
        table[:, -1, -1] = cb[:, -1] - ca[:, :-1].sum(axis=1) + block.sum(axis=(1, 2))
        digests = _digests(b"t", np.sort(table.reshape(len(a), -1), axis=1))
        out[a, categorical[b]] = digests
        out[b, categorical[a]] = digests
    return out


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """Each column's doubled average ranks minus ``m + 1``, ``m`` its count of values that are not
    NaN: integers summing to zero, which a sign flip negates. A NaN takes 0, the middle."""
    n, d = x.shape
    out = np.zeros((d, n))
    for col, ranks in zip(np.ascontiguousarray(x.T), out, strict=True):
        present = ~np.isnan(col)
        if n > 1 and present.all():  # without NaN or ties, a value's rank is its sorted place
            order = np.argsort(col)
            ordered = col[order]
            if (ordered[1:] != ordered[:-1]).all():
                ranks[order] = 2 * np.arange(1, n + 1) - (n + 1)
                continue
        _, inverse, counts = np.unique(col[present], return_inverse=True, return_counts=True)
        doubled = 2 * np.cumsum(counts) - counts + 1  # a value's first plus last place, from 1
        ranks[present] = doubled[inverse.reshape(-1)] - (counts.sum() + 1)
    return np.ascontiguousarray(out.T)


def _abs_products(ranks: np.ndarray) -> np.ndarray:
    """``|ranks.T @ ranks|`` as exact unsigned 64-bit integers.

    Every partial sum of two columns' products of centered ranks is an integer of magnitude at most
    ``(n**3 - n) / 3``, so float64 sums them exactly while that bound is below ``2**53``. Past it,
    unsigned integers sum them exactly modulo ``2**64``, and the smaller of a sum and its negative
    removes the sign; the two agree on every sum below ``2**63``.
    """
    n = len(ranks)
    if (n**3 - n) // 3 < 2**53:
        return np.abs(ranks.T @ ranks).astype(np.uint64)
    wide = ranks.astype(np.int64).view(np.uint64)
    sums = wide.T @ wide
    return np.minimum(sums, -sums)


def _marginal_key(col: np.ndarray, is_categorical: bool) -> bytes:
    """A column's marginal, which tells apart columns with equal relations (:func:`tie_ranks`)."""
    if is_categorical:
        return _digest(b"k", np.sort(np.unique(col, return_counts=True)[1]))
    values = np.sort(col[~np.isnan(col)])
    if values.size == 0 or values[0] == values[-1]:
        return f"V{values.size} constant".encode()
    z = values - values.mean()
    z /= np.sqrt(np.mean(z**2))
    return f"V{values.size} {abs(np.mean(z**3)):.9f} {np.mean(z**4):.9f}".encode()


def _row_pattern(col: np.ndarray, is_categorical: bool) -> bytes:
    """A column's pattern in row order, which tells apart columns with equal relations and marginals
    (:func:`tie_ranks`)."""
    if is_categorical:
        return b"R" + _first_appearance(col).astype("<i8").tobytes()
    steps = (col[1:] > col[:-1]).astype(np.int8) - (col[1:] < col[:-1])
    return b"R" + steps.tobytes()


def _values(col: np.ndarray, is_categorical: bool) -> bytes:
    """A continuous column's standardized values rounded to ``_Z_DP`` decimal places, which tell
    apart columns with equal relations, marginals, and patterns, or a categorical column's
    categories numbered in order of first appearance, which its pattern already holds
    (:func:`tie_ranks`)."""
    if is_categorical:
        return b"X" + _first_appearance(col).astype("<f8").tobytes()
    return b"X" + (np.round(_standardized(col), _Z_DP) + 0.0).astype("<f8").tobytes()


def _first_appearance(col: np.ndarray) -> np.ndarray:
    """A categorical column's categories numbered from zero in order of first appearance."""
    _, first, codes = np.unique(col, return_index=True, return_inverse=True)
    seen = np.empty(first.size, dtype=np.int64)
    seen[np.argsort(first)] = np.arange(first.size)
    return seen[codes.reshape(-1)]


def _standardized(col: np.ndarray) -> np.ndarray:
    """A continuous column minus its mean over its root mean square, zero where all its values are
    equal; a NaN stays NaN."""
    present = ~np.isnan(col)
    values = col[present]
    z = np.full(col.shape, np.nan)
    z[present] = 0.0
    if values.size and values.min() != values.max():
        centered = values - values.mean()
        z[present] = centered / np.sqrt(np.mean(centered**2))
    return z


def _first_difference(a: np.ndarray, b: np.ndarray) -> int:
    """-1 or 1 as standardized column ``a`` or ``b`` is lower at the first row where they differ by
    more than ``_Z_TOL``, or 0 where no row does (:func:`tie_ranks`)."""
    far = np.flatnonzero(np.abs(a - b) > _Z_TOL)
    if not far.size:
        return 0
    return -1 if a[far[0]] < b[far[0]] else 1


def _digest(tag: bytes, ints: np.ndarray) -> bytes:
    """A 16-byte hash of ``tag`` and ``ints`` as little-endian 64-bit integers."""
    return hashlib.blake2b(tag + ints.astype("<i8").tobytes(), digest_size=16).digest()


def _digests(tag: bytes, rows: np.ndarray) -> np.ndarray:
    """:func:`_digest` of each row of ``rows``, a 2-D array of whole numbers, as 16-byte strings."""
    data = np.ascontiguousarray(rows, dtype="<i8")
    tagged = np.empty((len(data), 1 + 8 * data.shape[1]), dtype=np.uint8)
    tagged[:, 0] = tag[0]
    tagged[:, 1:] = data.view(np.uint8)
    raw, width, blake = memoryview(tagged.tobytes()), tagged.shape[1], hashlib.blake2b
    return np.array(
        [blake(raw[i : i + width], digest_size=16).digest() for i in range(0, len(raw), width)],
        dtype="S16",
    )
