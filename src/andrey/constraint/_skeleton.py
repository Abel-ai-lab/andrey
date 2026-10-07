"""Batched conditional-independence scanning for the PC and FCI skeleton searches.

:func:`scan_conditioning_sets` tests one adjacent pair against every size-``k`` subset of a
candidate neighborhood. It uses the test's vectorized ``batched_call`` when there is one, one
stacked ``np.linalg.inv`` per chunk in place of thousands of scalar calls, and the scalar loop
otherwise, so a test that implements only :class:`~andrey.core.ci.CITest` still works.
:meth:`~andrey.core.ci.FisherZ.batched_call` returns the scalar p-values exactly, so the skeleton
is the same either way. Subsets are consumed in fixed-size chunks, so the scan holds one chunk in
memory however many subsets it visits.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations, islice

import numpy as np

from andrey.core.ci import CITest

Removal = tuple[int, int, tuple[int, ...]]  # (x, y, separating set) for one removed pair

# One batched_call inverts a stack of (k+2, k+2) correlation submatrices; cap how many subsets that
# stack holds at once so a huge C(|nbr|, k) neighborhood is consumed in bounded chunks, not one
# array. Each row's inverse is computed independently, so the chunk boundary never changes a result.
_CHUNK = 4096


def scan_conditioning_sets(
    test: CITest,
    a: int,
    b: int,
    candidates: list[int],
    size: int,
    alpha: float,
    separating: set[int],
    skip_within: set[int] | None = None,
    trace: list[dict] | None = None,
) -> bool:
    """Test every size-``size`` subset of ``candidates`` as a conditioning set for ``(a, b)``.

    Each subset whose Fisher-Z p-value exceeds ``alpha`` (``a`` conditionally independent of ``b``
    given the subset) is unioned into ``separating`` in place. Returns whether any subset separated
    the pair. There is no early exit: the whole subset space is scanned so ``separating``
    accumulates the union over every separating set, exactly as the scalar loop it replaces did.

    With ``skip_within`` given, subsets lying entirely inside it are not tested: the caller passes
    the candidates the pair's other endpoint already scanned at this size, so each shared subset is
    evaluated once per pair. Skipping a repeat changes neither the union nor the return value.

    When ``test`` exposes ``batched_call(a, b, condition_sets)`` the subsets are evaluated in
    vectorized, size-bounded chunks; otherwise each is tested with a scalar ``test(a, b, subset)``
    call. Both paths visit the same subsets in the same order and take the same ``p > alpha``
    decisions, so the accumulated ``separating`` and the return value are identical.

    With ``trace`` given, one record per test is appended (see :func:`record_tests`); the check
    runs once per chunk.
    """
    subsets = combinations(candidates, size)
    if skip_within is not None and len(skip_within) >= size:
        subsets = (s for s in subsets if not skip_within.issuperset(s))
    batched = getattr(test, "batched_call", None) or (
        lambda a, b, chunk: [test(a, b, subset) for subset in chunk]
    )
    found = False
    while True:
        chunk = list(islice(subsets, _CHUNK))
        if not chunk:
            break
        pvalues = batched(a, b, chunk)
        if trace is not None:
            record_tests(trace, size, a, b, chunk, pvalues, alpha)
        for subset, p in zip(chunk, pvalues):
            if p > alpha:
                found = True
                separating.update(subset)
    return found


def record_tests(
    trace: list[dict],
    depth: int,
    a: int,
    b: int,
    subsets: Iterable[tuple[int, ...]],
    pvalues: Iterable[float],
    alpha: float,
) -> None:
    """Append one record per test of the pair ``(a, b)``, lower index first as ``(x, y)``.

    A record holds ``step="test"``, ``depth``, ``x``, ``y``, the conditioning set ``S``, the p-value
    ``p``, and ``removed``: whether this test separated the pair (``p > alpha``).
    """
    x, y = min(a, b), max(a, b)
    trace.extend(
        dict(step="test", depth=depth, x=x, y=y, S=tuple(s), p=float(p), removed=bool(p > alpha))
        for s, p in zip(subsets, pvalues)
    )


# ---- vectorized size-0/1 prescreen ---------------------------------------------------------------
#
# The marginal (|S| = 0) and first-order (|S| = 1) Fisher-Z tests are closed forms over the sample
# correlation matrix -- no inversion. A CI test that exposes those closed forms vectorized
# (:meth:`FisherZ.marginal_pvalues` / :meth:`FisherZ.first_order_pvalues`) lets the size-0 pass run
# as a single all-pairs matrix threshold and the size-1 pass as one vectorized read per surviving
# pair, in place of the dense per-pair scalar loop. The decisions (p > alpha) and separating sets
# are identical to the scalar scan (the p-values match exactly), so the recovered
# skeleton is unchanged.


def prescreen_capable(test: CITest) -> bool:
    """Whether ``test`` exposes the vectorized size-0/1 prescreen primitives (Fisher-Z does)."""
    return callable(getattr(test, "marginal_pvalues", None)) and callable(
        getattr(test, "first_order_pvalues", None)
    )


def marginal_removals(
    test: CITest, adjacent_upper: np.ndarray, alpha: float, trace: list[dict] | None = None
) -> list[tuple[int, int]]:
    """(size 0) The still-adjacent pairs the marginal test separates, as ``(x, y)`` with ``x < y``.

    ``adjacent_upper`` is the strictly-upper-triangular boolean adjacency mask. One vectorized read
    of the correlation matrix flags every pair with marginal ``p > alpha``; the return is those in
    row-major order (matching the serial loop's visitation). Each carries an empty separating set.
    With ``trace`` given, every adjacent pair's test is recorded.
    """
    pmat = test.marginal_pvalues()  # ty: ignore[unresolved-attribute]  # gated by prescreen_capable
    separated = adjacent_upper & (pmat > alpha)
    if trace is not None:
        for x, y in zip(*np.nonzero(adjacent_upper)):
            record_tests(trace, 0, int(x), int(y), [()], [pmat[x, y]], alpha)
    xs, ys = np.nonzero(separated)
    return list(zip(xs.tolist(), ys.tolist()))


def first_order_separating(
    test: CITest,
    x: int,
    y: int,
    candidates: list[int],
    alpha: float,
    trace: list[dict] | None = None,
) -> list[int]:
    """(size 1) The sorted candidates whose first-order test separates ``(x, y)`` (``p > alpha``).

    ``candidates`` is the pair's start-of-size snapshot conditioners (PC: either endpoint's
    neighbors minus the partner; FCI: the lower endpoint's). Vectorized over the candidates in one
    shot; no cache writes. Returns the separating conditioners sorted ascending (the separating-set
    key order); empty when none separate. With ``trace`` given, every candidate's test is recorded.
    """
    if not candidates:
        return []
    pvals = test.first_order_pvalues(x, y, candidates)  # ty: ignore[unresolved-attribute]
    if trace is not None:
        record_tests(trace, 1, x, y, [(c,) for c in candidates], pvals, alpha)
    passing = np.asarray(candidates)[pvals > alpha]
    return sorted(passing.tolist())


def scan_pairs(
    test: CITest,
    pairs: list[tuple[int, int]],
    neighbours: list[np.ndarray],
    size: int,
    alpha: float,
    trace: list[dict] | None = None,
) -> list[Removal]:
    """Scan adjacent ``pairs`` at conditioning ``size`` against a start-of-size neighbor snapshot.

    Returns ``(x, y, separating)`` for every pair some size-``size`` conditioning set separates.
    Each pair is tested from either endpoint and the separating sets are unioned, so the removals
    do not depend on column order; a subset both endpoints offer is tested once, from the first.
    This is the general ``|S| >= 2`` scan the vectorized prescreen does not cover; the serial loop
    and the parallel workers share it, so they stay identical. With ``trace`` given, every test is
    recorded in the order it runs.
    """
    removals: list[Removal] = []
    for x, y in pairs:
        separating: set[int] = set()
        found = False
        from_x = neighbours[x][neighbours[x] != y].tolist()
        from_y = neighbours[y][neighbours[y] != x].tolist()
        # Once the x side has scanned, the y side skips the subsets both sides offer.
        shared = set(from_x) & set(from_y) if len(from_x) >= size else None
        for a, b, candidates, seen in ((x, y, from_x, None), (y, x, from_y, shared)):
            if len(candidates) < size:
                continue
            if scan_conditioning_sets(test, a, b, candidates, size, alpha, separating, seen, trace):
                found = True
        if found:
            removals.append((x, y, tuple(sorted(separating))))
    return removals


def adjacent_pairs(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """The still-adjacent pairs ``(x, y)``, ``x < y``, row-major (ascending ``x`` then ``y``)."""
    xs, ys = np.nonzero(np.triu(adjacency != 0, 1))
    return list(zip(xs.tolist(), ys.tolist()))
