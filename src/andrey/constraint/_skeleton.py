"""Batched conditional-independence scanning for the PC and FCI skeleton searches.

:func:`scan_conditioning_sets` tests one adjacent pair against every size-``k`` subset of a
candidate neighborhood. It uses the test's vectorized path when there is one (one stacked
``np.linalg.inv`` per chunk for the partial-correlation tests, its subsets canonical as built; one
count of every chunk's contingency tables for the chi-square tests) in place of thousands of scalar
calls, and the scalar loop otherwise, so a test that implements only
:class:`~andrey.core.ci.CITest` still works. Every shipped vectorized path returns the scalar
p-values exactly, so the skeleton is the same either way. Subsets are consumed in fixed-size
chunks, so the scan holds one chunk in memory however many subsets it visits. :func:`scan_pairs`
scans every pair of one conditioning size; for FisherZ and StudentT it stacks the chunks of
consecutive pairs into one evaluation, up to ``_STACK_BYTES`` of submatrices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import nullcontext
from functools import partial
from itertools import chain, combinations, islice
from math import comb
from typing import TYPE_CHECKING

import numpy as np

from andrey.core import backend
from andrey.core.ci import CITest, FisherZ, StudentT, pins_blas

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

#: One removed pair: ``(x, y, separating set, the largest p-value among its separating sets)``; the
#: p-value is ``None`` when the scan does not record it.
Removal = tuple[int, int, tuple[int, ...], float | None]
# A pair's subsets that one call evaluates: ``(the pair's position, x, y, subsets)``, ``x < y``.
_Chunk = tuple[int, int, int, list[tuple[int, ...]]]

# One batched_call evaluates a chunk of subsets together; cap how many subsets a chunk holds so a
# huge C(|nbr|, k) neighborhood is consumed in bounded chunks, not one array. Each subset's p-value
# is computed independently, so the chunk boundary never changes a result.
_CHUNK = 4096

# The most bytes of correlation submatrices one stacked evaluation of scan_pairs holds: a stack of
# B queries of conditioning size k takes 8 * B * (k + 2) ** 2 bytes, and holds at least one chunk.
# Its inverse and index arrays take about twice as much again, in each process that scans.
_STACK_BYTES = 1 << 22


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
    largest: list[float] | None = None,
) -> bool:
    """Test every size-``size`` subset of ``candidates`` as a conditioning set for ``(a, b)``.

    Each subset whose p-value exceeds ``alpha`` (``a`` conditionally independent of ``b``
    given the subset) is unioned into ``separating`` in place. Returns whether any subset separated
    the pair. With ``largest`` given, the largest p-value among the separating subsets is appended
    to it when one separated the pair; without it, no p-value is compared. There is no early exit:
    the whole subset space is scanned so ``separating`` accumulates the union over every separating
    set.

    With ``skip_within`` given, subsets lying entirely inside it are not tested: the caller passes
    the candidates the pair's other endpoint already scanned at this size, so each shared subset is
    evaluated once per pair. Skipping a repeat changes neither the union nor the return value.

    ``candidates`` are ascending and exclude ``a`` and ``b``, so every subset is a sorted, distinct
    conditioning set. A FisherZ or StudentT test evaluates the subsets in size-bounded chunks of
    canonical query arrays (the test's ``_canonical_pvalues``), with no re-sorting. Another test
    that exposes ``batched_call(a, b, condition_sets)`` evaluates the same chunks through it;
    otherwise each subset is tested with a scalar ``test(a, b, subset)`` call, or
    ``test._bounded_pvalue(a, b, subset, alpha, exact_above)`` where the test has one, with
    ``exact_above`` true when ``largest`` is given. All paths visit the same subsets in the same
    order and take the same ``p > alpha`` decisions, so the accumulated ``separating``, the largest
    p-value, and the return value are identical.

    With ``trace`` given, one record per test is appended (see :func:`record_tests`); the check
    runs once per chunk.
    """
    subsets = _subsets(candidates, size, skip_within)
    # CMIknn's _bounded_pvalue stops its shuffles once the side of alpha is certain; a p-value
    # above alpha that `largest` records comes from every shuffle.
    bounded = getattr(test, "_bounded_pvalue", None)
    scalar = (
        test if bounded is None else partial(bounded, alpha=alpha, exact_above=largest is not None)
    )
    canonical = _canonical_route(test)
    if canonical is not None:
        batched = partial(_canonical_chunk, canonical, size)
    else:
        batched = getattr(test, "batched_call", None) or (
            lambda a, b, chunk: [scalar(a, b, subset) for subset in chunk]
        )
    found = False
    top = 0.0
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
                if largest is not None and p > top:
                    top = float(p)
    if found and largest is not None:
        largest.append(top)
    return found


def _subsets(
    candidates: list[int], size: int, skip_within: set[int] | None
) -> Iterator[tuple[int, ...]]:
    """The size-``size`` subsets of ``candidates`` in ``combinations`` order, less those lying
    entirely inside ``skip_within`` when it is given."""
    subsets = combinations(candidates, size)
    if skip_within is not None and len(skip_within) >= size:
        subsets = (s for s in subsets if not skip_within.issuperset(s))
    return subsets


def _canonical_route(test: CITest) -> Callable[..., np.ndarray] | None:
    """A FisherZ or StudentT test's ``_canonical_pvalues``, which the skeleton's canonical query
    arrays take, else ``None``.

    A subclass or a wrapper of either test keeps the ``batched_call`` and ``first_order_pvalues``
    it may override.
    """
    if type(test) in (FisherZ, StudentT):
        return test._canonical_pvalues  # ty: ignore[unresolved-attribute]  # gated by the type
    return None


def _canonical_chunk(
    pvalues: Callable[..., np.ndarray], size: int, a: int, b: int, chunk: list[tuple[int, ...]]
) -> np.ndarray:
    """``pvalues`` of the pair ``(a, b)`` given each subset of ``chunk``: ``size`` members, sorted
    and distinct, as the subsets of ascending candidates are."""
    n = len(chunk)
    x = np.full(n, min(a, b), dtype=np.intp)
    y = np.full(n, max(a, b), dtype=np.intp)
    return pvalues(x, y, np.array(chunk, dtype=np.intp).reshape(n, size))


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
# A CI test that exposes vectorized marginal (|S| = 0) and first-order (|S| = 1) p-values
# (``marginal_pvalues`` and ``first_order_pvalues``: the partial-correlation closed forms over the
# correlation matrix, the chi-square tables counted together, the conditional Gaussian sets fit
# together) lets the size-0 pass run as a single all-pairs matrix threshold and the size-1 pass as
# one vectorized read per surviving pair, or per stack of pairs for FisherZ and StudentT, in place
# of the dense per-pair scalar loop. The decisions (p > alpha) and separating sets are identical to
# the scalar scan (the p-values match exactly), so the recovered skeleton is unchanged.

# The most first-order queries one stack of pairs holds, so the size-1 pass holds a bounded array
# however many pairs it tests.
_FIRST_ORDER_STACK = 1 << 16


def prescreen_capable(test: CITest) -> bool:
    """Whether ``test`` exposes the vectorized size-0/1 prescreen primitives.

    The partial-correlation, chi-square, and conditional Gaussian tests do.
    """
    return callable(getattr(test, "marginal_pvalues", None)) and callable(
        getattr(test, "first_order_pvalues", None)
    )


def marginal_removals(
    test: CITest,
    adjacent_upper: np.ndarray,
    alpha: float,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[tuple[int, int, float | None]]:
    """(size 0) The still-adjacent pairs the marginal test separates, as ``(x, y, p)``, ``x < y``.

    ``adjacent_upper`` is the strictly-upper-triangular boolean adjacency mask. One vectorized
    all-pairs evaluation flags every pair with marginal ``p > alpha``; the return is those in
    row-major order (matching the serial loop's visitation), each with its p-value, or ``None``
    when ``record_pvalues`` is false; each pair's separating set is empty. With ``trace`` given,
    every adjacent pair's test is recorded.
    """
    pmat = test.marginal_pvalues()  # ty: ignore[unresolved-attribute]  # gated by prescreen_capable
    separated = adjacent_upper & (pmat > alpha)
    if trace is not None:
        for x, y in zip(*np.nonzero(adjacent_upper)):
            record_tests(trace, 0, int(x), int(y), [()], [pmat[x, y]], alpha)
    xs, ys = np.nonzero(separated)
    if not record_pvalues:
        return [(x, y, None) for x, y in zip(xs.tolist(), ys.tolist())]
    return list(zip(xs.tolist(), ys.tolist(), pmat[xs, ys].tolist()))


def first_order_removals(
    test: CITest,
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    alpha: float,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """(size 1) The ``pairs`` a single conditioner separates, as ``(x, y, separating, p)``.

    Each pair ``x < y`` is tested against its start-of-size snapshot conditioners: either
    endpoint's ``neighbors`` minus the pair, in PC and FCI alike. ``separating`` holds the
    conditioners whose test separates the pair, ascending (the separating-set key order), and ``p``
    is the largest p-value among the pair's tests, or ``None`` when ``record_pvalues`` is false. A
    FisherZ or StudentT test evaluates the queries of consecutive pairs in one stack
    (:func:`_stacked_first_order`); another test evaluates each pair's candidates in one
    ``first_order_pvalues`` call. With ``trace`` given, every test is recorded, pair by pair.
    """
    queries = []
    for x, y in pairs:
        candidates = sorted((set(neighbors[x].tolist()) | set(neighbors[y].tolist())) - {x, y})
        if candidates:
            queries.append((x, y, candidates))
    canonical = _canonical_route(test)
    tested = (
        _stacked_first_order(canonical, queries)
        if canonical is not None
        else (
            (x, y, c, test.first_order_pvalues(x, y, c))  # ty: ignore[unresolved-attribute]
            for x, y, c in queries
        )
    )
    removals: list[Removal] = []
    for x, y, candidates, pvals in tested:
        if trace is not None:
            record_tests(trace, 1, x, y, [(c,) for c in candidates], pvals, alpha)
        passing = np.asarray(candidates)[pvals > alpha]
        if passing.size:
            # Some candidate separates, so the largest p-value is a separating one.
            p = max(pvals.tolist()) if record_pvalues else None
            removals.append((x, y, tuple(sorted(passing.tolist())), p))
    return removals


def _stacked_first_order(
    pvalues: Callable[..., np.ndarray], queries: list[tuple[int, int, list[int]]]
) -> Iterator[tuple[int, int, list[int], np.ndarray]]:
    """Yield each ``(x, y, candidates)`` query with the first-order p-values of its candidates.

    ``pvalues`` takes canonical query arrays. Consecutive queries are stacked, whole, up to
    ``_FIRST_ORDER_STACK`` conditioners at a time; the first-order closed form is elementwise, so
    each p-value is the one a query evaluated alone gets.
    """
    start = 0
    while start < len(queries):
        stop, held = start + 1, len(queries[start][2])
        while stop < len(queries) and held + len(queries[stop][2]) <= _FIRST_ORDER_STACK:
            held += len(queries[stop][2])
            stop += 1
        block = queries[start:stop]
        counts = [len(candidates) for _, _, candidates in block]
        x = np.repeat(np.array([q[0] for q in block], dtype=np.intp), counts)
        y = np.repeat(np.array([q[1] for q in block], dtype=np.intp), counts)
        z = np.fromiter((c for _, _, candidates in block for c in candidates), np.intp, held)
        stacked = pvalues(x, y, z[:, None])
        for query, pvals in zip(block, np.split(stacked, np.cumsum(counts)[:-1])):
            yield (*query, pvals)
        start = stop


def scan_pairs(
    test: CITest,
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    alpha: float,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """Scan adjacent ``pairs`` at conditioning ``size`` against a start-of-size neighbor snapshot.

    Returns ``(x, y, separating, p)`` for every pair some size-``size`` conditioning set separates,
    with ``p`` the largest p-value among the separating tests, or ``None`` when ``record_pvalues``
    is false. Each pair is tested from either endpoint and the separating sets are unioned, so the
    removals do not depend on column order; a subset both endpoints offer is tested once, from the
    first.
    This is the general ``|S| >= 2`` scan the vectorized prescreen does not cover; the serial loop
    and the parallel workers share it, so they stay identical. A FisherZ or StudentT test evaluates
    the chunks of consecutive pairs in one stack (:func:`_stacked_scan`); another test scans each
    pair with :func:`scan_conditioning_sets`. Both test the same subsets in the same order. With
    ``trace`` given, every test is recorded in the order it runs.
    """
    canonical = _canonical_route(test)
    if canonical is not None:
        return _stacked_scan(canonical, pairs, neighbors, size, alpha, trace, record_pvalues)
    removals: list[Removal] = []
    for x, y in pairs:
        separating: set[int] = set()
        found = False
        largest: list[float] | None = [] if record_pvalues else None
        for a, b, candidates, seen in _sides(neighbors, x, y, size):
            if scan_conditioning_sets(
                test, a, b, candidates, size, alpha, separating, seen, trace, largest
            ):
                found = True
        if found:
            removals.append((x, y, tuple(sorted(separating)), max(largest) if largest else None))
    return removals


def _sides(
    neighbors: list[np.ndarray], x: int, y: int, size: int
) -> Iterator[tuple[int, int, list[int], set[int] | None]]:
    """The sides :func:`scan_pairs` tests the pair ``(x, y)`` from, as ``(a, b, candidates,
    skip_within)`` for :func:`scan_conditioning_sets`: ``x`` first, then ``y``, each side's
    candidates its other neighbors, and a side with fewer than ``size`` of them left out."""
    from_x = neighbors[x][neighbors[x] != y].tolist()
    from_y = neighbors[y][neighbors[y] != x].tolist()
    # Once the x side has scanned, the y side skips the subsets both sides offer.
    shared = set(from_x) & set(from_y) if len(from_x) >= size else None
    for a, b, candidates, seen in ((x, y, from_x, None), (y, x, from_y, shared)):
        if len(candidates) >= size:
            yield a, b, candidates, seen


def _stacked_scan(
    pvalues: Callable[..., np.ndarray],
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    alpha: float,
    trace: list[dict] | None,
    record_pvalues: bool,
) -> list[Removal]:
    """:func:`scan_pairs` for a test whose ``pvalues`` take canonical query arrays.

    Each chunk is the one :func:`scan_conditioning_sets` evaluates in one call, in the per-pair
    order (:func:`_pair_chunks`); :func:`_stacks` evaluates consecutive chunks together. Every
    p-value is the one its chunk gets alone, and the chunks are read back in order, so the
    separating sets, the largest p-values, and the trace are those of the per-pair scan.
    """
    separated: dict[int, tuple[set[int], list[float]]] = {}
    for stack, values in _stacks(pvalues, size, _pair_chunks(pairs, neighbors, size)):
        bounds = np.cumsum([0, *(len(chunk) for _, _, _, chunk in stack)]).tolist()
        passing = values > alpha
        if trace is not None:
            for (_, x, y, chunk), start, stop in zip(stack, bounds, bounds[1:]):
                record_tests(trace, size, x, y, chunk, values[start:stop], alpha)
        for k in np.flatnonzero(np.logical_or.reduceat(passing, bounds[:-1])).tolist():
            i, _, _, chunk = stack[k]
            rows = np.flatnonzero(passing[bounds[k] : bounds[k + 1]])
            members, tops = separated.setdefault(i, (set(), []))
            for row in rows.tolist():
                members.update(chunk[row])
            if record_pvalues:
                tops.append(float(values[bounds[k] + rows].max()))
    removals: list[Removal] = []
    for i, (members, tops) in separated.items():
        x, y = pairs[i]
        removals.append((x, y, tuple(sorted(members)), max(tops) if record_pvalues else None))
    return removals


def _pair_chunks(
    pairs: list[tuple[int, int]], neighbors: list[np.ndarray], size: int
) -> Iterator[_Chunk]:
    """Yield ``(i, x, y, chunk)``, ``x < y``, for each chunk of subsets the per-pair scan of
    :func:`scan_pairs` evaluates in one call, in its order; ``i`` is the pair's position."""
    for i, (a, b) in enumerate(pairs):
        x, y = min(a, b), max(a, b)
        for _, _, candidates, seen in _sides(neighbors, a, b, size):
            subsets = _subsets(candidates, size, seen)
            while chunk := list(islice(subsets, _CHUNK)):
                yield i, x, y, chunk


def _stacks(
    pvalues: Callable[..., np.ndarray],
    size: int,
    chunks: Iterable[_Chunk],
) -> Iterator[tuple[list[_Chunk], np.ndarray]]:
    """Yield ``(stack, values)``: consecutive ``chunks``, whole, and the p-values of their queries.

    A stack holds as many chunks as fit in ``_STACK_BYTES`` of correlation submatrices, at least
    one (:func:`_evaluated`).
    """
    limit = max(1, _STACK_BYTES // (8 * (size + 2) ** 2))
    stack: list[_Chunk] = []
    rows = 0
    for item in chunks:
        if stack and rows + len(item[3]) > limit:
            yield from _evaluated(pvalues, size, stack, rows)
            stack, rows = [], 0
        stack.append(item)
        rows += len(item[3])
    if stack:
        yield from _evaluated(pvalues, size, stack, rows)


def _evaluated(
    pvalues: Callable[..., np.ndarray],
    size: int,
    stack: list[_Chunk],
    rows: int,
) -> Iterator[tuple[list[_Chunk], np.ndarray]]:
    """Yield ``(stack, values)`` for the ``rows`` queries of ``stack`` evaluated in one call.

    When that call raises, each chunk is evaluated in a call of its own and yielded alone, so the
    error comes from the chunk the per-pair scan raises on, after the same calls.
    """
    counts = [len(chunk) for _, _, _, chunk in stack]
    xs = np.repeat(np.array([x for _, x, _, _ in stack], dtype=np.intp), counts)
    ys = np.repeat(np.array([y for _, _, y, _ in stack], dtype=np.intp), counts)
    members = chain.from_iterable(subset for _, _, _, chunk in stack for subset in chunk)
    zs = np.fromiter(members, np.intp, rows * size).reshape(rows, size)
    try:
        values = pvalues(xs, ys, zs)
    except Exception:
        values = None
    if values is not None:
        yield stack, values
        return
    for item in stack:
        yield [item], _canonical_chunk(pvalues, size, *item[1:])


def pair_queries(neighbors: list[np.ndarray], x: int, y: int, size: int) -> int:
    """The tests :func:`scan_pairs` runs on the adjacent pair ``(x, y)`` at conditioning ``size``.

    It counts the size-``size`` subsets of either endpoint's other neighbors, a subset both offer
    once.
    """
    shared = len(set(neighbors[x].tolist()) & set(neighbors[y].tolist()))
    return (
        comb(neighbors[x].size - 1, size) + comb(neighbors[y].size - 1, size) - comb(shared, size)
    )


def separate_apart(
    test: CITest,
    adjacency: np.ndarray,
    apart: np.ndarray,
    alpha: float,
    *,
    max_depth: int | None = None,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """Separating sets, over the final skeleton, for the pairs background knowledge makes
    non-adjacent.

    ``adjacency`` is the boolean skeleton the search ends with and ``apart`` the boolean pairs the
    knowledge makes non-adjacent, which the search never tested. Every such pair that shares a
    neighbor, and so forms an unshielded triple, is tested as the search tests a pair it removes:
    against subsets of either endpoint's neighbors by increasing size, up to ``max_depth`` when one
    is given, with :func:`scan_pairs`. Returns ``(x, y, separating, p)``, ``x < y``, for every pair
    some size separates: the union of the separating subsets at the first size that separates it
    and their largest p-value (``None`` when ``record_pvalues`` is false). A pair no tested subset
    separates is left out: the subsets come from the final neighborhoods and stop at
    ``max_depth``, so a separating set elsewhere is not found. With ``trace`` given, every test is
    recorded as :func:`scan_pairs` records it.
    """
    common = adjacency.astype(np.float32) @ adjacency.astype(np.float32) > 0
    pairs = adjacent_pairs(np.triu(apart & common, 1))
    neighbors = [np.flatnonzero(row) for row in adjacency]
    found: list[Removal] = []
    size = 0
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        while pairs and (max_depth is None or size <= max_depth):
            removals = scan_pairs(test, pairs, neighbors, size, alpha, trace, record_pvalues)
            found.extend(removals)
            separated = {(x, y) for x, y, _, _ in removals}
            pairs = [
                (x, y)
                for x, y in pairs
                if (x, y) not in separated and max(neighbors[x].size, neighbors[y].size) > size
            ]
            size += 1
    return found


def unseparated_triples(
    adjacency: np.ndarray,
    sepsets: Mapping[tuple[int, int], object],
    constraints: Constraints | None,
) -> frozenset[tuple[int, int, int]]:
    """The unshielded triples ``(a, b, c)``, ``a < c``, over a non-adjacent pair with no
    separating set.

    ``adjacency`` is the boolean skeleton and ``sepsets`` the separating sets, keyed in either
    order. A pair ``constraints`` make non-adjacent that ``sepsets`` leaves out has no separating
    set the search found, so whether ``b`` is a collider is unknown: no rule reads the triple as a
    collider or a non-collider. Without constraints the search tested every pair it left
    non-adjacent, and there are none.
    """
    if constraints is None:
        return frozenset()
    apart = ~constraints.support()
    triples = set()
    for b, row in enumerate(adjacency):
        for a, c in combinations(np.flatnonzero(row).tolist(), 2):
            if apart[a, c] and (a, c) not in sepsets and (c, a) not in sepsets:
                triples.add((a, b, c))
    return frozenset(triples)


def adjacent_pairs(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """The still-adjacent pairs ``(x, y)``, ``x < y``, row-major (ascending ``x`` then ``y``)."""
    xs, ys = np.nonzero(np.triu(adjacency != 0, 1))
    return list(zip(xs.tolist(), ys.tolist()))
