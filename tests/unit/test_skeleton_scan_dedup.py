"""Skeleton scan dedup and the lazy possible-d-sep generator leave every decision unchanged.

``scan_pairs`` scans a pair from either endpoint and the second endpoint skips the subsets lying
inside the first's candidates, so a shared subset is tested once. A dense random graph, where the
two neighborhoods overlap heavily, is compared against the unfiltered scan and the query count is
checked against the closed form. ``_iter_dsep_choices`` yields subsets one at a time; it is compared
exhaustively against the eager list-building form, aliasing included.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from andrey.constraint._skeleton import scan_conditioning_sets, scan_pairs
from andrey.constraint.fci import _iter_dsep_choices
from andrey.core.ci import FisherZ

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch):
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _unfiltered_scan_pairs(test, pairs, neighbours, size, alpha):
    """Both directions scanned in full, no subset skipped."""
    removals = []
    for x, y in pairs:
        separating: set[int] = set()
        found = False
        for a, b in ((x, y), (y, x)):
            candidates = neighbours[a][neighbours[a] != b]
            if candidates.size < size:
                continue
            if scan_conditioning_sets(test, a, b, candidates.tolist(), size, alpha, separating):
                found = True
        if found:
            removals.append((x, y, tuple(sorted(separating))))
    return removals


def _dense_case(seed: int, d: int = 9, n: int = 300, p: float = 0.7):
    """A dense random adjacency over a sample with a few linear dependencies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.7 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 0] + 0.5 * X[:, 2]
    upper = np.triu(rng.random((d, d)) < p, 1)
    adj = upper | upper.T
    neighbours = [np.flatnonzero(adj[i]) for i in range(d)]
    pairs = [(x, y) for x in range(d) for y in range(x + 1, d) if adj[x, y]]
    return X, pairs, neighbours


class _CountingTest:
    """Wraps a CI test and counts the conditioning sets it is asked about."""

    def __init__(self, inner: FisherZ) -> None:
        self.inner = inner
        self.queries = 0

    def __call__(self, x, y, condition_set=()):
        self.queries += 1
        return self.inner(x, y, condition_set)

    def batched_call(self, x, y, condition_sets):
        condition_sets = list(condition_sets)
        self.queries += len(condition_sets)
        return self.inner.batched_call(x, y, condition_sets)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_dense_scan_matches_unfiltered_scan(seed, size):
    X, pairs, neighbours = _dense_case(seed)
    shared = [len(set(neighbours[x].tolist()) & set(neighbours[y].tolist())) for x, y in pairs]
    assert max(shared) >= size, "the case must offer subsets both endpoints share"
    got = scan_pairs(FisherZ(X), pairs, neighbours, size, ALPHA)
    want = _unfiltered_scan_pairs(FisherZ(X), pairs, neighbours, size, ALPHA)
    assert got == want
    assert got, "the case must remove something"


@pytest.mark.parametrize("size", [2, 3])
def test_shared_subsets_are_tested_once(size):
    X, pairs, neighbours = _dense_case(0)
    counting = _CountingTest(FisherZ(X))
    scan_pairs(counting, pairs, neighbours, size, ALPHA)
    expected = 0
    for x, y in pairs:
        from_x = set(neighbours[x].tolist()) - {y}
        from_y = set(neighbours[y].tolist()) - {x}
        expected += comb(len(from_x), size) + comb(len(from_y), size)
        expected -= comb(len(from_x & from_y), size)
    assert counting.queries == expected


def test_skip_within_below_size_is_a_no_op():
    X, pairs, neighbours = _dense_case(1)
    x, y = pairs[0]
    candidates = neighbours[x][neighbours[x] != y].tolist()
    test = FisherZ(X)
    plain: set[int] = set()
    skipped: set[int] = set()
    found = scan_conditioning_sets(test, x, y, candidates, 2, ALPHA, plain)
    found_skip = scan_conditioning_sets(test, x, y, candidates, 2, ALPHA, skipped, {candidates[0]})
    assert (found, plain) == (found_skip, skipped)


# ---- lazy possible-d-sep subsets ----------------------------------------------------------------


def _eager_dsep_choices(dsep, adj_a, adj_b):
    """The eager list-building generator, aliasing included."""
    raw: list[tuple[int, ...]] = []
    for size in range(len(dsep) + 1):
        raw.extend(combinations(range(len(dsep)), size))
    m = len(raw)
    out = []
    for j in range(1, m + 1):
        positions = raw[j] if j < m and len(raw[j]) == len(raw[j - 1]) else raw[j - 1]
        if len(positions) < 2:
            continue
        values = [dsep[i] for i in positions]
        if all(v in adj_a for v in values):
            continue
        if all(v in adj_b for v in values):
            continue
        out.append(tuple(values))
    return out


@pytest.mark.parametrize("seed", range(24))
def test_lazy_dsep_choices_match_eager(seed):
    rng = np.random.default_rng(seed)
    length = int(rng.integers(2, 11))
    dsep = rng.permutation(30)[:length].tolist()
    adj_a = set(rng.choice(30, size=int(rng.integers(0, 12)), replace=False).tolist())
    adj_b = set(rng.choice(30, size=int(rng.integers(0, 12)), replace=False).tolist())
    assert list(_iter_dsep_choices(dsep, adj_a, adj_b)) == _eager_dsep_choices(dsep, adj_a, adj_b)


def test_lazy_dsep_choices_aliasing():
    # The first combination of every size is skipped and the last emitted twice; a size with a
    # single combination (the full set) is emitted once.
    assert list(_iter_dsep_choices([4, 7], set(), set())) == [(4, 7)]
    assert list(_iter_dsep_choices([4, 7, 9], set(), set())) == [(4, 9), (7, 9), (7, 9), (4, 7, 9)]
