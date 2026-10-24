"""The skeleton scan stacks the chunks of many pairs into one evaluation, with unchanged results.

For FisherZ, StudentT, and RCoT, ``scan_pairs`` evaluates the chunks of consecutive pairs
together, up to ``_STACK_BYTES`` of correlation submatrices. Every cap, from one chunk per stack to
the default, gives the removals, separating sets, largest p-values, and trace of the per-pair scan,
which a subclass of each test takes; a query that raises raises the per-pair scan's error after the
same trace. The size-1 pass, ``first_order_removals``, stacks the same way and prepares its queries
only as the stacks need them.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.constraint import _skeleton
from andrey.constraint._skeleton import (
    canonical_capable,
    first_order_removals,
    pair_queries,
    scan_pairs,
)
from andrey.core.ci import FisherZ, SingularCorrelationError, StudentT
from andrey.core.ci_kernel import RCoT

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


class _PerPairFisherZ(FisherZ):
    """A FisherZ subclass, which ``scan_pairs`` scans pair by pair."""


class _PerPairStudentT(StudentT):
    """A StudentT subclass, which ``scan_pairs`` scans pair by pair."""


class _PerPairRCoT(RCoT):
    """An RCoT subclass, which ``scan_pairs`` scans pair by pair."""


TESTS = [(FisherZ, _PerPairFisherZ), (StudentT, _PerPairStudentT)]


def _case(seed: int, d: int = 12, n: int = 300, p: float = 0.6):
    """A dense random adjacency over a sample with a few linear dependencies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.7 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 0] + 0.5 * X[:, 2]
    upper = np.triu(rng.random((d, d)) < p, 1)
    adj = upper | upper.T
    neighbors = [np.flatnonzero(adj[i]) for i in range(d)]
    pairs = [(x, y) for x in range(d) for y in range(x + 1, d) if adj[x, y]]
    return X, pairs, neighbors


def _scan(test, pairs, neighbors, size, record_pvalues=True):
    trace: list[dict] = []
    removals = scan_pairs(test, pairs, neighbors, size, ALPHA, trace, record_pvalues)
    return removals, trace


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, 4000, None])
@pytest.mark.parametrize("chunk", [2, None])
@pytest.mark.parametrize("size", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_stacks_match_the_per_pair_scan(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap, chunk, size, seed
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    if chunk is not None:
        monkeypatch.setattr(_skeleton, "_CHUNK", chunk)
    X, pairs, neighbors = _case(seed)
    for record_pvalues in (True, False):
        got = _scan(stacked(X), pairs, neighbors, size, record_pvalues)
        want = _scan(per_pair(X), pairs, neighbors, size, record_pvalues)
        assert got == want
    assert got[0], "the case must remove something"


def test_a_stack_holds_many_pairs_within_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    size, limit = 2, 200
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * (size + 2) ** 2)
    X, pairs, neighbors = _case(0)
    test = FisherZ(X)
    rows: list[int] = []
    canonical = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return canonical(x, y, conditions)

    test._canonical_pvalues = spy
    scan_pairs(test, pairs, neighbors, size, ALPHA)
    assert sum(rows) == sum(pair_queries(neighbors, x, y, size) for x, y in pairs)
    assert max(rows) <= limit
    assert len(rows) < len(pairs), "a stack must hold the chunks of several pairs"


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, None])
def test_a_singular_stack_raises_the_per_pair_error(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    d = 10
    X = np.random.default_rng(4).standard_normal((200, d))
    X[:, -1] = X[:, -3] + X[:, -2]  # a query over the last three columns is singular
    pairs = [(x, y) for x in range(d) for y in range(x + 1, d)]
    neighbors = [np.array([j for j in range(d) if j != i]) for i in range(d)]
    errors, traces = [], []
    for test in (stacked(X), per_pair(X)):
        trace: list[dict] = []
        with pytest.raises(SingularCorrelationError) as raised:
            scan_pairs(test, pairs, neighbors, 2, ALPHA, trace)
        errors.append((type(raised.value), str(raised.value)))
        traces.append(trace)
    assert errors[0] == errors[1]
    assert traces[0] == traces[1]
    assert traces[0], "the singular query must come after other pairs' tests"


def _first_order(test, pairs, neighbors, record_pvalues=True):
    trace: list[dict] = []
    removals = first_order_removals(test, pairs, neighbors, ALPHA, trace, record_pvalues)
    return removals, trace


def _candidate_rows(pairs, neighbors):
    """The first-order queries of ``pairs``: either endpoint's neighbors, less the pair."""
    return sum(len((set(neighbors[x]) | set(neighbors[y])) - {x, y}) for x, y in pairs)


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, 4000, None])
@pytest.mark.parametrize("seed", range(3))
def test_first_order_stacks_match_the_per_pair_pass(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap, seed
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, pairs, neighbors = _case(seed)
    for record_pvalues in (True, False):
        got = _first_order(stacked(X), pairs, neighbors, record_pvalues)
        want = _first_order(per_pair(X), pairs, neighbors, record_pvalues)
        assert got == want
    assert got[0], "the case must remove something"
    assert len(got[1]) == _candidate_rows(pairs, neighbors)


def test_a_first_order_stack_holds_many_pairs_within_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 100
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * 3**2)
    X, pairs, neighbors = _case(0)
    test = FisherZ(X)
    rows: list[int] = []
    canonical = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return canonical(x, y, conditions)

    test._canonical_pvalues = spy
    first_order_removals(test, pairs, neighbors, ALPHA)
    assert sum(rows) == _candidate_rows(pairs, neighbors)
    assert max(rows) <= limit
    assert len(rows) < len(pairs), "a stack must hold the queries of several pairs"


class _Watched(FisherZ):
    """A FisherZ subclass whose first-order calls report how many rows were prepared before each."""

    def __init__(self, X, prepared):
        super().__init__(X)
        self.prepared = prepared
        self.waiting: list[int] = []
        self.evaluated = 0

    def first_order_pvalues(self, x, y, zs):
        zs = list(zs)
        self.waiting.append(self.prepared() - self.evaluated)
        self.evaluated += len(zs)
        return super().first_order_pvalues(x, y, zs)


@pytest.mark.parametrize("stacked", [True, False])
def test_first_order_queries_are_prepared_as_they_are_evaluated(
    monkeypatch: pytest.MonkeyPatch, stacked: bool
) -> None:
    """Rows built but not yet evaluated stay within one stack and one pair, whatever the pairs."""
    limit, d = 100, 30
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * 3**2)
    prepared = 0
    real = _skeleton._first_order_queries

    def counting(pairs, neighbors):
        nonlocal prepared
        for query in real(pairs, neighbors):
            prepared += len(query[2])
            yield query

    monkeypatch.setattr(_skeleton, "_first_order_queries", counting)
    X, pairs, neighbors = _case(1, d=d, p=1.0)
    total = _candidate_rows(pairs, neighbors)
    assert total > 20 * limit
    if stacked:
        test = FisherZ(X)
        rows: list[int] = []
        canonical = test._canonical_pvalues
        waiting: list[int] = []

        def spy(x, y, conditions):
            waiting.append(prepared - sum(rows))
            rows.append(len(x))
            return canonical(x, y, conditions)

        test._canonical_pvalues = spy
        first_order_removals(test, pairs, neighbors, ALPHA)
        assert sum(rows) == total
        assert max(waiting) <= limit + d - 2
    else:
        test = _Watched(X, lambda: prepared)
        first_order_removals(test, pairs, neighbors, ALPHA)
        assert test.evaluated == total
        assert max(test.waiting) <= d - 2


def test_canonical_capable_is_the_exact_type_rule() -> None:
    X, _, _ = _case(0)

    class Scalar:
        def __call__(self, x, y, condition_set=()):
            return 1.0

    assert canonical_capable(FisherZ(X)) and canonical_capable(StudentT(X))
    assert not canonical_capable(_PerPairFisherZ(X))
    assert not canonical_capable(_PerPairStudentT(X))
    assert not canonical_capable(Scalar())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("cap", [1, None])
@pytest.mark.parametrize("size", [0, 1, 2])
def test_rcot_stacks_match_the_per_pair_scan(monkeypatch: pytest.MonkeyPatch, cap, size) -> None:
    """RCoT's stacks run each conditioning set's queries together; every p-value, removal, and
    trace record is the per-pair scan's, bit for bit."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, pairs, neighbors = _case(0, d=8, n=500)
    test, rows = RCoT(X), []
    grouped = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return grouped(x, y, conditions)

    test._canonical_pvalues = spy
    got = _scan(test, pairs, neighbors, size)
    want = _scan(_PerPairRCoT(X), pairs, neighbors, size)
    assert got == want
    assert got[0], "the case must remove something"
    assert sum(rows) == sum(pair_queries(neighbors, x, y, size) for x, y in pairs)


def test_a_failing_rcot_stack_raises_the_per_pair_error(monkeypatch: pytest.MonkeyPatch) -> None:
    X, pairs, neighbors = _case(0, d=8, n=500)
    failing = pairs[len(pairs) // 2]
    call = RCoT.__call__

    def raising(test, x, y, condition_set=()):
        if (min(x, y), max(x, y)) == failing:
            raise FloatingPointError(f"query {x}, {y}, {tuple(condition_set)}")
        return call(test, x, y, condition_set)

    monkeypatch.setattr(RCoT, "__call__", raising)
    errors, traces = [], []
    for test in (RCoT(X), _PerPairRCoT(X)):
        trace: list[dict] = []
        with pytest.raises(FloatingPointError) as raised:
            scan_pairs(test, pairs, neighbors, 2, ALPHA, trace)
        errors.append(str(raised.value))
        traces.append(trace)
    assert errors[0] == errors[1]
    assert traces[0] == traces[1]
    assert traces[0], "the failing query must come after other pairs' tests"
