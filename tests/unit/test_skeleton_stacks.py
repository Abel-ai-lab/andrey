"""The skeleton scan stacks the chunks of many pairs into one evaluation, with unchanged results.

For FisherZ and StudentT, ``scan_pairs`` evaluates the chunks of consecutive pairs together, up to
``_STACK_BYTES`` of correlation submatrices. Every cap, from one chunk per stack to the default,
gives the removals, separating sets, largest p-values, and trace of the per-pair scan, which a
subclass of either test takes; a singular query raises the per-pair scan's error after the same
trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.constraint import _skeleton
from andrey.constraint._skeleton import pair_queries, scan_pairs
from andrey.core.ci import FisherZ, SingularCorrelationError, StudentT

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


class _PerPairFisherZ(FisherZ):
    """A FisherZ subclass, which ``scan_pairs`` scans pair by pair."""


class _PerPairStudentT(StudentT):
    """A StudentT subclass, which ``scan_pairs`` scans pair by pair."""


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
