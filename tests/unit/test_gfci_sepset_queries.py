"""GFCI's separating-set search takes the canonical query path with unchanged results.

For FisherZ and StudentT, ``_derive_sepsets`` evaluates each nonempty size's subsets of a pair in
chunks of canonical query arrays. It asks the queries of the scalar loop, which a subclass of either
test takes, in the same order, and records the same separating sets; a singular query raises the
scalar loop's error after the same queries.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import TAIL
from andrey.core.ci import FisherZ, SingularCorrelationError, StudentT
from andrey.search import gfci
from andrey.search.gfci import _derive_sepsets

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


class _ScalarFisherZ(FisherZ):
    """A FisherZ subclass, which ``_derive_sepsets`` queries one call at a time."""


class _ScalarStudentT(StudentT):
    """A StudentT subclass, which ``_derive_sepsets`` queries one call at a time."""


TESTS = [(FisherZ, _ScalarFisherZ), (StudentT, _ScalarStudentT)]


def _case(seed: int, d: int = 12, n: int = 300, p: float = 0.35):
    """A random CPDAG adjacency over a sample with a few linear dependencies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.7 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 0] + 0.5 * X[:, 2]
    X[:, 4] += 0.5 * X[:, 3]
    upper = np.triu(rng.random((d, d)) < p, 1)
    cpdag = np.where(upper | upper.T, np.int8(TAIL), np.int8(0))
    return X, cpdag


def _logged(test):
    """``test`` with every query it answers appended to the returned log, as ``(x, y, set)``."""
    log: list[tuple[int, int, tuple[int, ...]]] = []
    scalar, canonical = test._partial_correlation, test._canonical_pvalues

    def partial_correlation(x, y, cond):
        log.append((x, y, tuple(cond)))
        return scalar(x, y, cond)

    def canonical_pvalues(xs, ys, conditions):
        values = canonical(xs, ys, conditions)
        log.extend(zip(xs.tolist(), ys.tolist(), map(tuple, conditions.tolist())))
        return values

    test._partial_correlation = partial_correlation
    test._canonical_pvalues = canonical_pvalues
    return test, log


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
@pytest.mark.parametrize("chunk", [2, None])
@pytest.mark.parametrize("max_depth", [None, 1])
@pytest.mark.parametrize("in_column_order", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_canonical_queries_match_the_scalar_loop(
    monkeypatch: pytest.MonkeyPatch, canonical, scalar, chunk, max_depth, in_column_order, seed
) -> None:
    if chunk is not None:
        monkeypatch.setattr(gfci, "_CHUNK", chunk)
    X, cpdag = _case(seed)
    results = []
    for kind in (canonical, scalar):
        test, log = _logged(kind(X))
        sepsets = _derive_sepsets(
            cpdag, test, ALPHA, max_depth=max_depth, in_column_order=in_column_order
        )
        results.append((sepsets, log))
    assert results[0] == results[1]
    assert any(len(cond) >= 2 for _, _, cond in results[0][1]) or max_depth == 1


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
def test_a_singular_query_raises_the_scalar_error(canonical, scalar) -> None:
    d = 8
    X = np.random.default_rng(4).standard_normal((200, d))
    X[:, 5] = X[:, 0] + X[:, 1]  # (3, 4) given {0, 1, 5}, after {0, 1, 2}, is singular
    cpdag = np.zeros((d, d), dtype=np.int8)
    for u in (0, 1, 2, 5):
        for v in (3, 4):
            cpdag[u, v] = cpdag[v, u] = TAIL
    errors, logs = [], []
    for kind in (canonical, scalar):
        test, log = _logged(kind(X))
        with pytest.raises(SingularCorrelationError) as raised:
            _derive_sepsets(cpdag, test, 1.0)
        errors.append((type(raised.value), str(raised.value)))
        logs.append(log)
    assert errors[0] == errors[1]
    assert logs[0] == logs[1]
    assert logs[0], "the singular query must come after other queries"
