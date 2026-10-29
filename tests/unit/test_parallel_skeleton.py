"""Skeleton worker pool: the gate, the tests it takes, its order, its errors, and the worker count.

The pool runs an ``|S| >= 2`` pass of a FisherZ or StudentT test when the pass holds at least
``MIN_POOL_QUERIES`` tests. A pooled pass returns the serial scan's removals in the serial order,
so the adjacency, the separating sets, their insertion order, and the p-values are the serial
ones, and a singular query raises the serial exception. ``andrey.pc``, ``fci``, ``gfci``, and
``cdnod`` read the worker count from ``andrey.config.n_jobs`` or ``ANDREY_N_JOBS``. Most
tests that need the pool set the gate to 0, so a small case still reaches it, and every one spies
on the pool's creation, so a pooled comparison cannot silently run serial against serial.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey.constraint import _parallel_skeleton
from andrey.constraint.fci import _fast_adjacency_search
from andrey.core.ci import FisherZ, SingularCorrelationError, make_indep_test
from andrey.data import sample_discrete

pc_module = importlib.import_module("andrey.constraint.pc")


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def pools(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The worker count of each skeleton pool created during the test."""
    created: list[int] = []
    real = _parallel_skeleton.ProcessPoolExecutor

    def spy(*args, **kwargs):
        created.append(kwargs["max_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(_parallel_skeleton, "ProcessPoolExecutor", spy)
    return created


@pytest.fixture
def open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every ``|S| >= 2`` pass of a pool-capable test to the pool."""
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "0")


def _gaussian_dag(seed: int, n: int, d: int, density: float) -> np.ndarray:
    """A random strictly-lower-triangular linear-Gaussian SEM with edge probability ``density``."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.4, 1.2, (d, d)) * rng.choice([-1.0, 1.0], (d, d))
    b = np.tril(weights * (rng.random((d, d)) < density), -1)
    x = np.zeros((n, d))
    noise = rng.standard_normal((n, d))
    for j in range(d):
        x[:, j] = noise[:, j] + x @ b[j]
    return x


def _ordered(skeleton: tuple) -> tuple:
    """The skeleton with each dict as its items in insertion order."""
    return (skeleton[0].tolist(), *(list(part.items()) for part in skeleton[1:]))


def test_a_pass_below_the_gate_runs_in_process(pools: list[int]) -> None:
    x = _gaussian_dag(0, 500, 15, 0.3)
    trace: list[dict] = []
    serial = pc_module._discover_skeleton(x, 0.05, "fisherz", workers=1, _trace=trace)
    gated = pc_module._discover_skeleton(x, 0.05, "fisherz", workers=3)
    assert any(record["depth"] >= 2 for record in trace), "no pass of size two or more ran"
    assert pools == []
    assert _ordered(gated) == _ordered(serial)


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
def test_pooled_skeletons_keep_the_serial_order(pools: list[int], indep_test: str) -> None:
    x = _gaussian_dag(1, 600, 20, 0.25)
    serial = pc_module._discover_skeleton(x, 0.05, indep_test, workers=1)
    pooled = pc_module._discover_skeleton(x, 0.05, indep_test, workers=3)
    assert _ordered(pooled) == _ordered(serial)
    test = make_indep_test(indep_test, x)
    fci_serial = _fast_adjacency_search(x, test, 0.05, workers=1)
    fci_pooled = _fast_adjacency_search(x, test, 0.05, workers=3)
    assert _ordered(fci_pooled) == _ordered(fci_serial)
    assert pools == [3, 3]


@pytest.mark.usefixtures("open_gate")
def test_other_tests_scan_in_process(pools: list[int]) -> None:
    data = sample_discrete(d=10, n=300, seed=0, density=3.0, categories=(2, 4))
    x = data.data.astype(np.float64)
    serial = pc_module._discover_skeleton(x, 0.05, "chisq", workers=1)
    assert _ordered(pc_module._discover_skeleton(x, 0.05, "chisq", workers=3)) == _ordered(serial)

    class Wrapped(FisherZ):
        """A FisherZ subclass, which may change any query path."""

    y = _gaussian_dag(1, 600, 20, 0.25)
    _fast_adjacency_search(y, Wrapped(y), 0.05, workers=3)
    assert pools == []


def _singular_data() -> np.ndarray:
    # Column 2 is column 0 plus column 1, and column 1 depends on column 0, so the three stay
    # adjacent and a size-2 query of two of them given the third is singular.
    x = _gaussian_dag(2, 400, 30, 0.2)
    x[:, 1] = 0.7 * x[:, 0] + np.random.default_rng(3).standard_normal(len(x))
    x[:, 2] = x[:, 0] + x[:, 1]
    return x


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("indep_test", ["fisherz", "student_t"])
def test_a_singular_query_raises_the_serial_error(pools: list[int], indep_test: str) -> None:
    x = _singular_data()
    with pytest.raises(SingularCorrelationError) as serial:
        pc_module._discover_skeleton(x, 0.05, indep_test, workers=1)
    with pytest.raises(SingularCorrelationError) as pooled:
        pc_module._discover_skeleton(x, 0.05, indep_test, workers=3)
    assert type(pooled.value) is type(serial.value)
    assert str(pooled.value) == str(serial.value)
    assert pools == [3]


def _domain_index(n: int) -> np.ndarray:
    return np.repeat([0.0, 1.0], n // 2)


_FACADES = {
    "pc": andrey.pc,
    "fci": andrey.fci,
    "gfci": andrey.gfci,
    "cdnod": lambda x: andrey.cdnod(x, _domain_index(len(x))),
}


def _same_output(a: andrey.StructureOutput, b: andrey.StructureOutput) -> bool:
    return np.array_equal(a.structure.to_numpy(), b.structure.to_numpy()) and repr(
        a.metadata
    ) == repr(b.metadata)


@pytest.mark.usefixtures("open_gate")
@pytest.mark.parametrize("method", sorted(_FACADES))
def test_facades_run_the_configured_workers_on_the_pool(pools: list[int], method: str) -> None:
    x = _gaussian_dag(1, 600, 20, 0.25)
    fit = _FACADES[method]
    serial = fit(x)
    with andrey.config(n_jobs=2):
        pooled = fit(x)
    assert pools == [2]
    assert _same_output(pooled, serial)


@pytest.mark.usefixtures("open_gate")
def test_the_environment_sets_the_workers(
    pools: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    x = _gaussian_dag(1, 600, 20, 0.25)
    serial = andrey.fci(x)
    monkeypatch.setenv("ANDREY_N_JOBS", "3")
    pooled = andrey.fci(x)
    assert pools == [3]
    assert _same_output(pooled, serial)


def test_a_pass_above_the_gate_runs_on_the_pool(
    pools: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dense enough that the size-2 pass holds more than 10,000 tests; capped there.
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "10000")
    x = _gaussian_dag(4, 800, 40, 0.3)
    serial = andrey.pc(x, max_depth=2)
    with andrey.config(n_jobs=2):
        pooled = andrey.pc(x, max_depth=2)
    assert pools == [2]
    assert _same_output(pooled, serial)


@pytest.mark.usefixtures("open_gate")
def test_each_pass_is_counted_once(pools: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    """PC counts a pass's tests once for its warning and the pool's cutoff; FCI's pool counts it."""
    counted = {"pc": 0, "pool": 0}
    scans: list[tuple[int, list[int] | None]] = []

    def counting(module, name):
        real = module.pair_queries

        def count(*args):
            counted[name] += 1
            return real(*args)

        monkeypatch.setattr(module, "pair_queries", count)

    counting(pc_module, "pc")
    counting(_parallel_skeleton, "pool")
    real_scan = _parallel_skeleton.SkeletonWorkerPool.scan

    def scan(self, pairs, neighbors, adj_mask, size, alpha, record_pvalues=True, weights=None):
        scans.append((len(pairs), weights))
        return real_scan(self, pairs, neighbors, adj_mask, size, alpha, record_pvalues, weights)

    monkeypatch.setattr(_parallel_skeleton.SkeletonWorkerPool, "scan", scan)
    x = _gaussian_dag(1, 600, 20, 0.25)

    pc_module._discover_skeleton(x, 0.05, "fisherz", workers=3, warn_expensive=True)
    assert scans and all(weights is not None and len(weights) == n for n, weights in scans)
    assert counted == {"pc": sum(n for n, _ in scans), "pool": 0}

    counted.update(pc=0, pool=0)
    scans.clear()
    _fast_adjacency_search(x, make_indep_test("fisherz", x), 0.05, workers=3)
    assert scans and all(weights is None for _, weights in scans)
    assert counted == {"pc": 0, "pool": sum(n for n, _ in scans)}

    counted.update(pc=0, pool=0)
    pc_module._discover_skeleton(x, 0.05, "fisherz", workers=1)
    assert counted == {"pc": 0, "pool": 0}, "a serial fit that does not warn counts nothing"
