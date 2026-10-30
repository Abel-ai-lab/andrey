"""PNL's batched pairs: the blocked HSIC equals the reference test, rows equal single calls, and the
torch path matches NumPy's to rounding."""

from __future__ import annotations

import functools
import importlib
import math
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.special import gammaincc

import andrey
from andrey.core import backend, warning_policy
from andrey.core.independence import (
    _center_kernel,
    _empirical_bandwidth,
    _gaussian_kernel,
    _kernel_into,
    _standardize,
    hsic_statistic_test,
)

_pnl = importlib.import_module("andrey.irregular.pnl")


def _table(n: int) -> np.ndarray:
    """A cause, its post-nonlinear effect, a column with ties, and a copy of the cause."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = np.tanh(x + 0.4 * rng.laplace(size=n)) ** 3
    return np.column_stack([x, y, np.round(rng.normal(size=n)), x])


def _reference_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The HSIC gamma test from whole-matrix kernels, spelled out, for 1-D ``x`` and ``y``."""
    n = len(x)
    kxc, kyc = (
        _center_kernel(
            _gaussian_kernel(_standardize(v.reshape(-1, 1)), _empirical_bandwidth(v[:, None]))
        )
        for v in (x, y)
    )
    statistic = np.sum(kxc * kyc)
    mean = np.trace(kxc) * np.trace(kyc) / n
    var = 2 * np.sum(kxc**2) * np.sum(kyc**2) / n / n
    return float(statistic / mean), float(gammaincc(mean**2 / var, statistic / (var / mean)))


@pytest.mark.parametrize("n", [150, 600, 1300])  # each kernel width of the HSIC test
def test_blocked_hsic_equals_the_whole_matrix_test(n):
    X = _table(n)
    cause, effect = _pnl._column(X[:, 0]), _pnl._column(X[:, 1])
    disturbance = _pnl._fit_disturbance(cause.phi, effect.outer, effect.outer_deriv)
    buffers = _pnl._Buffers.for_rows(n)
    kxc = _kernel_into(cause.standardized, cause.width, buffers.cause, buffers.block)
    blocked = _pnl._hsic(kxc, np.trace(kxc), np.sum(kxc**2), disturbance, buffers)
    reference = _reference_test(_pnl._gaussianize(X[:, 0]), disturbance)
    assert hsic_statistic_test(_pnl._gaussianize(X[:, 0]), disturbance) == reference
    assert blocked == reference


def test_pair_rows_equal_single_calls():
    """On the numpy backend, pinned because a torch device reduces a batch in another order."""
    X = _table(120)
    pairs = [(0, 1), (1, 0), (2, 3), (3, 3), (0, 3)]
    with andrey.config(backend="numpy"):
        rows = _pnl.pnl_pairs(X, pairs)
        assert [tuple(row) for row in rows] == [_pnl.pnl(X[:, i], X[:, j]) for i, j in pairs]
        assert np.array_equal(_pnl.pnl_pairs(X, pairs, n_jobs=3), rows)


def test_worker_threads_are_capped_by_a_byte_budget(monkeypatch):
    """Each thread holds three n x n matrices: a budget below the asked-for threads lowers them."""
    n = 120
    X = _table(n)
    per_thread = 3 * 8 * n * n
    runs = []
    run_tasks = _pnl._run_tasks
    monkeypatch.setattr(
        _pnl,
        "_run_tasks",
        lambda columns, tasks: runs.append(len(tasks)) or run_tasks(columns, tasks),
    )
    with andrey.config(backend="numpy"), warnings.catch_warnings():
        warnings.simplefilter("error")  # the default budget holds four threads at 120 rows
        full = _pnl.pnl_pairs(X, n_jobs=4)
        assert len(runs) == 4
        for budget, threads in ((2 * per_thread, 2), (per_thread // 2, 1)):
            runs.clear()
            warning_policy._warned.clear()
            monkeypatch.setattr(_pnl, "_THREAD_BYTES", budget)
            with pytest.warns(andrey.PerformanceWarning, match=f"runs {threads} of the 4 worker"):
                capped = _pnl.pnl_pairs(X, n_jobs=4)
            assert len(runs) == threads
            if threads > 1:  # every task's result is the same on any number of limited threads
                assert np.array_equal(capped, full)
        # `pnl` fits two directions, so 64 workers ask for two threads, which this budget holds.
        monkeypatch.setattr(_pnl, "_THREAD_BYTES", 2 * per_thread)
        runs.clear()
        with andrey.config(n_jobs=64):
            _pnl.pnl(X[:, 0], X[:, 1])
        assert len(runs) == 2


def test_pairs_default_to_every_column_pair_and_are_checked():
    X = _table(50)
    with andrey.config(backend="numpy"):  # a torch device batches a cause's pairs differently
        rows = _pnl.pnl_pairs(X)
        assert np.array_equal(rows[1], _pnl.pnl_pairs(X, [(0, 2)])[0])
    assert rows.shape == (6, 4)
    assert _pnl.pnl_pairs(X, []).shape == (0, 4)
    with pytest.raises(ValueError, match="out of range"):
        _pnl.pnl_pairs(X, [(0, 4)])
    with pytest.raises(ValueError, match="table"):
        _pnl.pnl_pairs(X[:, 0])


def test_device_dispatch_gates_on_rows_squared(monkeypatch):
    """A pair asks for a device by rows squared without starting one; a table may start one."""
    seen = []

    def spy(pinnable, auto_order, *, metric=None, thresholds=None, amortized=True, op=None):
        seen.append((metric, amortized, op, "mps" in pinnable))
        return "numpy"

    monkeypatch.setattr(_pnl.backend, "select", spy)
    X = _table(60)
    _pnl.pnl(X[:, 0], X[:, 1])
    _pnl.pnl_pairs(X)
    assert seen == [(60 * 60, False, "pnl", False), (60 * 60, True, "pnl", False)]


def test_device_cutoff_reads_its_variable_and_never_chooses_mps(monkeypatch):
    for name in ("ANDREY_BACKEND", "ANDREY_DEVICE", "ANDREY_PNL_GPU_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "mps", "numpy"))
    first = math.isqrt(_pnl._PNL_GPU_THRESHOLDS["cuda"] - 1) + 1  # fewest rows reaching it
    choose = functools.partial(_pnl._pnl_backend, amortized=True)
    assert (choose(first - 1), choose(first)) == ("numpy", "cuda")
    monkeypatch.setenv("ANDREY_PNL_GPU_THRESHOLD", "100")
    assert (choose(9), choose(10)) == ("numpy", "cuda")
    with andrey.config(backend="mps"):
        assert choose(10) == "cuda"  # float32 mps is not a PNL backend
    assert _pnl._PNL_GPU_THRESHOLDS["mps"] is None


def _decisions(rows: np.ndarray) -> np.ndarray:
    """``andrey.pnl``'s choice per row with ``alpha=None``: 1 forward, -1 backward, 0 no edge."""
    by_pvalue = np.sign(rows[:, 0] - rows[:, 1])
    return np.where(by_pvalue != 0, by_pvalue, np.sign(rows[:, 3] - rows[:, 2]))


@pytest.mark.parametrize("n", [150, 600, 1300])
def test_torch_path_matches_numpy(n):
    """On torch's CPU backend, the p-values and statistics match NumPy's to 1e-8, a batch smaller
    than a cause's effects included, and every decision is the same, except where NumPy's two
    statistics tie exactly (a column against itself or its copy) and rounding may break the tie.

    The tolerance covers torch's float64 ``exp``, which on some CPUs is off by up to 3e-9 relative
    in a fraction of processes.
    """
    pytest.importorskip("torch")
    X = _table(n)
    pairs = [(0, 1), (1, 0), (2, 3), (3, 3), (0, 3), (1, 2)]
    with andrey.config(backend="numpy"):
        expected = _pnl.pnl_pairs(X, pairs)
    backend._dispatch_counts.clear()
    with andrey.config(backend="cpu"):
        got = _pnl.pnl_pairs(X, pairs)
        with mock.patch.object(_pnl, "_DEVICE_BATCH_BYTES", 8 * n * n):  # one kernel per batch
            one_per_batch = _pnl.pnl_pairs(X, pairs, n_jobs=2)
    assert backend.dispatch_stats() == {("pnl", "cpu"): 2}
    tied = expected[:, 2] == expected[:, 3]
    assert tied.sum() == 2 and not tied.all()
    for rows in (got, one_per_batch):
        assert np.allclose(rows[:, :2], expected[:, :2], rtol=0, atol=1e-8)
        assert np.allclose(rows[:, 2:], expected[:, 2:], rtol=1e-8, atol=0)
        assert np.array_equal(_decisions(rows)[~tied], _decisions(expected)[~tied])
