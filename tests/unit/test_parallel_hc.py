"""Parallel Hill-Climbing is bit-identical to serial HC.

HC instantiates the shared ``andrey.search._parallel`` substrate: one pass scans every add/remove/
reverse move, a move ``u -> v`` is owned by the chunk holding its head ``v``, and the chunk bests
reduce by the strict total order ``(delta, u, v, op_rank)``. So the reduced move -- and the whole
applied sequence -- must equal serial HC's exactly, independent of worker count. The delta is
quantized in the comparison key (:data:`hc._KEY_DP`) so that ULP-level differences in the
non-bit-pure Schur delta (a symmetric ``add x->y`` vs ``add y->x`` tie computed by different worker
scorers) break identically to serial; ``test_symmetric_tie_parity`` pins that case.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pytest

from andrey.core import backend
from andrey.search import _parallel_hc as ph
from andrey.search.hc import hc


def _gaussian_sem(seed: int, n: int, d: int, p: float) -> np.ndarray:
    """A random strictly-lower-triangular linear-Gaussian SEM (edge density ``p``)."""
    rng = np.random.default_rng(seed)
    b = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                b[j, i] = rng.uniform(0.4, 1.2) * rng.choice([-1.0, 1.0])
    e = rng.standard_normal((n, d))
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = e[:, j] + x @ b[j]
    return x


@contextmanager
def _force_parallel():
    """Force the HC work gate open so every run fans out to the workers."""
    prev = os.environ.get("ANDREY_HC_PARALLEL_MIN_WORK")
    os.environ["ANDREY_HC_PARALLEL_MIN_WORK"] = "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("ANDREY_HC_PARALLEL_MIN_WORK", None)
        else:
            os.environ["ANDREY_HC_PARALLEL_MIN_WORK"] = prev


def _parallel(x: np.ndarray, workers: int):
    """Run ``hc`` on the pool-backed path (gate forced open) via the ``num_workers`` axis."""
    with _force_parallel(), backend.config(num_workers=workers):
        return hc(x)


_SWEEP = [(seed, d, p) for seed in range(4) for (d, p) in [(10, 0.3), (16, 0.2), (24, 0.12)]]


@pytest.mark.parametrize("seed,d,p", _SWEEP)
@pytest.mark.parametrize("workers", [2, 4, 8])
def test_parallel_bit_identical_to_serial(seed, d, p, workers):
    """The pool-backed path reproduces serial HC's CPDAG and objective exactly."""
    x = _gaussian_sem(seed, 600, d, p)
    serial_struct, serial_score = hc(x)
    par_struct, par_score = _parallel(x, workers)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy()), (
        f"CPDAG differs at seed={seed}, d={d}, workers={workers}"
    )
    assert serial_score == par_score  # bit-identical objective


def test_worker_count_invariance():
    """Every worker count gives the identical result (an exact partition of the move space)."""
    x = _gaussian_sem(0, 600, 24, 0.15)
    reference = hc(x)[0].to_numpy()
    for workers in (2, 3, 5, 8, 24):
        assert np.array_equal(_parallel(x, workers)[0].to_numpy(), reference), (
            f"worker count {workers} changed the result"
        )


def test_symmetric_tie_parity():
    """A symmetric ``add x->y`` / ``add y->x`` ULP tie must break identically.

    At d=30 the whole-range scan and the per-worker scan could pick opposite directions of the
    same edge because the Schur delta is not bit-pure across scorers; the quantized key prevents it.
    """
    x = _gaussian_sem(0, 600, 30, 0.1)
    serial_struct, serial_score = hc(x)
    par_struct, par_score = _parallel(x, 8)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_large_n_samples_parity():
    """Bit-identical at large n_samples with a correlated block -- what the delta/n key guards.

    The Schur-delta cross-scorer noise scales as ~n_samples * cond(cov) * eps, so a fixed grid on
    the raw delta under-quantizes here; keying on delta/n keeps the grid tracking the signal.
    n=40000 with near-collinear columns pushes n*cond past where a raw-delta key would flake.
    """
    rng = np.random.default_rng(5)
    n, d = 40000, 24
    base = rng.standard_normal((n, d))
    base[:, 1] = base[:, 0] + 1e-3 * rng.standard_normal(n)  # a highly-correlated pair (large cond)
    base[:, 5] = base[:, 4] + 1e-3 * rng.standard_normal(n)
    serial_struct, serial_score = hc(base)
    par_struct, par_score = _parallel(base, 8)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_larger_d_parity():
    """A larger graph where the target axis genuinely splits across workers stays bit-identical."""
    x = _gaussian_sem(1, 800, 45, 2.0 / 45)
    serial_struct, serial_score = hc(x)
    par_struct, par_score = _parallel(x, 6)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_spawn_context_parity(monkeypatch):
    """Forcing the ``spawn`` start method (entry points + initargs must pickle) stays exact."""
    x = _gaussian_sem(1, 600, 16, 0.2)
    serial_struct, serial_score = hc(x)
    monkeypatch.setattr(
        ph, "_pick_context", lambda: __import__("multiprocessing").get_context("spawn")
    )
    par_struct, par_score = _parallel(x, 3)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_no_pool_below_work_gate(monkeypatch):
    """A workers>1 run below the work gate stays in-process: the pool seam never fires; exact."""
    monkeypatch.setenv("ANDREY_HC_PARALLEL_MIN_WORK", str(10**18))  # unreachable gate

    def _boom(*args, **kwargs):
        raise AssertionError("worker set created below the work gate")

    monkeypatch.setattr(ph, "_create_pool", _boom)
    x = _gaussian_sem(0, 600, 16, 0.2)
    serial_struct, serial_score = hc(x)
    with backend.config(num_workers=4):
        par_struct, par_score = hc(x)  # gate never crossed -> in-process path, no pool
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_pool_created_once_spans_passes(monkeypatch):
    """One persistent worker set serves every pass, so the Schur caches survive across passes."""
    calls: list[int] = []
    real = ph._create_pool

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ph, "_create_pool", counting)
    x = _gaussian_sem(0, 600, 24, 0.15)  # many passes: a per-pass set would show many calls
    with _force_parallel(), backend.config(num_workers=4):
        hc(x)
    assert sum(calls) == 1  # created lazily once, reused -> not re-spawned per pass


def test_run_token_guards_worker_reuse():
    """A worker rejects a task whose cov token does not match its seeded one."""
    from andrey.core.score import BICScore

    x = _gaussian_sem(0, 600, 8, 0.3)
    score = BICScore(x, lambda_value=1.0)
    d = x.shape[1]
    wset = ph._create_pool(score.cov, score.n, d, score.lambda_value, "token-A", workers=2)
    try:
        adj_int = [0] * d
        parents_of = {i: [] for i in range(d)}
        # Reach one worker directly with a mismatched token (each worker is its own 1-process pool).
        fut = wset._pools[0].submit(ph._scan_task, [0], adj_int, parents_of, "token-B")
        with pytest.raises(RuntimeError, match="reused across runs"):
            fut.result()
    finally:
        wset.close()
