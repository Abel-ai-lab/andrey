"""Parallel hill-climbing workers using :mod:`andrey.search._parallel`.

This module supplies picklable worker functions and a work estimate for parallel dispatch. Each
pass scans legal add/remove/reverse moves and applies the best improvement before the next pass.
Tabu search (:mod:`andrey.search.tabu`) runs its passes on the same workers and work gate.

A move ``u -> v`` belongs to the worker owning its head ``v`` (see :class:`hc.MoveScanner`). Fixed
target partitions cover every move and each worker keeps its scanner, with its deltas, across
passes. The total-order ``(delta, u, v, op_rank)`` reduction matches the serial scan regardless of
partitioning or worker assignment.
"""

from __future__ import annotations

import numpy as np

from andrey.core import env
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import _parallel as par
from andrey.search import calibrate
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search.hc import MoveScanner, ScannedMove, StructureKey

reduce_best = par.reduce_best  # re-exported so callers/tests reach it here
_pick_context = par.pick_context  # patchable seam (mirrors GES) for the spawn-context parity test

# Each pass scans d * (d - 1) moves, mostly with deltas kept from earlier passes. Below this
# threshold, a pass runs in-process because IPC costs exceed the savings from HC's cheap scan.
# ANDREY_HC_PARALLEL_MIN_WORK overrides it; 0 forces workers for serial/parallel equality tests.
# The default sits between the cutoffs for one fit per process and for many; `andrey calibrate`
# measures the latter.
_DEFAULT_MIN_WORK = 30_000

# _init_worker installs state once; the scanner and its Schur-delta caches persist across passes.
_WORKER_SCANNER: MoveScanner | None = None
_WORKER_TOKEN: object = None
_WORKER_BLAS: object = None  # a held threadpoolctl controller, caps this worker's BLAS to 1 thread


def _init_worker(
    cov: np.ndarray, n_samples: int, n_vars: int, lambda_value: float, run_token: object
) -> None:
    """Seed a worker with the shared covariance and pin its numeric environment.

    Runs once per worker process. Caps BLAS to one thread (multi-thread reduction order is not
    bit-stable), pins the numpy backend, and rebuilds the Schur-delta scorer over the parent's exact
    ``cov`` bytes -- so every worker feeds identical bytes into the delta and is bit-identical to
    serial. ``n_samples`` is the observation count the BIC penalty uses; ``n_vars`` the node count.
    """
    global _WORKER_SCANNER, _WORKER_TOKEN, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend as _backend

    _WORKER_BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime (never restored)
    _backend.config.backend = "numpy"
    delta = DeltaBICScore(BICScore.from_cov(cov, n_samples, lambda_value=lambda_value))
    _WORKER_SCANNER = MoveScanner(delta, n_vars)
    _WORKER_TOKEN = run_token


def _scan_task(
    targets: list[int],
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    run_token: object,
    floor: float = -_EPS_IMPROVE,
    tabu: frozenset[StructureKey] | None = None,
) -> ScannedMove | None:
    """Worker side of one chunk: the best move whose head is in ``targets``.

    ``floor`` and ``tabu`` pass through to :meth:`hc.MoveScanner.best`; tabu search sets them, and
    hill climbing keeps the defaults.
    """
    if run_token != _WORKER_TOKEN:
        raise RuntimeError("HC worker pool reused across runs (cov/token mismatch)")
    assert _WORKER_SCANNER is not None  # _init_worker runs before any task
    return _WORKER_SCANNER.best(adj_int, parents_of, targets=targets, floor=floor, tabu=tabu)


def _min_work() -> int:
    """The crossover gate in estimated moves scanned (env override, fixed default)."""
    return par.min_work(env.HC_PARALLEL_MIN_WORK, _DEFAULT_MIN_WORK)


def _estimate_work(d: int) -> int:
    """An O(1) estimate of the pass's scanned-move count -- the gate's predictor."""
    return d * (d - 1)


def _create_pool(
    cov: np.ndarray,
    n_samples: int,
    n_vars: int,
    lambda_value: float,
    run_token: object,
    workers: int,
) -> par.StableWorkerSet:
    """Build the persistent stable-assignment worker set -- the seam tests spy on (created once)."""
    return par.StableWorkerSet(
        n_vars,
        workers,
        _init_worker,
        (cov, n_samples, n_vars, lambda_value, run_token),
        context=_pick_context(),
    )


class _LazyWorkerSet:
    """The stable-assignment worker set, created on first use and reused across every pass.

    Never constructed if no pass crosses the work gate. Holds the ``token`` (the cov identity a task
    carries so a worker can reject a mismatched run).
    """

    def __init__(self, score: BICScore, n_vars: int, workers: int) -> None:
        self._score = score
        self._n_vars = n_vars
        self._workers = workers
        self._wset: par.StableWorkerSet | None = None
        self.token: object = None

    def get(self) -> par.StableWorkerSet:
        if self._wset is None:
            self.token = hash(self._score.cov.tobytes())
            self._wset = _create_pool(
                self._score.cov,
                self._score.n,
                self._n_vars,
                self._score.lambda_value,
                self.token,
                self._workers,
            )
        return self._wset

    def close(self) -> None:
        if self._wset is not None:
            self._wset.close()
            self._wset = None


def run_hc(
    adj_int: list[int],
    parents_of: dict[int, list[int]],
    delta: DeltaBICScore,
    d: int,
    max_iter: int,
    *,
    workers: int,
) -> int:
    """Run the greedy HC move loop with per-pass parallelism, mutating ``adj_int``/``parents_of``.

    Returns the number of moves applied, ``max_iter`` when the loop stopped at its limit.

    Below the gate a pass scans the whole move range in-process (the same scan as serial, proven
    bit-identical by the decomposition tests); above it the pass fans out to one lazily-created
    worker set that spans all remaining passes (so the Schur-delta caches persist). Either way the
    reduce keys on ``(delta, u, v, op_rank)``, so the applied move sequence is serial-identical.
    """
    from andrey.search.hc import _apply_move  # local import breaks the hc <-> _parallel_hc cycle

    workers = max(1, min(workers, d))
    w_min = _min_work()
    near_cutoff = _estimate_work(d) >= calibrate.WARN_FRACTION * w_min
    if workers > 1 and near_cutoff and calibrate.cutoff_is_default("hc"):
        calibrate.warn_default_cutoff("hc", w_min)
    fan_out = workers > 1 and _estimate_work(d) > w_min
    lazy = _LazyWorkerSet(delta.score_obj, d, workers)
    scanner = MoveScanner(delta, d)  # the in-process (below-gate) scan; workers hold their own
    moves = 0
    try:
        for _ in range(max_iter):
            if fan_out:
                wset = lazy.get()
                best = reduce_best(wset.dispatch(_scan_task, adj_int, parents_of, lazy.token))
            else:
                best = scanner.best(adj_int, parents_of)
            if best is None:
                break
            _apply_move(adj_int, parents_of, best[4])
            moves += 1
    finally:
        lazy.close()
    return moves
