"""Test GES parity across serial, chunked, and pooled execution.

The tests also cover worker startup, work-based dispatch, and failure propagation.
"""

from __future__ import annotations

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import _parallel as par
from andrey.search import _parallel_ges as pg
from andrey.search import operators as ops
from andrey.search.ges import ges
from tests.unit.ges_chunks import backward_inprocess, forward_inprocess, ges_inprocess

# Random linear-Gaussian datasets spanning sparse/dense and small/large graphs (the test_pc_ges
# sweep): the general-input regime a single fixed baseline cannot reach.
_SWEEP = [
    (seed, d, n, p)
    for seed in range(6)
    for (d, n, p) in [(5, 400, 0.4), (8, 600, 0.4), (10, 600, 0.4)]
]


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


@pytest.mark.parametrize("seed,d,n,p", _SWEEP)
@pytest.mark.parametrize("n_chunks", [2, 3, 8])
@pytest.mark.parametrize("ebic_gamma", [0.0, 1.0])
def test_decomposition_bit_identical_to_serial(seed, d, n, p, n_chunks, ebic_gamma):
    """The N-chunk in-process reduce reproduces serial GES's CPDAG and score exactly."""
    x = _gaussian_sem(seed, n, d, p)
    serial_struct, serial_score = ges(x, ebic_gamma=ebic_gamma)
    decomp_marks, decomp_score = ges_inprocess(x, n_chunks, ebic_gamma=ebic_gamma)
    assert np.array_equal(serial_struct.to_numpy(), decomp_marks), (
        f"CPDAG differs at seed={seed}, d={d}, n_chunks={n_chunks}"
    )
    assert serial_score == decomp_score  # bit-identical accumulated BIC


@pytest.mark.parametrize("seed,d,n,p", _SWEEP)
def test_chunk_count_invariance(seed, d, n, p):
    """Every chunk count (2, 3, one-per-target, more-than-targets) gives the identical result."""
    x = _gaussian_sem(seed, n, d, p)
    reference = ges_inprocess(x, 2)[0]
    for n_chunks in (3, d, d + 5):
        assert np.array_equal(ges_inprocess(x, n_chunks)[0], reference), (
            f"chunk count {n_chunks} changed the result at seed={seed}, d={d}"
        )


# --- determinism: the Schur delta search is deterministic despite non-bit-pure deltas -------------


@pytest.mark.parametrize("seed,d,n,p", [(0, 30, 500, 0.3), (2, 40, 400, 0.1), (5, 10, 600, 0.4)])
def test_serial_run_to_run_byte_identical(seed, d, n, p):
    """Serial GES is byte-identical run to run: same CPDAG bytes and same objective bits."""
    x = _gaussian_sem(seed, n, d, p)
    s1, sc1 = ges(x)
    s2, sc2 = ges(x)
    assert np.array_equal(s1.to_numpy(), s2.to_numpy())
    assert sc1 == sc2  # objective recomputed from an identical CPDAG -> identical bits


def _move_sequence(x: np.ndarray, n_chunks: int):
    """The ordered sequence of applied Insert/Delete moves under an N-chunk decomposition.

    Records ``("I", x, y, T)`` / ``("D", x, y, H)`` in application order by wrapping the operators,
    so a divergence in *which* move each pass selects (not just the final CPDAG) is observable.
    """
    X = np.asarray(x, dtype=np.float64)
    n = X.shape[1]
    engine = DeltaBICScore(BICScore(X, penalty_discount=1.0))
    seq: list[tuple] = []
    orig_ins, orig_del = pg.ops.apply_insert, pg.ops.apply_delete

    def rec_ins(adj, xx, yy, T):
        seq.append(("I", xx, yy, tuple(T)))
        orig_ins(adj, xx, yy, T)

    def rec_del(adj, xx, yy, H):
        seq.append(("D", xx, yy, tuple(H)))
        orig_del(adj, xx, yy, H)

    pg.ops.apply_insert, pg.ops.apply_delete = rec_ins, rec_del
    try:
        adj = np.zeros((n, n), dtype=np.int8)
        with threadpool_limits(limits=1):
            adj = forward_inprocess(adj, engine, n, n / 2, n_chunks, range(n))
            backward_inprocess(adj, engine, n, n_chunks, range(n))
    finally:
        pg.ops.apply_insert, pg.ops.apply_delete = orig_ins, orig_del
    return seq


@pytest.mark.parametrize("seed,d,n,p", [(1, 20, 400, 0.4)])
def test_chunk_count_invariance_on_move_sequence(seed, d, n, p):
    """Chunk count changes neither the final CPDAG nor the ordered sequence of selected moves."""
    x = _gaussian_sem(seed, n, d, p)
    reference = _move_sequence(x, 2)
    for n_chunks in (3, 7, d):
        assert _move_sequence(x, n_chunks) == reference, (
            f"chunk count {n_chunks} changed the move sequence at seed={seed}, d={d}"
        )


@pytest.mark.parametrize("seed,d,n,p", _SWEEP)
@pytest.mark.parametrize("ebic_gamma", [0.0, 1.0])
def test_objective_is_self_consistent(seed, d, n, p, ebic_gamma):
    """GES returns the score recomputed from its emitted CPDAG."""
    from andrey.core.orient import pdag2dag
    from andrey.search import operators as ops

    x = _gaussian_sem(seed, n, d, p)
    struct, total = ges(x, ebic_gamma=ebic_gamma)
    score = BICScore(np.asarray(x, dtype=np.float64), penalty_discount=1.0, ebic_gamma=ebic_gamma)
    dag = pdag2dag(struct.to_numpy())
    independent = float(sum(score.score(v, ops.parents(dag, v).tolist()) for v in range(d)))
    assert independent == total  # byte-identical: the objective is a pure function of the CPDAG


def test_ties_are_exercised_by_the_sweep():
    """The input sweep produces tied candidate deltas within ``1e-12``.

    These ties exercise the move-order rule in serial and chunked GES comparisons.
    """
    x = _gaussian_sem(0, 600, 10, 0.4)
    n = x.shape[1]
    score = BICScore(x, penalty_discount=1.0)
    from andrey.search.ges import _reachable

    adj = np.zeros((n, n), dtype=np.int8)
    state = pg.neighborhood_state(adj)
    reachable = _reachable(adj)
    deltas: list[float] = []
    for y in range(n):
        for xx in range(n):
            if xx == y:
                continue
            nbrs_y = state[2][y]
            na = nbrs_y & state[0][xx]
            for T in pg.ops.subsets(sorted(nbrs_y - state[0][xx])):
                if pg.ops.is_clique(adj, na | set(T)) and not reachable(y, xx, na | set(T)):
                    base = na | set(T) | set(state[1][y])
                    with_x, without_x = tuple(sorted(base | {xx})), tuple(sorted(base))
                    deltas.append(round(score.score(y, with_x) - score.score(y, without_x), 12))
    assert len(deltas) - len(set(deltas)) > 0  # at least one delta tie (to 1e-12) exists to break


# --- pool-backed parity (the IPC layer preserves the in-process exactness) ------------------------

import os  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from andrey.core import backend  # noqa: E402


@contextmanager
def _force_pool():
    """Force the crossover gate open (``ANDREY_GES_PARALLEL_MIN_WORK=0``) so every pass fans out."""
    prev = os.environ.get("ANDREY_GES_PARALLEL_MIN_WORK")
    os.environ["ANDREY_GES_PARALLEL_MIN_WORK"] = "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("ANDREY_GES_PARALLEL_MIN_WORK", None)
        else:
            os.environ["ANDREY_GES_PARALLEL_MIN_WORK"] = prev


def _parallel(x: np.ndarray, workers: int, max_parents: float | None = None):
    """Run ``ges`` on the pool-backed path (gate forced open) via the ``num_workers`` axis."""
    with _force_pool(), backend.config(num_workers=workers):
        return ges(x, max_parents=max_parents)


# Exhaustive exactness across the sweep is the in-process tests' job (cheap); the pool tests only
# prove the IPC layer (adj pickling, scoring from the shared cov, as_completed reduce) preserves it,
# so a handful of pool-creating cases suffice -- each forkserver worker re-imports Andrey.
@pytest.mark.parametrize("seed,workers", [(0, 2), (1, 4), (2, 3)])
def test_pool_parallel_bit_identical_to_serial(seed, workers):
    """The pool-backed path (workers > 1) reproduces serial GES's CPDAG and score exactly."""
    x = _gaussian_sem(seed, 600, 10, 0.4)
    serial_struct, serial_score = ges(x)
    par_struct, par_score = _parallel(x, workers)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy()), (
        f"CPDAG differs at seed={seed}, workers={workers}"
    )
    assert serial_score == par_score


def test_pool_parallel_matches_serial_under_a_cap():
    """Workers test the cap on their own candidates and return serial's CPDAG where it binds."""
    x = _gaussian_sem(0, 400, 30, 0.2)
    serial_struct, serial_score = ges(x, max_parents=2)
    par_struct, par_score = _parallel(x, 4, 2)
    assert ops.max_parents_in_class(ges(x)[0].to_numpy()).max() > 2, "the cap does not bind"
    assert not ops.exceeds_parent_cap(par_struct.to_numpy(), 2)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_pool_parity_at_larger_d():
    """A larger graph where the target axis genuinely splits across workers stays bit-identical."""
    x = _gaussian_sem(0, 400, 40, 0.1)
    serial_struct, serial_score = ges(x)
    par_struct, par_score = _parallel(x, 4)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_run_token_guards_pool_reuse():
    """A worker rejects a task with a covariance token different from its initialized token."""
    x = _gaussian_sem(0, 600, 8, 0.4)
    score = BICScore(x, penalty_discount=1.0)
    n = x.shape[1]
    wset = pg._create_pool(
        score.cov,
        score.n,
        n,
        score.penalty_discount,
        score.ebic_gamma,
        "token-A",
        workers=2,
        ranks=list(range(n)),
    )
    try:
        adj = np.zeros((n, n), dtype=np.int8)
        # Reach one worker directly with a mismatched token (each worker is its own 1-process pool).
        fut = wset._pools[0].submit(pg._forward_task, [0], adj, n / 2, "token-B")  # wrong token
        with pytest.raises(RuntimeError, match="reused across runs"):
            fut.result()
    finally:
        wset.close()


# --- safety (start-method policy, BLAS pinning, fail-loud) ----------------------------------------

import sys  # noqa: E402
import types  # noqa: E402


def test_fork_context_policy(monkeypatch):
    """``fork`` only when no CUDA context exists; a live CUDA context rules ``fork`` out."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert par.fork_safe() is True  # torch not imported
    uninit = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", uninit)
    assert par.fork_safe() is True  # torch present, CUDA cold
    initialized = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", initialized)
    assert par.fork_safe() is False  # a live CUDA context -> never fork


def test_spawn_context_parity(monkeypatch):
    """Forcing the ``spawn`` start method (entry points + initargs must pickle) stays exact."""
    x = _gaussian_sem(1, 600, 10, 0.4)
    serial_struct, serial_score = ges(x)
    monkeypatch.setattr(
        pg, "_pick_context", lambda: __import__("multiprocessing").get_context("spawn")
    )
    par_struct, par_score = _parallel(x, 2)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_local_score_bic_thread_pinned_is_stable():
    """Under a 1-thread BLAS pin the score is identical whatever the ambient thread cap was."""
    from threadpoolctl import threadpool_limits

    from andrey.core.score import local_score_bic

    rng = np.random.default_rng(0)
    cov = np.cov(rng.standard_normal((500, 300)), rowvar=False)
    parents = list(range(1, 130))
    with threadpool_limits(limits=1):
        pinned = local_score_bic(cov, 500, 0, parents, penalty_discount=1.0)
    for ambient in (1, 2, 4, 8):
        with threadpool_limits(limits=ambient), threadpool_limits(limits=1):
            assert local_score_bic(cov, 500, 0, parents, penalty_discount=1.0) == pinned


def test_ges_result_independent_of_ambient_blas_threads():
    """``ges`` pins BLAS internally, so its result does not depend on the caller's thread cap."""
    from threadpoolctl import threadpool_limits

    x = _gaussian_sem(0, 600, 10, 0.4)
    with threadpool_limits(limits=1):
        r1 = ges(x)
    with threadpool_limits(limits=4):
        r4 = ges(x)
    assert np.array_equal(r1[0].to_numpy(), r4[0].to_numpy())
    assert r1[1] == r4[1]


def test_parity_on_degenerate_covariance():
    """Serial and pooled GES agree on near-duplicate columns the column check accepts.

    GES never scores a parent set that holds both columns of a near-duplicate pair here, so every
    delta stays on the fast path; the test covers the near-collinear scores, not the fallback to
    full scores.
    """
    rng = np.random.default_rng(3)
    base = rng.standard_normal((400, 6))
    # Near-duplicate columns, kept just far enough apart for the column check to accept them.
    x = np.hstack([base, base[:, :2] + 1e-5 * rng.standard_normal((400, 2))])
    serial_struct, serial_score = ges(x)
    par_struct, par_score = _parallel(x, 3)
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_collect_reraises_worker_exception():
    """Collecting a failed worker task raises its original exception."""
    from concurrent.futures import Future

    fut: Future = Future()
    fut.set_exception(ValueError("worker boom"))
    with pytest.raises(ValueError, match="worker boom"):
        par.collect([fut])


def test_collect_wraps_broken_pool():
    """A killed worker surfaces loudly, naming the serial workaround."""
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    fut: Future = Future()
    fut.set_exception(BrokenProcessPool("worker died"))
    with pytest.raises(RuntimeError, match="ANDREY_NUM_WORKERS=1"):
        par.collect([fut])


# --- the measured density-aware crossover gate ----------------------------------------------------


def test_no_pool_below_work_gate(monkeypatch):
    """GES stays in-process below the work threshold and matches serial output."""
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", str(10**18))  # unreachable gate

    def _boom(*args, **kwargs):
        raise AssertionError("pool created below the work gate")

    monkeypatch.setattr(pg, "_create_pool", _boom)
    x = _gaussian_sem(0, 600, 10, 0.4)
    serial_struct, serial_score = ges(x)
    with backend.config(num_workers=4):
        par_struct, par_score = ges(x)  # gate never crossed -> in-process path, no pool
    assert np.array_equal(serial_struct.to_numpy(), par_struct.to_numpy())
    assert serial_score == par_score


def test_parity_mixed_gate(monkeypatch):
    """A GES run that crosses the work threshold matches serial output."""
    # Below the work of the first and the busiest forward passes, above that of most others.
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "200")
    created: list[int] = []
    real = pg._create_pool
    monkeypatch.setattr(pg, "_create_pool", lambda *a, **k: (created.append(1), real(*a, **k))[1])
    x = _gaussian_sem(0, 600, 30, 0.3)
    serial_struct, serial_score = ges(x)  # workers=1: the serial oracle
    with backend.config(num_workers=4):
        mixed_struct, mixed_score = ges(x)
    assert created, "no pass crossed the gate -- not a mixed run"
    assert np.array_equal(serial_struct.to_numpy(), mixed_struct.to_numpy())
    assert serial_score == mixed_score


def test_forward_work_estimate_is_density_aware():
    """The forward work estimate is higher for a dense graph of the same size."""
    n = 20
    sparse = np.zeros((n, n), dtype=np.int8)
    dense = np.zeros((n, n), dtype=np.int8)
    from andrey.core.structure import TAIL

    for i in range(8):  # an undirected clique among the first 8 nodes
        for j in range(i + 1, 8):
            dense[i, j] = dense[j, i] = TAIL
    w_sparse = sum(pg.forward_work_per_target(n, pg.neighborhood_state(sparse), None))
    w_dense = sum(pg.forward_work_per_target(n, pg.neighborhood_state(dense), None))
    assert w_dense > 10 * w_sparse  # 2^7 per clique candidate dwarfs the sparse all-singleton scan


def test_work_estimate_never_overflows_at_high_degree():
    """The degree cap keeps the estimate an ``int`` even on a large near-complete graph."""
    n = 60
    adj = np.ones((n, n), dtype=np.int8)  # every off-diagonal TAIL -> undirected near-clique
    np.fill_diagonal(adj, 0)
    state = pg.neighborhood_state(adj)
    assert isinstance(sum(pg.forward_work_per_target(n, state, None)), int)
    assert isinstance(sum(pg.backward_work_per_target(n, state)), int)


# --- load-balanced degree-aware chunking + cross-pass cache ---------------------------------------


def test_balanced_chunks_exact_cover():
    """Balanced chunks contain every target exactly once and have no empty chunk."""
    from itertools import chain

    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(1, 60))
        work = rng.integers(0, 1000, size=n).tolist()
        for k in (1, 2, 3, 7, n, n + 3):
            chunks = par.balanced_chunks(work, k)
            assert sorted(chain.from_iterable(chunks)) == list(range(n))
            assert all(chunks)  # no empty chunk


def test_balanced_chunks_reduce_skew():
    """On a work-skewed profile the balanced chunker lowers the heaviest chunk vs even ranges."""
    work = [100] * 10 + [1] * 30  # ten heavy targets clustered at one end
    n = len(work)

    def heaviest(chunks):
        return max(sum(work[y] for y in c) for c in chunks)

    assert heaviest(par.balanced_chunks(work, 4)) < heaviest(par.even_chunks(n, 4))


def test_forward_scan_rescores_group_with_mixed_bounds():
    """A group whose passing sources mix a zero bound with a positive one is scored again.

    Target 0 has an empty base with sources 1, 2, and 3. A semi-directed path reaches 3, so 1 and 2
    pass. Source 1's bound is 0 (its delta does not depend on the batch) and source 2's is positive,
    so the group's screening deltas are not exact and the passing sources are scored on their own.
    """
    x = _gaussian_sem(0, 200, 4, 0.5)
    engine = DeltaBICScore(BICScore(x, penalty_discount=1.0))
    scan = pg._ForwardScan([0], engine, 4, None, range(4))
    screening = {1: 1.0, 2: 2.0, 3: 3.0}  # not deltas, so a returned screening value shows
    scan.groups[0] = {(): (0, 0b1110, frozenset({1, 2, 3}), screening, {1: 0.0, 2: 0.5, 3: 0.5})}
    scan.exact[0] = {}
    scan.reach[(0, 0)] = 0b1001  # y = 0 reaches source 3
    deltas, passing = scan._exact(0, ())
    assert passing == 0b0110
    expected = engine.delta_many_with_base(0, (), [("add", 1), ("add", 2)])
    assert [deltas[1], deltas[2]] == expected


def test_forward_scan_refreshes_a_reach_map_before_the_first_graph():
    """The successor masks start as the empty graph's, one per node, so a reach set kept before
    the first pass meets masks of the same length."""
    x = _gaussian_sem(0, 200, 4, 0.5)
    scan = pg._ForwardScan([0], DeltaBICScore(BICScore(x, penalty_discount=1.0)), 4, None, range(4))
    assert scan.succ == [0] * 4
    scan.reach[(0, 0)] = 0b1001
    scan._refresh(np.zeros((4, 4), dtype=np.int8))
    assert scan.reach == {(0, 0): 0b1001}


def test_pool_created_once_spans_passes(monkeypatch):
    """A GES run with multiple passes creates only one worker pool."""
    calls: list[int] = []
    real = pg._create_pool

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pg, "_create_pool", counting)
    x = _gaussian_sem(0, 400, 40, 0.1)  # many passes: a per-pass pool would show many calls
    with _force_pool(), backend.config(num_workers=4):
        ges(x)
    assert sum(calls) == 1  # created lazily once, reused -> not re-spawned per pass


# --- transitive parity (GIES / GFCI delegate to ges()) --------------------------------------------


@pytest.mark.parametrize("facade_name", ["gies", "gfci"])
def test_gies_gfci_transitive_parity(facade_name):
    """Worker count does not change the GIES or GFCI output structure."""
    import andrey

    facade = getattr(andrey, facade_name)
    x = _gaussian_sem(0, 600, 10, 0.4)
    serial = facade(x)
    with _force_pool(), backend.config(num_workers=4):
        parallel = facade(x)
    assert np.array_equal(serial.structure.to_numpy(), parallel.structure.to_numpy()), facade_name
    assert serial.structure.kind == parallel.structure.kind


# --- worker boot (preloaded forkserver template, eager concurrent spawn) --------------------------

import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402


def test_forkserver_template_preloads_worker_modules():
    """The forkserver preloads importable worker modules and preserves ``__main__``.

    Invalid preload names are silently skipped, so each name is checked for importability.
    """
    if "forkserver" not in mp.get_all_start_methods():
        pytest.skip("forkserver unavailable on this platform")
    ctx = par.pick_context()
    assert ctx is mp.get_context("forkserver")
    for name in par._FORKSERVER_PRELOAD:
        assert importlib.util.find_spec(name) is not None, f"preload names a missing module: {name}"
    from multiprocessing import forkserver

    registered = set(forkserver._forkserver._preload_modules)
    assert set(par._FORKSERVER_PRELOAD) <= registered
    assert "__main__" in registered, "registering a preload dropped the stdlib default"


# d=100: the covariance (80 KB) exceeds the 64 KiB pipe buffer, so each boot write blocks until its
# worker reads it.
@pytest.mark.parametrize("d", [6, 100])
def test_workers_are_live_at_construction(d):
    """Each pool has a live worker before dispatch, including with a large boot payload."""
    x = _gaussian_sem(0, 200, d, 0.3 if d < 10 else 0.02)
    score = BICScore(x, penalty_discount=1.0)
    wset = pg._create_pool(
        score.cov,
        score.n,
        d,
        score.penalty_discount,
        score.ebic_gamma,
        "tok",
        workers=3,
        ranks=list(range(d)),
    )
    try:
        assert len(wset._pools) == 3
        for pool in wset._pools:
            procs = list(pool._processes.values())
            assert len(procs) == 1 and procs[0].is_alive()
    finally:
        wset.close()


def test_broken_worker_surfaces_at_construction():
    """A worker that cannot boot fails the constructor, not the first mid-pass dispatch."""
    with pytest.raises(RuntimeError, match="ANDREY_NUM_WORKERS=1"):
        par.StableWorkerSet(4, 2, pg._init_worker, ("not a cov",))  # wrong arity: the boot dies


# --- a script without the ``__main__`` guard ------------------------------------------------------

import pathlib  # noqa: E402
import subprocess  # noqa: E402

_SCRIPT = """\
import numpy as np
import andrey

X = np.random.default_rng(0).standard_normal((200, {d}))
{call}
"""


# d=120: the covariance exceeds the 64 KiB pipe buffer, so the payload write finds no reader
# (``BrokenPipeError``). d=8 fits, so the parent waits for a pid or result that never arrives
# (``EOFError`` under forkserver, ``BrokenProcessPool`` under spawn). GIES and GFCI call ``ges()``,
# so GES stands for them.
@pytest.mark.parametrize("d,guarded", [(8, False), (120, False), (8, True)])
def test_script_without_main_guard(tmp_path, d, guarded):
    """An unguarded script gets the guard-first message; the same script guarded runs."""
    call = "andrey.ges(X)"
    if guarded:
        call = f'if __name__ == "__main__":\n    {call}'
    script = tmp_path / "fit.py"  # a real file: forkserver and spawn re-import ``__main__`` by path
    script.write_text(_SCRIPT.format(d=d, call=call))
    src_dir = str(pathlib.Path(pg.__file__).resolve().parents[2])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")])),
        "ANDREY_NUM_WORKERS": "2",
        "ANDREY_GES_PARALLEL_MIN_WORK": "0",
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )
    if guarded:
        assert proc.returncode == 0, proc.stderr
        return
    assert proc.returncode != 0
    last = proc.stderr.strip().splitlines()[-1]
    assert last.startswith("RuntimeError: parallel-search workers failed to start"), last
    assert last.index('if __name__ == "__main__":') < last.index("out of memory"), last
    assert "ANDREY_NUM_WORKERS=1" in last, last


# --- the forkserver's socket on a file system that rejects chmod ----------------------------------
# The forkserver chmods its Unix socket in ``TMPDIR``; a parallel file system can reject that call
# (``EOPNOTSUPP``). The tests stand in for it with a rejecting ``chmod``, or with a stub for the
# probe and the start.

import errno  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402

from andrey.core.warning_policy import PerformanceWarning  # noqa: E402

needs_forkserver = pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(), reason="forkserver unavailable on this platform"
)


def _reject_chmod(path, *args, **kwargs):
    raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP), str(path))


@pytest.fixture
def forkserver_decision():
    """Start and end a test with no recorded decision on where the forkserver's socket goes."""
    par._forkserver_starts.cache_clear()
    yield
    par._forkserver_starts.cache_clear()


@needs_forkserver
def test_socket_probe_reports_whether_the_directory_holds_it(monkeypatch):
    """The probe accepts a usable directory and rejects a missing one, a socket path past the
    platform's limit, or a rejected chmod."""
    # pytest's tmp_path can itself overrun the limit (104 bytes on macOS), so the base is short.
    with tempfile.TemporaryDirectory(dir="/tmp") as base:
        assert par._holds_forkserver_socket(base)
        assert not par._holds_forkserver_socket(os.path.join(base, "missing"))
        deep = os.path.join(base, "d" * 120)
        os.mkdir(deep)
        assert not par._holds_forkserver_socket(deep)
        os.rmdir(deep)
        with monkeypatch.context() as patch:
            patch.setattr(os, "chmod", _reject_chmod)
            assert not par._holds_forkserver_socket(base)
        assert os.listdir(base) == [], "the probe left files behind"


@needs_forkserver
def test_forkserver_socket_moves_off_a_tmpdir_that_rejects_it(
    forkserver_decision, monkeypatch, tmp_path
):
    """The server starts with its socket in a node-local directory and the temporary one returns."""
    from multiprocessing import forkserver

    bad, local = str(tmp_path), par._NODE_LOCAL_DIRS[0]
    monkeypatch.setattr(tempfile, "tempdir", bad)
    monkeypatch.setattr(par, "_holds_forkserver_socket", lambda base: base != bad)
    started = []
    monkeypatch.setattr(forkserver, "ensure_running", lambda: started.append(tempfile.gettempdir()))
    with pytest.warns(PerformanceWarning, match=re.escape(repr(local))) as caught:
        ctx = par.pick_context()
    assert repr(bad) in str(caught[0].message)
    assert ctx.get_start_method() == "forkserver"
    assert started == [local]
    assert tempfile.gettempdir() == bad, "the directory change outlived the start"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert par.pick_context() is ctx  # decided once per process: no second start or warning
    assert started == [local]


@needs_forkserver
@pytest.mark.parametrize("start_fails", [False, True], ids=["no directory holds it", "start fails"])
@pytest.mark.parametrize("safe,method", [(True, "fork"), (False, "spawn")])
def test_forkserver_is_replaced_when_its_socket_cannot_be_placed(
    forkserver_decision, monkeypatch, tmp_path, start_fails, safe, method
):
    """Without a socket directory the pool starts with ``fork`` or ``spawn``, and says so."""
    from multiprocessing import forkserver

    bad = str(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", bad)
    monkeypatch.setattr(par, "_holds_forkserver_socket", lambda base: start_fails and base != bad)

    def start():
        if not start_fails:
            raise AssertionError("no directory holds the socket, so the server must not start")
        raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP))

    monkeypatch.setattr(forkserver, "ensure_running", start)
    monkeypatch.setattr(par, "fork_safe", lambda: safe)
    with pytest.warns(PerformanceWarning, match=re.escape(repr(method))) as caught:
        ctx = par.pick_context()
    assert repr(bad) in str(caught[0].message)
    assert ctx.get_start_method() == method
    assert tempfile.gettempdir() == bad


_REJECTED_TMPDIR_SCRIPT = """\
import errno
import os
import warnings

import numpy as np

import andrey


def rejecting(real_chmod, tmpdir):
    def chmod(path, *args, **kwargs):
        if str(path).startswith(tmpdir):
            raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP), str(path))
        return real_chmod(path, *args, **kwargs)

    return chmod


if __name__ == "__main__":
    os.chmod = rejecting(os.chmod, os.environ["TMPDIR"])
    X = np.random.default_rng(0).standard_normal((300, 10))
    X[:, 1:] += 0.6 * X[:, :-1]
    serial = andrey.ges(X)
    with andrey.config(num_workers=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pooled = andrey.ges(X)
    print(np.array_equal(serial.structure.to_numpy(), pooled.structure.to_numpy()))
    print(*(str(w.message) for w in caught), sep="\\n")
"""


@needs_forkserver
def test_pool_starts_with_tmpdir_on_a_file_system_that_rejects_chmod(tmp_path):
    """A fit on two workers completes with the serial answer when ``TMPDIR`` rejects the socket."""
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    script = tmp_path / "fit.py"
    script.write_text(_REJECTED_TMPDIR_SCRIPT)
    src_dir = str(pathlib.Path(pg.__file__).resolve().parents[2])
    env = {
        **os.environ,
        "TMPDIR": str(rejected),
        "PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")])),
        "ANDREY_GES_PARALLEL_MIN_WORK": "0",
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    identical, *messages = proc.stdout.splitlines()
    assert identical == "True"
    assert any(str(rejected) in m and "forkserver" in m for m in messages), proc.stdout


@pytest.mark.parametrize("seed", range(30))
def test_kept_reach_sets_match_a_fresh_traversal(seed):
    """Every reach set kept across a change of successors equals the traversal on the new ones.

    The graphs reach past 64 nodes. An entry none of whose nodes changed successors is always kept.
    """
    from andrey.core import _bitset

    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 150))
    tails = rng.random((d, d)) < 2.0 / d
    np.fill_diagonal(tails, False)
    before = _bitset.row_masks(tails)
    reach = {}
    for _ in range(20):
        start = int(rng.integers(0, d))
        clique = _bitset.bitmask(v for v in range(d) if v != start and rng.random() < 0.1)
        reach[(start, clique)] = _bitset.semidirected_reach_set(before, start, clique)
    flips = rng.integers(0, d, size=(int(rng.integers(1, 6)), 2))
    for u, v in flips.tolist():
        if u != v:
            tails[u, v] = not tails[u, v]
    after = _bitset.row_masks(tails)
    changed = _bitset.bitmask(u for u in range(d) if before[u] != after[u])
    kept = pg._kept_reach(dict(reach), before, after)
    for (start, clique), seen in kept.items():
        assert seen == _bitset.semidirected_reach_set(after, start, clique), (seed, start)
    for key, seen in reach.items():
        if not seen & changed:
            assert key in kept, (seed, key)
