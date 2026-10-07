"""The memory cap is enforced, and the parent — not the exit code — says so.

Validation-table row **memory cap**: ``status="oom"`` from a cap *above* the import+load footprint
and *below* a known in-fit allocation. These tests are the mechanism's specification as much as its
regression net.

Everything here is deliberately paired. A test that only shows a capped run dying would pass just as
well if the cap killed everything, so each kill is matched with a control that must survive:

* an over-cap allocation dies **and** the same allocation under a generous cap returns ``ok``;
* a pool whose *tree* crosses the cap dies **and** the same per-process allocation in the leader
  alone, under the same cap, returns ``ok`` — which is what pins summing over the tree;
* a 4 GB address-space *reservation* that never touches a page survives a cap far below it: the
  JVM shape that ``RLIMIT_AS`` would kill, and the reason the cap does not use it.

The footprint the caps are set against is **measured**, not hardcoded: a literal would drift with
numpy and turn a real failure into a resize of the constant.

Stub adapters only — no causal-discovery algorithm runs. As in :mod:`test_runner`, the child
re-imports these classes from this module to unpickle them, so module-level imports stay stdlib +
numpy and ``andrey_bench`` is imported inside the tests.
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np
import pytest

_ADJ = np.zeros((4, 4), dtype=np.int8)

#: How long an allocating stub holds its memory before returning. At the poller's 0.05 s period this
#: is ~20 samples, so a breach cannot be missed because the parent happened to look away.
_HOLD_S = 1.0

#: One over-cap allocation, and the per-worker allocation of the pooled variant. Small enough to be
#: cheap to fault in, large enough to clear any sampling noise.
_ALLOC_MB = 600
_WORKER_MB = 350


class _BaseStub:
    """Minimal SolutionAdapter-shaped stub; only ``fit`` and the runner-visible metadata matter."""

    name = "stub.ges"
    algorithm = "ges"
    package = "stub"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    output_type = "dag"

    def to_structure(self, native):  # pragma: no cover - runner never calls this
        raise NotImplementedError


def _touch(mb: int) -> np.ndarray:
    """Allocate ``mb`` MiB and fault every page in, so it lands in RSS rather than in address space."""
    arr = np.ones(mb * 1024 * 1024 // 8, dtype=np.float64)
    arr += 1.0
    return arr


class StubFootprintAdapter(_BaseStub):
    """Allocates nothing. Its peak is the import + load footprint every cap must sit above."""

    def fit(self, data, params):
        time.sleep(0.4)  # long enough for the poller to take a reading
        return _ADJ.copy()


class StubAllocAdapter(_BaseStub):
    """Touches ~600 MB inside the fit and holds it — a known in-fit allocation."""

    def fit(self, data, params):
        arr = _touch(_ALLOC_MB)
        time.sleep(_HOLD_S)
        return _ADJ.copy() if arr is not None else _ADJ.copy()


class StubWorkerSizedAllocAdapter(_BaseStub):
    """Touches one *worker's* share (~350 MB) in the leader — the control for the pool test."""

    def fit(self, data, params):
        arr = _touch(_WORKER_MB)
        time.sleep(_HOLD_S)
        return _ADJ.copy() if arr is not None else _ADJ.copy()


class StubReserveAdapter(_BaseStub):
    """Reserves 4 GB of address space and never touches it — the shape ``RLIMIT_AS`` kills wrongly.

    ``np.empty`` gets anonymous pages that are not faulted until written, so RSS stays at the
    footprint. A JVM does the same thing on a far larger scale, which is why the cap has to be a
    resident-memory cap.
    """

    def fit(self, data, params):
        reserved = np.empty(4_000_000_000 // 8, dtype=np.float64)
        time.sleep(0.8)
        return _ADJ.copy() if reserved is not None else _ADJ.copy()


class StubSelfKillAdapter(_BaseStub):
    """SIGKILLs itself — a bare ``rc = -9`` that no cap was observed to cause."""

    def fit(self, data, params):
        os.kill(os.getpid(), signal.SIGKILL)
        return _ADJ.copy()  # pragma: no cover - unreachable


def _hold_in_worker(_) -> float:
    """Pool-worker body (module-level so forkserver can import it): touch ~350 MB and hold it."""
    import time as _t

    import numpy as _np

    arr = _np.ones(_WORKER_MB * 1024 * 1024 // 8, dtype=_np.float64)
    arr += 1.0
    _t.sleep(_HOLD_S)
    return float(arr[0])


class StubPoolAllocAdapter(_BaseStub):
    """A 2-worker forkserver pool where the *workers* hold the memory and the leader stays small.

    The same shape as Andrey's parallel ``StableWorkerSet``. The leader never approaches the cap on
    its own, so only a total summed over the tree can fire.
    """

    mode = "parallel"

    def fit(self, data, params):
        import concurrent.futures as _cf
        import multiprocessing as _mp

        ctx = _mp.get_context("forkserver")
        ctx.set_forkserver_preload(["numpy"])
        with _cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            list(ex.map(_hold_in_worker, [0, 1]))
        return _ADJ.copy()


def _tiny_data() -> np.ndarray:
    return np.zeros((16, 4), dtype=np.float64)


def _run(adapter, **kwargs):
    """One repeat of ``adapter`` with a wall cap far above every hold, so a timeout cannot pose as
    an OOM: every ``status`` these tests read is about memory."""
    from andrey_bench.runner import run_task

    return run_task(adapter, _tiny_data(), {}, cap_wall_s=60.0, repeats=1, warmup=0, **kwargs)[0]


@pytest.fixture(scope="module")
def footprint_mb() -> float:
    """The child's measured import + load footprint (MB) — what every cap below is set relative to."""
    measured = _run(StubFootprintAdapter(), cap_mem_mb=16384.0)
    assert measured["status"] == "ok", measured
    assert measured["tree_peak_mem_mb"] > 0.0, measured  # the poller really sampled a live tree
    return float(measured["tree_peak_mem_mb"])


# --------------------------------------------------------------------------------------------------
# The cap fires — and does not fire on a run that stayed inside it.
# --------------------------------------------------------------------------------------------------


def test_a_cap_below_an_in_fit_allocation_reports_oom(footprint_mb):
    """Above the footprint, below the allocation -> the parent kills the tree and calls it ``oom``."""
    cap = footprint_mb + 300.0
    m = _run(StubAllocAdapter(), cap_mem_mb=cap)

    assert m["status"] == "oom", m
    assert m["oom_source"] == "poller", m  # the parent witnessed it; nothing was read off rc
    # The tree was actually *killed*, not merely flagged. Without this the whole module passes with
    # the kill removed: the breach flag alone discards the result and the record still reads ``oom``,
    # so a benchmark would be labeled capped while running unbounded. A poller-killed child is -9;
    # one that ran to completion under a flag-only breach exits 0.
    assert m["exit_code"] == -signal.SIGKILL, m
    assert m["mem_route"] == "poll", m
    assert m["cap_mem_mb"] == cap, m  # the cap is on the record ...
    assert m["tree_peak_mem_mb"] > cap, m  # ... beside the peak that crossed it
    assert "adj" not in m  # no output survives a censored run


def test_the_same_allocation_under_a_generous_cap_is_not_an_oom(footprint_mb):
    """The control that makes the test above mean something: the cap killed *this* run, not any run."""
    m = _run(StubAllocAdapter(), cap_mem_mb=footprint_mb + 3000.0)

    assert m["status"] == "ok", m
    assert m.get("oom_source") is None, m
    # The poller watched the same allocation happen and let it through.
    assert m["tree_peak_mem_mb"] > footprint_mb + 0.8 * _ALLOC_MB, m


def test_an_uncapped_run_declares_no_route_and_polls_nothing(footprint_mb):
    """No cap is a recorded fact, not a missing field — and nothing is watching."""
    m = _run(StubAllocAdapter(), cap_mem_mb=None)

    assert m["status"] == "ok", m
    assert m["mem_route"] == "none", m
    assert m["cap_mem_mb"] is None, m
    assert m["tree_peak_mem_mb"] is None, m


# --------------------------------------------------------------------------------------------------
# The total is summed over the tree: a cap breached only by pooled workers still fires.
# --------------------------------------------------------------------------------------------------


def test_a_cap_breached_only_by_pooled_workers_still_fires(footprint_mb):
    """Two workers holding ~350 MB each cross a cap the leader alone never approaches.

    The paired control is the load-bearing half: the *same* per-process allocation, made in the
    leader, under the *same* cap, must come back ``ok``. A poller that watched only the child it
    launched would pass both — it is the difference between them that says the tree was summed.
    """
    cap = footprint_mb + 500.0

    leader_only = _run(StubWorkerSizedAllocAdapter(), cap_mem_mb=cap)
    assert leader_only["status"] == "ok", leader_only  # one worker's share fits under the cap

    pooled = _run(StubPoolAllocAdapter(), cap_mem_mb=cap, num_workers=2)
    assert pooled["status"] == "oom", pooled
    assert pooled["oom_source"] == "poller", pooled
    assert pooled["exit_code"] == -signal.SIGKILL, pooled  # the tree was killed, not just flagged
    assert pooled["tree_peak_mem_mb"] > cap, pooled
    # And the breach really came from the children, not from a leader that quietly grew.
    assert pooled["tree_peak_mem_mb"] > leader_only["tree_peak_mem_mb"], (pooled, leader_only)


# --------------------------------------------------------------------------------------------------
# What the cap must NOT do: kill on reserved-but-untouched address space.
# --------------------------------------------------------------------------------------------------


def test_a_reservation_that_touches_nothing_survives_a_cap_far_below_it(footprint_mb):
    """4 GB reserved, 0 faulted, cap at footprint+300 MB -> ``ok``.

    This is the whole argument against ``RLIMIT_AS``: it would have raised ``MemoryError`` at the
    ``np.empty`` and censored a run that never used the memory. A JVM reserves this way by design,
    so the Tetrad contestant depends on this staying true.
    """
    m = _run(StubReserveAdapter(), cap_mem_mb=footprint_mb + 300.0)

    assert m["status"] == "ok", m
    assert m["tree_peak_mem_mb"] < footprint_mb + 300.0, m  # resident memory never moved


# --------------------------------------------------------------------------------------------------
# Attribution: who says it was an OOM.
# --------------------------------------------------------------------------------------------------


def test_a_bare_sigkill_is_an_error_when_nothing_was_capped():
    """``rc = -9`` on its own means nothing. Uncapped, it is an ``error``.

    Mapping every ``-9`` to ``oom`` would turn any external kill — a node drain, an admin, a crashed
    pool — into a recorded memory limit.
    """
    m = _run(StubSelfKillAdapter(), cap_mem_mb=None)

    assert m["status"] == "error", m
    assert m.get("oom_source") is None, m
    assert m["exit_code"] == -signal.SIGKILL, m


def test_a_bare_sigkill_under_a_cap_is_a_labelled_fallback(footprint_mb):
    """Under a declared cap the same ``-9`` is an ``oom`` — the kernel OOM-killer can land between
    two samples — but it is labeled ``external`` so it is never confused with one the parent saw."""
    m = _run(StubSelfKillAdapter(), cap_mem_mb=footprint_mb + 3000.0)

    assert m["status"] == "oom", m
    assert m["oom_source"] == "external", m  # inferred, not witnessed — and it says so


# --------------------------------------------------------------------------------------------------
# Route selection and the SLURM half (unit level: no allocation is held here).
# --------------------------------------------------------------------------------------------------


def test_the_route_is_selectable_and_an_unknown_one_is_refused():
    from andrey_bench import memcap

    assert memcap.resolve_route(None) == memcap.NONE  # no cap, nothing to enforce
    assert memcap.resolve_route(None, memcap.SLURM) == memcap.NONE
    assert memcap.resolve_route(4096.0) == memcap.POLL  # the portable default
    assert memcap.resolve_route(4096.0, memcap.SLURM) == memcap.SLURM
    with pytest.raises(ValueError, match="unknown memory-cap route"):
        memcap.resolve_route(4096.0, "cgroup-v2")


def test_a_cap_nobody_enforces_is_refused():
    """``--cap-mem-mb N --mem-route none`` writes ``cap_mem_mb=N`` on records nothing bounded.

    ``mem_route`` is not a ``run_id`` field, so such a record is indistinguishable from an enforced one to
    anything pooling by id — the cap on it would be a claim about the measurement that is not true.
    """
    from andrey_bench import memcap

    with pytest.raises(ValueError, match="records a cap nothing enforces"):
        memcap.resolve_route(4096.0, memcap.NONE)


def test_the_slurm_route_asks_for_the_cap_and_rounds_it_up():
    """A cap that rounded *down* would enforce something tighter than the number on the record."""
    from andrey_bench import memcap

    assert memcap.srun_prefix(4096.0) == ["srun", "--ntasks=1", "--mem=4096M"]
    assert memcap.srun_prefix(1536.5)[-1] == "--mem=1537M"


def test_the_slurm_route_refuses_a_cap_the_job_cannot_schedule():
    """`srun --mem=` above the job's own memory is queued forever, not refused.

    The runner must say so at startup: the failure is otherwise a node held to the wall with no step,
    no record and no error.
    """
    from andrey_bench import memcap

    starved = {"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "16"}  # 32 GB
    with pytest.raises(ValueError, match="exceeds the 32000 MB"):
        memcap.resolve_route(64_000, memcap.SLURM, env=starved)

    # Enough memory, and off-scheduler where nothing can be read, both stay quiet.
    assert memcap.resolve_route(64_000, memcap.SLURM, env={"SLURM_MEM_PER_NODE": "262144"})
    assert memcap.resolve_route(64_000, memcap.SLURM, env={}) == memcap.SLURM


@pytest.mark.parametrize(
    "line",
    [
        "slurmstepd: error: Detected 1 oom_kill event in StepId=1234.0.",
        "slurmstepd: error: Some of your processes may have been killed by the cgroup "
        "out-of-memory handler.",
        "slurmstepd: error: StepId=1234.0 Exceeded step memory limit at some point.",
        "slurmstepd: error: Exceeded job memory limit",
    ],
)
def test_a_cgroup_kill_is_read_back_off_the_step(line):
    """On the SLURM route the kernel does the killing, so the reason exists only in what the step
    printed — the exit status is a bare signal, exactly as under polling."""
    from andrey_bench import memcap

    assert memcap.stderr_reports_oom(line)


@pytest.mark.parametrize(
    "line",
    [
        "",
        None,
        "Traceback (most recent call last):\n  ValueError: adjacency must be square",
        "slurmstepd: error: *** JOB 1234 CANCELLED AT 2026-08-14T12:00:00 DUE TO TIME LIMIT ***",
    ],
)
def test_a_non_memory_failure_is_not_read_as_a_cgroup_kill(line):
    """The other direction, and the one that would quietly manufacture a memory limit: a preemption
    or a plain crash must not be filed as an OOM."""
    from andrey_bench import memcap

    assert not memcap.stderr_reports_oom(line)


def test_the_tree_total_sums_a_descendant_this_process_did_not_allocate():
    """:func:`~andrey_bench.memcap.tree_rss_mb` over a live child, checked directly.

    The poller's kill decision is only as good as this sum, and a version that read the leader alone
    would still pass every "capped run dies" assertion above.
    """
    import subprocess
    import sys

    from andrey_bench import memcap

    before = memcap.tree_rss_mb(os.getpid())
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import numpy as np, time; a = np.ones(300*1024*1024//8); a += 1.0; "
            "print('up', flush=True); time.sleep(5)",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline().strip() == b"up"  # the 300 MB is resident before the read
        after = memcap.tree_rss_mb(os.getpid())
    finally:
        child.kill()
        child.wait()

    assert after - before > 250.0, (before, after)  # the descendant's pages are in the total
