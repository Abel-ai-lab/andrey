"""Runner tests using picklable stub adapters.

Children import this module to reconstruct adapters. Module-level imports stay limited to the
standard library and numpy; tests import the runner and contracts locally.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# A fixed tiny 4x4 DAG adjacency (0 -> 1 -> 2 -> 3). Returned by every stub so output hashes are
# stable and comparable across repeats.
_ADJ = np.array(
    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
    dtype=np.int8,
)


class _BaseStub:
    """Stub adapter; the runner never calls ``to_structure``."""

    name = "stub.ges"
    algorithm = "ges"
    package = "stub"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    output_type = "dag"

    def to_structure(self, native):  # pragma: no cover - runner never calls this
        raise NotImplementedError


class StubSleepAdapter(_BaseStub):
    """Sleep 0.3 s per fit to distinguish timed fits from warmups."""

    def fit(self, data, params):
        time.sleep(0.3)
        return _ADJ.copy()


class StubTimeoutAdapter(_BaseStub):
    def fit(self, data, params):
        time.sleep(5.0)
        return _ADJ.copy()


#: Counts ``setup()`` calls inside one child. Each repeat is a fresh process, so it starts at 0.
_SETUP_CALLS = [0]


class StubSetupAdapter(_BaseStub):
    """Sleep 0.4 s in setup and 0.2 s in fit; return the setup call count."""

    def setup(self):
        _SETUP_CALLS[0] += 1
        time.sleep(0.4)

    def fit(self, data, params):
        time.sleep(0.2)
        return np.array([[_SETUP_CALLS[0]]], dtype=np.int64)


class StubPidThenHangAdapter(_BaseStub):
    """Write the worker PID to ``params["pid_path"]``, then exceed the cap.

    The PID identifies the worker after driver death reparents it.
    """

    def fit(self, data, params):
        Path(params["pid_path"]).write_text(str(os.getpid()))
        time.sleep(600.0)
        return _ADJ.copy()


class StubExitAdapter(_BaseStub):
    """Exit without writing a result, using the requested process exit code."""

    def fit(self, data, params):
        os._exit(params["exit_code"])


def _sleep_through_the_cap(_) -> int:
    """Sleep past the cap; module-level so forkserver workers can import it."""
    time.sleep(600.0)
    return 0


class StubPoolPidThenHangAdapter(_BaseStub):
    """Write the worker PID, then hang with a forkserver pool.

    Fork clears the parent-death signal, leaving pool workers dependent on group cleanup.
    """

    mode = "parallel"

    def fit(self, data, params):
        import concurrent.futures as cf
        import multiprocessing as mp

        Path(params["pid_path"]).write_text(str(os.getpid()))
        ctx = mp.get_context("forkserver")
        with cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            list(ex.map(_sleep_through_the_cap, [0, 1]))
        return _ADJ.copy()


class StubDtypeEchoAdapter(_BaseStub):
    def fit(self, data, params):
        return np.array([[data.dtype.itemsize]], dtype=np.int64)


class EnvEchoAdapter(_BaseStub):
    """Return a 1x3 array of child environment values.

    Columns are ``ANDREY_NUM_WORKERS``, ``OMP_NUM_THREADS`` and whether
    ``ANDREY_GES_PARALLEL_MIN_WORK`` is set.
    """

    def fit(self, data, params):
        import os as _os

        nw = int(_os.environ.get("ANDREY_NUM_WORKERS", "-1"))
        omp = int(_os.environ.get("OMP_NUM_THREADS", "-1"))
        has_minwork = 1 if "ANDREY_GES_PARALLEL_MIN_WORK" in _os.environ else 0
        return np.array([[nw, omp, has_minwork]], dtype=np.int64)


def _spin_and_touch(_):
    """Touch about 150 MB and consume 0.5 s of CPU in a pool worker.

    This usage is absent from the parent's self-rusage. Kept module-level for forkserver imports.
    """
    import time as _t

    import numpy as _np

    arr = _np.ones(150 * 1024 * 1024 // 8, dtype=_np.float64)  # 150 MiB
    arr += 1.0  # fault every page in
    start = _t.process_time()
    while _t.process_time() - start < 0.5:
        pass
    return float(arr.sum())


class StubForkserverPoolAdapter(_BaseStub):
    """Run a 2-worker forkserver pool, matching Andrey's ``StableWorkerSet``.

    CPU and RSS usage are in child processes.
    """

    mode = "parallel"

    def fit(self, data, params):
        import concurrent.futures as _cf
        import multiprocessing as _mp

        ctx = _mp.get_context("forkserver")
        # Preload numpy into the (persistent) forkserver so its import cost is paid ONCE, at
        # forkserver startup — which, under warmup, happens before the timed snapshots and is excluded.
        # Workers then fork with numpy already resident, so their CPU is essentially the spin.
        ctx.set_forkserver_preload(["numpy"])
        with _cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            list(ex.map(_spin_and_touch, [0, 1]))
        return _ADJ.copy()


def _tiny_data() -> np.ndarray:
    return np.zeros((16, 4), dtype=np.float64)


def test_fit_only_timing_excludes_warmup():
    from andrey_bench.runner import run_task

    res = run_task(StubSleepAdapter(), _tiny_data(), {}, cap_wall_s=15.0, repeats=1, warmup=1)

    assert len(res) == 1
    m = res[0]
    assert m["status"] == "ok", m
    assert m["wall_s"] >= 0.25, m  # the timed fit really ran (~0.3s)
    assert m["wall_s"] < 0.55, m  # ~0.3, NOT ~0.6 => warmup excluded from the clock
    assert m["warmup_s"] >= 0.25, m  # warmup measured, and separated out
    assert m["output_hash"]
    assert m["cpu_s"] is not None
    assert m["peak_rss_mb"] > 0.0


def test_wall_cap_timeouts_and_kills_child():
    from andrey_bench.runner import run_task

    start = time.perf_counter()
    res = run_task(StubTimeoutAdapter(), _tiny_data(), {}, cap_wall_s=1.0, repeats=1, warmup=0)
    elapsed = time.perf_counter() - start

    assert len(res) == 1
    m = res[0]
    assert m["status"] == "timeout", m
    assert m["elapsed_at_kill_s"] is not None
    assert m["elapsed_at_kill_s"] >= 0.9, m  # killed at ~the cap
    assert elapsed < 4.0, elapsed  # the 5s fit did NOT run to completion
    assert "adj" not in m and m.get("wall_s") is None


#: A driver: one 600 s fit under a 600 s cap, the campaign a signal interrupts.
_DRIVER = """
import sys

import test_runner
from andrey_bench.runner import run_task

run_task(
    getattr(test_runner, sys.argv[2])(), test_runner._tiny_data(), {"pid_path": sys.argv[1]},
    cap_wall_s=600.0, repeats=1, warmup=0,
)
"""


def _alive(pid: int) -> bool:
    """Return whether the process exists and is not a zombie."""
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until(condition, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _live_group_members(pgid: int) -> list[int]:
    """Read live group members from ``/proc``, excluding zombies.

    Group membership survives reparenting after driver death.
    """
    members = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as fh:
                # The comm field holds spaces and parentheses; after it, index 0 is state and 2 pgrp.
                fields = fh.read().rsplit(") ", 1)[1].split()
            state, pgrp = fields[0], int(fields[2])
        except (OSError, IndexError, ValueError):
            continue
        if pgrp == pgid and state != "Z":
            members.append(int(name))
    return members


def _kill_survivor(pid: int) -> None:
    """Kill a surviving process group during test cleanup."""
    if _alive(pid):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def _spawn_driver(adapter: str, pid_path: Path) -> subprocess.Popen:
    """Spawn a driver using the named adapter class."""
    from andrey_bench.runner import BENCH_ROOT

    env = dict(os.environ)
    # A bare interpreter: give it the harness root and this file's directory, which the conftest
    # and the runner's child env put on the path for the test process.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BENCH_ROOT), str(Path(__file__).resolve().parent)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.Popen([sys.executable, "-c", _DRIVER, str(pid_path), adapter], env=env)


def test_a_parallel_pool_dies_with_a_sigkilled_driver(tmp_path):
    """Group cleanup must reach the forkserver and pool, which lack the parent-death signal."""
    pid_path = tmp_path / "worker.pid"
    driver = _spawn_driver("StubPoolPidThenHangAdapter", pid_path)
    pgid = None
    try:
        assert _wait_until(pid_path.exists, 120.0), "the worker never reached its fit"
        pgid = os.getpgid(int(pid_path.read_text()))
        # The supervisor, worker, forkserver, resource tracker and two pool workers.
        assert _wait_until(lambda: len(_live_group_members(pgid)) >= 6, 120.0), (
            f"the pool never started: {_live_group_members(pgid)}"
        )
        driver.kill()
        driver.wait(timeout=30.0)
        assert _wait_until(lambda: not _live_group_members(pgid), 10.0), (
            f"the driver died and its fit's group did not: {_live_group_members(pgid)}"
        )
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)


def test_a_parallel_pool_dies_when_its_worker_dies_first(monkeypatch, tmp_path):
    """Preserve the worker exit status while cleaning up its pool before driver exit."""
    from andrey_bench.runner import run_task

    communicate = subprocess.Popen.communicate
    for signum in (signal.SIGKILL, signal.SIGTERM):
        pid_path = tmp_path / f"worker-{signum}.pid"
        pgid = None

        def kill_worker(self, *args, **kwargs):
            nonlocal pgid
            if pgid is not None:
                return communicate(self, *args, **kwargs)
            pgid = self.pid
            assert _wait_until(lambda: pid_path.exists() and pid_path.stat().st_size > 0, 120.0), (
                "the worker never reached its fit"
            )
            assert _wait_until(lambda: len(_live_group_members(pgid)) >= 6, 120.0), (
                f"the pool never started: {_live_group_members(pgid)}"
            )
            os.kill(int(pid_path.read_text()), signum)
            return communicate(self, *args, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(subprocess.Popen, "communicate", kill_worker)
                measures = run_task(
                    StubPoolPidThenHangAdapter(),
                    _tiny_data(),
                    {"pid_path": str(pid_path)},
                    cap_wall_s=5.0,
                    repeats=1,
                    warmup=0,
                )[0]
            assert measures["status"] == "error", measures
            assert measures["exit_code"] == -signum, measures
            assert _wait_until(lambda: not _live_group_members(pgid), 5.0), (
                f"the worker died and its pool did not: {_live_group_members(pgid)}"
            )
        finally:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)


def test_worker_exit_codes_are_preserved_without_a_result():
    from andrey_bench.runner import run_task

    for code in (0, 7):
        measures = run_task(
            StubExitAdapter(),
            _tiny_data(),
            {"exit_code": code},
            cap_wall_s=15.0,
            repeats=1,
            warmup=0,
        )[0]
        assert measures["status"] == "error", measures
        assert measures["exit_code"] == code, measures


def test_poller_start_failure_cleans_up_the_fit_and_death_pipe(monkeypatch, tmp_path):
    """An interrupt after the poller thread starts still stops the thread, the fit, and the pipe."""
    import pytest

    from andrey_bench import memcap
    from andrey_bench.runner import run_task

    start = memcap.TreeMemoryPoller.start
    pipe = os.pipe
    pid_path = tmp_path / "worker.pid"
    pollers = []
    pipes = []

    def capture_pipe():
        fds = pipe()
        pipes.append(fds)
        return fds

    def interrupted_start(self):
        pollers.append(self)
        start(self)
        assert _wait_until(lambda: pid_path.exists() and pid_path.stat().st_size > 0, 120.0), (
            "the worker never reached its fit"
        )
        raise KeyboardInterrupt("poller startup interrupted")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "pipe", capture_pipe)
            patch.setattr(memcap.TreeMemoryPoller, "start", interrupted_start)
            with pytest.raises(KeyboardInterrupt, match="poller startup interrupted"):
                run_task(
                    StubPidThenHangAdapter(),
                    _tiny_data(),
                    {"pid_path": str(pid_path)},
                    cap_wall_s=600.0,
                    cap_mem_mb=16384.0,
                    repeats=1,
                    warmup=0,
                )
        assert _wait_until(lambda: not _live_group_members(pollers[0].pid), 5.0), (
            f"fit survived poller startup failure: {_live_group_members(pollers[0].pid)}"
        )
        assert not pollers[0].is_alive(), "the poller thread survived its failed start"
        with pytest.raises(OSError):
            os.fstat(pipes[0][1])
    finally:
        for poller in pollers:
            if poller.ident is not None:
                poller.stop()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(poller.pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(poller.pid, 0)
        if pipes:
            with contextlib.suppress(OSError):
                os.close(pipes[0][1])


def test_an_unwinding_driver_kills_the_group_while_the_caller_lives(monkeypatch, tmp_path):
    import pytest

    from andrey_bench.runner import run_task

    pid_path = tmp_path / "worker.pid"

    def interrupted(self, *args, **kwargs):
        assert _wait_until(pid_path.exists, 120.0), "the worker never reached its fit"
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_task(
            StubPidThenHangAdapter(),
            _tiny_data(),
            {"pid_path": str(pid_path)},
            cap_wall_s=600.0,
            repeats=1,
            warmup=0,
        )
    worker = int(pid_path.read_text())
    try:
        # The worker's exit can finish after the supervisor has been reaped.
        assert _wait_until(lambda: not _alive(worker), 5.0), (
            f"worker {worker} survived the driver's unwind"
        )
    finally:
        _kill_survivor(worker)


def test_setup_is_untimed_and_runs_once_per_child():
    from andrey_bench.runner import run_task

    for warmup in (0, 2):
        m = run_task(
            StubSetupAdapter(), _tiny_data(), {}, cap_wall_s=15.0, repeats=1, warmup=warmup
        )[0]
        assert m["status"] == "ok", m
        assert m["wall_s"] < 0.35, m  # About 0.2 s; excludes the 0.4 s setup and any warm-up fit.
        assert m["setup_s"] >= 0.35, m  # setup measured on its own
        assert int(m["adj"][0, 0]) == 1, m  # one setup call for every fit in the child
        if warmup:
            assert m["warmup_s"] >= m["setup_s"] + 0.35, m  # Setup plus two warm-up fits.
        else:
            assert m["warmup_s"] == m["setup_s"], m  # No warm-up fit; `warmup_s` is setup alone.


def test_dtype_cast_applied_child_side(tmp_path):
    from andrey_bench.runner import run_task

    data_file = tmp_path / "data.npz"
    np.savez(data_file, data=np.zeros((16, 4), dtype=np.float64))

    f64 = run_task(StubDtypeEchoAdapter(), data_file, {}, cap_wall_s=15.0, repeats=1, warmup=0)
    assert f64[0]["status"] == "ok", f64
    assert int(f64[0]["adj"][0, 0]) == 8  # default float64 -> the cast is a no-op on an f64 file
    # The default names the precision it ran at, so the f64 fit has one identity and not two.
    assert f64[0]["dtype"] == "float64", f64

    f32 = run_task(
        StubDtypeEchoAdapter(), data_file, {}, cap_wall_s=15.0, repeats=1, warmup=0, dtype="float32"
    )
    assert f32[0]["status"] == "ok", f32
    assert int(f32[0]["adj"][0, 0]) == 4  # dtype='float32' cast applied child-side


# --------------------------------------------------------------------------------------------------
# cpu_s / peak_rss_mb capture a parallel (forkserver) worker pool, not just this process.
# --------------------------------------------------------------------------------------------------


def test_forkserver_pool_cpu_and_rss_are_captured():
    from andrey_bench.runner import run_task

    res = run_task(
        StubForkserverPoolAdapter(), _tiny_data(), {}, cap_wall_s=90.0, repeats=1, warmup=1
    )

    assert len(res) == 1
    m = res[0]
    assert m["status"] == "ok", m
    # Two workers x ~0.5s CPU each == ~1.0s of *child* CPU the pool burned during the timed fit.
    assert m["child_cpu_s"] >= 0.9, m  # a self-only delta would be ~0
    assert m["child_cpu_s"] < 1.6, m  # the warmup pool (~1.0s) is excluded from the delta
    assert m["cpu_s"] >= m["child_cpu_s"], m  # total CPU includes the child term
    # Two live workers holding ~150MB each -> ~300MB of child RSS the sampler must catch.
    assert m["child_peak_rss_mb"] >= 200, m  # ru_maxrss (self only) would miss the pool
    assert m["peak_rss_mb"] >= m["child_peak_rss_mb"], m


# --------------------------------------------------------------------------------------------------
# The cap-tier worker budget reaches the child as ANDREY_NUM_WORKERS, scrubbed of inherited state.
# --------------------------------------------------------------------------------------------------


def test_inherited_andrey_env_is_scrubbed(monkeypatch):
    from andrey_bench.runner import run_task

    monkeypatch.setenv("ANDREY_NUM_WORKERS", "97")
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")

    res = run_task(
        EnvEchoAdapter(),
        _tiny_data(),
        {},
        cap_wall_s=15.0,
        repeats=1,
        warmup=0,
        threads=1,
        num_workers=4,
    )
    nw, omp, has_minwork = (int(x) for x in res[0]["adj"][0])
    assert (nw, omp, has_minwork) == (4, 1, 0), res[0][
        "adj"
    ]  # 97 scrubbed, work-gate not inherited


def test_extra_env_overlays_child_env():
    from andrey_bench.runner import run_task

    res = run_task(
        EnvEchoAdapter(),
        _tiny_data(),
        {},
        cap_wall_s=15.0,
        repeats=1,
        warmup=0,
        threads=1,
        num_workers=4,
        extra_env={"ANDREY_GES_PARALLEL_MIN_WORK": "0"},
    )
    nw, omp, has_minwork = (int(x) for x in res[0]["adj"][0])
    assert (nw, omp, has_minwork) == (4, 1, 1), res[0]["adj"]  # overlay applied


def test_split_cores_gives_both_modes_the_same_machine():
    from andrey_bench.runner import split_cores

    assert split_cores(16, "parallel") == (1, 16)
    assert split_cores(16, "serial") == (16, 1)
