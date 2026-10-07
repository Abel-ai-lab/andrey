"""Embedded R timing and resource-cap tests.

R startup and package loading precede timing. Data transfer, the R call, and graph retrieval are
timed. A wall- or memory-cap kill terminates R with the fit and leaves no R process behind.

Every test that needs R skips without one, because CI has no R; `ANDREY_BENCH_REQUIRE_R=1` makes
them fail instead. The stub adapters live at module level so the worker can unpickle them - the
runner puts this file's directory on the child's ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import psutil
import pytest
from conftest import needs_r as _needs_r

needs_r = _needs_r()

_ADJ = np.zeros((4, 4), dtype=np.int8)


class _RStub:
    """`SolutionAdapter` stub whose `setup` starts R as an R adapter does."""

    name = "stub.r"
    algorithm = "ges"
    package = "stub"
    package_version = "0"
    backend = "native"
    mode = "serial"
    output_type = "dag"

    def to_structure(self, native):  # pragma: no cover - the runner never calls this
        raise NotImplementedError


class StubRStartupAdapter(_RStub):
    """Start R and load pcalg in `setup`; run a trivial R call in `fit`.

    Including startup in fit timing would add interpreter and package loading to `wall_s`.
    """

    def setup(self):
        from andrey_bench import rsession

        rsession.start("pcalg")

    def fit(self, data, params):
        from andrey_bench import rsession

        session = rsession.session()
        return np.asarray(session.matrix(data)).astype(np.int8)[:1, :1]


class StubRHangAdapter(_RStub):
    """Start R in `setup`, then hang inside R; stopping the fit requires terminating R."""

    def setup(self):
        from andrey_bench import rsession

        rsession.start("pcalg")

    def fit(self, data, params):
        from andrey_bench import rsession

        rsession.session().define("Sys.sleep(600)")
        return _ADJ.copy()


class StubRGrowAdapter(_RStub):
    """Start R in `setup`, then allocate beyond a small cap in R's heap.

    The worker's Python allocator does not track this allocation.
    """

    def setup(self):
        from andrey_bench import rsession

        rsession.start("pcalg")

    def fit(self, data, params):
        from andrey_bench import rsession

        rsession.session().define("x <- matrix(0, nrow = 20000, ncol = 20000); sum(x)")
        return _ADJ.copy()


def _tiny_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 4))


def _worker_processes() -> set[int]:
    """PIDs of every live benchmark worker descended from this process."""
    me = psutil.Process(os.getpid())
    found = set()
    for proc in me.children(recursive=True):
        try:
            if "andrey_bench._worker" in " ".join(proc.cmdline()):
                found.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


# --------------------------------------------------------------------------------------------------
# Without setup, fitting fails instead of starting R inside the timed region.
# --------------------------------------------------------------------------------------------------


def test_a_fit_without_setup_refuses_to_start_r(monkeypatch):
    from andrey_bench import rsession

    # Clear any session left by another test to reproduce a worker that skipped setup.
    monkeypatch.setattr(rsession, "_SESSION", None)
    with pytest.raises(RuntimeError, match="untimed"):
        rsession.session()


def test_library_paths_follow_the_order_r_searches(monkeypatch):
    """`R_LIBS`, then `R_LIBS_USER`, the site library, and R's own."""
    from pathlib import Path

    from andrey_bench import rsession

    monkeypatch.setenv("R_LIBS", os.pathsep.join(["/libs/a", "/libs/b"]))
    monkeypatch.setenv("R_LIBS_USER", "/user")
    monkeypatch.setenv("R_LIBS_SITE", "/site")
    monkeypatch.setenv("R_HOME", "/r")
    expected = ["/libs/a", "/libs/b", "/user", "/site", "/r/library"]
    assert rsession.library_paths() == [Path(p) for p in expected]


def test_a_package_in_two_libraries_reports_the_version_r_loads(monkeypatch, tmp_path):
    """R takes a package from the first library that has it, so the record must too."""
    from andrey_bench import rsession

    for lib, version in (("libs", "2.0"), ("user", "1.0")):
        (tmp_path / lib / "pcalg").mkdir(parents=True)
        (tmp_path / lib / "pcalg" / "DESCRIPTION").write_text(
            f"Package: pcalg\nVersion: {version}\n"
        )
    monkeypatch.setenv("R_LIBS", str(tmp_path / "libs"))
    monkeypatch.setenv("R_LIBS_USER", str(tmp_path / "user"))
    monkeypatch.delenv("R_LIBS_SITE", raising=False)
    monkeypatch.delenv("R_HOME", raising=False)
    assert rsession.package_version("pcalg") == "2.0"


@needs_r
def test_the_environment_check_reports_the_versions_r_loaded():
    from andrey_bench import rsession

    versions = rsession.check_environment("pcalg")
    assert versions["pcalg"] == rsession.package_version("pcalg") != "unknown"
    assert versions["rpy2"] != "unknown"


# --------------------------------------------------------------------------------------------------
# Data transfer into R and graph retrieval.
# --------------------------------------------------------------------------------------------------


@needs_r
def test_data_reaches_r_in_the_same_orientation():
    """R must see the same n x d table, not its transpose: a transposed matrix still fits."""
    import rpy2.robjects as ro

    from andrey_bench import rsession

    data = np.arange(24, dtype=np.float64).reshape(6, 4)
    session = rsession.start()
    matrix = session.matrix(data)

    assert list(ro.r["dim"](matrix)) == [6, 4]
    assert np.allclose(list(ro.r["colMeans"](matrix)), data.mean(axis=0))
    assert list(matrix.colnames) == ["V0", "V1", "V2", "V3"]


@needs_r
def test_a_returned_graph_is_reindexed_by_its_own_dimnames():
    """Restore variable order from the result names when a method reorders variables."""
    import rpy2.robjects as ro

    from andrey_bench import rsession

    session = rsession.start()
    ro.r("""
        m <- matrix(0, 3, 3)
        m[1, 2] <- 1
        dimnames(m) <- list(c("V2", "V0", "V1"), c("V2", "V0", "V1"))
    """)
    out = session.square_matrix(ro.r["m"], names=["V0", "V1", "V2"])

    # The R matrix encodes `V2 -> V0`: row 2, column 0 in `V0, V1, V2` order.
    expected = np.zeros((3, 3))
    expected[2, 0] = 1
    assert np.array_equal(out, expected)


@needs_r
def test_a_graph_over_the_wrong_variables_is_rejected():
    import rpy2.robjects as ro

    from andrey_bench import rsession

    session = rsession.start()
    ro.r('m <- matrix(0, 2, 2); dimnames(m) <- list(c("A", "B"), c("A", "B"))')
    with pytest.raises(ValueError, match="expected"):
        session.square_matrix(ro.r["m"], names=["V0", "V1"])


# --------------------------------------------------------------------------------------------------
# The timing boundary and the cleanup, through the runner that enforces both.
# --------------------------------------------------------------------------------------------------


@needs_r
def test_r_startup_and_package_loading_are_outside_the_timed_fit():
    from andrey_bench.runner import run_task

    res = run_task(StubRStartupAdapter(), _tiny_data(), {}, cap_wall_s=300.0, repeats=1, warmup=0)

    m = res[0]
    assert m["status"] == "ok", m
    # Loading pcalg takes about a second; the fit transfers one small matrix.
    assert m["setup_s"] > 0.2, m
    assert m["wall_s"] < m["setup_s"] / 2, m
    assert m["warmup_s"] == m["setup_s"], m


@needs_r
def test_a_fit_hanging_inside_r_times_out_and_leaves_no_process():
    """The cap must reach code executing in R, not just in Python."""
    from andrey_bench.runner import run_task

    before = _worker_processes()
    start = time.perf_counter()
    res = run_task(StubRHangAdapter(), _tiny_data(), {}, cap_wall_s=20.0, repeats=1, warmup=0)
    elapsed = time.perf_counter() - start

    m = res[0]
    assert m["status"] == "timeout", m
    assert elapsed < 60.0, elapsed  # the 600 s R sleep did not run to completion
    assert m["setup_s"] is not None, m  # R did start, so the timeout is the fit's
    assert _worker_processes() == before, "a benchmark worker survived the timeout"


@needs_r
def test_memory_r_allocates_is_capped_and_the_tree_dies():
    """R's heap contributes to worker RSS, allowing the tree poller to enforce the cap."""
    from andrey_bench.runner import run_task

    before = _worker_processes()
    res = run_task(
        StubRGrowAdapter(),
        _tiny_data(),
        {},
        cap_wall_s=300.0,
        cap_mem_mb=2000.0,
        mem_route="poll",
        repeats=1,
        warmup=0,
    )

    m = res[0]
    assert m["status"] == "oom", m
    assert m["oom_source"] == "poller", m
    assert _worker_processes() == before, "a benchmark worker survived the memory cap"
