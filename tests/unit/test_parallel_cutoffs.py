"""The worker-pool cutoffs of GES and the skeleton search: the default warns once, explicit values
do not, results hold.

Controlled inputs only; the calibrator's search runs on fake timing functions, and its real timers
run on the smallest passes.
"""

from __future__ import annotations

import importlib
import json
import os
import time
import warnings

import numpy as np
import pytest

import andrey
from andrey.cli import calibration_report, main, render_calibration
from andrey.constraint._skeleton import scan_pairs
from andrey.search import calibrate as cal

pg = importlib.import_module("andrey.search._parallel_ges")
sk = importlib.import_module("andrey.constraint._parallel_skeleton")
SKELETON_VARIABLE = cal.METHODS["skeleton"].variable

METHODS = {"ges": pg}


@pytest.fixture(autouse=True)
def unset_cutoffs(monkeypatch):
    for var in (m.variable for m in cal.METHODS.values()):
        monkeypatch.delenv(var, raising=False)


def _fit(method: str, workers: int, d: int = 6):
    with andrey.config(num_workers=workers):
        return getattr(andrey, method)(cal.sample(d)).structure.to_numpy()


def _cutoff_warnings(caught):
    return [w for w in caught if w.category is andrey.PerformanceWarning]


def _small_default(monkeypatch, method, cutoff):
    """Set the built-in cutoff of ``method`` so a six-variable fit reaches the warning threshold."""
    monkeypatch.setattr(METHODS[method], "_DEFAULT_MIN_WORK", cutoff)


@pytest.mark.parametrize("method", METHODS)
def test_default_cutoff_warns_once_naming_the_setting_and_command(method, monkeypatch):
    _small_default(monkeypatch, method, 300)  # the first pass over 6 variables has work 30
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(method, workers=2)
        _fit(method, workers=2)
    [warning] = _cutoff_warnings(caught)
    assert f"{cal.METHODS[method].variable}=300" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert "If this process runs many fits" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("method", METHODS)
def test_a_fit_far_below_the_default_cutoff_does_not_warn(method):
    """Six variables: every pass is far below a tenth of the real default."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(method, workers=2)
    assert not _cutoff_warnings(caught)


@pytest.mark.parametrize("d, warns", [(20, False), (40, True)])
def test_the_default_cutoff_warns_once_the_largest_pass_reaches_a_tenth_of_it(d, warns):
    """On the built-in default, the warning follows the work the gate counts, not the size."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit("ges", workers=2, d=d)
    reached = pg.largest_pass_work >= cal.WARN_FRACTION * pg._DEFAULT_MIN_WORK
    assert reached is warns  # the case lands on the side it names
    assert len(_cutoff_warnings(caught)) == int(reached)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", ["0", "30", "default"])
def test_an_explicit_cutoff_never_warns(method, value, monkeypatch):
    """Any set value counts, including one equal to the built-in default.

    Every value here would warn if treated as a default: the fit's work (30) reaches a tenth of it.
    """
    _small_default(monkeypatch, method, 300)
    if value == "default":
        value = str(METHODS[method]._DEFAULT_MIN_WORK)
    monkeypatch.setenv(cal.METHODS[method].variable, value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(method, workers=2)
    assert not _cutoff_warnings(caught)


@pytest.mark.parametrize("method", METHODS)
def test_a_serial_fit_never_warns(method):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(method, workers=1)
    assert not _cutoff_warnings(caught)


@pytest.mark.parametrize("method, offset, pooled", [("ges", -1, True), ("ges", 0, None)])
def test_serial_and_pooled_fits_agree_around_a_forced_cutoff(method, offset, pooled, monkeypatch):
    """A first pass over d variables has work d * (d - 1); a cutoff one below it pools that pass.

    A later pass can count more work, for a target with many undirected neighbors, so at the cutoff
    only the results are compared.
    """
    d = 8
    created = []
    real = METHODS[method]._create_pool
    monkeypatch.setattr(METHODS[method], "_create_pool", lambda *a: created.append(a) or real(*a))
    monkeypatch.setenv(cal.METHODS[method].variable, str(d * (d - 1) + offset))
    parallel = _fit(method, workers=2, d=d)
    if pooled is not None:
        assert bool(created) is pooled
    np.testing.assert_array_equal(parallel, _fit(method, workers=1, d=d))


def test_a_fit_on_the_default_cutoff_runs_in_one_process(monkeypatch):
    """No pass of a sparse fit over 64 variables reaches the default cutoff."""
    monkeypatch.setattr(pg, "_create_pool", lambda *a: pytest.fail("a pass reached the pool"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.PerformanceWarning)
        parallel = _fit("ges", workers=2, d=64)
    assert 0 < pg.largest_pass_work < pg._DEFAULT_MIN_WORK
    np.testing.assert_array_equal(parallel, _fit("ges", workers=1, d=64))


def test_the_calibrator_states_the_cutoff_in_the_largest_pass_work():
    """A size's work is the largest pass of its pooled fit; the first pass is a lower bound."""
    serial, pooled, work = cal.fit_timers("ges", workers=2)
    pooled(12)
    assert work(12) >= 12 * 11
    c = cal.bracket(lambda d: 2.0, lambda d: 1.0, grid=(16, 32), work=lambda d: 7 * d)
    assert c.cutoff == 7 * cal._MIN_D and c.work(cal._MIN_D) == 7 * cal._MIN_D


def _timers(serial_cost, pooled_cost, calls):
    def serial(d):
        calls.append(d)
        return serial_cost(d)

    return serial, lambda d: pooled_cost(d)


def test_bracket_finds_a_crossover_inside_the_grid():
    calls = []
    serial, pooled = _timers(lambda d: d * d * 1e-6, lambda d: 5e-3 + d * d * 1e-7, calls)
    c = cal.bracket(serial, pooled, grid=(16, 32, 64, 128))
    # serial = pooled at d = sqrt(5e-3 / 9e-7), about 74.5
    assert c.below < 74.5 < c.above
    assert c.above / c.below <= 2 ** (1 / 8) + 0.01  # three log bisections of (64, 128)
    assert c.cutoff == cal._round2((c.below * (c.below - 1) * c.above * (c.above - 1)) ** 0.5)
    assert all(cal._MIN_D <= d <= cal._MAX_D for d in calls)


def test_bracket_extends_below_the_grid():
    calls = []
    serial, pooled = _timers(lambda d: d * d * 1e-6, lambda d: 2e-5 + d * d * 1e-7, calls)
    c = cal.bracket(serial, pooled, grid=(16, 32, 64, 128))
    assert c.below is not None and c.above is not None
    assert c.below < 16 and c.below < 4.72 < c.above  # crossover at sqrt(2e-5 / 9e-7), about 4.7
    assert min(calls) >= cal._MIN_D


def test_bracket_extends_above_the_grid_and_stops_at_the_cap():
    calls = []
    serial, pooled = _timers(lambda d: 1.0, lambda d: 2.0, calls)  # the pool never pays
    c = cal.bracket(serial, pooled, grid=(16, 32, 64, 128))
    assert (c.below, c.above) == (cal._MAX_D, None)
    assert c.cutoff == max(cal._MAX_D * (cal._MAX_D - 1), pg._DEFAULT_MIN_WORK)  # never below it
    assert max(calls) == cal._MAX_D


def test_a_never_faster_cutoff_keeps_the_slowest_measured_pass_off_the_pool(monkeypatch):
    """GES uses the pool above its cutoff and the skeleton search at it, so a calibration that
    never finds the pool faster sets the skeleton's cutoff one past the slowest pass measured."""
    ges = cal.Crossover(256, None, (), ((256, 2_000_000),), "ges")
    assert ges.cutoff == 2_000_000
    p = cal.skeleton_pass(100, 2)
    assert p.tests > sk.MIN_POOL_QUERIES
    never = cal.Crossover(256, None, (), ((256, p.tests),), "skeleton")
    assert never.cutoff == p.tests + 1
    monkeypatch.setenv(SKELETON_VARIABLE, str(never.cutoff))
    monkeypatch.setattr(sk.SkeletonWorkerPool, "_ensure", lambda self: pytest.fail("pool started"))
    pool = sk.SkeletonWorkerPool(p.test, 2)
    assert pool.scan(p.pairs, p.neighbors, p.adjacency, p.size, 0.05) is None


def test_bracket_when_the_pool_wins_down_to_the_smallest_size():
    c = cal.bracket(lambda d: 2.0, lambda d: 1.0, grid=(16, 32))
    assert (c.below, c.above) == (None, cal._MIN_D)
    assert c.cutoff == cal._MIN_D * (cal._MIN_D - 1)


def test_fits_never_run_the_calibrator(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a fit started calibration")

    for name in ("bracket", "calibrate", "fit_timers", "skeleton_timers", "skeleton_pass", "_time"):
        monkeypatch.setattr(cal, name, refuse)
    for method in METHODS:
        _fit(method, workers=2)
    with andrey.config(num_workers=2):
        andrey.pc(cal.sample(12))


def test_calibrate_command_prints_the_settings(capsys, monkeypatch):
    seen = {}

    def fake(workers, methods):
        seen.update(workers=workers, methods=list(methods))
        return {m: cal.Crossover(32, 35, ((32, 0.2, 0.25), (35, 0.27, 0.26))) for m in methods}

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--method", "ges", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert seen == {"workers": 4, "methods": ["ges"]}
    assert report["settings"] == {"ANDREY_NUM_WORKERS": "4", "ANDREY_GES_PARALLEL_MIN_WORK": "1100"}
    assert report["methods"]["ges"]["pool_faster_at"] == {"d": 35, "work": 1190}
    assert report["schema_version"] == "2"


def test_calibrate_command_rejects_a_malformed_environment(capsys, monkeypatch):
    monkeypatch.setenv("ANDREY_NUM_WORKERS", "many")
    assert main(["calibrate", "--workers", "4"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "invalid_configuration" and "ANDREY_NUM_WORKERS" in error["message"]


@pytest.mark.parametrize("args", [["--workers", "1"], ["--method", "pc"], ["--method", "hc"]])
def test_calibrate_command_rejects_bad_options(args, capsys):
    assert main(["calibrate", *args]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsupported_value"


# --- the skeleton cutoff ---------------------------------------------------------------------
#
# A pass of the first probe, eight variables at conditioning size 2, holds C(8, 2) * C(6, 2) = 420
# tests. The pool's gate reads the cutoff when the pool is created, so a pass is scanned directly.


@pytest.fixture
def probe():
    return cal.skeleton_pass(8, 2)


def _scan(pool, p):
    return pool.scan(p.pairs, p.neighbors, p.adjacency, p.size, 0.05)


def _skeleton_warnings(probe, workers=2, scans=2):
    """The cutoff warnings of ``scans`` passes over ``probe``, each on a pool of ``workers``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(scans):
            pool = sk.SkeletonWorkerPool(probe.test, workers)
            try:
                assert _scan(pool, probe) is None  # every case here keeps the pass in this process
            finally:
                pool.close()
    return _cutoff_warnings(caught)


def test_the_skeleton_probe_counts_the_tests_the_gate_counts():
    assert cal.skeleton_pass(8, 2).tests == 420
    assert cal.skeleton_pass(4, 3).tests == 0  # a neighborhood of three offers no set of three
    assert cal.skeleton_pass(16, 2).tests > cal.skeleton_pass(8, 2).tests


@pytest.mark.parametrize("offset, pooled", [(0, True), (1, False)])
def test_the_skeleton_override_sets_the_pass_size_that_uses_the_pool(
    offset, pooled, probe, monkeypatch
):
    monkeypatch.setenv(SKELETON_VARIABLE, str(probe.tests + offset))
    pool = sk.SkeletonWorkerPool(probe.test, 2)
    try:
        removals = _scan(pool, probe)
    finally:
        pool.close()
    assert (removals is not None) is pooled
    if pooled:
        serial = scan_pairs(probe.test, probe.pairs, probe.neighbors, probe.size, 0.05)
        assert [r[:2] for r in removals] == [r[:2] for r in serial]


def test_a_bad_skeleton_override_fails_the_fit_before_any_pass(monkeypatch):
    monkeypatch.setenv(SKELETON_VARIABLE, "many")
    with andrey.config(num_workers=2), pytest.raises(ValueError, match=SKELETON_VARIABLE):
        andrey.pc(cal.sample(6))


@pytest.mark.parametrize("variable", [m.variable for m in cal.METHODS.values()])
def test_the_command_line_rejects_a_bad_cutoff(variable, capsys, monkeypatch):
    monkeypatch.setenv(variable, "many")
    assert main(["config"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "invalid_configuration" and f"{variable} must be" in error["message"]


def test_the_default_skeleton_cutoff_warns_once_naming_the_setting_and_command(probe, monkeypatch):
    monkeypatch.setattr(sk, "MIN_POOL_QUERIES", 8 * probe.tests)  # the pass reaches a tenth of it
    [warning] = _skeleton_warnings(probe)  # two pools, one warning
    assert f"{SKELETON_VARIABLE}={8 * probe.tests}" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert "If this process runs many fits" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("factor, warns", [(8, True), (20, False)])
def test_the_default_skeleton_cutoff_warns_once_a_pass_reaches_a_tenth_of_it(
    factor, warns, probe, monkeypatch
):
    monkeypatch.setattr(sk, "MIN_POOL_QUERIES", factor * probe.tests)
    assert bool(_skeleton_warnings(probe)) is warns


@pytest.mark.parametrize("factor", [2, 8])
def test_an_explicit_skeleton_cutoff_never_warns(factor, probe, monkeypatch):
    """Any set value counts, including one equal to the built-in default (factor 8). The pass stays
    in process under each, and each would warn if treated as a default."""
    monkeypatch.setattr(sk, "MIN_POOL_QUERIES", 8 * probe.tests)
    monkeypatch.setenv(SKELETON_VARIABLE, str(factor * probe.tests))
    assert not _skeleton_warnings(probe)


def test_a_serial_skeleton_pool_never_warns(probe, monkeypatch):
    monkeypatch.setattr(sk, "MIN_POOL_QUERIES", 8 * probe.tests)
    assert not _skeleton_warnings(probe, workers=1)


def test_a_pc_fit_warns_on_the_default_skeleton_cutoff_and_not_on_an_override(monkeypatch):
    """Forty sparse variables reach a size-2 pass; a default of 1 test warns and pools it."""
    monkeypatch.setattr(sk, "MIN_POOL_QUERIES", 1)
    x = cal.sample(40)
    with warnings.catch_warnings(record=True) as caught, andrey.config(num_workers=2):
        warnings.simplefilter("always")
        warned = andrey.pc(x).structure.to_numpy()
    assert len(_cutoff_warnings(caught)) == 1
    monkeypatch.setenv(SKELETON_VARIABLE, "1")
    with warnings.catch_warnings(record=True) as caught, andrey.config(num_workers=2):
        warnings.simplefilter("always")
        explicit = andrey.pc(x).structure.to_numpy()
    assert not _cutoff_warnings(caught)
    np.testing.assert_array_equal(warned, explicit)


# --- the skeleton calibrator -----------------------------------------------------------------


def test_the_skeleton_timers_pool_only_the_pooled_arm_and_restore_the_cutoff(monkeypatch):
    created = []
    real = sk.ProcessPoolExecutor
    monkeypatch.setattr(
        sk, "ProcessPoolExecutor", lambda *a, **k: created.append(k) or real(*a, **k)
    )
    monkeypatch.setenv(SKELETON_VARIABLE, "123")
    serial, pooled, work = cal.skeleton_timers(workers=2, size=2)
    assert serial(8) > 0 and not created
    assert pooled(8) > 0 and created
    assert work(8) == 420
    assert os.environ[SKELETON_VARIABLE] == "123"


def test_the_skeleton_calibrator_measures_a_cutoff_and_the_pool_start(monkeypatch):
    """Run on the smallest passes: one timed run each, a probe that stops at 16 variables."""

    def once(fn, repeats):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    monkeypatch.setattr(cal, "_time", once)
    monkeypatch.setattr(cal, "GRID", (8, 12))
    monkeypatch.setattr(cal, "_MAX_D", 16)
    monkeypatch.setattr(cal, "SKELETON_SIZES", {2: 6})
    c = cal.calibrate(2, ["skeleton"])["skeleton"]
    assert c.method == "skeleton" and c.cutoff > 0 and c.start_s > 0
    assert cal._MIN_D <= min(d for d, _, _ in c.timings) and max(d for d, _, _ in c.timings) <= 16
    assert SKELETON_VARIABLE not in os.environ


def test_the_skeleton_cutoff_is_the_largest_of_the_sizes(monkeypatch):
    """The pool wins from 50 variables at size 2 and from 20 at size 3; the cutoff is size 2's."""

    def timers(pool_wins_from):
        pooled = lambda d: 0.5 if d >= pool_wins_from else 2.0  # noqa: E731
        return (lambda d: 1.0), pooled, (lambda d: 10 * d)

    by_size = {2: timers(50), 3: timers(20)}
    cutoffs = {
        size: cal.bracket(t[0], t[1], work=t[2], method="skeleton").cutoff
        for size, t in by_size.items()
    }
    monkeypatch.setattr(cal, "SKELETON_SIZES", dict.fromkeys(by_size, 0))
    monkeypatch.setattr(cal, "skeleton_timers", lambda workers, size: by_size[size])
    assert cutoffs[3] < cutoffs[2]
    assert cal.calibrate(2, ["skeleton"])["skeleton"].cutoff == cutoffs[2]


def test_calibrate_command_measures_every_cutoff_by_default(capsys, monkeypatch):
    seen = {}

    def fake(workers, methods):
        seen["methods"] = list(methods)
        return {
            m: cal.Crossover(32, 35, ((32, 0.2, 0.25), (35, 0.27, 0.26)), method=m, start_s=0.12345)
            for m in methods
        }

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert seen["methods"] == ["ges", "skeleton"]
    assert set(report["settings"]) == {
        "ANDREY_NUM_WORKERS",
        *(m.variable for m in cal.METHODS.values()),
    }
    assert report["methods"]["skeleton"]["variable"] == SKELETON_VARIABLE
    assert report["methods"]["skeleton"]["pool_start_s"] == 0.1235
    text = render_calibration(calibration_report(fake(4, ["skeleton"]), 4))
    assert (
        f"export {SKELETON_VARIABLE}=" in text and "Starting the pool takes about 0.12 s." in text
    )


def test_the_skeleton_cutoff_defaults_to_the_built_in_when_the_pool_never_wins():
    c = cal.bracket(lambda d: 1.0, lambda d: 2.0, grid=(16,), method="skeleton", work=lambda d: d)
    assert c.below == cal._MAX_D and c.above is None
    assert c.cutoff == sk.MIN_POOL_QUERIES  # the largest probe's work is far below the default
