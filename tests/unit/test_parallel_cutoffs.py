"""The GES and HC worker-pool cutoffs: the default warns once, explicit values do not, results hold.

Controlled inputs only; the calibrator's search runs on fake timing functions.
"""

from __future__ import annotations

import importlib
import json
import warnings

import numpy as np
import pytest

import andrey
from andrey.cli import main
from andrey.search import calibrate as cal

pg = importlib.import_module("andrey.search._parallel_ges")
ph = importlib.import_module("andrey.search._parallel_hc")

METHODS = {"ges": pg, "hc": ph}


@pytest.fixture(autouse=True)
def unset_cutoffs(monkeypatch):
    for var in cal.CUTOFFS.values():
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
    assert f"{cal.CUTOFFS[method]}=300" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert "If this process runs many fits" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("method", METHODS)
def test_a_fit_far_below_the_default_cutoff_does_not_warn(method):
    """Six variables: every pass is far below a tenth of either real default."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(method, workers=2)
    assert not _cutoff_warnings(caught)


@pytest.mark.parametrize("cutoff, warns", [(300, True), (301, False)])
def test_hc_warns_from_a_tenth_of_the_cutoff(cutoff, warns, monkeypatch):
    """HC's every pass has work d * (d - 1) = 30 at d = 6: the threshold is a tenth of 300."""
    _small_default(monkeypatch, "hc", cutoff)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit("hc", workers=2)
    assert bool(_cutoff_warnings(caught)) is warns


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", ["0", "30", "default"])
def test_an_explicit_cutoff_never_warns(method, value, monkeypatch):
    """Any set value counts, including one equal to the built-in default.

    Every value here would warn if treated as a default: the fit's work (30) reaches a tenth of it.
    """
    _small_default(monkeypatch, method, 300)
    if value == "default":
        value = str(METHODS[method]._DEFAULT_MIN_WORK)
    monkeypatch.setenv(cal.CUTOFFS[method], value)
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


@pytest.mark.parametrize(
    "method, offset, pooled",
    [("ges", -1, True), ("ges", 0, None), ("hc", -1, True), ("hc", 0, False)],
)
def test_serial_and_pooled_fits_agree_around_a_forced_cutoff(method, offset, pooled, monkeypatch):
    """A first pass over d variables has work d * (d - 1); a cutoff one below it pools that pass.

    HC's passes all have that work, so at the cutoff it stays serial. GES's later passes do more
    work, so at the cutoff only the results are compared.
    """
    d = 8
    created = []
    real = METHODS[method]._create_pool
    monkeypatch.setattr(METHODS[method], "_create_pool", lambda *a: created.append(a) or real(*a))
    monkeypatch.setenv(cal.CUTOFFS[method], str(d * (d - 1) + offset))
    parallel = _fit(method, workers=2, d=d)
    if pooled is not None:
        assert bool(created) is pooled
    np.testing.assert_array_equal(parallel, _fit(method, workers=1, d=d))


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
    assert c.cutoff == cal._MAX_D * (cal._MAX_D - 1)
    assert max(calls) == cal._MAX_D


def test_bracket_when_the_pool_wins_down_to_the_smallest_size():
    c = cal.bracket(lambda d: 2.0, lambda d: 1.0, grid=(16, 32))
    assert (c.below, c.above) == (None, cal._MIN_D)
    assert c.cutoff == cal._MIN_D * (cal._MIN_D - 1)


def test_fits_never_run_the_calibrator(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a fit started calibration")

    for name in ("bracket", "calibrate", "fit_timers", "_time"):
        monkeypatch.setattr(cal, name, refuse)
    for method in METHODS:
        _fit(method, workers=2)


def test_calibrate_command_prints_the_settings(capsys, monkeypatch):
    seen = {}

    def fake(workers, methods):
        seen.update(workers=workers, methods=list(methods))
        return {m: cal.Crossover(32, 35, ((32, 0.2, 0.25), (35, 0.27, 0.26))) for m in methods}

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--method", "hc", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert seen == {"workers": 4, "methods": ["hc"]}
    assert report["settings"] == {"ANDREY_NUM_WORKERS": "4", "ANDREY_HC_PARALLEL_MIN_WORK": "1100"}
    assert report["methods"]["hc"]["pool_faster_at"] == {"d": 35, "work": 1190}
    assert report["schema_version"] == "2"


def test_calibrate_command_rejects_a_malformed_environment(capsys, monkeypatch):
    monkeypatch.setenv("ANDREY_NUM_WORKERS", "many")
    assert main(["calibrate", "--workers", "4"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "invalid_configuration" and "ANDREY_NUM_WORKERS" in error["message"]


@pytest.mark.parametrize("args", [["--workers", "1"], ["--method", "pc"]])
def test_calibrate_command_rejects_bad_options(args, capsys):
    assert main(["calibrate", *args]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsupported_value"
