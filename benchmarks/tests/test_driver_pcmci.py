"""The PCMCI driver: its field, its rungs, the series it refuses, and one smoke run end to end."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey_bench import datasets

BENCH = Path(__file__).resolve().parents[1]


def _driver():
    """Import `pcmci_cpu.py` by path - it is a script beside the package, not inside it."""
    spec = importlib.util.spec_from_file_location("pcmci_cpu_driver", BENCH / "pcmci_cpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _driver()


def test_the_field_is_both_andrey_tests_tigramite_and_the_floor() -> None:
    names = [a.name for a in driver.solutions(2)]
    assert names == [
        "andrey.pcmci.student_t.tau2",
        "andrey.pcmci.fisherz.tau2",
        "tigramite.pcmci.parcorr.tau2",
        "baseline.empty.temporal.tau2",
    ]
    assert len(set(names)) == len(names)


def test_the_rungs_are_every_size_at_every_length_on_the_lagged_regime() -> None:
    keys = driver.ladder_keys([5, 10], [200, 500], seeds=2)
    assert [(k.d, k.n, k.seed) for k in keys] == [
        (d, n, s) for d in (5, 10) for n in (200, 500) for s in (0, 1)
    ]
    assert {(k.functional, k.density, k.standardize) for k in keys} == {("var", 0.0, "raw")}
    assert driver.SLICE["functional"] == datasets.REGIMES[driver.REGIME][1]


def test_the_shortest_series_is_the_shortest_andrey_accepts() -> None:
    d, tau_max = 3, 2
    shortest = driver.min_length(d, tau_max)
    x = np.random.default_rng(0).standard_normal((shortest, d))
    andrey.pcmci(x, tau_max=tau_max)
    with pytest.raises(ValueError):
        andrey.pcmci(x[:-1], tau_max=tau_max)


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--ladder", "10", "--lengths", "40"], "too short"),
        (["--smoke", "--solutions", "andrey.pcmci.studentt.tau2"], "unknown solutions"),
        (["--ladder", "5"], "required"),
    ],
)
def test_a_run_that_cannot_measure_what_it_names_stops_before_it_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture, flags: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        driver.main(["--out", str(tmp_path / "run"), *flags])
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_the_smoke_run_writes_a_scored_table(tmp_path: Path) -> None:
    """The Andrey solutions and the floor; the tigramite fit is checked in its own test file."""
    out = tmp_path / "run"
    names = ["andrey.pcmci.student_t.tau2", "andrey.pcmci.fisherz.tau2"]
    floor = "baseline.empty.temporal.tau2"
    code = driver.main(["--out", str(out), "--smoke", "--solutions", *names, floor])
    assert code == 0

    rows = pd.read_parquet(out / "runs.parquet").set_index("name")
    assert sorted(rows.index) == sorted([*names, floor])
    assert set(rows["status"]) == {"ok"}
    assert set(rows["tau_max"]) == {2}
    assert (rows.loc[names, "temporal_shd"] <= rows.loc[floor, "temporal_shd"]).all()
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["ladder"] == [[5, 200]]
    status = json.loads((out / "STATUS.json").read_text())
    assert status["complete"] and status["units_done"] == 3


def test_each_method_runs_its_own_field_on_its_own_regime() -> None:
    fields = {m: [a.name for a in driver.METHODS[m].solutions(1)] for m in driver.METHODS}
    assert fields["pcmciplus"] == [
        "andrey.pcmciplus.student_t.tau1",
        "tigramite.pcmciplus.parcorr.tau1",
        "baseline.empty.window.tau1",
    ]
    assert fields["lpcmci"] == [
        "andrey.lpcmci.student_t.tau1",
        "tigramite.lpcmci.parcorr.tau1",
        "baseline.empty.window.tau1",
    ]
    assert driver.METHODS["pcmci"].regime == "var_lagged"
    for method in ("pcmciplus", "lpcmci"):
        assert driver.METHODS[method].regime == "var_same_step"
        assert driver.METHODS[method].tau_max == 1
    keys = driver.ladder_keys([5], [200], seeds=1, regime="var_same_step")
    assert [k.density for k in keys] == [datasets.TEMPORAL_REGIMES["var_same_step"]]


@pytest.mark.parametrize("method", ["pcmci", "pcmciplus", "lpcmci"])
def test_each_method_refuses_only_the_series_andrey_refuses(method: str) -> None:
    d, tau_max = 3, 2
    shortest = driver.METHODS[method].min_length(d, tau_max)
    x = np.random.default_rng(0).standard_normal((shortest, d))
    facade = getattr(andrey, method)
    facade(x, tau_max=tau_max)
    with pytest.raises(ValueError):
        facade(x[:-1], tau_max=tau_max)


@pytest.mark.parametrize("method", ["pcmciplus", "lpcmci"])
def test_a_same_step_smoke_run_writes_a_window_scored_table(tmp_path: Path, method: str) -> None:
    out = tmp_path / "run"
    ours, floor = f"andrey.{method}.student_t.tau1", "baseline.empty.window.tau1"
    code = driver.main(
        ["--out", str(out), "--method", method, "--smoke", "--solutions", ours, floor]
    )
    assert code == 0

    rows = pd.read_parquet(out / "runs.parquet").set_index("name")
    assert set(rows["status"]) == {"ok"} and set(rows["tau_max"]) == {1}
    assert set(rows["family"]) == {"temporal_window"}
    assert rows.loc[ours, "endpoint_shd"] < rows.loc[floor, "endpoint_shd"]
    assert rows.loc[floor, "endpoint_shd"] == 2 * rows.loc[floor, "n_true_links"]
    meta = json.loads((out / "run_meta.json").read_text())
    assert (meta["method"], meta["regime"], meta["tau_max"]) == (method, "var_same_step", 1)
