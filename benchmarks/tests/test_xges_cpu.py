"""The XGES campaign: Andrey's adapter, the package's arm without the extended search, and the BIC
columns.

The campaign compares graphs by their BIC, so its claims rest on three facts checked here: every
arm searches under the one penalty, the ``bic`` column is the score Andrey's search reports for its
own graph, and every arm's ``bic_gap`` is measured against the same true graph. The package arms
need their own environment; the test that runs them skips when it has not been built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xges_cpu as driver

import andrey
from andrey.core.score import BICScore
from andrey_bench import datasets, rundir, runner
from andrey_bench.adapters.andrey_xges import AndreyXGES
from andrey_bench.adapters.xges_ges import PYTHON_ENV_VAR, XgesGES, xges_adapter
from andrey_bench.contracts import SolutionAdapter

XGES_PYTHON = Path(runner.BENCH_ROOT) / ".venv-xges" / "bin" / "python"

#: The Andrey arms and the floor: what a tiny run measures without the package's environment.
ANDREY_ARMS = [
    "andrey.xges.numpy",
    "andrey.xges.numpy.no_extended_search",
    "andrey.ges.numpy.serial.uncapped",
    "baseline.empty",
]
TINY = ["--ladder", "8", "--densities", "2", "4", "--seeds", "1", "--warmup", "0", "--cores", "1"]


def _dataset() -> np.ndarray:
    """A dense linear-Gaussian sample on which the extended search changes the graph."""
    key = driver.ladder_keys([8], [4.0], 1)[0]
    data, _marks, _kind = datasets.generate_one(key)
    return data


def test_the_test_dataset_separates_the_extended_search_from_the_base_search() -> None:
    """Otherwise an arm that ignored ``extended_search`` would pass the comparisons below."""
    data = _dataset()
    graphs = [
        andrey.xges(data, lambda_value=2.0, extended_search=flag).structure.to_numpy()
        for flag in (True, False)
    ]
    assert not np.array_equal(*graphs)


# --- the Andrey adapter --------------------------------------------------------------------------


def test_the_andrey_adapter_reports_one_search_per_name() -> None:
    extended = AndreyXGES()
    base = AndreyXGES(name="andrey.xges.numpy.no_extended_search", extended_search=False)
    for adapter in (extended, base):
        assert isinstance(adapter, SolutionAdapter)
        assert (adapter.package, adapter.backend, adapter.mode) == ("andrey", "numpy", "serial")
        assert adapter.output_type == "cpdag"
        assert adapter.params()["max_parents"] is None
    assert extended.params()["extended_search"] is True
    assert base.params()["extended_search"] is False
    assert AndreyXGES(lambda_value=2.0).params()["lambda_value"] == 2.0


@pytest.mark.parametrize("extended_search", [True, False])
@pytest.mark.parametrize("lambda_value", [1.0, 2.0])
def test_the_andrey_fit_is_the_public_xges(extended_search: bool, lambda_value: float) -> None:
    data = _dataset()
    adapter = AndreyXGES(extended_search=extended_search, lambda_value=lambda_value)
    marks = adapter.fit(data, adapter.params())
    public = andrey.xges(data, lambda_value=lambda_value, extended_search=extended_search)
    np.testing.assert_array_equal(marks, public.structure.to_numpy())
    assert adapter.to_structure(marks).kind == "cpdag"


def test_the_package_adapter_runs_no_extended_search_as_its_own_arm() -> None:
    assert XgesGES().params()["extended_search"] is True
    base = xges_adapter(name="xges.xges.no_extended_search", extended_search=False)
    assert base.params()["extended_search"] is False
    assert base.name == "xges.xges.no_extended_search"


# --- the field -----------------------------------------------------------------------------------


def test_every_arm_searches_under_the_campaign_penalty_with_no_parent_cap() -> None:
    field = {a.name: a for a in driver.solutions(2.0)}
    assert list(field) == [
        "andrey.xges.numpy",
        "andrey.xges.numpy.no_extended_search",
        "xges.xges",
        "xges.xges.no_extended_search",
        "andrey.ges.numpy.serial.uncapped",
        "baseline.empty",
    ]
    for name, adapter in field.items():
        params = adapter.params()
        if name == "baseline.empty":
            continue
        # The package's alpha is Andrey's lambda_value, as `andrey.xges` documents.
        assert params.get("lambda_value", params.get("alpha")) == 2.0, name
        assert params["max_parents"] is None, name
        if "xges" in name:
            assert params["extended_search"] is ("no_extended_search" not in name), name


def test_two_densities_are_two_datasets_at_ten_rows_per_node() -> None:
    keys = driver.ladder_keys([10, 20], [2.0, 4.0], seeds=2)
    assert len({k.digest for k in keys}) == len(keys) == 8
    assert all(k.n == 10 * k.d for k in keys)
    assert {k.n for k in driver.ladder_keys([10], [2.0], 1, n=500)} == {500}


def test_an_unknown_solution_name_stops_the_run(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        driver.main(["--out", "unused", "--solutions", "andrey.xges.numpy", "xges.xges.extended"])
    assert "unknown solutions" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seeds", "0"),
        ("--repeats", "0"),
        ("--warmup", "-1"),
        ("--n", "0"),
        ("--lambda-value", "-1"),
        ("--densities", "0"),
    ],
)
def test_a_setting_that_measures_nothing_stops_the_run(
    flag: str, value: str, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit):
        driver.main(["--out", "unused", flag, value])
    assert flag in capsys.readouterr().err


# --- one tiny run --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Andrey arms and the floor at d=8 on both densities, fitted once for the tests below."""
    out = tmp_path_factory.mktemp("xges-cpu") / "run"
    assert driver.main(["--out", str(out), *TINY, "--solutions", *ANDREY_ARMS]) == 0
    return out


def _table(out: Path) -> pd.DataFrame:
    return pd.read_parquet(out / "runs.parquet")


#: The tiny run's datasets by ``dataset_id``.
KEYS = {k.digest: k for k in driver.ladder_keys([8], [2.0, 4.0], 1)}


def test_every_unit_is_measured_once_and_scored(tiny_run: Path) -> None:
    table = _table(tiny_run)
    assert len(table) == len(ANDREY_ARMS) * 2
    assert set(table["status"]) == {"ok"}
    assert table["run_id"].is_unique
    assert table[list(driver.BIC_COLUMNS)].notna().all().all()
    # Both densities count toward completion, not just the first.
    status = json.loads((tiny_run / rundir.STATUS_NAME).read_text())
    assert status["units_expected"] == status["units_done"] == len(table)
    assert status["complete"] is True


def test_bic_is_the_score_andrey_xges_reports_for_its_graph(tiny_run: Path) -> None:
    table = _table(tiny_run)
    for row in table[table["name"] == "andrey.xges.numpy"].itertuples():
        data, _ = datasets.load(datasets.store_dir(tiny_run), KEYS[row.dataset_id])
        assert row.bic == pytest.approx(andrey.xges(data).metadata["score"], rel=1e-12)


def test_every_arm_is_measured_against_one_true_graph(tiny_run: Path) -> None:
    table = _table(tiny_run)
    for dataset_id, rows in table.groupby("dataset_id"):
        data, marks = datasets.load(datasets.store_dir(tiny_run), KEYS[dataset_id])
        truth = driver.total_bic(marks, BICScore(data))
        np.testing.assert_allclose(rows["bic"] - rows["bic_gap"], truth, rtol=1e-12)
        # The floor: on data generated from a graph, the empty graph scores worse than the truth.
        assert rows.loc[rows["name"] == "baseline.empty", "bic_gap"].item() > 0


def test_a_resume_measures_nothing_twice(tiny_run: Path, capsys: pytest.CaptureFixture) -> None:
    before = _table(tiny_run)
    parts = {p: p.stat().st_mtime_ns for p in (tiny_run / rundir.RECORDS_DIRNAME).glob("*")}
    assert (
        driver.main(["--out", str(tiny_run), *TINY, "--solutions", *ANDREY_ARMS, "--resume"]) == 0
    )
    assert f"0 records, {len(before)} units skipped" in capsys.readouterr().out
    after = _table(tiny_run)
    pd.testing.assert_frame_equal(after, before)
    assert {p: p.stat().st_mtime_ns for p in parts} == parts


def test_a_rescore_restores_the_bic_columns(tiny_run: Path) -> None:
    before = _table(tiny_run)
    for part in (tiny_run / rundir.RECORDS_DIRNAME).glob("*.parquet"):
        frame = pd.read_parquet(part)
        frame.drop(columns=list(driver.BIC_COLUMNS)).to_parquet(part, index=False)
    assert driver.main(["--out", str(tiny_run), *TINY, "--rescore"]) == 0
    after = _table(tiny_run)
    pd.testing.assert_frame_equal(after[before.columns], before)


def test_a_rescore_at_another_penalty_is_refused(tiny_run: Path) -> None:
    with pytest.raises(SystemExit, match="lambda-value"):
        driver.main(["--out", str(tiny_run), *TINY, "--rescore", "--lambda-value", "2"])


# --- the package, across the environment boundary ------------------------------------------------


@pytest.mark.skipif(not XGES_PYTHON.exists(), reason="xges-env not built - run start_xges.sh")
@pytest.mark.parametrize("extended_search", [True, False])
def test_the_package_finds_andreys_graph_at_the_matched_penalty(
    extended_search: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At ``alpha == lambda_value`` the two searches return one CPDAG on this dataset."""
    monkeypatch.setenv(PYTHON_ENV_VAR, str(XGES_PYTHON))
    data = _dataset()
    package = XgesGES(alpha=2.0, extended_search=extended_search)
    measures = runner.run_task(
        package, data, package.params(), cap_wall_s=600.0, repeats=1, warmup=0, threads=1
    )
    assert measures[0]["status"] == "ok", measures[0].get("error_text")
    theirs = package.to_structure(np.asarray(measures[0]["adj"])).to_numpy()
    ours = AndreyXGES(extended_search=extended_search, lambda_value=2.0)
    np.testing.assert_array_equal(theirs, ours.fit(data, ours.params()))
