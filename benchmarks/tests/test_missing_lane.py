"""The missing-value lanes: masked datasets, the baseline arms, the reference adapters, the driver.

The MissDAG and MVPC tests run the authors' code and skip unless its environment is built
(``benchmarks/start_missdag.sh``, ``benchmarks/start_mvpc_r.sh``).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import missing_cpu
import numpy as np
import pytest
from conftest import needs_r

from andrey_bench import datasets, scoring
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.missing_arms import MIN_ROWS, AndreyMissing, Listwise, MeanImpute
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import run_task

COMPLETE = DatasetKey(
    topology="er",
    functional="linear",
    noise="gaussian",
    standardize="standardized",
    density=2.0,
    d=6,
    n=300,
    seed=1,
)
MAR = replace(COMPLETE, missing="mar", missing_form="quantile", missing_rate=0.2)


def test_a_key_without_missing_keeps_its_identity():
    """The new fields stay out of the label and the generator call of a key that leaves them unset."""
    assert "missing" not in datasets.generation_kwargs(COMPLETE)
    assert COMPLETE.label == "er-linear-gaussian-deg2-d6-n300-s1-standardized-shuffled"
    assert MAR.label == f"{COMPLETE.label}-mar-quantile-r0.2"
    assert MAR.digest != COMPLETE.digest


@pytest.mark.parametrize(
    "fields",
    [
        {"missing": "mcar", "missing_form": "logistic"},
        {"missing": "mar"},
        {"missing": "censored", "missing_form": "logistic"},
    ],
)
def test_a_mask_needs_a_known_mechanism_and_its_form(fields):
    with pytest.raises(ValueError):
        datasets.generation_kwargs(replace(COMPLETE, missing_rate=0.2, **fields))


def test_a_mask_is_drawn_over_sample_scm_samples_only():
    key = replace(MAR, functional="camuv2021", noise="uniform", standardize="raw")
    with pytest.raises(ValueError, match="sample_scm"):
        datasets.generation_kwargs(key)


def test_a_masked_dataset_holds_the_complete_values_and_records_its_m_graph(tmp_path):
    registry = datasets.materialize([COMPLETE, MAR], tmp_path)
    full, truth = datasets.load(tmp_path, COMPLETE)
    masked, masked_truth = datasets.load(tmp_path, MAR)
    absent = np.isnan(masked)
    assert 0.1 < absent.mean() < 0.3
    np.testing.assert_array_equal(masked[~absent], full[~absent])
    np.testing.assert_array_equal(masked_truth, truth)
    record = registry["keys"][MAR.label]["missing"]
    assert record["mechanism"] == "mar" and record["form"] == "quantile"
    assert record["complete_rows"] == int((~absent).all(axis=1).sum())
    # Each indicator's parents are always-observed columns, and only masked columns have parents.
    for column, parents in record["parents"].items():
        assert parents and not absent[:, parents].any()
        assert absent[:, int(column)].any() or not parents
    assert "missing" not in registry["keys"][COMPLETE.label]


def test_listwise_deletion_fits_the_complete_rows_and_fails_below_its_floor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    X[:10, 0] = np.nan
    seen = Listwise(_Recorder()).prepare(X)
    assert seen.shape == (10, 3) and not np.isnan(seen).any()
    X[: 20 - MIN_ROWS + 1, 1] = np.nan
    with pytest.raises(ValueError, match="listwise deletion kept"):
        Listwise(_Recorder()).prepare(X)


def test_mean_imputation_fills_continuous_means_and_categorical_modes():
    X = np.array([[1.0, 0.0], [3.0, 1.0], [np.nan, 1.0], [5.0, np.nan], [3.0, 0.0], [2.0, 1.0]])
    filled = MeanImpute(_Recorder(), categorical=[False, True]).prepare(X)
    assert filled[2, 0] == pytest.approx(np.nanmean(X[:, 0]))
    assert filled[3, 1] == 1.0
    assert MeanImpute(_Recorder()).name == "impute_mean.recorder"


def test_a_failed_listwise_fit_is_a_recorded_error_not_an_empty_graph(tmp_path):
    sparse = replace(MAR, n=12, missing="mcar", missing_form="", missing_rate=0.5)
    arm = Listwise(AndreyMissing("pc"))
    result = run_batch(tmp_path, [arm], [sparse], machine_id="test", cap_wall_s=120)
    [record] = result.records
    assert record.status == "error"
    assert "listwise deletion kept" in record.error_text
    assert not record.metrics


def test_mvpc_is_scored_on_the_parents_it_detects(tmp_path):
    arms = [AndreyMissing("pc", "mvpc", seed=0), AndreyMissing("pc"), BaselineEmpty()]
    result = run_batch(tmp_path, arms, [MAR], machine_id="test", cap_wall_s=120)
    by_name = {r.name: r for r in result.records}
    assert {r.status for r in result.records} == {"ok"}
    mvpc = by_name["andrey.pc.mvpc.seed0"].metrics
    assert {"mparent_precision", "mparent_recall"} <= set(mvpc)
    assert "mparent_recall" not in by_name["andrey.pc.default"].metrics


def test_parent_scores_count_detected_pairs_against_the_m_graph():
    detected = np.zeros((4, 4), dtype=np.int8)
    detected[0, 2] = detected[1, 2] = detected[3, 1] = 1
    scores = scoring.missingness_parent_scores(detected, {"2": [0], "1": [3, 0]})
    assert scores["mparent_precision"] == pytest.approx(2 / 3)
    assert scores["mparent_recall"] == pytest.approx(2 / 3)
    none = scoring.missingness_parent_scores(np.zeros((2, 2)), {"0": []})
    assert np.isnan(none["mparent_precision"]) and np.isnan(none["mparent_recall"])


@pytest.mark.parametrize("lane", ["m1-generic", "m2-generic", "m2-reduced"])
def test_every_masked_unit_has_its_complete_data_unit(lane):
    units = missing_cpu.plan(lane, seeds=2)
    keys = {u.key for u in units}
    masked = [u.key for u in units if u.key.missing]
    assert masked and all(
        replace(k, missing="", missing_form="", missing_rate=0.0) in keys for k in masked
    )
    assert all(any(isinstance(a, BaselineEmpty) for a in u.arms) for u in units)
    names = [a.name for u in units for a in u.arms]
    assert any(n.startswith("listwise.") for n in names)
    assert any(n.startswith("impute_mean.") for n in names)


def test_the_parts_of_a_lane_cover_its_units_once():
    units = missing_cpu.plan("m2-reduced", seeds=3)
    parts = [missing_cpu.plan("m2-reduced", seeds=3)[i::4] for i in range(4)]
    assert sorted(u.key.label for p in parts for u in p) == sorted(u.key.label for u in units)


class _Recorder:
    """A stand-in solution adapter that returns the data it fits."""

    name = "recorder"
    package = package_version = backend = mode = algorithm = output_type = "stub"

    def params(self):
        return {}

    def fit(self, data, params):
        return data


def _missdag_env() -> bool:
    return bool(
        os.environ.get("ANDREY_BENCH_MISSDAG") and os.environ.get("ANDREY_BENCH_MISSDAG_PYTHON")
    )


@pytest.mark.skipif(not _missdag_env(), reason="MissDAG is not built (start_missdag.sh)")
def test_missdag_fits_a_dag_on_data_with_missing_values():
    from andrey_bench.adapters.missdag import MissDagGaussian, MissDagNotears

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    X[:, 1] += 1.5 * X[:, 0]
    X[:, 3] += -1.5 * X[:, 1]
    complete = X.copy()
    X[rng.random(X.shape) < 0.1] = np.nan
    for adapter, data in ((MissDagGaussian(), X), (MissDagNotears(), complete)):
        params = {**adapter.params(), **({"em_iter": 2} if "em_iter" in adapter.params() else {})}
        [measures] = run_task(adapter, data, params, cap_wall_s=300, repeats=1, warmup=0)
        assert measures["status"] == "ok", measures.get("error_text")
        adj = np.asarray(measures["adj"])
        assert adj.shape == (4, 4) and adj.sum() >= 1
        assert adapter.to_structure(adj).kind == "dag"


def _mvpc_env() -> bool:
    return bool(os.environ.get("ANDREY_BENCH_MVPC_R"))


@needs_r()
@pytest.mark.skipif(not _mvpc_env(), reason="the MVPC authors' code is absent (start_mvpc_r.sh)")
def test_the_authors_mvpc_reads_the_lane_files_and_reports_detected_parents(tmp_path):
    from andrey_bench.adapters.mvpc_r import MvpcR

    key = DatasetKey(
        topology="er",
        functional=datasets.MVPC_PAPER,
        noise="gaussian",
        standardize="raw",
        density=2.0,
        d=20,
        n=1000,
        seed=3,
        missing="mar",
        missing_form="quantile",
    )
    complete = replace(key, missing_form="complete")
    registry = datasets.materialize([key, complete], tmp_path)
    masked, truth = datasets.load(tmp_path, key)
    full, _ = datasets.load(tmp_path, complete)
    absent = np.isnan(masked)
    assert absent.any() and not np.isnan(full).any()
    np.testing.assert_array_equal(masked[~absent], full[~absent])
    parents = registry["keys"][key.label]["missing"]["parents"]
    assert parents and all(absent[:, int(c)].any() for c in parents)
    assert json.dumps(registry["keys"][key.label]["generation_call"])
    arms = [MvpcR("permc"), MvpcR("testwise")]
    result = run_batch(tmp_path, arms, [key], machine_id="test", cap_wall_s=600)
    by_name = {r.name: r for r in result.records}
    assert {r.status for r in result.records} == {"ok"}, result.errors
    assert "mparent_recall" in by_name["mvpc-r.permc"].metrics
    assert "mparent_recall" not in by_name["mvpc-r.testwise"].metrics
