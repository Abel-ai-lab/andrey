"""The missing-value lanes: masked datasets, the baseline arms, the reference adapters, the driver.

The MissDAG and MVPC tests run the authors' code and skip unless its environment is built
(``benchmarks/start_missdag.sh``, ``benchmarks/start_mvpc_r.sh``); the bnlearn tests that fit skip
without R.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import replace

import missing_cpu
import missing_summary
import numpy as np
import pandas as pd
import pytest
from conftest import needs_r

from andrey_bench import datasets, scoring
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_structural_em import BnlearnStructuralEM
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


def _bnlearn_cells(seeds: int = 1) -> set[tuple]:
    """The masked cells of ``m2-bnlearn``, written out: m2-b, the 15-variable cells of m2-c, the
    cells of m2-reduced, and the larger-sample grid. A cell is ``(topology, density, d, n,
    standardize, mechanism, form, rate)``."""
    paper = [("er", 1.0), ("er", 2.0), ("scale_free", 2.0)]
    cells = {
        (g, dens, 15, 100, "raw", "mcar", "", r)
        for (g, dens) in paper
        for r in (0.1, 0.2, 0.3, 0.4, 0.5)
    }
    cells |= {
        (g, dens, 15, 100, "raw", m, "logistic", r)
        for (g, dens) in paper
        for m in ("mar", "self_masking")
        for r in (0.1, 0.3, 0.5)
    }
    cells |= {
        ("er", 2.0, 20, 500, "standardized", m, f, 0.2)
        for m, f in (("mcar", ""), ("mar", "logistic"))
    }
    cells |= {
        ("er", dens, d, n, "standardized", m, f, r)
        for dens, d, n, (m, f), r in itertools.product(
            (2.0, 4.0),
            (10, 20, 50),
            (500, 2000),
            (("mcar", ""), ("mar", "logistic")),
            (0.1, 0.2, 0.3),
        )
    }
    return cells


def test_the_bnlearn_lane_runs_each_cell_once_on_the_datasets_of_the_lanes_it_extends():
    units = missing_cpu.plan("m2-bnlearn", seeds=2)
    keys = [u.key for u in units]
    assert len(set(keys)) == len(keys)
    masked = [k for k in keys if k.missing]
    cells = {
        (k.topology, k.density, k.d, k.n, k.standardize, k.missing, k.missing_form, k.missing_rate)
        for k in masked
    }
    assert cells == _bnlearn_cells()
    assert len(masked) == 2 * len(cells)
    assert {k.seed for k in keys} == {0, 1}
    # Each masked dataset has its complete data in the plan, and the datasets of the lanes it
    # extends are the same files, so the shared store reuses them.
    complete = {k for k in keys if not k.missing}
    assert all(
        replace(k, missing="", missing_form="", missing_rate=0.0) in complete for k in masked
    )
    for lane in ("m2-b", "m2-reduced"):
        assert {u.key for u in missing_cpu.plan(lane, seeds=2)} <= set(keys)
    assert {u.key for u in missing_cpu.plan("m2-c", seeds=2) if u.key.d == 15} <= set(keys)


def test_the_bnlearn_lane_fits_bnlearn_beside_andrey_on_every_unit():
    units = missing_cpu.plan("m2-bnlearn", seeds=1)

    def names(d: int, masked: bool) -> list[str]:
        [arms] = {
            tuple(a.name for a in u.arms)
            for u in units
            if u.key.d == d and bool(u.key.missing) == masked
        }
        return list(arms)

    structural_em = [
        "bnlearn.structural_em.hc",
        "bnlearn.structural_em.tabu",
        "bnlearn.structural_em.hc.iter10",
        "bnlearn.structural_em.tabu.iter10",
    ]
    masked = names(15, True)
    # Andrey's hc and tabu run at the standard BIC bnlearn scores, as tabu_cpu.py pairs them.
    expected = [
        *structural_em,
        "andrey.pc.default",
        "andrey.hc.em.ebic_gamma0",
        "listwise.andrey.hc.default.ebic_gamma0",
        "impute_mean.andrey.hc.default.ebic_gamma0",
        "andrey.tabu.em.ebic_gamma0",
        "listwise.andrey.tabu.default.ebic_gamma0",
        "impute_mean.andrey.tabu.default.ebic_gamma0",
        "andrey.ges.em",
        "andrey.boss.em",
        "andrey.grasp.em",
        "andrey.xges.em",
        "andrey.exact_search.em",
        "baseline.empty",
    ]
    assert masked == expected
    assert names(10, True) == masked and names(20, True) == masked
    assert names(50, True) == [n for n in masked if n != "andrey.exact_search.em"]
    complete = names(15, False)
    assert complete == [
        "bnlearn.hc",
        "bnlearn.tabu",
        "andrey.hc.default.ebic_gamma0",
        "andrey.tabu.default.ebic_gamma0",
        "andrey.ges.default",
        "andrey.boss.default",
        "andrey.grasp.default",
        "andrey.xges.default",
        "andrey.exact_search.default",
        "baseline.empty",
    ]
    assert names(50, False) == [n for n in complete if n != "andrey.exact_search.default"]


def test_the_structural_em_arms_are_named_for_their_search_and_iterations():
    arms = {(m, i): BnlearnStructuralEM(m, i) for m in ("hc", "tabu") for i in (5, 10)}
    assert [a.name for a in arms.values()] == [
        "bnlearn.structural_em.hc",
        "bnlearn.structural_em.hc.iter10",
        "bnlearn.structural_em.tabu",
        "bnlearn.structural_em.tabu.iter10",
    ]
    params = arms["tabu", 10].params()
    assert json.loads(json.dumps(params)) == params
    assert params["max_iter"] == 10 and params["impute"] == "exact"
    assert arms["hc", 5].output_type == "dag" and arms["hc", 5].algorithm == "hc"
    with pytest.raises(ValueError, match="maximize"):
        BnlearnStructuralEM("ges")


@needs_r()
@pytest.mark.parametrize(
    "arm",
    [BnlearnStructuralEM(m, i) for m in ("hc", "tabu") for i in (5, 10)],
    ids=lambda a: a.name,
)
def test_structural_em_recovers_a_chain_from_data_with_missing_values(arm):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    for j in (1, 2, 3):
        X[:, j] += 1.5 * X[:, j - 1]
    X[rng.random(X.shape) < 0.1] = np.nan
    assert np.isnan(X).any()
    [measures] = run_task(arm, X, arm.params(), cap_wall_s=300, repeats=1, warmup=0)
    assert measures["status"] == "ok", measures.get("error_text")
    adjacency = np.asarray(measures["adj"])
    assert adjacency.shape == (4, 4) and set(np.unique(adjacency)) <= {0, 1}
    assert not (adjacency & adjacency.T).any() and not adjacency.diagonal().any()
    skeleton = adjacency | adjacency.T
    assert all(skeleton[j - 1, j] for j in (1, 2, 3))
    assert arm.to_structure(adjacency).kind == "dag"


def test_the_summary_pairs_hc_and_tabu_with_bnlearn_and_the_other_searches_with_its_hc():
    arms = missing_summary.arms_of
    em = "andrey.{}.em.ebic_gamma0"
    plain = "andrey.{}.default.ebic_gamma0"
    for method in ("hc", "tabu"):
        got = arms("m2-bnlearn", em.format(method))
        assert got == {
            "method": em.format(method),
            "testwise": "andrey.pc.default",
            "complete": plain.format(method),
            "reference": f"bnlearn.structural_em.{method}",
            "reference_iter10": f"bnlearn.structural_em.{method}.iter10",
            "listwise": f"listwise.{plain.format(method)}",
            "mean": f"impute_mean.{plain.format(method)}",
        }
    for method in ("ges", "boss", "grasp", "xges", "exact_search"):
        got = arms("m2-bnlearn", f"andrey.{method}.em")
        assert got["same_data"] == "bnlearn.structural_em.hc"
        assert "reference" not in got
    # Other lanes keep their references, and a bnlearn arm is no method arm.
    assert arms("m2-a", "andrey.notears.em")["reference"] == "missdag.miss_dag_gaussian.notears"
    assert "reference" not in arms("m2-b", "andrey.ges.em.ebic_gamma1")
    assert arms("m1-paper", "andrey.pc.mvpc.alpha0.01.seed0")["reference"] == "mvpc-r.permc"
    assert arms("m2-bnlearn", "bnlearn.structural_em.hc") is None


def _records(arm: str, seeds: int, output_type: str, shd: float, mec: float | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": arm,
            "data_seed": range(seeds),
            "status": "ok",
            "output_type": output_type,
            "shd": shd,
            "mec_shd": np.nan if mec is None else mec,
            "skeleton_f1": 0.8,
            "wall_s": 1.0,
            "complete_rows": 60.0,
        }
    )


def test_the_bnlearn_lane_compares_the_distance_between_equivalence_classes():
    """bnlearn's DAG is scored on ``mec_shd``, Andrey's CPDAG on ``shd``: with 3 against 2 the
    method is worse by 1, where the directed ``shd`` of the DAG (10) would call it better."""
    assert missing_summary.shd_column("m2-bnlearn") == "class_shd"
    assert missing_summary.shd_column("m2-b") == "shd"
    seeds = 12
    masked = pd.concat(
        [
            _records("andrey.hc.em.ebic_gamma0", seeds, "cpdag", 3.0, None),
            _records("andrey.ges.em", seeds, "cpdag", 4.0, None),
            _records("bnlearn.structural_em.hc", seeds, "dag", 10.0, 2.0),
            _records("bnlearn.structural_em.hc.iter10", seeds, "dag", 10.0, 3.0),
            _records("baseline.empty", seeds, "cpdag", 20.0, None),
        ]
    )
    whole = pd.concat(
        [
            _records("andrey.hc.default.ebic_gamma0", seeds, "cpdag", 2.0, None),
            _records("andrey.ges.default", seeds, "cpdag", 2.0, None),
            _records("bnlearn.hc", seeds, "dag", 9.0, 1.0),
        ]
    )
    masked["class_shd"], whole["class_shd"] = (
        missing_summary.class_shd(masked),
        missing_summary.class_shd(whole),
    )
    cell = {
        "lane": "m2-bnlearn",
        "graph": "er",
        "density": 2.0,
        "d": 15,
        "n": 100,
        "mechanism": "mcar",
        "form": "",
        "rate": 0.2,
    }
    rows = {r["arm"]: r for r in missing_summary.cell_rows(cell, masked, whole)}
    assert set(rows) == {"andrey.hc.em.ebic_gamma0", "andrey.ges.em"}
    hc = rows["andrey.hc.em.ebic_gamma0"]
    assert hc["shd_column"] == "class_shd"
    assert hc["reference"]["arm"] == "bnlearn.structural_em.hc"
    assert hc["reference"]["mean_shd"] == 2.0 and hc["reference"]["mean_D"] == 1.0
    assert hc["reference"]["verdict"] == "worse" and hc["judged"]
    assert hc["reference_iter10"]["mean_D"] == 0.0 and hc["reference_iter10"]["verdict"] == "tie"
    # The control: both plain searches on the complete data, 2 against bnlearn's 1.
    assert hc["control"]["other"] == "bnlearn.hc" and hc["control"]["mean_D"] == 1.0
    assert "same_data" not in hc
    ges = rows["andrey.ges.em"]
    same = ges["same_data"]
    assert same["package"] == "bnlearn" and same["arm"] == "bnlearn.structural_em.hc"
    assert same["mean_D"] == 2.0 and same["verdict"] == "worse"
    assert same["control"]["other"] == "bnlearn.hc" and same["control"]["mean_D"] == 1.0
    assert "reference" not in ges and not ges["judged"] and ges["result"] == "not judged"


def test_the_class_distance_is_the_cpdag_distance_of_every_arm():
    table = pd.DataFrame(
        {
            "output_type": ["dag", "cpdag", "dag", "cpdag"],
            "shd": [10.0, 3.0, 7.0, np.nan],
            "mec_shd": [2.0, np.nan, np.nan, np.nan],
        }
    )
    got = missing_summary.class_shd(table)
    assert got.iloc[:2].tolist() == [2.0, 3.0]
    assert got.iloc[2:].isna().all()  # a DAG with no CPDAG is never scored on its directed shd


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
