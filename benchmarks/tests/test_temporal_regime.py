"""The lagged regime: its datasets, its truth, its scores, and the identities it must not move.

``var_lagged`` is the first regime whose truth is not one graph but a stack of lags. Adding it
touched the dataset store, the truth builder, the scorer, the stored estimates, and the record, so
the tests below cover each of those, and pin the digests of the regimes that existed before it:
a dataset whose digest moves is a different dataset to every record already measured.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from tiny_keys import TINY_KEYS

import andrey.data as data
from andrey import GraphStructure, TemporalStructure
from andrey_bench import datasets, rundir, scoring
from andrey_bench.adapters.baseline_empty import BaselineEmpty, baseline_empty_temporal
from andrey_bench.contracts import (
    RunRecord,
    lag_marks,
    links_from_temporal,
    temporal_from_lag_marks,
    temporal_from_links,
)
from andrey_bench.integration import _truth, run_batch, to_dataframe

_TAIL, _ARROW = 1, 2


def _lagged_key(**overrides) -> datasets.DatasetKey:
    topology, functional, noise, standardize = datasets.REGIMES["var_lagged"]
    coordinates = {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.TEMPORAL_REGIMES["var_lagged"],
        "d": 5,
        "n": 300,
        "seed": 0,
    }
    return datasets.DatasetKey(**{**coordinates, **overrides})


# --- identity --------------------------------------------------------------------------------------


def test_the_datasets_measured_before_the_lagged_regime_keep_their_identity():
    """Every digest and label of the existing regimes and the tiny keys, written out."""
    assert [(k.digest, k.label) for k in TINY_KEYS] == [
        ("8fc43b7bfa82", "er-linear-gaussian-deg2-d6-n200-s0-standardized"),
        ("5c97ae989c9b", "er-linear-gaussian-deg2-d6-n200-s1-standardized"),
        ("f94d1499d8f6", "er-linear-gaussian-deg2-d10-n200-s0-standardized"),
        ("4694426d2468", "er-linear-gaussian-deg2-d10-n200-s1-standardized"),
    ]
    digests = {}
    for regime, (topology, functional, noise, standardize) in datasets.REGIMES.items():
        if regime in datasets.TEMPORAL_REGIMES:
            continue
        key = datasets.DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=2.0,
            d=10,
            n=500,
            seed=0,
            latents=datasets.latents_for(regime, 10),
        )
        digests[regime] = key.digest
    assert digests == {
        "linear_gauss_er": "d25169e594f4",
        "lingam_sf": "830725644ac9",
        "nonlinear_hub": "839a214ce06a",
        "pnl_small_world": "e8b438a8b9ab",
        "latent_gauss_er": "72e95ac72b33",
        "latent_lingam_er": "a04ce0fceceb",
        "latent_anm_er": "ac7e8fa030c0",
    }


def test_a_lagged_key_is_a_sample_scm_call_of_its_own():
    key = _lagged_key()
    assert datasets.generation_kwargs(key) == {
        "graph": "erdos_renyi",
        "functional": "var",
        "noise": "gaussian",
        "d": 5,
        "n": 300,
        "seed": 0,
        "scale": "raw",
        "density": 0.0,
    }
    assert key.label == "er-var-gaussian-deg0-d5-n300-s0-raw"
    assert key.digest != _lagged_key(seed=1).digest != _lagged_key(n=301).digest


def test_the_lagged_datasets_keep_their_identity_beside_the_paper_regimes():
    """The digests the lagged regime had before the paper regimes existed."""
    assert _lagged_key().digest == "d74ca17b53ba"
    assert _lagged_key(d=20, n=1000, seed=2).digest == "e09ae500537f"


# --- the stored truth ------------------------------------------------------------------------------


def test_a_lagged_key_stores_its_lag_only_truth_as_lag_marks(tmp_path):
    key = _lagged_key()
    registry = datasets.materialize([key], tmp_path)
    entry = registry["keys"][key.label]
    assert entry["graph_kind"] == "temporal"
    assert entry["graph_shape"] == [1, key.d, key.d]
    assert entry["data_shape"] == [key.n, key.d]

    series, marks = datasets.load(tmp_path, key)
    truth = _truth(marks, entry["graph_kind"])
    generated = data.sample_scm(**datasets.generation_kwargs(key)).graph
    # The stored truth is the generator's lag 1, and its lag 0 had nothing to drop.
    assert truth.lags == (1,)
    assert truth.lag(1) == generated.lag(1)
    assert not np.asarray(generated.lag(0).to_numpy()).any()
    # Every series keeps its own past: the autoregressive self-loops are links.
    assert links_from_temporal(truth)[0].diagonal().all()


def test_a_lagged_key_with_instantaneous_edges_is_refused():
    """A lag-0 edge has no place in the stored stack, so the key fails instead of losing it."""
    with pytest.raises(ValueError, match="instantaneous graph has edges"):
        datasets.generate_one(_lagged_key(density=2.0))


# --- links and marks -------------------------------------------------------------------------------


def test_a_link_stack_survives_the_structure_both_ways():
    rng = np.random.default_rng(0)
    links = (rng.random((3, 6, 6)) < 0.3).astype(np.int8)
    links[0, 1, 2] = links[0, 2, 1] = 1  # both directions at one lag
    links[1, 4, 4] = 1  # a self-loop
    structure = temporal_from_links(links)
    assert structure.lags == (1, 2, 3)
    np.testing.assert_array_equal(links_from_temporal(structure), links)


def test_lag_marks_follow_the_endpoint_convention():
    """The mark at node i of edge i-j is marks[i, j]: a tail at the cause, an arrow at the effect."""
    links = np.zeros((1, 3, 3), dtype=np.int8)
    links[0, 0, 1] = 1  # X0(t-1) -> X1(t)
    links[0, 1, 2] = links[0, 2, 1] = 1  # both ways
    links[0, 2, 2] = 1  # X2's own past
    marks = lag_marks(links)[0]
    assert (marks[0, 1], marks[1, 0]) == (_TAIL, _ARROW)
    assert (marks[1, 2], marks[2, 1]) == (_ARROW, _ARROW)
    assert marks[2, 2] == _ARROW
    assert temporal_from_lag_marks(lag_marks(links)) == temporal_from_links(links)


def test_a_stack_holds_lags_one_to_l_only():
    g = GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="digraph")
    with pytest.raises(ValueError, match="lags 1..L"):
        links_from_temporal(TemporalStructure.from_lag_graphs([g, g], lags=[0, 1]))
    with pytest.raises(ValueError, match="shape"):
        lag_marks(np.zeros((2, 2)))


# --- scores ----------------------------------------------------------------------------------------


def _truth_with(links: list[tuple[int, int, int]], d: int = 4, lags: int = 2) -> TemporalStructure:
    stack = np.zeros((lags, d, d), dtype=np.int8)
    for lag, i, j in links:
        stack[lag - 1, i, j] = 1
    return temporal_from_links(stack)


def test_the_truth_scores_perfectly_and_the_empty_stack_scores_its_link_count():
    truth = _truth_with([(1, 0, 1), (1, 2, 3), (2, 1, 2)])
    perfect = scoring.score_result(truth, truth)
    assert perfect["family"] == "temporal"
    assert perfect["temporal_shd"] == 0.0
    assert perfect["lag1_f1"] == perfect["lag2_f1"] == perfect["summary_f1"] == 1.0

    empty = scoring.score_result(_truth_with([]), truth)
    assert empty["temporal_shd"] == 3.0
    assert (empty["lag1_shd"], empty["lag2_shd"]) == (2.0, 1.0)
    assert empty["lag1_recall"] == 0.0


def test_the_scores_are_lag_resolved():
    """A link at the wrong lag is a miss at one lag and a false link at the other, while the
    lag-merged summary graph does not see it."""
    truth = _truth_with([(1, 0, 1)])
    wrong_lag = scoring.score_result(_truth_with([(2, 0, 1)]), truth)
    assert wrong_lag["temporal_shd"] == 2.0
    assert (wrong_lag["lag1_recall"], wrong_lag["lag2_precision"]) == (0.0, 0.0)
    assert wrong_lag["summary_f1"] == 1.0
    # A link in the wrong direction is two differences at its lag.
    assert scoring.score_result(_truth_with([(1, 1, 0)]), truth)["lag1_shd"] == 2.0


def test_a_static_estimate_is_not_scored_against_a_lagged_truth():
    static = GraphStructure.from_numpy(np.zeros((4, 4), dtype=np.int8), kind="cpdag")
    with pytest.raises(TypeError):
        scoring.score_result(static, _truth_with([(1, 0, 1)]))


def test_the_structure_hash_tells_lags_apart():
    a, b = _truth_with([(1, 0, 1)]), _truth_with([(2, 0, 1)])
    assert scoring.structure_hash(a) == scoring.structure_hash(_truth_with([(1, 0, 1)]))
    assert scoring.structure_hash(a) != scoring.structure_hash(b)


def test_sortability_is_not_computed_on_a_series():
    assert scoring.sortability(np.zeros((10, 4)), _truth_with([(1, 0, 1)])) == {}


def test_a_lagged_estimate_survives_the_run_that_measured_it(tmp_path):
    estimate = _truth_with([(1, 0, 1), (2, 3, 3)])
    rundir.write_structure(tmp_path, "abc:andrey.pcmci.student_t.tau2:0:def", estimate)
    restored = rundir.read_structure(tmp_path, "abc:andrey.pcmci.student_t.tau2:0:def")
    assert restored == estimate
    assert scoring.structure_hash(restored) == scoring.structure_hash(estimate)


# --- the record ------------------------------------------------------------------------------------


def _required_fields() -> dict:
    """A placeholder for every `RunRecord` field without a default."""
    return {
        f.name: 0
        for f in dataclasses.fields(RunRecord)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }


def test_a_record_written_before_the_lagged_lane_stays_valid(tmp_path):
    """``tau_max`` defaults to ``None``: a record built without it, and a part written before the
    field existed, still assemble into one table with the lagged lane's rows."""
    assert RunRecord(**_required_fields()).tau_max is None

    old = to_dataframe([RunRecord(**{**_required_fields(), "name": "old"})])
    old.drop(columns="tau_max").to_parquet(tmp_path / "a.parquet", index=False)
    new = RunRecord(**{**_required_fields(), "name": "new", "tau_max": 2})
    rundir.write_part(tmp_path / "b.parquet", [new])
    table = rundir.assemble(tmp_path).set_index("name")
    assert np.isnan(table.loc["old", "tau_max"]) and table.loc["new", "tau_max"] == 2


def test_a_lagged_run_records_its_window_and_lag_resolved_scores(tmp_path):
    """Through the real pipeline: materialize, fit in a child, convert, score, record, flatten.

    The empty stack's lag-resolved SHD is exactly the truth's link count, so the scores are checked
    against a number known before the run.
    """
    key = _lagged_key(d=4, n=200)
    result = run_batch(
        tmp_path,
        [baseline_empty_temporal(2)],
        [key],
        machine_id="tiny-cpu",
        cap_wall_s=240.0,
        repeats=1,
        warmup=0,
        run_dir=tmp_path,
    )
    assert result.errors == []
    (record,) = result.records
    assert (record.status, record.output_type, record.tau_max) == ("ok", "temporal", 2)
    _, marks = datasets.load(tmp_path, key)
    true_links = int(links_from_temporal(_truth(marks, "temporal")).sum())
    assert record.metrics["temporal_shd"] == record.metrics["lag1_shd"] == true_links
    assert record.metrics["lag2_shd"] == 0.0

    table = to_dataframe(result.records)
    assert table["shd"].isna().all()  # the CPDAG column is not the lag-resolved one
    assert {"lag1_shd", "lag2_shd", "temporal_shd", "summary_f1"} <= set(table.columns)
    table.to_parquet(tmp_path / "runs.parquet", index=False)
    assert len(pd.read_parquet(tmp_path / "runs.parquet")) == 1
    assert rundir.read_structure(tmp_path, record.run_id).lags == (1, 2)


def test_a_static_solution_records_no_window(tmp_path):
    result = run_batch(
        tmp_path,
        [BaselineEmpty()],
        [k for k in TINY_KEYS if k.d == 6][:1],
        machine_id="tiny-cpu",
        cap_wall_s=240.0,
        repeats=1,
        warmup=0,
    )
    assert [r.tau_max for r in result.records] == [None]
