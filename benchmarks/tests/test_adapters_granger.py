"""The Granger adapters, the series they run on, and the lagged-edge scores.

Every adapter returns the coefficients in one layout, so a lagged edge reads the same from either
package. The scores are checked on small stacks whose answer is known.
"""

from __future__ import annotations

import json
import pickle

import granger_var
import numpy as np
import pytest

from andrey_bench.adapters.andrey_granger import andrey_granger_adapters
from andrey_bench.adapters.causal_learn_granger import CausalLearnGranger

SERIAL, PARALLEL = andrey_granger_adapters()
ADAPTERS = [SERIAL, PARALLEL, CausalLearnGranger()]


def test_both_packages_report_one_method_the_same_way() -> None:
    andrey_adapter, other = SERIAL, ADAPTERS[-1]
    for field in ("algorithm", "output_type", "mode"):
        assert getattr(andrey_adapter, field) == getattr(other, field), field
    assert andrey_adapter.params() == other.params()


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_an_adapter_survives_the_record_and_the_worker(adapter) -> None:
    assert json.loads(json.dumps(adapter.params(), sort_keys=True)) == adapter.params()
    assert pickle.loads(pickle.dumps(adapter)).name == adapter.name


@pytest.fixture(scope="module")
def series() -> tuple[np.ndarray, np.ndarray]:
    """A standardized series from the driver's generator, where both packages fit one model."""
    data, truth = granger_var.sample(d=4, n=400, seed=0)
    return (data - data.mean(axis=0)) / data.std(axis=0), truth


def test_both_packages_return_the_same_coefficients_on_standardized_data(series) -> None:
    data, _ = series
    natives = []
    for adapter in (SERIAL, ADAPTERS[-1]):
        adapter.setup()
        natives.append(np.asarray(adapter.fit(data, {"maxlag": 2})))
    assert natives[0].shape == (4, 8)
    np.testing.assert_allclose(natives[0], natives[1], atol=1e-8)


def test_the_parallel_variant_returns_the_serial_coefficients(series, monkeypatch) -> None:
    data, _ = series
    monkeypatch.setenv("ANDREY_N_JOBS", "3")
    assert PARALLEL.mode == "parallel" and SERIAL.mode == "serial"
    assert np.array_equal(PARALLEL.fit(data, {"maxlag": 2}), SERIAL.fit(data, {"maxlag": 2}))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_the_structure_holds_the_nonzero_coefficients_as_lagged_edges(adapter, series) -> None:
    data, _ = series
    native = np.asarray(adapter.fit(data, {"maxlag": 2}))
    structure = adapter.to_structure(native)
    assert structure.type == "temporal" and structure.lags == (1, 2)
    support = granger_var.lag_support(native)
    for k in structure.lags:
        edges = {(i, j) for i, j, _ in structure.lag(k).oriented_edges()}
        assert edges == {(int(i), int(j)) for i, j in zip(*np.nonzero(support[k - 1]))}


def test_a_series_has_lagged_effects_only_and_every_variable_its_own_past(series) -> None:
    data, truth = series
    assert data.shape == (400, 4) and truth.shape == (granger_var.LAGS, 4, 4)
    assert truth[0].diagonal().all()  # sample() raises on instantaneous effects


def test_the_layout_reads_lag_k_cause_j_effect_i() -> None:
    coeff = np.zeros((2, 4))
    coeff[1, 0] = 0.5  # x0(t-1) -> x1(t)
    coeff[0, 3] = -0.2  # x1(t-2) -> x0(t)
    support = granger_var.lag_support(coeff)
    assert support.shape == (2, 2, 2)
    assert {tuple(map(int, e)) for e in np.argwhere(support)} == {(0, 0, 1), (1, 1, 0)}


def test_lag_scores_count_every_lag_and_leave_self_loops_out_of_cross() -> None:
    truth = np.zeros((2, 3, 3), dtype=bool)
    truth[0, 0, 0] = truth[0, 0, 1] = truth[1, 1, 2] = True
    perfect = granger_var.lag_scores(truth, truth)
    assert (perfect["f1"], perfect["cross_f1"], perfect["cross_n_true"]) == (1.0, 1.0, 2.0)
    one_lag = truth[:1].copy()  # misses lag 2; padded rather than misaligned
    scores = granger_var.lag_scores(one_lag, truth)
    assert (scores["precision"], scores["recall"]) == (1.0, 2 / 3)
    empty = granger_var.lag_scores(np.zeros_like(truth), truth)
    assert np.isnan(empty["precision"]) and empty["recall"] == 0.0 and empty["f1"] == 0.0


def test_a_run_drawn_in_another_column_order_is_not_rescored(tmp_path) -> None:
    """A run's settings name its column order, so its stored fits meet only the truth they had."""
    meta = {
        "driver": "granger_var.py",
        "lags": granger_var.LAGS,
        "lag_degree": granger_var.LAG_DEGREE,
        "input": granger_var.INPUT,
    }
    (tmp_path / "run_meta.json").write_text(json.dumps(meta))
    (tmp_path / granger_var.RECORDS).write_text("")
    with pytest.raises(SystemExit, match="other process settings"):
        granger_var.rescore(tmp_path)
    (tmp_path / "run_meta.json").write_text(json.dumps({**meta, "shuffle": granger_var.SHUFFLE}))
    assert granger_var.rescore(tmp_path) == []
