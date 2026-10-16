"""Cross-link F1 of the window causal graph, and the rescore that adds it to a finished run.

Each score is checked on graphs small enough to count by hand, for every structure a time-series
lane returns: a lag stack, a window of marks, and a coefficient stack. The rescore is checked on a
real run folder: it reads the stored graphs and truths, fits nothing, and leaves out a truth with
unobserved series.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from andrey_bench import scoring
from andrey_bench.coefficients import CoefficientStack
from andrey_bench.contracts import temporal_from_links
from andrey_bench.window import ARROW, TAIL, WindowGraph

BENCH = Path(__file__).resolve().parents[1]


def _lags(links, n: int = 3, lags: int = 2):
    stack = np.zeros((lags, n, n), dtype=np.int8)
    for lag, i, j in links:
        stack[lag - 1, i, j] = 1
    return temporal_from_links(stack)


def _window(same=(), lagged=(), n: int = 3, lags: int = 1, lag0: str = "dag", undirected=()):
    marks = np.zeros((lags + 1, n, n), dtype=np.int8)
    for i, j in same:  # i -> j
        marks[0, i, j], marks[0, j, i] = TAIL, ARROW
    for i, j in undirected:
        marks[0, i, j] = marks[0, j, i] = TAIL
    for lag, i, j in lagged:
        marks[lag, i, j] = TAIL
    return WindowGraph(marks, lag0=lag0)


def test_lagged_cross_links_leave_autodependencies_out():
    truth = _lags([(1, 0, 1), (2, 1, 2), (1, 0, 0), (1, 2, 2)])
    estimate = _lags([(1, 0, 1), (1, 1, 2), (1, 1, 1)])
    scores = scoring.cross_link_scores(estimate, truth)
    assert (scores["cross_n_true"], scores["cross_n_est"], scores["cross_tp"]) == (2, 2, 1)
    assert scores["cross_precision"] == scores["cross_recall"] == scores["cross_f1"] == 0.5
    assert scores["cross_oriented_f1"] == 0.5  # lagged links are oriented by time


def test_a_same_step_link_counts_by_adjacency_and_its_direction_counts_apart():
    truth = _window(same=[(0, 1), (1, 2)], lagged=[(1, 2, 0)])
    reversed_ = _window(same=[(1, 0)], undirected=[(1, 2)], lagged=[(1, 2, 0)], lag0="cpdag")
    scores = scoring.cross_link_scores(reversed_, truth)
    assert scores["cross_f1"] == 1.0  # both same-step pairs are adjacent, the lagged link holds
    # Oriented: the reversed link is a miss and an extra, the unoriented one a miss.
    assert scores["cross_oriented_precision"] == 0.5
    assert scores["cross_oriented_recall"] == pytest.approx(1 / 3)


def test_the_empty_window_finds_nothing_and_claims_nothing():
    truth = _window(same=[(0, 1)], lagged=[(1, 1, 2)])
    scores = scoring.cross_link_scores(_window(), truth)
    assert scores["cross_recall"] == 0.0 and scores["cross_precision"] == 1.0
    assert scores["cross_f1"] == 0.0


def test_a_window_shorter_than_the_truth_misses_the_later_lags():
    truth = _window(lagged=[(1, 0, 1), (3, 1, 2)], lags=3)
    scores = scoring.cross_link_scores(_window(lagged=[(1, 0, 1)], lags=1), truth)
    assert (scores["cross_precision"], scores["cross_recall"]) == (1.0, 0.5)


def test_a_coefficient_stack_reads_effect_rows_and_cause_columns():
    stack = np.zeros((3, 3, 3))
    stack[0, 1, 0] = 0.5  # 0 -> 1 at the same step
    stack[1, 2, 1] = 0.3  # 1 at t-1 -> 2 at t
    stack[1, 0, 0] = 0.4  # an autodependency
    stack[2, 2, 2] = 0.2  # moving-average weights are not links of the window graph
    truth = CoefficientStack(stack, "varma")
    swapped = stack.copy()
    swapped[0] = swapped[0].T  # 1 -> 0 instead
    scores = scoring.cross_link_scores(CoefficientStack(swapped, "varma"), truth)
    assert scores["cross_n_true"] == 2 and scores["cross_f1"] == 1.0
    assert scores["cross_oriented_f1"] == 0.5


def test_a_panel_pools_its_time_points_from_the_second():
    truth = np.zeros((3, 2, 2, 2))
    truth[:, 0, 1, 0] = 1.0  # 0 -> 1 at every time point
    estimate = truth.copy()
    estimate[0] = np.nan  # not estimated
    estimate[2, 0] = 0.0  # missed at the last time point
    scores = scoring.cross_link_scores(
        CoefficientStack(estimate, "panel"), CoefficientStack(truth, "panel")
    )
    assert (scores["cross_n_true"], scores["cross_tp"]) == (2, 1)


def test_a_truth_with_unobserved_series_is_not_scored():
    truth = WindowGraph(np.zeros((2, 4, 4), np.int8), lag0="dag", latents=1)
    with pytest.raises(ValueError, match="unobserved"):
        scoring.cross_link_scores(WindowGraph(np.zeros((2, 3, 3), np.int8), "pag"), truth)


# --- the rescore -----------------------------------------------------------------------------------


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_rescore_scores_a_finished_run_from_its_stored_graphs(tmp_path: Path) -> None:
    driver, rescore = _script("pcmci_cpu"), _script("rescore_cross_links")
    out = tmp_path / "run"
    ours, floor = "andrey.pcmciplus.student_t.tau5", "baseline.empty.window.tau5"
    args = ["--out", str(out), "--method", "pcmciplus", "--smoke", "--solutions", ours, floor]
    assert driver.main(args) == 0

    scored = rescore.rescore(out).set_index("run_id")
    table = pd.read_parquet(out / "runs.parquet").set_index("name")
    assert set(scored.index) == set(table["run_id"])
    assert (out / rescore.OUTPUT).exists()
    ours_f1 = scored.loc[table.loc[ours, "run_id"], "cross_f1"]
    floor_row = scored.loc[table.loc[floor, "run_id"]]
    assert 0 < ours_f1 <= 1 and floor_row["cross_recall"] == 0.0
    assert floor_row["cross_n_true"] == scored.loc[table.loc[ours, "run_id"], "cross_n_true"]


def test_the_rescore_leaves_out_a_truth_with_unobserved_series(tmp_path: Path) -> None:
    driver, rescore = _script("pcmci_cpu"), _script("rescore_cross_links")
    out = tmp_path / "run"
    args = ["--out", str(out), "--method", "lpcmci", "--smoke"]
    args += ["--solutions", "andrey.lpcmci.student_t.tau3"]
    assert driver.main(args) == 0
    assert rescore.rescore(out).empty
