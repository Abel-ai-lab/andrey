"""The true time-series PAG over the observed series, on models small enough to work out by hand.

Each model lists its links as ``(lag, cause, effect)`` over all series, the unobserved ones last;
the expected PAG is derived from the model in each test's docstring.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from andrey_bench import scoring
from andrey_bench.window import ARROW, CIRCLE, NULL, TAIL, WindowGraph
from andrey_bench.window_pag import true_window_pag

BENCH = Path(__file__).resolve().parents[1]

#: Gerhardus and Runge (2020), Fig. 1: X, Y, Z observed, white-noise L hidden.
FIG1 = [(1, 0, 0), (1, 1, 1), (1, 2, 2), (0, 3, 0), (0, 3, 1), (1, 1, 2)]


def _truth(links, n: int, latents: int = 0, lags: int = 1) -> WindowGraph:
    marks = np.zeros((lags + 1, n, n), dtype=np.int8)
    for lag, i, j in links:
        marks[lag, i, j] = TAIL
        if lag == 0:
            marks[0, j, i] = ARROW
    return WindowGraph(marks, lag0="dag", latents=latents)


def _links(pag: WindowGraph) -> dict[tuple[int, int, int], tuple[int, int]]:
    """``(lag, i, j) -> (mark at X_i(t-lag), mark at X_j(t))`` for every link, lag 0 once."""
    out = {}
    for k, i, j in zip(*np.nonzero(pag.marks), strict=True):
        if k == 0 and i > j:
            continue
        later = ARROW if k else int(pag.marks[0, j, i])
        out[(int(k), int(i), int(j))] = (int(pag.marks[k, i, j]), later)
    return out


def test_independent_autoregressions_keep_their_self_links():
    """Two AR(1) series: the self-links only, oriented once two copies give R1 a triple."""
    truth = _truth([(1, 0, 0), (1, 1, 1)], n=2)
    assert _links(true_window_pag(truth, tau_max=2)) == {
        (1, 0, 0): (TAIL, ARROW),
        (1, 1, 1): (TAIL, ARROW),
    }
    # One copy of each self-link: nothing tells its earlier end apart from a hidden cause.
    assert _links(true_window_pag(truth, tau_max=1)) == {
        (1, 0, 0): (CIRCLE, ARROW),
        (1, 1, 1): (CIRCLE, ARROW),
    }


def test_hidden_common_cause_within_a_step_is_bidirected():
    """X, Y, Z observed, white-noise L hidden: L_t -> X_t, L_t -> Y_t, Y_{t-1} -> Z_t.

    Gerhardus and Runge (2020), Fig. 1: X_t <-> Y_t, the self-links, and Y_{t-1} -> Z_t.
    """
    pag = true_window_pag(_truth(FIG1, n=4, latents=1), tau_max=2)
    assert _links(pag) == {
        (0, 0, 1): (ARROW, ARROW),
        (1, 0, 0): (TAIL, ARROW),
        (1, 1, 1): (TAIL, ARROW),
        (1, 2, 2): (TAIL, ARROW),
        (1, 1, 2): (TAIL, ARROW),
    }


def test_hidden_mediator_leaves_a_direct_lagged_link():
    """X_{t-1} -> H_t -> Y_t with H hidden: X_{t-1} -> Y_t, oriented through X's self-link."""
    links = [(1, 0, 0), (1, 0, 2), (0, 2, 1)]
    pag = true_window_pag(_truth(links, n=3, latents=1), tau_max=2)
    assert _links(pag) == {(1, 0, 0): (TAIL, ARROW), (1, 0, 1): (TAIL, ARROW)}


def test_autocorrelated_hidden_cause_links_every_lag():
    """An AR(1) hidden L drives X and Y within a step: X and Y are confounded at every lag.

    No conditioning on observed series blocks the hidden chain ``L_{t-k} -> ... -> L_t``, so every
    pair of window nodes stays adjacent, self-links included, and no node is an ancestor of
    another, so no end of a link is a tail.
    """
    links = [(1, 2, 2), (0, 2, 0), (0, 2, 1)]
    pag = true_window_pag(_truth(links, n=3, latents=1), tau_max=2)
    found = _links(pag)
    assert set(found) == {(0, 0, 1)} | {(k, i, j) for k in (1, 2) for i in (0, 1) for j in (0, 1)}
    assert all(TAIL not in ends for ends in found.values())


def test_the_window_bounds_the_hidden_past():
    """A hidden past makes the earliest copies of a link more connected than the latest one.

    X_{t-2} -> H_{t-1} -> Y_t with H hidden and the observed X_{t-1} not on that path: at
    ``tau_max=1`` the cause X_{t-2} is outside the window, so the window has no X-Y link; at
    ``tau_max=2`` the link ``X_{t-2} -> Y_t`` appears.
    """
    links = [(1, 0, 2), (1, 2, 1)]
    narrow = true_window_pag(_truth(links, n=3, latents=1), tau_max=1)
    wide = true_window_pag(_truth(links, n=3, latents=1), tau_max=2)
    assert not narrow.marks.any()
    assert set(_links(wide)) == {(2, 0, 1)}


def test_a_dag_lag0_truth_is_required():
    with pytest.raises(ValueError):
        true_window_pag(WindowGraph(np.zeros((2, 2, 2), np.int8), lag0="pag"), tau_max=1)


def test_no_hidden_series_matches_the_directed_truth_where_oriented():
    """Without hidden series every link of the truth is found, and no oriented mark contradicts it."""
    links = [(1, 0, 0), (1, 1, 1), (1, 2, 2), (0, 0, 1), (2, 1, 2)]
    truth = _truth(links, n=3, lags=2)
    pag = true_window_pag(truth, tau_max=2)
    assert set(_links(pag)) == {(1, 0, 0), (1, 1, 1), (1, 2, 2), (0, 0, 1), (2, 1, 2)}
    oriented = np.isin(pag.marks, (TAIL, ARROW))
    assert np.array_equal(pag.marks[oriented], truth.marks[oriented])
    assert not np.any((pag.marks != NULL) & (truth.marks == NULL))


# --- scores against the true PAG ------------------------------------------------------------------


def test_a_truth_with_hidden_series_is_scored_against_its_true_pag():
    truth = _truth(FIG1, n=4, latents=1, lags=2)
    pag = true_window_pag(truth, tau_max=2)
    perfect = scoring.window_scores(pag, truth)
    assert perfect["endpoint_shd"] == perfect["wrong_marks"] == 0.0
    assert perfect["adjacency_f1"] == 1.0 and perfect["n_true_links"] == 5.0

    empty = scoring.window_scores(WindowGraph(np.zeros_like(pag.marks), lag0="pag"), truth)
    assert empty["endpoint_shd"] == 10.0 and empty["adjacency_recall"] == 0.0

    # A circle where the true PAG has a tail leaves one end open; it claims nothing wrong.
    marks = pag.marks.copy()
    marks[1, 1, 2] = CIRCLE
    opened = scoring.window_scores(WindowGraph(marks, lag0="pag"), truth)
    assert opened["endpoint_shd"] == 1.0 and opened["wrong_marks"] == 0.0
    # A tail at X_t on X_t <-> Y_t contradicts the true arrowhead.
    marks = pag.marks.copy()
    marks[0, 0, 1] = TAIL
    wrong = scoring.window_scores(WindowGraph(marks, lag0="pag"), truth)
    assert wrong["endpoint_shd"] == 1.0 and wrong["wrong_marks"] == 1.0


def test_the_rescore_scores_lpcmci_fits_against_the_true_pag(tmp_path: Path) -> None:
    def script(name: str):
        spec = importlib.util.spec_from_file_location(f"{name}_script", BENCH / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    driver, rescore = script("pcmci_cpu"), script("rescore_window_pag")
    out = tmp_path / "run"
    ours, floor = "andrey.lpcmci.student_t.tau3", "baseline.empty.window.tau3"
    args = ["--out", str(out), "--method", "lpcmci", "--smoke", "--solutions", ours, floor]
    assert driver.main(args) == 0

    scored = rescore.rescore(out).set_index("run_id")
    table = pd.read_parquet(out / "runs.parquet").set_index("name")
    assert set(scored.index) == set(table["run_id"])
    ours_row = scored.loc[table.loc[ours, "run_id"]]
    floor_row = scored.loc[table.loc[floor, "run_id"]]
    assert floor_row["adjacency_recall"] == 0.0
    assert floor_row["endpoint_shd"] == 2 * floor_row["n_true_links"]
    assert ours_row["endpoint_shd"] < floor_row["endpoint_shd"]
