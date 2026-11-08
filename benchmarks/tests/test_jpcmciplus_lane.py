"""The J-PCMCI+ lane: its simulation, its patch, its Andrey adapters, its scores, and its rows.

Runs without tigramite: the reference arms need its environment and run in the lane's own jobs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import jpcmciplus_cpu as lane
import numpy as np
import pytest

from andrey_bench.adapters.andrey_jpcmciplus import AndreyJPCMCIPlus, AndreyPooledPCMCIPlus
from andrey_bench.adapters.tigramite_jpcmciplus import apply_patch
from andrey_bench.context_models import (
    AUTO_WEIGHTS,
    NOISE_SD,
    WEIGHTS,
    breaks_assumption_3,
    jpcmciplus2023,
    jpcmciplus2023_nonlinear,
)

N, TIME, SPACE = 5, (5, 6), (7, 8)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("observed", [1.0, 0.5])
def test_the_draw_follows_the_authors_design(seed, observed):
    d = jpcmciplus2023(N, 4, 30, observed, seed)
    by_effect: dict[int, list] = {}
    for cause, effect, lag, weight, function in d.model:
        by_effect.setdefault(effect, []).append((cause, lag, weight))
        assert function == "linear"
    system = [link for link in d.model if link[1] < N and link[0] < N]
    cross = [link for link in system if link[0] != link[1]]
    assert len(cross) == N and len({(c, e) for c, e, *_ in cross}) == N
    assert all(1 <= lag <= 2 and w in WEIGHTS for _, _, lag, w, _ in cross)
    for j in (*range(N), *TIME):  # every system variable and time context has its own past
        assert [w for c, lag, w in by_effect[j] if c == j and lag == 1][0] in AUTO_WEIGHTS
    between = {kind: [link for link in d.model if link[0] in kind and link[1] in kind and
                      link[0] != link[1]] for kind in (TIME, SPACE)}  # fmt: skip
    assert len(between[TIME]) == 1 and 1 <= between[TIME][0][2] <= 2
    assert len(between[SPACE]) == 1 and between[SPACE][0][2] == 0
    from_contexts = [link for link in d.model if link[0] >= N and link[1] < N]
    assert len(from_contexts) in (8, 10)  # twice the context slots that are no context's parent
    assert all(link[2] == 0 for link in from_contexts if link[0] in SPACE)
    assert set(d.noise_sd) <= set(NOISE_SD)
    k = int(observed * 2)
    assert len(d.time_contexts) == len(d.space_contexts) == k
    assert d.datasets.shape == (4, 30, N + 2 * k)
    x = d.datasets
    for c in d.time_contexts:
        assert np.array_equal(x[0, :, c], x[-1, :, c])  # one series in every dataset
    for c in d.space_contexts:
        assert np.all(x[:, :, c] == x[:, :1, c])  # fixed within a dataset
    assert observed == 0.5 or not d.breaks_assumption_3


def test_the_cross_links_follow_one_causal_order():
    for seed in range(20):
        d = jpcmciplus2023(10, 2, 10, 1.0, seed)
        edges = {(c, e) for c, e, *_ in d.model if c < 10 and e < 10 and c != e}
        reach = {v: {e for c, e in edges if c == v} for v in range(10)}
        for _ in range(10):
            reach = {v: reach[v] | set().union(*(reach[u] for u in reach[v])) for v in reach}
        assert not any(v in reach[v] for v in reach)


def test_the_true_links_are_over_the_observed_columns():
    d = jpcmciplus2023(N, 3, 20, 0.5, 2)
    observed = {node: N + k for k, node in enumerate(d.observed)}
    expected = {(observed[c], lag, e) for c, e, lag, *_ in d.model if c in observed and e < N}
    assert set(d.context_links) == expected
    assert set(d.links) == {(c, e, lag) for c, e, lag, *_ in d.model if c < N and e < N}


def test_the_nonlinear_model_has_one_context_of_each_kind():
    d = jpcmciplus2023_nonlinear(5, 20, 0)
    assert d.datasets.shape == (5, 20, 4)
    assert (d.time_contexts, d.space_contexts) == ((2,), (3,))
    assert set(d.links) == {(1, 0, 1), (1, 1, 1)}
    assert set(d.context_links) == {(2, 1, 0), (2, 1, 1), (3, 0, 0), (3, 0, 1)}


def _model(links: list[tuple[int, int]]) -> dict:
    """A model over two system variables and contexts 2 and 3 from (cause, effect) pairs."""
    model = {j: [] for j in range(4)}
    for cause, effect in links:
        model[effect].append(((cause, -1), 0.5, "linear"))
    return model


@pytest.mark.parametrize(
    ("links", "observed", "broken"),
    [
        ([(2, 3), (2, 0), (3, 1)], {3}, True),  # hidden 2 drives observed 3 and variable 0
        ([(2, 3), (3, 1)], {3}, False),  # hidden 2 reaches the system through 3 only
        ([(3, 2), (2, 0)], {3}, False),  # observed 3 -> hidden 2 -> 0: a mediator, not Assumption 3
        ([(2, 3), (2, 0)], {2, 3}, False),  # nothing hidden
    ],
)
def test_assumption_3_is_a_hidden_context_confounding_an_observed_one(links, observed, broken):
    assert breaks_assumption_3(_model(links), 2, observed) is broken


def test_about_half_the_half_observed_draws_break_assumption_3():
    """The rate the cells' 40 seeds are sized for."""
    rate = np.mean([jpcmciplus2023(N, 2, 10, 0.5, s).breaks_assumption_3 for s in range(100)])
    assert 0.35 < rate < 0.65


def test_the_patch_applies_to_its_hunk_once_and_fails_elsewhere():
    source = "def f(x):\n    a = 1\n\n    return x + a\n"
    # The empty line is a context line whose leading space an editor stripped.
    patch = (
        "header\n--- a/f.py\n+++ b/f.py\n@@ -1,4 +1,4 @@\n"
        " def f(x):\n-    a = 1\n+    a = 2\n\n     return x + a\n"
    )
    assert apply_patch(source, patch) == source.replace("a = 1", "a = 2")
    with pytest.raises(RuntimeError, match="matches 0 places"):
        apply_patch(source.replace("a = 1", "a = 3"), patch)
    with pytest.raises(RuntimeError, match="no hunk"):
        apply_patch(source, "header only\n")


def test_the_andrey_adapters_return_window_marks_over_their_columns():
    d = jpcmciplus2023(N, 4, 60, 1.0, 1)
    columns = {"time_contexts": list(d.time_contexts), "space_contexts": list(d.space_contexts)}
    marks = AndreyJPCMCIPlus().fit(d.datasets, {**AndreyJPCMCIPlus().params(), **columns})
    assert marks.shape == (3, 9, 9)
    found = lane.found_context_links(marks, d)
    assert found and all(
        marks[0, j, c] == 2 for c, k, j in found if k == 0
    )  # arrow at the variable
    assert not marks[:, N:, N:].any()  # no link between contexts
    pooled = AndreyPooledPCMCIPlus()
    assert pooled.fit(d.datasets, {**pooled.params(), "n_system": N}).shape == (3, N, N)


def _data(links, context_links=frozenset()) -> lane.Dataset:
    cell = lane.CELLS["all-M10-T10"]
    draw = SimpleNamespace(
        n_system=2, links=frozenset(links), context_links=frozenset(context_links),
        time_contexts=(2,), space_contexts=(),
    )  # fmt: skip
    data = lane.Dataset(cell, 0, draw, None)
    same, lagged = np.zeros((2, 2), bool), np.zeros((2, 2, 2), bool)
    for c, e, lag in links:
        (lagged[lag - 1] if lag else same)[c, e] = True
    data.truth = lane.WindowGraph(lane.marks_from_edges(same, lagged), lag0="dag")
    return data


def test_the_scores_count_marks_and_context_links():
    data = _data({(0, 1, 1)}, {(2, 0, 0), (2, 1, 1)})
    est = np.zeros((3, 3, 3), dtype=np.int8)
    est[1, 0, 1] = 1  # the true X0(t-1) -> X1(t)
    est[0, 0, 1], est[0, 1, 0] = 1, 1  # an extra unoriented same-step link
    est[0, 2, 0], est[0, 0, 2] = 1, 2  # the context link at lag 0, found
    est[2, 2, 1] = 1  # a context link at lag 2 where the truth has lag 1
    s = lane.scores(est, data, contexts=True)
    assert (s["shd"], s["shd_lag0"], s["shd_lagged"]) == (2, 2, 0)
    assert (s["context_hits"], s["context_misses"], s["context_extras"]) == (1, 1, 1)
    assert s["context_error"] == 2
    empty = lane.scores(None, data, contexts=True)
    assert (empty["shd"], empty["context_error"]) == (2, 2)


@pytest.mark.parametrize(
    ("arm", "stderr", "status"),
    [
        ("andrey", "Traceback\n  ...\nValueError: jpcmciplus: no degree of freedom\n", "refused"),
        ("andrey", "Traceback\nRuntimeError: boom\n", "error"),
        ("tigramite", "Traceback\nValueError: Inconsistent link assumptions\n", "error"),
    ],
)
def test_only_an_andrey_value_error_is_a_refusal(arm, stderr, status):
    assert lane._status(arm, {"status": "error", "stderr": stderr}) == status


def _write(tmp_path, cell: str, seeds, values, broken=(), stock=None):
    """Datasets and records of `cell`: Andrey's error is ``values[0]``, tigramite's ``values[1]``,
    the empty graph's ``values[2]``; a seed in `broken` breaks Assumption 3."""
    out = tmp_path / cell
    out.mkdir()
    rng = np.random.default_rng(0)
    with (out / "datasets.jsonl").open("w") as ds, (out / "records.jsonl").open("w") as rec:
        for s in seeds:
            noise = float(rng.normal())
            empty = {"shd": values[2], "context_error": values[2]}
            ds.write(json.dumps({"cell": cell, "seed": s, "breaks_assumption_3": s in broken,
                                 "empty": empty, "data_sha1": str(s)}) + "\n")  # fmt: skip
            for arm, value in (("andrey", values[0]), ("tigramite", values[1])):
                for alpha in lane.ALPHAS:
                    r = {"cell": cell, "seed": s, "arm": arm, "alpha": alpha, "status": "ok",
                         "shd": value + noise, "context_error": value + noise, "wall_s": 1.0,
                         "task_s": 2.0, "marks_sha1": f"{arm}{s}", "build": {}}  # fmt: skip
                    rec.write(json.dumps(r) + "\n")
            if stock is not None:
                sha = f"tigramite{s}" if s in stock else "other"
                r = {"cell": cell, "seed": s, "arm": "tigramite.stock", "alpha": 0.05,
                     "status": "ok", "marks_sha1": sha, "wall_s": 1.0, "task_s": 2.0,
                     "build": {}}  # fmt: skip
                rec.write(json.dumps(r) + "\n")
    return out


def _summary(tmp_path, *runs, replications=()):
    args = SimpleNamespace(
        runs=[str(r) for r in runs],
        replications=[str(r) for r in replications] or None,
        out=str(tmp_path / "summary.json"),
    )
    lane.summarize(args)
    return json.loads((tmp_path / "summary.json").read_text())


def _row(summary, regime):
    return next(r for r in summary["rows"] if r["regime"] == regime)


@pytest.mark.parametrize(("andrey", "result"), [(5.0, "tie"), (3.0, "better"), (7.0, "worse")])
def test_a_claimed_row_is_judged_by_its_paired_interval(tmp_path, andrey, result):
    run = _write(tmp_path, "all-M10-T100", range(20), (andrey, 5.0, 12.0))
    row = _row(_summary(tmp_path, run), "all-M10-T100|shd|alpha=0.05|all")
    assert row["judged"] and row["claim"] == "in" and row["paired"] == 20
    assert row["result"] == result and row["rerun"] == (result != "tie")


def test_a_rerun_counts_only_if_it_agrees(tmp_path):
    run = _write(tmp_path, "all-M10-T100", range(20), (3.0, 5.0, 12.0))
    (tmp_path / "again").mkdir()
    again = _write(tmp_path / "again", "all-M10-T100", range(100, 120), (5.0, 5.0, 12.0))
    summary = _summary(tmp_path, run, replications=[again])
    row = _row(summary, "all-M10-T100|shd|alpha=0.05|all")
    assert row["result"] == "tie (did not replicate)" and not summary["rerun"]


def test_a_refusal_where_tigramite_finished_fails_the_row(tmp_path):
    run = _write(tmp_path, "all-M10-T100", range(20), (5.0, 5.0, 12.0))
    with (run / "records.jsonl").open("a") as rec:
        rec.write(json.dumps({"cell": "all-M10-T100", "seed": 4, "arm": "andrey", "alpha": 0.05,
                              "status": "refused", "task_s": 2.0, "build": {}}) + "\n")  # fmt: skip
    row = _row(_summary(tmp_path, run), "all-M10-T100|shd|alpha=0.05|all")
    assert row["paired"] == 19 and row["failures"] == [4]
    assert row["result"].startswith("fail")


def test_the_floor_and_the_level_bound_what_is_judged(tmp_path):
    run = _write(tmp_path, "all-M10-T100", range(20), (5.0, 5.0, 5.0))
    summary = _summary(tmp_path, run)
    row = _row(summary, "all-M10-T100|shd|alpha=0.05|all")
    assert not row["judged"] and row["reason"] == "reference not better than the empty graph"
    assert _row(summary, "all-M10-T100|shd|alpha=0.01|all")["claim"] == "out"


def test_draws_that_break_assumption_3_form_a_row_that_is_not_judged(tmp_path):
    run = _write(tmp_path, "half-M10-T100", range(30), (5.0, 5.0, 12.0), broken=range(15, 30))
    summary = _summary(tmp_path, run)
    holds = _row(summary, f"half-M10-T100|shd|alpha=0.05|{lane.HOLDS}")
    broken = _row(summary, f"half-M10-T100|shd|alpha=0.05|{lane.BROKEN}")
    assert holds["judged"] and holds["seeds"] == list(range(15))
    assert broken["claim"] == "out" and not broken["judged"] and broken["paired"] == 15
    assert not any(r["metric"] == "context_error" for r in summary["rows"])


def test_the_patch_check_compares_the_stock_fits_that_finished(tmp_path):
    run = _write(tmp_path, "all-M10-T100", range(12), (5.0, 5.0, 12.0), stock=range(11))
    (check,) = _summary(tmp_path, run)["patch_check"]
    assert (check["stock_finished"], check["same_as_patched"], check["differ"]) == (12, 11, [11])


def test_every_claimed_cell_is_at_the_papers_level_and_model():
    claimed = [c for c in lane.CELLS.values() if c.claim == "in"]
    assert len(claimed) == 21
    assert all(c.model == "jpcmciplus2023" and c.n_system == 5 and c.test == "parcorr"
               for c in claimed)  # fmt: skip
    assert {c.observed for c in claimed} == {1.0, 0.5}
    assert all(lane.cell_arms(c)[:2] == ["andrey", "tigramite"] for c in lane.CELLS.values())
