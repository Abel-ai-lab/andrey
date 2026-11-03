"""The UT-IGSP lane: its grid, data, truth, scores, decision rules, and children.

The reference child's tests run only where ``ANDREY_BENCH_UTIGSP_PYTHON`` names the interpreter
``start_utigsp.sh`` builds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import ut_igsp_cpu as lane

# The summaries are read by the repository's parity check, qa/reference_parity.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qa import reference_parity  # noqa: E402


def test_the_grid_holds_the_designed_cells():
    cells = lane.grid()
    assert len({c.name for c in cells}) == len(cells)
    in_claim = [c for c in cells if c.arm.startswith("G")]
    assert len(in_claim) == 56 and all(c.claim == "in" and c.draws == 5 for c in in_claim)
    # G5 refits G1's datasets without their targets; the others are new datasets.
    assert sum(c.datasets for c in in_claim if c.arm != "G5") == 1040
    block_a = [c for c in cells if c.arm == "A"]
    assert (len(block_a), sum(c.datasets for c in block_a)) == (15, 290)
    assert all(c.seeds[0] == 1000 for c in block_a)
    assert all(c.claim == "out" and c.draws == 1 for c in cells if c.arm.startswith("O"))


def _known(cell, seed):
    X, env, truth = lane.dataset(cell, seed)
    return X, env, truth


def test_the_designed_targets_are_the_true_ones_without_off_targets():
    cell = lane.cell_named("G1_shift_l0_n1000")
    X, env, truth = _known(cell, 0)
    assert X.shape == (6 * 1000, 20) and np.array_equal(np.unique(env), np.arange(6))
    assert truth["true"] == truth["designed"] == truth["given"]
    assert lane.target_floor(truth) == 0
    # Three off-targets per setting: the given targets miss 15 true ones.
    _, _, three = _known(lane.cell_named("G1_shift_l3_n1000"), 0)
    assert [t[:1] for t in three["designed"]] == three["designed"]
    assert all(set(d) < set(t) and len(t) == 4 for d, t in zip(three["designed"], three["true"]))
    assert lane.target_floor(three) == 15
    # Without targets (G5) every true target is to be learned.
    _, _, unknown = _known(lane.cell_named("G5_shift_l3_n1000"), 0)
    assert unknown["given"] == [[]] * 5 and lane.target_floor(unknown) == 20


def test_cells_of_a_block_share_their_dags_and_designed_targets():
    _, _, a = _known(lane.cell_named("G1_shift_l0_n1000"), 3)
    _, _, b = _known(lane.cell_named("G2_perfect_l3_n2000"), 3)
    assert a["dag"] == b["dag"] and a["designed"] == b["designed"]


def test_a_weight_change_targets_only_variables_with_parents():
    X, env, truth = _known(lane.cell_named("G6_weight_l3_n1000"), 0)
    parents = lane.has_parent(np.array(truth["dag"]))
    assert all(parents[v] for t in truth["true"] for v in t)


def test_a_true_estimate_scores_zero_and_the_empty_graph_its_edge_count():
    _, _, truth = _known(lane.cell_named("G1_shift_l3_n1000"), 0)
    dag = np.array(truth["dag"])
    exact = lane.essential(dag, truth["true"])
    line = {"algorithm": "ut_igsp", "marks": exact.tolist(), "targets": truth["true"]}
    scores = lane.score(line, "andrey", truth)
    assert scores["shd"] == scores["target_error"] == scores["shd_true_targets"] == 0
    assert scores["consistent"] and scores["iess_exact"] and scores["in_true_imec"]
    arcs = np.argwhere(dag == 1)
    reference = {"algorithm": "ut_igsp", "arcs": arcs.tolist(), "targets": truth["true"]}
    assert lane.score(reference, "reference", truth)["shd"] == 0
    empty = {
        "algorithm": "ut_igsp",
        "marks": np.zeros_like(dag).tolist(),
        "targets": truth["given"],
    }
    scores = lane.score(empty, "andrey", truth)
    assert scores["shd"] == scores["empty_shd"] == 15
    assert scores["target_error"] == scores["target_floor"] == 15
    assert scores["target_fn_off"] == 15 and scores["target_fp"] == 0


def test_scores_do_not_depend_on_how_the_variables_are_numbered():
    _, _, truth = _known(lane.cell_named("G2_perfect_l3_n1000"), 2)
    dag = np.array(truth["dag"])
    rng = np.random.default_rng(0)
    # An estimate with errors: one true arrow dropped, one wrong target per setting.
    est = dag.copy()
    i, j = np.argwhere(dag == 1)[0]
    est[i, j] = est[j, i] = 0
    targets = [sorted(set(t) | {int(rng.integers(20))}) for t in truth["true"]]
    line = {"algorithm": "ut_igsp", "arcs": np.argwhere(est == 1).tolist(), "targets": targets}
    base = lane.score(line, "reference", truth)
    perm = rng.permutation(20)  # variable v is renamed perm[v]
    inverse = np.argsort(perm)
    moved = {
        **truth,
        "dag": dag[np.ix_(inverse, inverse)].tolist(),
        **{
            key: [sorted(int(perm[v]) for v in t) for t in truth[key]]
            for key in ("true", "designed", "given")
        },
    }
    line_moved = {
        "algorithm": "ut_igsp",
        "arcs": [[int(perm[a]), int(perm[b])] for a, b in line["arcs"]],
        "targets": [sorted(int(perm[v]) for v in t) for t in targets],
    }
    assert lane.score(line_moved, "reference", moved) == base
    assert base["shd"] > 0 and base["target_error"] > 0


def _records(cell, package, algorithm, config, shd, target_error=None, draws=5, seeds=range(20)):
    """One finished record per draw of each seed, with the given per-seed errors."""
    out = []
    for seed in seeds:
        for draw in range(draws):
            out.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "package": package,
                    "algorithm": algorithm,
                    "config": config,
                    "draw": draw,
                    "status": "ok",
                    "wall_s": 1.0,
                    "shd": shd(seed),
                    "target_error": None if target_error is None else target_error(seed),
                    "empty_shd": 15,
                    "target_floor": 15,
                    "shd_true_targets": shd(seed),
                    "skeleton_shd": 0,
                    "marks": [[seed]],
                    "targets": [[draw]],
                }
            )
    return out


def test_the_summary_entries_are_read_and_judged_by_the_parity_check():
    cell = "G1_shift_l3_n1000"
    records = _records(cell, "andrey", "ut_igsp", "J1", lambda s: 2 + s % 2, lambda s: 3)
    records += _records(cell, "reference", "ut_igsp", "J1", lambda s: 0, lambda s: 3 + s % 3)
    records += _records(cell, "andrey", "gies", "default", lambda s: 9, draws=1)
    summary = lane.summarize(records)
    entries = {(e["regime"], e["comparator"]): e for e in summary["methods"]}
    shd = entries[("G1_shift_l3_J1", "graphical-model-learning.unknown_target_igsp")]
    [row] = shd["rows"]
    assert (row["d"], row["n"], row["paired"]) == (20, 1000, 20)
    assert row["multi_draw"]["above_zero"] and row["coverage"]["andrey_short"] == 0
    assert row["floor_95"][1] < 0
    assert (
        entries[("G1_shift_l3_J1_targets", "graphical-model-learning.unknown_target_igsp")][
            "metric"
        ]
        == "target_error"
    )
    assert entries[("G1_shift_l3_J1", "andrey.gies")]["role"] == "context"
    rows = reference_parity.paired_rows("lane", summary)
    failures, _ = reference_parity.check(rows, [])
    # Andrey has 2.5 more SHD on average: worse. Its targets are better: not a failure.
    [failure] = failures
    assert "metric=shd " in failure


def test_a_worse_cell_counts_only_when_its_rerun_on_fresh_seeds_is_worse():
    cell = "G2_perfect_l3_n1000"
    records = _records(cell, "andrey", "ut_igsp", "J1", lambda s: 2, lambda s: 3)
    records += _records(cell, "reference", "ut_igsp", "J1", lambda s: 1, lambda s: 3)
    fresh = range(2000, 2020)
    reruns = _records(cell, "andrey", "ut_igsp", "J1", lambda s: s % 2, lambda s: 3, seeds=fresh)
    reruns += _records(cell, "reference", "ut_igsp", "J1", lambda s: 1, lambda s: 3, seeds=fresh)
    summary = lane.summarize(records, reruns)
    [entry] = [e for e in summary["methods"] if e["regime"] == "G2_perfect_l3_J1"]
    assert entry["rows"][0]["replication"]["seeds"] == "2000-2019"
    rows = reference_parity.paired_rows("lane", summary)
    [row] = [r for r in rows if r.key["metric"] == "shd"]
    assert row.replicated and not row.failed
    assert reference_parity.check(
        reference_parity.paired_rows("lane", lane.summarize(records)), []
    )[0]


def test_a_timeout_on_a_dataset_the_reference_finished_fails_the_cell():
    cell = "G1_shift_l0_n1000"
    records = _records(cell, "andrey", "ut_igsp", "J1", lambda s: 0, lambda s: 0)
    records[0]["status"] = "timeout"
    records += _records(cell, "reference", "ut_igsp", "J1", lambda s: 0, lambda s: 0)
    summary = lane.summarize(records)
    [entry] = [e for e in summary["methods"] if e["regime"] == "G1_shift_l0_J1"]
    assert entry["rows"][0]["coverage"]["andrey_short"] == 1
    assert entry["rows"][0]["paired"] == 19
    failures, _ = reference_parity.check(reference_parity.paired_rows("lane", summary), [])
    assert any("finished by" in f for f in failures)


def test_a_resumed_run_may_change_host_and_workers_but_not_its_fits():
    meta = {"arm": "G1", "andrey_fits": {"c": [{"alpha": 1e-5}]}, "machine_id": "a", "workers": 4}
    assert lane.same_run(meta, {**meta, "machine_id": "b", "workers": 2})
    assert not lane.same_run(meta, {**meta, "andrey_fits": {"c": [{"alpha": 1e-3}]}})


def test_block_a_picks_the_candidate_better_in_the_most_cells():
    cells = ["A_shift_l0_n1000", "A_shift_l3_n1000", "A_perfect_l0_n1000"]
    base = lane.config_name(lane.J1_ANDREY)
    records = []
    for cell in cells:
        records += _records(cell, "andrey", "ut_igsp", base, lambda s: 3, lambda s: 2)
        # Better on SHD everywhere, the same targets: eligible.
        records += _records(cell, "andrey", "ut_igsp", "a0.001_sum_s5", lambda s: 1, lambda s: 2)
        # Better on SHD everywhere, worse targets everywhere: not eligible.
        records += _records(cell, "andrey", "ut_igsp", "a0.01_sum_s5", lambda s: 0, lambda s: 5)
        # As good as the first, farther from the baseline: loses the tie.
        records += _records(
            cell, "andrey", "ut_igsp", "a0.001_lexicographic_s5", lambda s: 1, lambda s: 2
        )
    result = lane.select(records, base)
    assert result["winner"] == "a0.001_sum_s5"
    assert result["eligible"] == ["a0.001_sum_s5", "a0.001_lexicographic_s5"]
    assert result["candidates"]["a0.01_sum_s5"]["counts"]["target_error"]["worse"] == 3
    # With no eligible candidate the baseline stays.
    only_worse = [r for r in records if r["config"] in (base, "a0.01_sum_s5")]
    assert lane.select(only_worse, base)["winner"] == base


def test_the_andrey_child_records_every_draw_and_switches_the_objective(tmp_path):
    cell = lane.Cell("G1", "shift", 0, 300, p=6)
    X, env, truth = lane.dataset(cell, 4)
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    fits = [
        lane._ut_igsp_fit({**lane.J1_ANDREY, "objective": o}, o, [0, 1])
        for o in ("sum", "lexicographic")
    ]
    request = {
        "mode": "fit",
        "X": X,
        "env": env,
        "known": known,
        "fits": json.dumps(fits),
        "warmup_rows": 50,
    }
    lines, killed, code, err = lane.run_child(
        sys.executable, lane.ANDREY_CHILD, request, 600, None, tmp_path
    )
    assert not killed and code == 0, err
    assert [(x["config"], x["draw"], x["status"]) for x in lines] == [
        ("sum", 0, "ok"),
        ("sum", 1, "ok"),
        ("lexicographic", 0, "ok"),
        ("lexicographic", 1, "ok"),
    ]
    assert isinstance(lines[0]["score"], int) and isinstance(lines[2]["score"], list)
    assert all(lane.score(x, "andrey", truth)["consistent"] for x in lines)


REFERENCE = os.environ.get(lane.REFERENCE_PYTHON)
needs_reference = pytest.mark.skipif(not REFERENCE, reason=f"{lane.REFERENCE_PYTHON} is not set")


@needs_reference
def test_the_truth_agrees_with_the_reference_library(tmp_path):
    assert lane.main(["check-truth", "--out", str(tmp_path), "--pairs", "20"]) == 0
    assert json.loads((tmp_path / "truth-check.json").read_text())["agree"] == 20


@needs_reference
def test_the_reference_child_runs_every_algorithm(tmp_path):
    cell = lane.Cell("G1", "shift", 1, 1000, p=8)
    X, env, truth = lane.dataset(cell, 1)
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    fits = [{**f, "draws": [0]} for f in lane.reference_fits(cell)]
    assert {f["algorithm"] for f in fits} == {"ut_igsp", "igsp", "jci_gsp"}
    request = {
        "mode": "fit",
        "X": X,
        "env": env,
        "known": known,
        "fits": json.dumps(fits),
        "warmup_rows": 50,
    }
    lines, killed, code, err = lane.run_child(
        REFERENCE, lane.GML_CHILD, request, 600, None, tmp_path
    )
    assert not killed and code == 0, err
    assert [x["status"] for x in lines] == ["ok"] * len(fits), [x.get("error_text") for x in lines]
    for line in lines:
        scores = lane.score(line, "reference", truth)
        assert scores["shd"] < scores["empty_shd"]


def test_the_children_import_nothing_from_the_harness():
    # The reference child runs in an interpreter without Andrey or the harness.
    text = Path(lane.GML_CHILD).read_text()
    assert "import andrey" not in text and "andrey_bench" not in text.split('"""', 2)[2]
