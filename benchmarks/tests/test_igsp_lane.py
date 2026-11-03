"""The IGSP lane: its grid, data, truth, scores, decision rules, children, and patched reference.

The reference children's tests run only where ``ANDREY_BENCH_UTIGSP_PYTHON`` and
``ANDREY_BENCH_GML_DEF42_PYTHON`` name the interpreters ``start_utigsp.sh`` builds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import igsp_cpu as lane
import numpy as np
import pytest
import ut_igsp_cpu

# The summaries are read by the repository's parity check, qa/reference_parity.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qa import reference_parity  # noqa: E402

REFERENCE = os.environ.get(lane.REFERENCE_PYTHON)
DEF42 = os.environ.get(lane.DEF42_PYTHON)
needs_reference = pytest.mark.skipif(
    not (REFERENCE and DEF42), reason=f"{lane.REFERENCE_PYTHON} or {lane.DEF42_PYTHON} is not set"
)


def test_the_grid_holds_the_designed_cells():
    cells = lane.grid()
    assert len({c.name for c in cells}) == len(cells)
    in_claim = [c for c in cells if c.claim == "in"]
    arms = [c.arm for c in in_claim]
    assert (arms.count("W"), arms.count("S"), len(in_claim)) == (12, 27, 87)
    assert sum(c.multi_target for c in in_claim) == 27
    assert all(c.draws == 5 and c.seeds == list(range(20)) for c in in_claim)
    # Every row of a summary has its own key: the regime and the parity check's row keys.
    keys = {(c.group, c.degree, c.p, c.n) for c in cells}
    assert len(keys) == len(cells)
    confirm = [c for c in cells if c.arm == "Q"]
    assert [c.seeds[0] for c in confirm] == [3000, 3000] and all(c.claim == "out" for c in confirm)


def test_a_named_target_is_the_column_whose_law_changes():
    # Wang's intervention gives the target the noise N(0, 0.2) whatever its causes.
    for family in ("4", "5"):
        for seed in range(3):
            cell = lane.Cell("S", "wang", 20_000, p=7, family=family, c=0.25)
            X, env, truth = lane.dataset(cell, seed)
            [[target]] = truth["nominal"]
            variance = X[env == 1].var(axis=0)
            assert abs(variance[target] - 0.2) < 0.02
            assert np.argmin(variance / X[env == 0].var(axis=0)) == target


def test_an_imperfect_setting_has_its_rows_split_among_the_subsets_of_its_targets():
    cell = lane.cell_named("Y8_imperfect_p10_n1000")
    X, env, truth = lane.dataset(cell, 4)
    assert np.bincount(env).tolist() == [1000] * 46
    envs, rows, setting = lane.environments(cell, 4, truth["nominal"])
    # Four subsets per pair, in setting order, about a quarter of the rows each.
    assert setting[1:5].tolist() == [1] * 4 and [len(e) for e in envs[1:5]] == [2, 1, 1, 0]
    assert all(150 < r < 350 for r in rows[1:])
    again = lane.dataset(cell, 4)
    assert np.array_equal(again[0], X) and np.array_equal(again[1], env)


def test_a_null_target_is_a_root_under_yangs_kinds_and_none_under_wangs():
    from andrey.core import ARROW

    for name in ("Y5_perfect_p20_n1000", "Y5_inhibiting_p20_n1000", "W_K1_p20_n1000"):
        cell = lane.cell_named(name)
        _, _, truth = lane.dataset(cell, 2)
        dag = np.asarray(truth["dag"])
        roots = {v for v in range(cell.p) if not (dag[v] == ARROW).any()}
        null = roots if cell.kind != "wang" else set()
        assert truth["effective"] == [sorted(set(t) - null) for t in truth["nominal"]]
        assert truth["null_targets"] == sum(len(set(t) & null) for t in truth["nominal"])
        arcs = [[r, c] for r in sorted(null) for c in range(cell.p) if dag[c, r] == ARROW]
        assert truth["null_arcs"] == arcs


def test_any_member_of_the_nominal_class_gives_the_effective_class():
    from andrey.core.orient import pdag2dag

    rng = np.random.default_rng(655)
    for _ in range(50):
        p = int(rng.integers(4, 10))
        order = rng.permutation(p)
        arcs = [
            [int(order[a]), int(order[b])]
            for a in range(p)
            for b in range(a + 1, p)
            if rng.random() < 0.4
        ]
        dag = lane.dag_from_arcs(arcs, p)
        children = {i for i, _ in arcs}
        roots = [v for v in range(p) if v not in {j for _, j in arcs}]
        nominal = [[int(v)] for v in rng.choice(p, int(rng.integers(1, p)), replace=False)]
        effective = [[v for v in t if v not in roots] for t in nominal]
        assert children or roots
        member = pdag2dag(lane.essential(dag, nominal))
        assert np.array_equal(lane.essential(member, effective), lane.essential(dag, effective))


def test_a_true_estimate_scores_zero_and_the_empty_graph_its_edge_count():
    cell = lane.cell_named("Y6_perfect_k4_p10_n1000")
    _, _, truth = lane.dataset(cell, 5)
    dag = np.asarray(truth["dag"])
    true_arcs = [[i, j] for i in range(cell.p) for j in range(cell.p) if dag[j, i] == 2]
    marks = lane.essential(dag, truth["given"]).tolist()
    mine = lane.score({"algorithm": "igsp", "marks": marks}, "andrey", truth)
    theirs = lane.score({"algorithm": "igsp", "arcs": true_arcs}, "reference", truth)
    empty = lane.score({"algorithm": "igsp", "arcs": []}, "reference", truth)
    assert mine["shd"] == theirs["shd"] == theirs["dag_hamming"] == 0 and mine["consistent"]
    assert empty["shd"] == empty["empty_shd"] == len(true_arcs)
    assert mine["null_kept"] == theirs["null_kept"] == mine["null_arcs"] > 0


def test_a_reversed_arc_out_of_a_null_target_is_counted():
    # 0 -> 1 with 0 a root targeted by a perfect intervention: 0 is a null target.
    truth = {
        "dag": lane.dag_from_arcs([[0, 1]], 3).tolist(),
        "nominal": [[0], [2]],
        "effective": [[], []],
        "given": [[0], [2]],
        "null_targets": 2,
        "null_arcs": [[0, 1]],
    }
    reversed_ = lane.score({"algorithm": "igsp", "arcs": [[1, 0]]}, "reference", truth)
    assert (reversed_["null_reversed"], reversed_["null_kept"]) == (1, 0)
    # Under the effective targets the class is the skeleton: the reversal costs nothing.
    assert reversed_["shd"] == 0 and reversed_["shd_nominal"] == 1


def _records(cell, package, config, shd, algorithm="igsp", draws=5, seeds=range(20), block="B"):
    """One finished record per draw of each seed, with the given per-seed SHD."""
    return [
        {
            "cell": cell,
            "block": block,
            "arm": cell.split("_")[0],
            "claim": "in",
            "seed": seed,
            "package": package,
            "algorithm": algorithm,
            "config": config,
            "draw": draw,
            "status": "ok",
            "wall_s": 1.0,
            "shd": shd(seed),
            "empty_shd": 15,
            "marks": [[seed]],
        }
        for seed in seeds
        for draw in range(draws)
    ]


def test_the_summary_filters_rows_whose_reference_loses_to_the_empty_graph():
    mine = lane.config_name(lane.andrey_defaults()["alpha"], lane.andrey_defaults()["n_starts"])
    alpha = mine.split("_")[0]
    single, multi = "Y6_perfect_k4_p10_n1000", "W_K2_p10_n1000"
    records = _records(single, "andrey", mine, lambda s: 3)
    records += _records(single, "reference", f"{alpha}_r5", lambda s: 1 + s % 2)
    records += _records(multi, "andrey", mine, lambda s: 2)
    # The unpatched reference does no better than the empty graph; the patched copy does.
    records += _records(multi, "reference", f"{alpha}_r5", lambda s: 16)
    records += _records(multi, "def42", f"{alpha}_r5", lambda s: 2)
    records += _records(single, "andrey", "default", lambda s: 9, algorithm="gies", draws=1)
    summary = lane.summarize(records)
    entries = {(e["regime"], e["comparator"]): e for e in summary["methods"]}
    level = lane.alpha_text(lane.andrey_defaults()["alpha"])
    [worse] = entries[(f"Y6|perfect|k=4|alpha={level}", "graphical-model-learning")]["rows"]
    assert worse["verdict"] == "worse" and worse["floor_95"][1] < 0 and "claim" not in worse
    [out] = entries[(f"W|K=2|alpha={level}", "graphical-model-learning")]["rows"]
    assert out["claim"] == "out" and "floor_95" not in out
    [patched] = entries[(f"W|K=2|alpha={level}", "graphical-model-learning+def42")]["rows"]
    assert patched["verdict"] == "tie" and patched["floor_95"][1] < 0
    assert entries[("Y6|perfect|k=4", "andrey.gies")]["role"] == "context"
    assert summary["counts"] == {"better": 0, "tie": 1, "worse": 1} and summary["filtered"] == 1
    failures, _ = reference_parity.check(reference_parity.paired_rows("lane", summary), [])
    [failure] = failures
    assert "Y6|perfect|k=4" in failure


def test_a_worse_row_counts_only_when_its_rerun_on_fresh_seeds_is_worse():
    mine = lane.config_name(lane.andrey_defaults()["alpha"], lane.andrey_defaults()["n_starts"])
    theirs = mine.split("_")[0] + "_r5"
    cell = "Y5_perfect_p20_n1000"
    records = _records(cell, "andrey", mine, lambda s: 2) + _records(
        cell, "reference", theirs, lambda s: 1
    )
    fresh = range(100, 120)
    reruns = _records(cell, "andrey", mine, lambda s: s % 2, seeds=fresh, block="C")
    reruns += _records(cell, "reference", theirs, lambda s: 1, seeds=fresh, block="C")
    rows = reference_parity.paired_rows("lane", lane.summarize(records, reruns))
    [row] = [r for r in rows if r.comparator == "graphical-model-learning"]
    assert row.replicated and not row.failed


def test_the_default_is_the_candidate_never_worse_and_better_somewhere():
    cells = ["W_K1_p10_n1000", "Y6_perfect_k4_p10_n1000", "S_family4_c0.25_p7_n1000"]
    errors = {1e-4: 3, 5e-4: 3, 1e-3: 1, 5e-3: 1, 1e-2: 2, 5e-2: 6}
    records = []
    for cell in cells:
        for alpha, shd in errors.items():
            records += _records(
                cell, "andrey", f"a{alpha:g}_s5", lambda s, e=shd: e, draws=1, block="A"
            )
    for cell in ("Q_perfect_k8_p20_n1000_deg3", "Q_perfect_k8_p20_n10000_deg3"):
        for alpha, shd in errors.items():
            records += _records(
                cell, "andrey", f"a{alpha:g}_s5", lambda s, e=shd: e, draws=1, block="Q"
            )
    for r in records:
        r["claim"] = "out" if r["arm"] == "Q" else "in"
    result = lane.select(records, step=1)
    # 1e-3 and 5e-3 tie each other everywhere and beat the rest: speed decides, and they are equal.
    assert result["counts"]["a0.001_s5"]["eligible"] and result["counts"]["a0.005_s5"]["eligible"]
    assert result["winner"] in ("a0.001_s5", "a0.005_s5")
    assert result["confirmation"]["confirmed"] and not result["escalate"]
    # A candidate better than the winner on the confirmation family sends both tables to the user.
    for r in records:
        if r["arm"] == "Q" and r["config"] == "a0.05_s5":
            r["shd"] = 0
    assert lane.select(records, step=1)["escalate"]
    # Step 2: five starts stay unless 1 or 10 is never worse and better somewhere.
    winner = result["winner"]
    step2 = [r for r in records if r["config"] == winner and r["arm"] != "Q"]
    for cell in cells:
        step2 += _records(
            cell, "andrey", winner.replace("_s5", "_s1"), lambda s: 1, draws=1, block="A"
        )
        step2 += _records(
            cell, "andrey", winner.replace("_s5", "_s10"), lambda s: 1, draws=1, block="A"
        )
    assert lane.select(step2, step=2)["winner"] == winner


def test_the_andrey_child_fits_igsp_and_the_context_arms(tmp_path):
    cell = lane.Cell("S", "wang", 2000, p=7, family="45", c=0.5)
    X, env, truth = lane.dataset(cell, 1)
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    fits = [{**f, "draws": f["draws"][:2]} for f in lane.andrey_fits(cell)]
    assert {f["algorithm"] for f in fits} == {"igsp", "gies", "grasp", "ut_igsp"}
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
    assert len(lines) == sum(len(f["draws"]) for f in fits)
    assert all(x["status"] == "ok" for x in lines), [x.get("error_text") for x in lines]
    for line in lines:
        scores = lane.score(line, "andrey", truth)
        assert scores["shd"] <= scores["empty_shd"]
        if line["algorithm"] == "igsp":
            assert scores["consistent"]


@needs_reference
def test_the_truth_agrees_with_the_reference_library(tmp_path):
    assert ut_igsp_cpu.main(["check-truth", "--out", str(tmp_path), "--pairs", "50"]) == 0


@needs_reference
def test_the_patched_reference_passes_the_oracle_checks(tmp_path):
    # Design 1b: no true arc marked, every class-changing covered reversal marked, the {0, 2}
    # probe oriented on 20 of 20 seeds; the unpatched copy orients the probe backwards.
    assert lane.main(["validate-patch", "--out", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "patch-validation.json").read_text())
    assert result["reference"]["probe"]["hard"]["1->0"] > 10


@needs_reference
def test_the_reference_children_fit_both_copies(tmp_path):
    cell = lane.Cell("S", "wang", 2000, p=7, family="45", c=0.5)
    X, env, truth = lane.dataset(cell, 1)
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    for package, python in (("reference", REFERENCE), ("def42", DEF42)):
        fits = [{**f, "draws": [0]} for f in lane.reference_fits(cell, package)]
        request = {
            "mode": "fit",
            "X": X,
            "env": env,
            "known": known,
            "fits": json.dumps(fits),
            "warmup_rows": 50,
        }
        lines, killed, code, err = lane.run_child(
            python, lane.GML_CHILD, request, 600, None, tmp_path
        )
        assert not killed and code == 0, err
        assert [x["status"] for x in lines] == ["ok"] * len(fits)
        for line in lines:
            assert lane.score(line, package, truth)["shd"] <= 1
