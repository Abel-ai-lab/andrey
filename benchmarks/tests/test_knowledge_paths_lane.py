"""The path and confounder claims lanes: claims drawn from the truth, error counts, the oracles,
and the paired summary."""

from __future__ import annotations

import json

import knowledge_paths as kp
import numpy as np


def _dag(d: int, arrows) -> np.ndarray:
    marks = np.zeros((d, d), dtype=np.int8)
    for a, b in arrows:
        marks[a, b], marks[b, a] = kp.TAIL, kp.ARROW
    return marks


CHAIN = _dag(4, [(0, 1), (1, 2), (2, 3)])


def test_true_claims_hold_in_the_truth():
    reach = kp.dag_reach(CHAIN)
    facts = {(a, b): bool(reach[a, b]) for a in range(4) for b in range(4) if a != b}
    for fraction in (0.1, 0.5, 1.0):
        claims = kp.draw_claims(np.random.default_rng(0), facts, fraction, [])
        assert all(reach[a, b] for a, b in claims["required_paths"])
        assert not any(reach[a, b] for a, b in claims["forbidden_paths"])
    wrong = kp.draw_claims(np.random.default_rng(0), facts, 1.0, [], true=False)
    assert not any(reach[a, b] for a, b in wrong["required_paths"])
    assert all(reach[a, b] for a, b in wrong["forbidden_paths"])


def test_a_hidden_common_cause_names_the_pairs_it_reaches():
    # 4 is hidden and causes 0 and, through hidden 5, 2; 3 is a child of observed 2.
    marks = _dag(6, [(4, 0), (4, 5), (5, 2), (2, 3)])
    hidden = np.array([False] * 4 + [True] * 2)
    assert kp.hidden_cause_pairs(marks, hidden) == {(0, 2)}


def test_an_open_mark_is_never_wrong_and_a_reversed_edge_is():
    truth = CHAIN
    undirected = truth.copy()
    undirected[1, 0] = kp.TAIL  # 0 - 1 left undirected
    assert kp.static_errors(undirected, truth, undirected_open=True)["errors"] == 0
    reversed_ = _dag(4, [(1, 0), (1, 2), (2, 3)])
    scores = kp.static_errors(reversed_, truth, undirected_open=True)
    assert (scores["adjacency_errors"], scores["wrong_marks"]) == (0, 2)
    empty = np.zeros_like(truth)
    assert (
        kp.static_errors(empty, truth, undirected_open=True)["errors"]
        == 3
        == scores["empty_errors"]
    )


def test_a_cpdag_fit_is_scored_against_the_class_its_claims_leave():
    # The class of the chain leaves every edge undirected; that 0 is an ancestor of 3 directs all.
    line = CHAIN.copy()
    line[CHAIN == kp.ARROW] = kp.TAIL
    claims = {"required_paths": [[0, 3]], "forbidden_paths": []}
    reversed_ = _dag(4, [(1, 0), (1, 2), (2, 3)])
    cases = [  # fit, claims, errors, dag_errors
        (line, None, 0, 3),  # the true class: nothing more can be directed
        (line, claims, 3, 3),  # the claims direct three edges the fit leaves undirected
        (CHAIN, claims, 0, 0),  # an edge directed as the true DAG directs it is no error
        (reversed_, claims, 1, 2),
        (np.zeros_like(CHAIN), claims, 3, 6),
    ]
    for fit, taken, errors, dag_errors in cases:
        counts = kp.class_errors(fit, CHAIN, taken)
        assert (counts["errors"], counts["dag_errors"]) == (errors, dag_errors)
        assert (counts["empty_errors"], counts["empty_dag_errors"]) == (3, 6)
    # Directing one edge the right way lowers both counts, and the wrong way raises both.
    right, wrong = line.copy(), line.copy()
    right[1, 0] = wrong[0, 1] = kp.ARROW
    base = kp.class_errors(line, CHAIN, claims)
    for fit, sign in ((right, -1), (wrong, 1)):
        counts = kp.class_errors(fit, CHAIN, claims)
        assert np.sign(counts["dag_errors"] - base["dag_errors"]) == sign
        assert counts["errors"] - base["errors"] == min(sign, 0)


def test_every_dag_on_three_nodes_is_listed():
    assert len(list(kp.all_dags(3))) == 25


def test_the_cpdag_oracle_finds_no_mismatch():
    result = kp.oracle_cpdag(0, graphs=5, exhaustive=(3,))
    assert result["failures"] == 0
    assert result["exact"] + result["dropped_as_expected"] == result["claim_sets"]
    assert result["pc_true_cpdag"] == result["graphs"]


def test_the_pag_oracle_finds_every_mark_sound():
    result = kp.oracle_pag(0, graphs=4, exhaustive=())
    assert result["failures"] == 0
    for counts in result["methods"].values():
        assert counts["sound"] == counts["fits"] > 0


def test_summarize_pairs_each_claimed_fit_with_its_bare_fit(tmp_path):
    rows = []
    for seed in range(6):
        base = {
            "lane": "cpdag",
            "method": "pc",
            "cell": "c",
            "seed": seed,
            "dataset_id": f"x{seed}",
            "status": "ok",
            "definite_marks": 4,
            "empty_errors": 9,
            **{f"n_{p}": 1 for p in kp.PARTS},
        }
        rows.append({**base, "arm": "bare", "fraction": None, "errors": 5})
        rows.append({**base, "arm": "true", "fraction": 0.1, "errors": 3})
    rows.append({**rows[-1], "dataset_id": "x9", "status": "knowledge_error", "errors": None})
    rows.append({**rows[-2], "dataset_id": "x9", "arm": "bare", "fraction": None, "errors": 5})
    run = tmp_path / "run"
    run.mkdir()
    (run / kp.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "summary.json"
    kp.main(["summarize", "--runs", str(run), "--out", str(out)])
    (row,) = json.loads(out.read_text())["rows"]
    assert (row["paired"], row["knowledge_error"], row["verdict"]) == (6, 1, "better")
    assert row["mean_D"] == -2
    assert "dag_verdict" not in row and row["in_setting"]


def test_summarize_judges_the_distance_to_the_true_dag_and_flags_cells_outside_the_setting(
    tmp_path,
):
    cell = "d10-n500-deg4"
    rows, again = [], []
    for method in ("pc", "ges"):
        for seed in range(6):
            base = {
                "lane": "cpdag",
                "method": method,
                "cell": cell,
                "dataset_id": f"x{seed}",
                "status": "ok",
                "definite_marks": 4,
                "wrong_marks": 2,
                "empty_errors": 9,
                **{f"n_{p}": 1 for p in kp.PARTS},
            }
            bare = {**base, "arm": "bare", "fraction": None, "errors": 5, "dag_errors": 7}
            true = {**base, "arm": "true", "fraction": 0.1, "errors": 5}
            rows += [bare, {**true, "dag_errors": 9 + seed % 2}]
            again += [bare, {**true, "dag_errors": 7 if method == "pc" else 9}]
    for name, records in (("run", rows), ("rep", again)):
        (tmp_path / name).mkdir()
        (tmp_path / name / kp.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "summary.json"
    kp.main(
        [
            "summarize",
            *("--runs", str(tmp_path / "run")),
            *("--replicate", str(tmp_path / "rep")),
            *("--out", str(out)),
        ]
    )
    by_method = {row["method"]: row for row in json.loads(out.read_text())["rows"]}
    for method, row in by_method.items():
        assert (row["verdict"], row["dag_verdict"], row["dag_mean_D"]) == ("tie", "worse", 2.5)
        assert row["reversed_with"] == 1
    # Worse again on the fresh seeds counts; ges at degree 4 is outside its setting.
    assert not by_method["pc"]["counts"] and by_method["pc"]["in_setting"]
    assert by_method["ges"]["counts"] and not by_method["ges"]["in_setting"]


def test_path_claims_translate_to_the_edges_they_imply():
    claims = {"required_paths": [[0, 1], [0, 3]], "forbidden_paths": [[3, 0]], "confounded": []}
    edges = kp.edge_claims(claims, CHAIN)
    # 0 and 3 are not adjacent in the chain: that required path has no edge form.
    assert (edges["required"], edges["forbidden"]) == ([[0, 1]], [[3, 0]])
    assert (edges["no_edge_form"], edges["confounded_no_edge_form"]) == (1, 0)


def test_lagged_claims_become_link_assumptions_over_the_whole_window():
    claims = {
        "required_paths": [[0, 1, 0]],
        "forbidden_paths": [[1, 0, 1]],
        "confounded": [[0, 1, 1]],
    }
    links = {(j, i, lag): link for j, i, lag, link in kp.link_assumptions(claims, 2, 1)}
    assert links == {
        (1, 0, 0): "-?>",  # 0 is an ancestor of 1 at lag 0, read from both ends
        (0, 1, 0): "<?-",
        (0, 1, -1): "<?>",  # 1 one step back is not an ancestor of 0
        (1, 0, -1): "o->",  # a confounded link is there
        (0, 0, -1): "o?>",  # no claim
        (1, 1, -1): "o?>",
    }


def test_summarize_judges_a_method_against_its_reference_under_the_same_knowledge(tmp_path):
    rows = []
    for seed in range(6):
        base = {
            "lane": "cpdag",
            "cell": "d10-n500-deg2",
            "dataset_id": f"x{seed}",
            "status": "ok",
            "definite_marks": 4,
            "wrong_marks": 0,
            "errors": 3,
            "empty_errors": 9,
            "empty_dag_errors": 18,
            **{f"n_{p}": 1 for p in kp.PARTS},
        }
        for method, bare, edges in (("ges", 8, 5), ("tetrad.ges", 8, 7)):
            rows.append({**base, "method": method, "arm": "bare", "fraction": None})
            rows[-1]["dag_errors"] = bare
            for arm in ("true", "edges") if method == "ges" else ("edges",):
                row = {**base, "method": method, "arm": arm, "fraction": 0.1, "dag_errors": edges}
                rows.append({**row, "n_no_edge_form": 2})
    run = tmp_path / "run"
    run.mkdir()
    (run / kp.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "summary.json"
    kp.main(["summarize", "--runs", str(run), "--out", str(out)])
    summary = json.loads(out.read_text())
    by_arm = {row["arm"]: row for row in summary["reference_rows"]}
    assert {row["reference"] for row in by_arm.values()} == {"tetrad.ges"}
    assert by_arm["bare"]["verdict"] == "tie" and by_arm["bare"]["mean_D"] == 0
    for arm in ("true", "edges"):
        assert (by_arm[arm]["verdict"], by_arm[arm]["mean_D"]) == ("better", -2)
        assert by_arm[arm]["measure"] == "dag_errors" and by_arm[arm]["floor"][1] < 0
        assert by_arm[arm]["no_edge_form"] == 2
    # The reference's own change with knowledge is a row of its own.
    own = [r for r in summary["rows"] if r["method"] == "tetrad.ges"]
    assert [(r["arm"], r["dag_mean_D"]) for r in own] == [("edges", -1)]


def test_a_variant_is_an_andrey_method_at_its_settings_with_every_arm():
    rows = kp.run_unit(("cpdag", "d10-n500-deg2", 0, None, ("ges_bic",), ("bare", "true", "edges")))
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["arm"] for row in rows} == {"bare", "true", "edges"}
    assert kp.REFERENCES["ges_bic"] == kp.REFERENCES["ges"]
