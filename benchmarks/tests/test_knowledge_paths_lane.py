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


def test_every_dag_on_three_nodes_is_listed():
    assert len(list(kp.all_dags(3))) == 25


def test_the_cpdag_oracle_finds_no_mismatch():
    result = kp.oracle_cpdag(0, graphs=5, exhaustive=(3,))
    assert result["failures"] == 0
    assert result["exact"] + result["raised_as_expected"] == result["claim_sets"]
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
