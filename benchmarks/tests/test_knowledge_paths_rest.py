"""The path and confounder claims lanes for the methods the first lanes do not run: the lanes'
defaults stay as they were, each method gets every arm, and the data of CD-NOD, FCI-JCI, and
multi-group DirectLiNGAM carry the truth the scores need."""

from __future__ import annotations

import shutil

import knowledge_paths as kp
import numpy as np
import pytest

#: The methods whose path and confounder claims the lanes measure in addition to the first eight.
REST = {
    "cdnod",
    "xges",
    "hc",
    "tabu",
    "mmhc",
    "rsmax2",
    "grasp",
    "exact_search",
    "iamb",
    "inter_iamb",
    "mmpc",
    "fci_jci",
    "grasp_fci",
    "multi_group_direct_lingam",
}


def test_the_default_methods_of_the_published_lanes_stay_as_they_were():
    assert kp.LANES["cpdag"]["methods"] == ("pc", "ges", "boss")
    assert kp.LANES["pag"]["methods"] == ("fci", "gfci", "boss_fci")
    assert kp.LANES["lpcmci"]["methods"] == ("lpcmci",)
    assert kp.LANES["lingam"]["methods"] == ("direct_lingam", "lingam.direct_lingam")
    assert kp.LANES["cpdag"]["fractions"] == (0.05, 0.1, 0.3)
    assert kp.LANES["pag"]["fractions"] == (0.1, 0.3)


def test_every_remaining_method_runs_in_a_lane():
    runs = {m for lane in kp.LANES for m in kp.own_methods(lane)}
    assert REST <= runs
    assert {"direct_lingam", "pc", "fci", "lpcmci"} <= runs  # the first lanes still run
    # A lane's kind says how its fits are scored, and every lane has one.
    assert set(kp.KIND) == set(kp.LANES)


def test_a_new_method_is_an_andrey_method_with_every_arm():
    rows = kp.run_unit(
        ("cpdag", "d10-n500-deg2", 0, None, ("hc",), ("bare", "true", "edges", "false"))
    )
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["arm"] for row in rows} == {"bare", "true", "edges", "false"}
    assert all("dag_errors" in row for row in rows)


def test_mmpc_returns_the_cpdag_the_path_claims_need():
    assert kp.VARIANTS["mmpc"] == ("mmpc", {"structure": "cpdag"})
    rows = kp.run_unit(("cpdag", "d10-n500-deg2", 0, None, ("mmpc",), ("bare", "true")))
    assert {row["status"] for row in rows} == {"ok"}
    assert {row["arm"] for row in rows} == {"bare", "true"}


def test_fractions_keep_the_bare_fit_and_the_fractions_asked_for():
    task = ("cpdag", "d10-n500-deg2", 0, None, ("iamb",), ("bare", "true"), (0.3,))
    rows = kp.run_unit(task)
    assert {(r["arm"], r["fraction"]) for r in rows} == {("bare", None), ("true", 0.3)}


def test_the_domain_shift_dag_is_the_dag_behind_the_identifiable_class():
    from andrey.core.orient import dag2cpdag

    prob = kp.problem_cdnod({"d": 10, "n": 500, "degree": 2}, 3)
    dag, base = prob["truth"], prob["base"]
    assert prob["data"].shape == (500, 10) and set(prob["env"]) == {0, 1, 2, 3}
    assert ((dag != 0) == (base != 0)).all()  # the same adjacencies
    plain = dag2cpdag(dag)
    directed = (plain == kp.TAIL) & (plain.T == kp.ARROW)
    assert (base[directed] == kp.TAIL).all()  # the domain index only directs more edges
    arrows = (base == kp.TAIL) & (base.T == kp.ARROW)
    assert (dag[arrows] == kp.TAIL).all()  # and directs them as the DAG does


def test_a_cdnod_fit_is_scored_against_the_class_the_domain_index_identifies():
    dag = np.zeros((3, 3), dtype=np.int8)
    for a, b in ((0, 1), (1, 2)):
        dag[a, b], dag[b, a] = kp.TAIL, kp.ARROW
    est = dag.copy()  # a fit that directs the chain, as an index on variable 0 lets CD-NOD do
    assert kp.class_errors(est, dag, None)["errors"] == 2  # the chain's CPDAG leaves it undirected
    assert kp.class_errors(est, dag, None, base=dag)["errors"] == 0
    assert kp.class_errors(est, dag, None, base=dag)["dag_errors"] == 0


def test_the_jci_truth_is_the_mag_of_the_system_variables_with_its_confounders_adjacent():
    cell = kp.LANES["jci"]["cells"]["p10-n500"]
    for seed in range(4):
        prob = kp.problem_jci(cell, seed)
        truth = prob["truth"]
        assert truth.shape == (10, 10) and prob["data"].shape == (1000, 10)
        assert set(prob["env"]) == {0.0, 1.0}
        for claims in prob["claims"].values():
            for a, b in claims["confounded"]:
                assert truth[a, b] != kp.NULL and truth[b, a] != kp.NULL
            for part in ("required_paths", "forbidden_paths"):
                assert all(max(pair) < 10 for pair in claims[part])


def test_a_multigroup_fit_is_scored_over_all_its_groups():
    d, groups = 4, 3
    dag = np.zeros((d, d), dtype=np.int8)
    dag[0, 1], dag[1, 0] = kp.TAIL, kp.ARROW
    truth = kp._side_by_side([dag] * groups)
    wrong = np.zeros((d, d), dtype=np.int8)  # each group misses the edge
    scored = kp.scored(
        "multigroup", {"status": "ok", "marks": kp._side_by_side([wrong] * groups)}, truth
    )
    assert scored["errors"] == groups and scored["empty_errors"] == groups
    prob = kp.problem_multigroup({"d": 6, "n": 200, "degree": 2}, 0)
    assert prob["data"].shape == (200, 18) and prob["truth"].shape == (18, 18)
    assert all(max(pair) < 6 for pair in prob["claims"][0.3]["required_paths"])


def test_a_multigroup_method_fits_every_group_and_reports_their_graphs_side_by_side():
    cell = {"d": 5, "n": 300, "degree": 2}
    prob = kp.problem_multigroup(cell, 1)
    result = kp.fit("multi_group_direct_lingam", prob["data"], prob["claims"][0.3], cell, 1)
    assert result["status"] == "ok" and result["marks"].shape == (15, 15)
    assert not result["marks"][:5, 5:].any()  # no edge between groups


def test_summaries_dispatch_on_the_lane_kind():
    def record(lane, method, arm, errors, **extra):
        return {
            "lane": lane,
            "cell": "c",
            "part": "all",
            "method": method,
            "arm": arm,
            "fraction": None if arm == "bare" else 0.1,
            "dataset_id": "x",
            "status": "ok",
            "errors": errors,
            "empty_errors": 5,
            "definite_marks": 0,
            "n_required_paths": 1,
            "n_forbidden_paths": 1,
            "n_confounded": 0,
            **extra,
        }

    for lane, measure in (("cdnod", "dag_errors"), ("jci", "errors"), ("multigroup", "errors")):
        extra = {"dag_errors": 3, "empty_dag_errors": 5} if measure == "dag_errors" else {}
        own = [record(lane, "m", "bare", 3, **extra), record(lane, "m", "true", 2, **extra)]
        ref = [record(lane, "r", "bare", 4, **extra), record(lane, "r", "true", 3, **extra)]
        if lane == "multigroup":
            assert "lingam.multi_group_direct_lingam" in kp.TAKES_PATHS
        kp.REFERENCES["m"] = ("r",)
        try:
            rows = kp.against_references(own + ref)
        finally:
            del kp.REFERENCES["m"]
        assert {row["measure"] for row in rows} == {measure}


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R with bnlearn")
def test_bnlearn_takes_the_edges_as_its_whitelist_and_blacklist():
    prob = kp.problem("cpdag", kp.LANES["cpdag"]["cells"]["d10-n500-deg2"], 0)
    edges = kp.edge_claims(prob["claims"][0.3], prob["truth"])
    for method in ("bnlearn.hc", "bnlearn.iamb"):
        marks = kp._bnlearn_reference(method, prob["data"], edges)
        assert marks.shape == (10, 10)
        for a, b in edges["required"]:
            assert marks[a, b] != kp.NULL  # a required edge is kept
        for a, b in edges["forbidden"]:
            assert not (marks[a, b] == kp.TAIL and marks[b, a] == kp.ARROW)
