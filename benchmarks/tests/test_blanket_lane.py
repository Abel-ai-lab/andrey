"""The set-learner lane: its answers and scorers, Andrey's adapter, and the driver.

The checks that need no R run everywhere: the packing of an answer, the true sets of a hand-built
DAG, the scores of fixed cases, Andrey's adapter against the facade it calls (and the table that
adds a learner by name against the facade's own choices), the cells and the field of a run, and one
smoke run of the driver with Andrey's solutions. The bnlearn adapter's checks that need R are in
``test_adapters_bnlearn_blanket.py``.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blanket_cpu  # noqa: E402

import andrey  # noqa: E402
from andrey_bench import blanket_lane as lane  # noqa: E402
from andrey_bench.adapters import andrey_blanket, bnlearn_blanket  # noqa: E402
from andrey_bench.adapters.andrey_blanket import LEARNERS, AndreyBlanket, Learner  # noqa: E402

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"


def _dag(d: int, edges) -> np.ndarray:
    """Endpoint marks of the DAG with an edge ``i -> j`` for each ``(i, j)``."""
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j in edges:
        marks[i, j], marks[j, i] = 1, 2
    return marks


#: 0 -> 2 <- 1, 2 -> 3, and 4 alone: the sets of 0 are the child 2 and the spouse 1.
DAG = _dag(5, [(0, 2), (1, 2), (2, 3)])


def _collider_data(n: int = 1500, seed: int = 0) -> np.ndarray:
    """Columns ``[a, b, c, e]`` with ``a -> c <- b`` and ``e`` independent of the rest."""
    rng = np.random.default_rng(seed)
    a, b, e = rng.normal(size=(3, n))
    c = a + b + 0.5 * rng.normal(size=n)
    return np.column_stack([a, b, c, e])


# --- the answers and the scorers ----------------------------------------------------------------


def test_an_answer_and_its_counts_round_trip():
    answer = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int8)
    packed = lane.pack_answer(answer, np.array([4, lane.NO_COUNT, 7]), 11)
    assert packed.shape == (5, 3)
    back, counts, total = lane.unpack_answer(packed)
    assert np.array_equal(back, answer)
    assert counts.tolist() == [4, lane.NO_COUNT, 7] and total == 11
    _, counts, total = lane.unpack_answer(lane.pack_answer(answer))
    assert (counts == lane.NO_COUNT).all() and total is None


def test_a_bare_square_answer_has_no_counts_and_other_shapes_are_refused():
    back, counts, total = lane.unpack_answer(np.zeros((4, 4), dtype=np.int8))
    assert back.shape == (4, 4) and (counts == lane.NO_COUNT).all() and total is None
    with pytest.raises(ValueError, match="square"):
        lane.pack_answer(np.zeros((3, 4)))
    with pytest.raises(ValueError, match=r"\(d, d\) or a \(d \+ 2, d\)"):
        lane.unpack_answer(np.zeros((5, 4)))


def test_the_true_sets_of_a_dag():
    assert lane.true_set(DAG, 0, "pc") == [2]
    assert lane.true_set(DAG, 0, "mb") == [1, 2]
    assert lane.true_set(DAG, 2, "mb") == lane.true_set(DAG, 2, "pc") == [0, 1, 3]
    assert lane.true_set(DAG, 3, "mb") == [2]
    assert lane.true_set(DAG, 4, "mb") == []
    with pytest.raises(ValueError, match="kind"):
        lane.true_set(DAG, 0, "parents")


def test_the_true_moral_graph_marries_the_parents_and_the_skeleton_does_not():
    skeleton, moral = lane.true_graph(DAG, "skeleton"), lane.true_graph(DAG, "moral")
    assert skeleton[0, 1] == 0 and moral[0, 1] == moral[1, 0] == 1
    assert (moral >= skeleton).all() and (skeleton == skeleton.T).all()
    assert skeleton.sum() == 6 and moral.sum() == 8
    with pytest.raises(ValueError, match="moral"):
        lane.true_graph(DAG, "cpdag")


def test_set_scores_of_fixed_cases():
    exact = lane.set_scores([1, 2], [2, 1])
    assert exact["exact"] == 1 and exact["f1"] == exact["precision"] == exact["recall"] == 1
    assert exact["distance"] == 0
    part = lane.set_scores([1, 5], [1, 2, 3, 4])
    assert (part["tp"], part["fp"], part["fn"], part["est"], part["true"]) == (1, 1, 3, 2, 4)
    assert part["precision"] == 0.5 and part["recall"] == 0.25 and part["exact"] == 0
    assert part["f1"] == pytest.approx(2 / 6)
    assert part["distance"] == pytest.approx(np.hypot(0.5, 0.75))


def test_an_empty_answer_has_no_precision_and_an_isolated_target_no_recall():
    empty = lane.set_scores([], [1, 2])
    assert empty["recall"] == 0 and empty["f1"] == 0 and empty["exact"] == 0
    assert np.isnan(empty["precision"]) and np.isnan(empty["distance"])
    isolated = lane.set_scores([3], [])
    assert isolated["precision"] == 0 and np.isnan(isolated["recall"])
    nothing = lane.set_scores([], [])
    assert nothing["exact"] == 1 and np.isnan(nothing["f1"])


def test_pooled_scores_weigh_by_set_size_and_the_means_by_target():
    scores = [lane.set_scores([1], [1]), lane.set_scores([1, 2, 3, 4], [1, 2, 3, 9])]
    pooled = lane.pooled_set_scores(scores)
    assert pooled["precision"] == pytest.approx(4 / 5) and pooled["recall"] == pytest.approx(4 / 5)
    assert pooled["mean_precision"] == pytest.approx((1 + 0.75) / 2)
    assert pooled["exact"] == 0.5 and pooled["targets"] == 2
    # An undefined ratio is left out of the mean, and an empty pool is undefined.
    with_empty = lane.pooled_set_scores([*scores, lane.set_scores([], [7])])
    assert with_empty["mean_precision"] == pooled["mean_precision"]
    assert with_empty["mean_recall"] == pytest.approx((1 + 0.75 + 0) / 3)
    assert np.isnan(lane.pooled_set_scores([])["precision"])


def test_pair_scores_count_pairs_and_read_either_entry():
    truth = lane.true_graph(DAG, "skeleton")
    perfect = lane.pair_scores(truth, truth)
    assert (perfect["shd"], perfect["f1"], perfect["empty_shd"]) == (0, 1, 3)
    half = np.zeros_like(truth)
    half[0, 2] = 1  # one direction of one edge
    half[0, 4] = 1  # an edge the truth lacks
    scored = lane.pair_scores(half, truth)
    assert (scored["tp"], scored["fp"], scored["fn"], scored["shd"]) == (1, 1, 2, 3)
    assert scored["precision"] == 0.5 and scored["recall"] == pytest.approx(1 / 3)
    assert np.isnan(lane.pair_scores(np.zeros_like(truth), np.zeros_like(truth))["f1"])


def test_sets_are_read_from_the_rows_of_the_targets_asked_for():
    answer = np.zeros((4, 4), dtype=np.int8)
    answer[1, [0, 3]] = 1
    answer[2, 2] = 1  # a target is never in its own set
    assert lane.sets_of(answer, [1, 2]) == {1: [0, 3], 2: []}


def test_answers_become_graphs_of_the_kind_they_give():
    sets = np.zeros((4, 4), dtype=np.int8)
    sets[0, 2] = 1  # one target's row only
    assert lane.answer_structure(sets, None).kind == "undirected"
    cpdag = np.zeros((3, 3), dtype=np.int8)
    cpdag[0, 1], cpdag[1, 0] = 1, 2  # 0 -> 1 as endpoint marks
    assert lane.answer_structure(cpdag, "cpdag").to_numpy()[1, 0] == 2
    adjacency = np.zeros((3, 3), dtype=np.int8)
    adjacency[0, 1] = 1  # 0 -> 1 as bnlearn's 0/1 matrix
    assert lane.answer_structure(adjacency, "cpdag").to_numpy()[1, 0] == 2


# --- Andrey's adapter ----------------------------------------------------------------------------


@pytest.mark.parametrize("learner", sorted(LEARNERS))
def test_the_table_says_what_the_facade_offers(learner):
    """A learner is added by naming it, so the table must match the function it names."""
    X = _collider_data(300)
    for structure in lane.STRUCTURES:
        adapter = None
        if structure in LEARNERS[learner].structures:
            adapter = AndreyBlanket(learner, indep_test="fisherz", structure=structure)
            kind = adapter.to_structure(adapter.fit(X, adapter.params())).kind
            assert kind == ("cpdag" if structure == "cpdag" else "undirected")
        else:
            with pytest.raises(ValueError, match="offers the graphs"):
                AndreyBlanket(learner, indep_test="fisherz", structure=structure)
            with pytest.raises(ValueError):
                getattr(andrey, learner)(X, structure=structure)
    target = getattr(andrey, learner)(X, target=0)
    assert ("markov_blanket" if LEARNERS[learner].kind == "mb" else "parents_and_children") in (
        target.metadata
    )


def test_the_adapter_answers_the_sets_the_facade_returns():
    X = _collider_data()
    adapter = AndreyBlanket("iamb", indep_test="fisherz")
    assert (adapter.name, adapter.kind, adapter.package) == ("andrey.iamb", "mb", "andrey")
    native = adapter.fit(X, {**adapter.params(), "targets": [0, 1, 2, 3]})
    answer, counts, _ = lane.unpack_answer(native)
    found = lane.sets_of(answer, [0, 1, 2, 3])
    for t in range(4):
        assert found[t] == sorted(andrey.iamb(X, target=t).metadata["markov_blanket"])
    assert found[0] == [1, 2] and found[3] == []
    assert answer.shape == (4, 4)


def test_only_the_targets_asked_for_are_answered():
    adapter = AndreyBlanket("mmpc", indep_test="fisherz")
    answer, counts, total = lane.unpack_answer(
        adapter.fit(_collider_data(), {**adapter.params(), "targets": [2]})
    )
    assert answer[2].tolist() == [1, 1, 0, 0] and not answer[[0, 1, 3]].any()
    assert counts[0] == lane.NO_COUNT


def test_a_whole_graph_answer_is_the_graph_of_the_structure():
    X = _collider_data()
    adapter = AndreyBlanket("iamb", indep_test="fisherz", structure="moral")
    moral = adapter.fit(X, adapter.params())
    expected = andrey.iamb(X, structure="moral").structure.to_numpy()
    assert np.array_equal(lane.unpack_answer(moral)[0], expected)
    assert adapter.to_structure(moral).kind == "undirected"
    cpdag = AndreyBlanket("iamb", indep_test="fisherz", structure="cpdag")
    structure = cpdag.to_structure(cpdag.fit(X, cpdag.params()))
    assert structure.kind == "cpdag" and structure.to_numpy()[2, 0] == 2  # a -> c


def test_a_run_without_targets_or_structure_is_refused():
    adapter = AndreyBlanket("iamb", indep_test="fisherz")
    with pytest.raises(ValueError, match="structure"):
        adapter.fit(_collider_data(100), adapter.params())


class _Output(SimpleNamespace):
    """What a facade returns, for the parts the adapter reads."""


def _stub(received: list, n_tests=None):
    def function(X, *, target=None, **settings):
        received.append({"target": target, **settings})
        star = np.zeros((X.shape[1],) * 2, dtype=np.int8)
        if target is not None:
            star[target, (target + 1) % X.shape[1]] = 1
        metadata = {} if n_tests is None else {"n_tests": n_tests + (target or 0)}
        return _Output(structure=SimpleNamespace(to_numpy=lambda: star), metadata=metadata)

    return function


def test_every_setting_reaches_the_facade_and_the_counts_the_adapter_reports(monkeypatch):
    received: list = []
    monkeypatch.setattr(andrey, "iamb", _stub(received, n_tests=10))
    adapter = AndreyBlanket("iamb", indep_test="gsq", alpha=0.01, max_depth=2)
    answer, counts, total = lane.unpack_answer(
        adapter.fit(np.zeros((20, 4)), {**adapter.params(), "targets": [0, 2]})
    )
    assert received == [
        {"target": t, "indep_test": "gsq", "alpha": 0.01, "max_depth": 2} for t in (0, 2)
    ]
    assert counts[[0, 2]].tolist() == [10, 12] and counts[1] == lane.NO_COUNT
    assert total == 22  # the asked targets' counts; the others have none
    assert answer[0, 1] == answer[2, 3] == 1


def test_a_fit_without_counts_reports_none(monkeypatch):
    monkeypatch.setattr(andrey, "iamb", _stub([]))
    adapter = AndreyBlanket("iamb", indep_test="fisherz")
    _, counts, total = lane.unpack_answer(
        adapter.fit(np.zeros((20, 3)), {**adapter.params(), "targets": [0, 1]})
    )
    assert (counts == lane.NO_COUNT).all() and total is None


def test_a_learner_is_added_by_naming_it(monkeypatch):
    received: list = []
    monkeypatch.setitem(LEARNERS, "gs", Learner("mb", ("cpdag", "skeleton")))
    monkeypatch.setattr(andrey, "gs", _stub(received), raising=False)
    assert "gs" in andrey_blanket.available()
    adapter = AndreyBlanket("gs", indep_test="gsq", max_depth=3)
    assert adapter.name == "andrey.gs"
    adapter.fit(np.zeros((20, 3)), {**adapter.params(), "targets": [1]})
    assert received[0]["max_depth"] == 3
    with pytest.raises(ValueError, match="offers the graphs"):
        AndreyBlanket("gs", indep_test="gsq", structure="moral")
    solutions = {
        a.name for a in blanket_cpu.field(blanket_cpu.Cell.parse("net:asia:200"), _SETTINGS, None)
    }
    assert {"andrey.gs", "bnlearn.gs"} <= solutions


def test_a_learner_the_installed_andrey_lacks_is_not_available(monkeypatch):
    monkeypatch.setitem(LEARNERS, "pcmb", Learner("mb", ("cpdag",)))
    monkeypatch.delattr(andrey, "pcmb", raising=False)
    assert "pcmb" not in andrey_blanket.available()
    with pytest.raises(RuntimeError, match="no 'pcmb'"):
        AndreyBlanket("pcmb", indep_test="gsq").setup()
    with pytest.raises(ValueError, match="learner must be one of"):
        AndreyBlanket("nope", indep_test="gsq")


def test_params_are_complete_json_and_the_adapter_pickles():
    for adapter in (
        AndreyBlanket("mmpc", indep_test="chisq", max_depth=1, structure="skeleton"),
        bnlearn_blanket.BnlearnBlanket("pc.stable", indep_test="gsq", kind="mb"),
        bnlearn_blanket.BnlearnBlanket("iamb", indep_test="gsq", sparse_rule=False),
    ):
        params = adapter.params()
        assert json.loads(json.dumps(params, sort_keys=True)) == params
        assert {"indep_test", "alpha", "max_depth", "kind", "structure"} <= set(params)
        assert pickle.loads(pickle.dumps(adapter)).name == adapter.name


# --- the bnlearn adapter's table and options, without R -----------------------------------------


def test_every_learner_with_a_reference_names_a_function_of_the_table():
    assert set(bnlearn_blanket.COUNTERPART.values()) <= set(bnlearn_blanket.FUNCTIONS)
    assert set(LEARNERS) <= set(bnlearn_blanket.COUNTERPART)
    for learner, function in bnlearn_blanket.COUNTERPART.items():
        kind = bnlearn_blanket.FUNCTIONS[function].kind
        if learner in LEARNERS:
            assert kind == LEARNERS[learner].kind


def test_the_test_option_picks_bnlearns_form_and_the_run_records_it():
    on = bnlearn_blanket.BnlearnBlanket("mmpc", indep_test="gsq", sparse_rule=True)
    off = bnlearn_blanket.BnlearnBlanket("mmpc", indep_test="gsq", sparse_rule=False)
    assert (on.params()["bnlearn_test"], on.params()["sparse_rule"]) == ("mi-adf", True)
    assert (off.params()["bnlearn_test"], off.params()["sparse_rule"]) == ("mi", False)
    chisq = bnlearn_blanket.BnlearnBlanket("mmpc", indep_test="chisq").params()
    assert chisq["bnlearn_test"] == "x2-adf"
    gaussian = bnlearn_blanket.BnlearnBlanket("iamb", indep_test="fisherz").params()
    assert (gaussian["bnlearn_test"], gaussian["sparse_rule"]) == ("zf", False)


def test_the_bnlearn_adapter_refuses_what_has_no_reference_setting():
    make = bnlearn_blanket.BnlearnBlanket
    with pytest.raises(ValueError, match="at least 1"):
        make("iamb", indep_test="fisherz", max_depth=0)
    with pytest.raises(ValueError, match="function must be one of"):
        make("hpc", indep_test="fisherz")
    with pytest.raises(ValueError, match="either set"):
        make("pc.stable", indep_test="fisherz")
    with pytest.raises(ValueError, match="'mb' set"):
        make("iamb", indep_test="fisherz", kind="pc")
    with pytest.raises(ValueError, match="gives the graphs"):
        make("mmpc", indep_test="fisherz", structure="cpdag")
    with pytest.raises(ValueError, match="indep_test must be"):
        make("iamb", indep_test="kci")


def test_solutions_are_named_for_their_function_and_pc_stable_for_its_set():
    make = bnlearn_blanket.BnlearnBlanket
    assert make("fast.iamb", indep_test="gsq").name == "bnlearn.fast-iamb"
    assert make("si.hiton.pc", indep_test="gsq").name == "bnlearn.si-hiton-pc"
    assert make("pc.stable", indep_test="gsq", kind="pc").name == "bnlearn.pc-stable.pc"
    assert make("pc.stable", indep_test="gsq", structure="cpdag").name == "bnlearn.pc-stable.graph"


def test_the_lane_stops_on_another_bnlearn_version(monkeypatch):
    monkeypatch.setattr(bnlearn_blanket.rsession, "package_version", lambda name: "5.3")
    with pytest.raises(SystemExit, match="written for bnlearn 5.2.1, and the R library has 5.3"):
        bnlearn_blanket.check_version()
    monkeypatch.setattr(bnlearn_blanket.rsession, "package_version", lambda name: "5.2.1")
    bnlearn_blanket.check_version()


# --- the cells and the field of a run -----------------------------------------------------------

_SETTINGS = blanket_cpu.Settings("gsq", True, 0.05, None)


def test_a_cell_is_a_network_or_a_gaussian_graph():
    network = blanket_cpu.Cell.parse("net:child:2000")
    assert (network.kind, network.name, network.n, network.label) == (
        "network",
        "child",
        2000,
        "net:child:2000",
    )
    gaussian = blanket_cpu.Cell.parse("gauss:20:1000:4")
    assert (gaussian.d, gaussian.n, gaussian.degree) == (20, 1000, 4.0)
    assert blanket_cpu.Cell.parse("gauss:20:1000").degree == 2.0
    assert gaussian.label == "gauss:20:1000:4"
    for bad in ("net:child", "gauss:20", "ring:3:4", "gauss:x:5"):
        with pytest.raises(ValueError, match="a cell is"):
            blanket_cpu.Cell.parse(bad)
    assert network.indep_test("chisq") == "chisq" and gaussian.indep_test("chisq") == "fisherz"
    assert all(blanket_cpu.Cell.parse(text).label.startswith(text) for text in blanket_cpu.CELLS)
    assert blanket_cpu.Cell.parse(gaussian.label) == gaussian


def test_a_gaussian_cell_is_a_function_of_its_seed():
    cell = blanket_cpu.Cell.parse("gauss:12:300")
    from andrey_bench import datasets

    X, truth, kind, _ = datasets.generate_one(cell.key(3))
    again = datasets.generate_one(cell.key(3))[0]
    assert X.shape == (300, 12) and kind == "dag" and truth.shape == (12, 12)
    assert np.array_equal(X, again) and not np.array_equal(X, datasets.generate_one(cell.key(4))[0])


def test_targets_are_every_column_or_a_draw_from_the_seed():
    assert blanket_cpu.targets_of(5, "all", 0) == [0, 1, 2, 3, 4]
    assert blanket_cpu.targets_of(5, "9", 0) == [0, 1, 2, 3, 4]
    drawn = blanket_cpu.targets_of(30, "4", 7)
    assert drawn == sorted(drawn) and len(set(drawn)) == 4 == len(drawn)
    assert drawn == blanket_cpu.targets_of(30, "4", 7) != blanket_cpu.targets_of(30, "4", 8)


def test_the_field_pairs_each_learner_with_its_bnlearn_function():
    cell = blanket_cpu.Cell.parse("net:asia:200")
    names = [a.name for a in blanket_cpu.field(cell, _SETTINGS, None)]
    assert names[-1] == "baseline.empty"
    for learner in andrey_blanket.available():
        assert f"andrey.{learner}" in names
        assert blanket_cpu.reference_name(f"andrey.{learner}", None) in names
    assert {"bnlearn.pc-stable.mb", "bnlearn.pc-stable.pc"} <= set(names)
    assert len(names) == len(set(names))
    assert blanket_cpu.reference_name("bnlearn.iamb", None) is None


def test_a_whole_graph_field_keeps_the_learners_that_offer_the_graph():
    cell = blanket_cpu.Cell.parse("gauss:8:200")
    moral = {a.name for a in blanket_cpu.field(cell, _SETTINGS, "moral")}
    assert "andrey.iamb" in moral and "andrey.mmpc" not in moral
    assert not {n for n in moral if "pc-stable" in n} and "bnlearn.iamb" in moral
    skeleton = {a.name for a in blanket_cpu.field(cell, _SETTINGS, "skeleton")}
    assert {"andrey.mmpc", "bnlearn.mmpc", "bnlearn.pc-stable.graph"} <= skeleton


def test_the_limit_on_the_conditioning_set_is_none_or_at_least_one():
    assert blanket_cpu.parse_max_depth("none") is None and blanket_cpu.parse_max_depth("3") == 3
    with pytest.raises(Exception, match="at least 1"):
        blanket_cpu.parse_max_depth("0")


# --- the driver ----------------------------------------------------------------------------------


@pytest.fixture
def asia_cache(tmp_path, monkeypatch):
    """A data cache holding the asia network's file, so no load reaches the network."""
    (tmp_path / "cache" / "asia").mkdir(parents=True)
    shutil.copy(ASIA, tmp_path / "cache" / "asia" / "asia.bif.gz")
    monkeypatch.setenv("ANDREY_DATA_DIR", str(tmp_path / "cache"))


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]


def test_the_driver_records_sets_scores_and_counts_and_rescores_them(asia_cache, tmp_path):
    out = tmp_path / "run"
    flags = ["--out", str(out), "--smoke", "--solutions", "andrey.iamb,andrey.mmpc,baseline.empty"]
    assert blanket_cpu.main(flags) == 0
    records = _records(out)
    assert {(r["solution"], r["kind"]) for r in records} == {
        ("andrey.iamb", "mb"),
        ("andrey.mmpc", "pc"),
        ("baseline.empty", "mb"),
        ("baseline.empty", "pc"),
    }
    assert all(r["status"] == "ok" and r["cell"] == "net:asia:500" for r in records)
    by = {(r["solution"], r["kind"]): r for r in records}
    iamb, empty = by["andrey.iamb", "mb"], by["baseline.empty", "mb"]
    assert iamb["targets"] == list(range(8)) and set(iamb["sets"]) == {str(t) for t in range(8)}
    assert iamb["pooled_f1"] > 0 and empty["pooled_f1"] == 0 and empty["pooled_recall"] == 0
    assert all(s["est"] == 0 for s in empty["per_target"].values())
    assert iamb["params"]["max_depth"] is None and iamb["params"]["indep_test"] == "gsq"
    with pytest.raises(SystemExit, match="already holds a run"):
        blanket_cpu.main(flags)
    meta = json.loads((out / "run_meta.json").read_text())
    assert (
        meta["discrete_test"] == "adf" and meta["sparse_rule"] is True and meta["mode"] == "target"
    )
    keys = ("sets", "per_target", "pooled_f1", "pooled_precision", "pooled_distance")
    # Compared as text, since an empty answer's precision is NaN.
    before = [json.dumps({k: r[k] for k in keys}, sort_keys=True) for r in records]
    assert blanket_cpu.main(["--out", str(out), "--rescore"]) == 0
    after = [json.dumps({k: r[k] for k in keys}, sort_keys=True) for r in _records(out)]
    assert after == before


def test_the_driver_scores_whole_graphs_and_resumes(asia_cache, tmp_path):
    out = tmp_path / "run"
    flags = [
        "--out",
        str(out),
        "--smoke",
        "--mode",
        "graph",
        "--structures",
        "moral",
        "cpdag",
        "--solutions",
        "andrey.iamb,baseline.empty",
    ]
    assert blanket_cpu.main(flags) == 0
    records = _records(out)
    assert {(r["solution"], r["structure"]) for r in records} == {
        ("andrey.iamb", "moral"),
        ("andrey.iamb", "cpdag"),
        ("baseline.empty", "moral"),
        ("baseline.empty", "cpdag"),
    }
    by = {(r["solution"], r["structure"]): r for r in records}
    assert by["andrey.iamb", "moral"]["shd"] < by["andrey.iamb", "moral"]["empty_shd"]
    assert by["andrey.iamb", "cpdag"]["shd"] < by["andrey.iamb", "cpdag"]["empty_shd"]
    assert by["baseline.empty", "moral"]["shd"] == by["baseline.empty", "moral"]["empty_shd"]
    assert blanket_cpu.main([*flags, "--resume"]) == 0
    assert len(_records(out)) == len(records)  # every fit's record was on disk
    assert blanket_cpu.main(["--out", str(out), "--rescore"]) == 0


def test_the_driver_stops_on_a_solution_it_does_not_have(asia_cache, tmp_path):
    with pytest.raises(SystemExit):
        blanket_cpu.main(["--out", str(tmp_path / "x"), "--smoke", "--solutions", "andrey.nope"])
