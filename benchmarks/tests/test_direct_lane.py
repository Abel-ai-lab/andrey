"""The lane of target-centered methods on categorical networks: the two readings' counts, the
datasets, the adapters, and the driver's rules.

The driver tests replace ``run_task`` with an in-process fit, so they need no harness environment;
the tests of the bnlearn arm skip unless R is loaded, as the other R lanes' do.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import direct_cpu  # noqa: E402

from andrey_bench import chessboard, networks  # noqa: E402
from andrey_bench import direct_lane as D  # noqa: E402
from andrey_bench import local_lane as L  # noqa: E402
from andrey_bench.adapters.andrey_pc import AndreyPCGsq  # noqa: E402
from andrey_bench.adapters.andrey_target import AndreyPCCounted, AndreyTarget  # noqa: E402
from andrey_bench.adapters.baseline_empty import BaselineEmpty  # noqa: E402
from andrey_bench.adapters.bnlearn_blanket import BnlearnBlanket  # noqa: E402
from andrey_bench.adapters.bnlearn_pc_graph import BnlearnPCStableGraph  # noqa: E402

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"
_TAIL, _ARROW = 1, 2


def _dag(d: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((d, d), dtype=np.int8)
    for i, j in edges:
        adj[i, j] = 1
    return adj


def _answer(d: int, edges: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """``edges[(i, j)] = (mark at i, mark at j)``."""
    marks = np.zeros((d, d), dtype=np.int8)
    for (i, j), (at_i, at_j) in edges.items():
        marks[i, j], marks[j, i] = at_i, at_j
    return marks


# ----- the counts -------------------------------------------------------------------------------

# 0 -> 2 <- 1 and 2 -> 3, node 4 alone: the collider fixes every edge of the CPDAG.
COLLIDER = _dag(5, [(0, 2), (1, 2), (2, 3)])
# 0 -> 1 -> 2: the CPDAG leaves both edges undirected.
CHAIN = _dag(3, [(0, 1), (1, 2)])


def _counts(answer: np.ndarray, dag: np.ndarray, target: int) -> tuple[dict, dict]:
    marks, cpdag = D.marks_of(dag), D.cpdag_of(dag)
    return D.dag_counts(answer, marks, target), D.cpdag_counts(answer, cpdag, target)


def test_the_truth_of_a_collider_is_its_own_cpdag():
    np.testing.assert_array_equal(D.cpdag_of(COLLIDER), D.marks_of(COLLIDER))
    cpdag = D.cpdag_of(CHAIN)
    assert (cpdag[0, 1], cpdag[1, 0], cpdag[1, 2], cpdag[2, 1]) == (_TAIL,) * 4


def test_the_truth_itself_costs_nothing_in_either_reading():
    dag, cpdag = _counts(D.cpdag_of(COLLIDER), COLLIDER, 2)
    assert dag == {"undirected": 0, "miss": 0, "extra": 0, "reverse": 0, "shd": 0}
    assert cpdag == {
        "extra": 0,
        "missing": 0,
        "reversed": 0,
        "wrongdir": 0,
        "over": 0,
        "under": 0,
        "total": 0,
    }


def test_a_missing_and_an_extra_edge():
    # At node 2 the answer has 0 -> 2 and 3 <- 2 right, lacks 1 -> 2, and adds 2 -> 4.
    answer = _answer(
        5,
        {(0, 2): (_TAIL, _ARROW), (2, 3): (_TAIL, _ARROW), (2, 4): (_TAIL, _ARROW)},
    )
    dag, cpdag = _counts(answer, COLLIDER, 2)
    assert (dag["miss"], dag["extra"], dag["reverse"], dag["undirected"]) == (1, 1, 0, 0)
    assert dag["shd"] == 2
    assert (cpdag["missing"], cpdag["extra"], cpdag["reversed"]) == (1, 1, 0)
    assert cpdag["total"] == 2


def test_a_reversed_edge_is_a_wrong_direction_in_both_readings():
    # 2 -> 0 where the truth has 0 -> 2.
    answer = _answer(5, {(2, 0): (_TAIL, _ARROW)})
    answer |= _answer(5, {(1, 2): (_TAIL, _ARROW), (2, 3): (_TAIL, _ARROW)})
    dag, cpdag = _counts(answer, COLLIDER, 2)
    assert (dag["reverse"], dag["miss"], dag["extra"], dag["undirected"]) == (1, 0, 0, 0)
    assert dag["shd"] == 1
    assert (cpdag["wrongdir"], cpdag["over"], cpdag["under"], cpdag["reversed"]) == (1, 0, 0, 1)
    assert cpdag["total"] == 1


def test_an_undirected_edge_the_cpdag_directs_is_an_under_orientation():
    # 0 - 2 where the truth, DAG and CPDAG alike, has 0 -> 2.
    answer = _answer(5, {(0, 2): (_TAIL, _TAIL), (1, 2): (_TAIL, _ARROW), (2, 3): (_TAIL, _ARROW)})
    dag, cpdag = _counts(answer, COLLIDER, 2)
    assert (dag["undirected"], dag["reverse"], dag["shd"]) == (1, 0, 1)
    assert (cpdag["under"], cpdag["over"], cpdag["wrongdir"], cpdag["reversed"]) == (1, 0, 0, 1)
    assert cpdag["total"] == 1


def test_the_two_readings_total_an_answer_differently():
    """The chain's edges are undirected in the CPDAG and directed in the DAG."""
    # Undirected, as the CPDAG has them: the DAG reading counts both edges, the CPDAG's none.
    undirected = _answer(3, {(0, 1): (_TAIL, _TAIL), (1, 2): (_TAIL, _TAIL)})
    dag, cpdag = _counts(undirected, CHAIN, 1)
    assert (dag["undirected"], dag["shd"]) == (2, 2)
    assert (cpdag["reversed"], cpdag["total"]) == (0, 0)
    # Directed as the DAG has them: the DAG reading is right, the CPDAG's over-orients both.
    directed = _answer(3, {(0, 1): (_TAIL, _ARROW), (1, 2): (_TAIL, _ARROW)})
    dag, cpdag = _counts(directed, CHAIN, 1)
    assert (dag["undirected"], dag["reverse"], dag["shd"]) == (0, 0, 0)
    assert (cpdag["over"], cpdag["reversed"], cpdag["total"]) == (2, 2, 2)
    # Directed against the DAG, a collider the CPDAG does not allow: wrong in both.
    collider = _answer(3, {(0, 1): (_TAIL, _ARROW), (1, 2): (_ARROW, _TAIL)})
    dag, cpdag = _counts(collider, CHAIN, 1)
    assert (dag["reverse"], dag["shd"]) == (1, 1)
    assert (cpdag["over"], cpdag["reversed"], cpdag["total"]) == (2, 2, 2)


def test_the_empty_answer_costs_the_degree_of_the_target():
    empty = np.zeros_like(D.marks_of(COLLIDER))
    for target, degree in [(2, 3), (0, 1), (4, 0)]:
        dag, cpdag = _counts(empty, COLLIDER, target)
        assert dag["shd"] == dag["miss"] == degree
        assert cpdag["total"] == cpdag["missing"] == degree


def test_only_the_edges_at_the_target_are_read():
    answer = D.marks_of(COLLIDER)
    star = D.star_at(answer, 2)
    assert L.target_edges(star, 2) == L.target_edges(answer, 2)
    assert not star[[0, 1, 3, 4]][:, [0, 1, 3, 4]].any()
    # An edge away from the target changes neither count.
    wrong = _answer(5, {(3, 4): (_TAIL, _ARROW)}) | answer
    assert D.scores(wrong, D.marks_of(COLLIDER), D.cpdag_of(COLLIDER), 2) == D.scores(
        answer, D.marks_of(COLLIDER), D.cpdag_of(COLLIDER), 2
    )


# ----- the datasets -----------------------------------------------------------------------------


def test_the_chessboard_has_2025_squares_and_3960_edges_pointing_down_or_right():
    adj = chessboard.adjacency()
    assert adj.shape == (2025, 2025) and adj.sum() == 3960
    side = chessboard.SIDE
    for i, j in zip(*np.nonzero(adj), strict=True):
        assert j == i + side or (j == i + 1 and j // side == i // side)
    assert np.triu(adj).sum() == adj.sum()  # node order is topological
    assert adj.sum(0).max() == adj.sum(1).max() == 2
    assert chessboard.labels()[side + 1] == "r1c1"


def test_the_chessboard_targets_are_ten_fixed_squares():
    targets = chessboard.targets()
    assert len(targets) == len(set(targets)) == chessboard.N_TARGETS
    assert targets == sorted(targets) and targets == chessboard.targets()
    assert 0 <= targets[0] and targets[-1] < chessboard.SIDE**2


def test_the_chessboard_rows_are_binary_dependent_and_repeat():
    adj = chessboard.adjacency()
    rows = chessboard.sample(adj, 1500, seed=3)
    assert rows.shape == (1500, 2025) and set(np.unique(rows)) == {0.0, 1.0}
    np.testing.assert_array_equal(rows, chessboard.sample(adj, 1500, seed=3))
    assert not np.array_equal(rows[:30], chessboard.sample(adj, 1500, seed=4)[:30])
    # A square and the square to its right are strongly dependent: the weights are at least 0.8.
    assert abs(np.corrcoef(rows[:, 100], rows[:, 101])[0, 1]) > 0.1


@pytest.fixture
def asia_cache(tmp_path, monkeypatch):
    """The small published network of the tests' fixtures, as the lane's networks cache."""
    folder = tmp_path / "networks"
    folder.mkdir()
    shutil.copy(ASIA, folder / "asia.bif.gz")
    monkeypatch.setenv(networks.CACHE_ENV, str(folder))
    return networks.load_network("asia")


def test_a_permuted_network_keeps_its_data_its_truth_and_its_labels_together(asia_cache):
    net = asia_cache
    draw = D.draw("asia", 300, seed=5)
    assert draw.data.shape == (300, net.d) and draw.targets == list(range(net.d))
    assert sorted(draw.labels) == sorted(net.nodes) and draw.labels != list(net.nodes)
    arcs = {(draw.labels[i], draw.labels[j]) for i, j in zip(*np.nonzero(draw.dag), strict=True)}
    assert arcs == set(net.arcs)
    # Column j holds the draw of the node it is labeled with.
    original = net.sample(300, 5)
    for j, label in enumerate(draw.labels):
        np.testing.assert_array_equal(draw.data[:, j], original[:, net.nodes.index(label)])
    # The truth does not depend on the sample size.
    other = D.truth("asia", seed=5)
    np.testing.assert_array_equal(other.dag, draw.dag)
    assert other.labels == draw.labels and other.targets == draw.targets
    assert D.draw("asia", 300, seed=6).labels != draw.labels


def test_the_chessboard_draw_permutes_data_truth_and_targets_together():
    draw = D.draw(D.CHESSBOARD, 40, seed=2)
    assert draw.data.shape == (40, 2025) and draw.dag.sum() == 3960
    assert len(draw.targets) == 10 and draw.targets == sorted(draw.targets)
    assert {draw.labels[t] for t in draw.targets} == {
        chessboard.labels()[t] for t in chessboard.targets()
    }
    for i, j in zip(*np.nonzero(draw.dag), strict=True):
        (ri, ci), (rj, cj) = (tuple(int(x) for x in draw.labels[k][1:].split("c")) for k in (i, j))
        assert (rj, cj) in {(ri + 1, ci), (ri, ci + 1)}
    np.testing.assert_array_equal(D.truth(D.CHESSBOARD, seed=2).dag, draw.dag)


# ----- the adapters -----------------------------------------------------------------------------


def _collider_data(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Binary columns ``[a, t, b, c]`` with ``a -> t <- b`` and ``t -> c``."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 2, (2, n))
    t = rng.random(n) < 1 / (1 + np.exp(-(4 * (a + b) - 4)))
    c = rng.random(n) < 1 / (1 + np.exp(-(4 * t - 2)))
    return np.column_stack([a, t, b, c]).astype(np.float64)


def test_the_target_adapter_names_its_method_and_passes_every_setting():
    mb, cmb = AndreyTarget("mb_by_mb"), AndreyTarget("cmb")
    assert (mb.name, cmb.name) == ("andrey.mb_by_mb", "andrey.cmb")
    assert mb.params() == {"alpha": 0.05, "indep_test": "gsq", "blanket": "iamb_forward"}
    assert cmb.params() == {"alpha": 0.02, "indep_test": "gsq", "blanket": "hiton_mb"}
    assert all(isinstance(a, L.LocalAdapter) and a.local for a in (mb, cmb))
    assert AndreyTarget("mb_by_mb", blanket="iamb").name == "andrey.mb_by_mb.iamb"
    assert AndreyTarget("cmb", blanket="hiton_mb").name == "andrey.cmb"
    with pytest.raises(ValueError, match="method"):
        AndreyTarget("mmb_by_mmb")


@pytest.mark.parametrize("method", ["mb_by_mb", "cmb"])
def test_a_target_adapter_orients_the_collider_at_its_target(method):
    adapter = AndreyTarget(method)
    adapter.setup()
    answer, tests = L.unpack_answer(
        adapter.fit(_collider_data(), {**adapter.params(), L.TARGET_PARAM: 1})
    )
    assert tests is not None and tests > 0
    at = L.target_edges(answer, 1)
    assert at == {0: (_ARROW, _TAIL), 2: (_ARROW, _TAIL), 3: (_TAIL, _ARROW)}
    assert adapter.to_structure(L.pack_answer(answer)).kind == "cpdag"


def test_a_blanket_other_than_the_default_reaches_the_method():
    adapter = AndreyTarget("mb_by_mb", blanket="iamb")
    adapter.setup()
    answer, _ = L.unpack_answer(
        adapter.fit(_collider_data(), {**adapter.params(), L.TARGET_PARAM: 1})
    )
    assert set(L.target_edges(answer, 1)) == {0, 2, 3}
    with pytest.raises(ValueError):
        adapter.fit(
            _collider_data(), {**adapter.params(), "blanket": "nonesuch", L.TARGET_PARAM: 1}
        )


def test_andreys_pc_arm_keeps_its_name_counts_its_tests_and_takes_the_cap():
    adapter, plain = AndreyPCCounted(), AndreyPCGsq()
    assert adapter.name == plain.name == "andrey.pc.gsq" and not adapter.local
    assert adapter.params() == {"alpha": 0.05, "indep_test": "gsq", "max_depth": None}
    adapter.setup()
    X = _collider_data()
    native = adapter.fit(X, adapter.params())
    marks, tests = L.unpack_answer(native)
    np.testing.assert_array_equal(marks, plain.fit(X, plain.params()))
    assert (marks[0, 1], marks[1, 0], marks[3, 1], marks[1, 3]) == (_TAIL, _ARROW, _ARROW, _TAIL)
    assert tests is not None and tests > 0
    capped = L.unpack_answer(adapter.fit(X, {**adapter.params(), "max_depth": 0}))[1]
    assert capped is not None and capped < tests
    assert adapter.to_structure(native).kind == "cpdag"


def test_the_bnlearn_arm_is_the_whole_graph_cpdag_run_of_pc_stable():
    adapter = BnlearnPCStableGraph()
    assert isinstance(adapter, BnlearnBlanket) and not adapter.local
    assert adapter.name == "bnlearn.pc-stable.graph" and adapter.output_type == "cpdag"
    assert adapter.params()["bnlearn_test"] == "mi-adf" and adapter.params()["max_depth"] is None
    assert adapter.params()["drop_constant"] is True
    assert BnlearnPCStableGraph(alpha=0.02, max_depth=5).params()["max_depth"] == 5


@needs_r()
def test_bnlearn_pc_stable_answers_with_the_cpdag_and_leaves_a_constant_column_out():
    adapter = BnlearnPCStableGraph()
    adapter.setup()
    X = np.column_stack([_collider_data(), np.ones(4000)])
    native = adapter.fit(X, adapter.params())
    marks, tests = L.unpack_answer(native)
    assert marks.shape == (5, 5) and tests is not None and tests > 0
    assert not marks[4].any() and not marks[:, 4].any()  # the constant column has no edges
    assert L.target_edges(marks, 1) == {0: (_ARROW, _TAIL), 2: (_ARROW, _TAIL), 3: (_TAIL, _ARROW)}
    assert adapter.to_structure(native).kind == "cpdag"
    capped = adapter.fit(X, {**adapter.params(), "max_depth": 1})
    assert L.unpack_answer(capped)[1] is not None


@needs_r()
def test_bnlearn_pc_stable_answers_the_empty_graph_when_no_column_varies():
    adapter = BnlearnPCStableGraph()
    adapter.setup()
    marks, tests = L.unpack_answer(adapter.fit(np.ones((50, 3)), adapter.params()))
    assert not marks.any() and tests == 0


# ----- the driver, with the fit run in this process ---------------------------------------------


class Recorder:
    """The stand-in ``run_task``: fits in this process, notes each call, and can fail one solution."""

    def __init__(self, failing: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.failing = failing or {}

    def __call__(self, adapter, data, params, **kwargs):
        self.calls.append((adapter.name, params.get(L.TARGET_PARAM)))
        if adapter.name in self.failing:
            return [{"status": self.failing[adapter.name], "wall_s": None}]
        native = adapter.fit(data, params)
        return [{"status": "ok", "adj": native, "wall_s": 0.01, "cpu_s": 0.01}]


@pytest.fixture
def lane(tmp_path, asia_cache, monkeypatch):
    """The driver on asia, its fits run in this process, without the bnlearn arm."""
    recorder = Recorder()
    monkeypatch.setattr(direct_cpu, "run_task", recorder)
    monkeypatch.setattr(direct_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(direct_cpu, "GLOBAL_ARMS", [AndreyPCCounted()])
    flags = ["--out", str(tmp_path / "run"), "--networks", "asia", "--rows", "1000"]
    return tmp_path / "run", flags + ["--seeds", "2", "--targets", "2"], recorder


ALL_THREE = "andrey.mb_by_mb,andrey.pc.gsq,baseline.empty"


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]


def test_a_local_solution_fits_once_per_target_and_a_global_one_once_per_dataset(lane):
    out, flags, recorder = lane
    assert direct_cpu.main(flags + ["--solutions", ALL_THREE]) == 0
    # Two datasets and two targets: four fits of MB-by-MB, two of PC, two of the empty answer.
    calls = recorder.calls
    assert sorted(c for c in calls if c[0] == "andrey.mb_by_mb") == sorted(
        [("andrey.mb_by_mb", t) for t in (0, 1)] * 2
    )
    assert [c for c in calls if c[0] == "andrey.pc.gsq"] == [("andrey.pc.gsq", None)] * 2
    assert [c for c in calls if c[0] == "baseline.empty"] == [("baseline.empty", None)] * 2
    records = _records(out)
    for name, scope in [("andrey.mb_by_mb", "target"), ("andrey.pc.gsq", "graph")]:
        mine = [r for r in records if r["solution"] == name]
        assert len(mine) == 4 and {r["scope"] for r in mine} == {scope}
    assert len([r for r in records if r["solution"] == "baseline.empty"]) == 4
    assert {r["cell"] for r in records} == {"net:asia:1000"}


def test_a_record_holds_the_marks_at_its_target_and_both_readings(lane):
    out, flags, _ = lane
    direct_cpu.main(flags + ["--solutions", "andrey.cmb,andrey.pc.gsq,baseline.empty"])
    for r in _records(out):
        assert r["status"] == "ok" and r["d"] == 8 and set(r["scores"]) == {"dag", "cpdag"}
        marks = L.entries_to_marks(r["marks"], r["d"])
        np.testing.assert_array_equal(marks, D.star_at(marks, r["target"]))
        if r["solution"] == "andrey.cmb":
            assert r["n_tests"] > 0 and r["params"]["alpha"] == 0.05
            assert "max_depth" not in r["params"] and r["params"][L.TARGET_PARAM] == r["target"]
        if r["solution"] == "baseline.empty":
            assert r["n_tests"] is None and r["marks"] == []
            true = D.truth("asia", r["seed"])
            degree = int(true.dag[r["target"]].sum() + true.dag[:, r["target"]].sum())
            assert r["scores"]["dag"]["shd"] == r["scores"]["cpdag"]["total"] == degree


def test_the_papers_set_the_flags_the_flags_leave_open(lane):
    out, flags, _ = lane
    cell = flags[2:-4] + ["--seeds", "1", "--targets", "1"]

    def run(name: str, *more: str) -> tuple[dict, dict]:
        folder = out.parent / name
        direct_cpu.main(["--out", str(folder), *cell, *more])
        pc = next(r for r in _records(folder) if r["solution"] == "andrey.pc.gsq")
        return json.loads((folder / "run_meta.json").read_text()), pc["params"]

    meta, pc = run("default")
    assert (meta["alpha"], meta["max_depth"], meta["paper"]) == (0.05, 5, "mb_by_mb")
    assert meta["solution list"] == ["andrey.mb_by_mb", "andrey.pc.gsq", "baseline.empty"]
    assert (pc["alpha"], pc["max_depth"]) == (0.05, 5)
    meta, pc = run("cmb", "--paper", "cmb")
    assert (meta["alpha"], meta["max_depth"], meta["paper"]) == (0.02, None, "cmb")
    assert meta["solution list"] == ["andrey.cmb", "andrey.pc.gsq", "baseline.empty"]
    assert (pc["alpha"], pc["max_depth"]) == (0.02, None)
    _, pc = run("flags", "--paper", "cmb", "--alpha", "0.01", "--max-depth", "3")
    assert (pc["alpha"], pc["max_depth"]) == (0.01, 3)
    _, pc = run("none", "--max-depth", "none")
    assert pc["max_depth"] is None


def test_the_papers_defaults_are_their_networks_sizes_levels_and_caps():
    assert direct_cpu.PAPERS["mb_by_mb"]["networks"] == ["alarm", "hailfinder"]
    assert direct_cpu.PAPERS["mb_by_mb"]["rows"] == [500, 1000, 2000, 5000]
    assert direct_cpu.PAPERS["mb_by_mb"]["alpha"] == 0.05
    assert direct_cpu.PAPERS["mb_by_mb"]["max_depth"] == 5
    assert direct_cpu.PAPERS["cmb"] | {"solutions": None} == {
        "networks": ["alarm"],
        "rows": [1000],
        "alpha": 0.02,
        "max_depth": None,
        "solutions": None,
    }
    names = {a.name for a in direct_cpu.solutions()}
    assert {n for p in direct_cpu.PAPERS.values() for n in p["solutions"]} <= names
    assert direct_cpu.solutions()[-1].name == BaselineEmpty().name
    assert direct_cpu.cell_label("alarm", 500) == "net:alarm:500"


def test_a_fit_past_the_cap_or_one_that_fails_is_a_status_with_no_scores(lane, monkeypatch):
    out, flags, _ = lane
    recorder = Recorder({"andrey.pc.gsq": "timeout", "andrey.cmb": "error"})
    monkeypatch.setattr(direct_cpu, "run_task", recorder)
    assert direct_cpu.main(flags + ["--solutions", "andrey.cmb,andrey.pc.gsq,baseline.empty"]) == 0
    records = _records(out)
    for name, status in [("andrey.pc.gsq", "timeout"), ("andrey.cmb", "error")]:
        mine = [r for r in records if r["solution"] == name]
        assert len(mine) == 4  # one per target, though PC fitted once per dataset
        assert {r["status"] for r in mine} == {status}
        assert all(
            r["marks"] is None and r["scores"] is None and r["n_tests"] is None for r in mine
        )
    assert {r["status"] for r in records if r["solution"] == "baseline.empty"} == {"ok"}


def test_a_finished_run_is_not_overwritten_and_resumes_without_new_fits(lane):
    out, flags, recorder = lane
    direct_cpu.main(flags)
    done, written = len(recorder.calls), len(_records(out))
    with pytest.raises(SystemExit, match="already holds a run"):
        direct_cpu.main(flags)
    assert direct_cpu.main([*flags, "--resume"]) == 0
    assert len(recorder.calls) == done and len(_records(out)) == written
    with pytest.raises(SystemExit, match="differs"):
        direct_cpu.main([*flags, "--resume", "--alpha", "0.01"])


def test_a_resumed_run_fits_only_what_is_missing(lane):
    out, flags, recorder = lane
    direct_cpu.main(flags + ["--solutions", "andrey.mb_by_mb"])
    path = out / "records.jsonl"
    kept = [r for r in _records(out) if not (r["seed"] == 1 and r["target"] == 1)]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    before = len(recorder.calls)
    direct_cpu.main([*flags, "--solutions", "andrey.mb_by_mb", "--resume"])
    assert recorder.calls[before:] == [("andrey.mb_by_mb", 1)]
    assert len(_records(out)) == len(kept) + 1


def test_a_run_is_rescored_from_its_stored_marks(lane):
    out, flags, _ = lane
    direct_cpu.main(flags + ["--solutions", ALL_THREE])
    before = _records(out)
    for r in before:  # damage the stored scores, so the rescore has something to restore
        r["scores"] = {"dag": {"shd": -1}, "cpdag": {"total": -1}}
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in before))
    assert direct_cpu.main(["--out", str(out), "--rescore"]) == 0
    for r in _records(out):
        assert r["scores"]["dag"]["shd"] >= 0 and r["scores"]["cpdag"]["total"] >= 0
        truth = D.truth(r["network"], r["seed"]).dag
        marks = L.entries_to_marks(r["marks"], r["d"])
        assert r["scores"] == D.scores(marks, D.marks_of(truth), D.cpdag_of(truth), r["target"])


def test_the_parts_of_a_run_split_the_datasets(lane):
    out, flags, _ = lane
    direct_cpu.main(flags + ["--part", "1", "--parts", "2"])
    assert {r["seed"] for r in _records(out)} == {1}


def test_the_smoke_run_replaces_the_cell_flags(lane, monkeypatch):
    out, flags, recorder = lane
    smoke = {"networks": ["asia"], "rows": [1000], "seeds": 1, "targets": 2, "cap_wall_s": 300.0}
    monkeypatch.setattr(direct_cpu, "SMOKE", smoke)
    direct_cpu.main(["--out", str(out), "--smoke", "--networks", "alarm", "--seeds", "9"])
    records = _records(out)
    assert {(r["cell"], r["seed"]) for r in records} == {("net:asia:1000", 0)}
    assert {r["target"] for r in records} == {0, 1}


def test_an_unknown_network_solution_or_cap_is_refused(lane):
    out, flags, _ = lane
    with pytest.raises(SystemExit):
        direct_cpu.main(["--out", str(out), "--networks", "nonesuch"])
    with pytest.raises(SystemExit):
        direct_cpu.main([*flags, "--solutions", "nonesuch"])
    with pytest.raises(SystemExit):
        direct_cpu.main([*flags, "--max-depth", "0"])
    with pytest.raises(SystemExit):
        direct_cpu.main(["--out", str(out), "--paper", "nonesuch"])
