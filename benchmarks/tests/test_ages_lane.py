"""The AGES lane: the paper's simulation, the scores of a partially directed graph, the adapters,
and the per-setting comparison.

A graph that directs more edges than the CPDAG is scored against the graphs between the true
CPDAG and the true DAG, so the checks pin which edges count as errors there, and that the two
packages' graphs reach the scores in one encoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ages_cpu  # noqa: E402
import ages_summary  # noqa: E402

from andrey_bench.adapters.andrey_ages import AndreyAGES  # noqa: E402
from andrey_bench.adapters.pcalg_ages import PcalgAGES  # noqa: E402


def _marks(d: int, directed=(), undirected=()) -> np.ndarray:
    m = np.zeros((d, d), dtype=np.int8)
    for a, b in directed:
        m[a, b], m[b, a] = 1, 2
    for a, b in undirected:
        m[a, b] = m[b, a] = 1
    return m


def test_the_lane_holds_the_papers_23_settings():
    assert len(set(ages_cpu.CELLS)) == len(ages_cpu.CELLS) == 23
    assert sum(qw > 0 for _, qw in ages_cpu.CELLS) == 13
    assert all(qs + qw <= 1 + 1e-9 for qs, qw in ages_cpu.CELLS)
    assert ages_cpu.parse_cell("0.3:0.7") == (0.3, 0.7)
    with pytest.raises(ValueError, match="not a setting of the paper"):
        ages_cpu.parse_cell("0.2:0.2")


def test_a_dataset_is_a_function_of_its_setting_and_seed():
    X, dag = ages_cpu.sample(0.3, 0.3, 200, 4)
    again, same = ages_cpu.sample(0.3, 0.3, 200, 4)
    other, _ = ages_cpu.sample(0.3, 0.3, 200, 5)
    assert X.shape == (200, ages_cpu.D) and dag.shape == (ages_cpu.D, ages_cpu.D)
    assert np.array_equal(X, again) and np.array_equal(dag, same)
    assert not np.array_equal(X, other)


def test_the_edge_shares_are_the_settings():
    """Without a missing share every pair is an edge; else the share of pairs is ``qs + qw``."""
    pairs = ages_cpu.D * (ages_cpu.D - 1) // 2
    _, complete = ages_cpu.sample(0.3, 0.7, 10, 0)
    assert np.count_nonzero(complete == 2) == pairs
    edges = [np.count_nonzero(ages_cpu.sample(0.1, 0.3, 10, s)[1] == 2) for s in range(200)]
    assert abs(np.mean(edges) / pairs - 0.4) < 0.02


def test_weak_edges_are_weak_and_strong_edges_strong():
    """The coefficient of a parent in its child's regression on all its parents is the weight."""
    X, dag = ages_cpu.sample(0.5, 0.5, 200_000, 1)
    magnitudes = []
    for child in range(ages_cpu.D):
        parents = np.flatnonzero(dag[child] == 2)
        if parents.size:
            magnitudes += np.abs(
                np.linalg.lstsq(X[:, parents], X[:, child], rcond=None)[0]
            ).tolist()
    magnitudes = np.array(magnitudes)
    weak, strong = magnitudes[magnitudes < 0.5], magnitudes[magnitudes >= 0.5]
    assert weak.size and strong.size and weak.size + strong.size == 45
    assert weak.min() > 0.08 and weak.max() < 0.32 and strong.min() > 0.78 and strong.max() < 1.22


def test_the_generating_dags_point_to_earlier_and_later_columns():
    """The columns come in an order drawn from the seed, not in the causal order."""
    forward = backward = 0
    for seed in range(3):
        _, dag = ages_cpu.sample(0.3, 0.3, 10, seed)
        parent, child = np.nonzero((dag == 1) & (dag.T == 2))
        forward += int((parent < child).sum())
        backward += int((parent > child).sum())
    assert forward and backward


def test_a_correct_extra_direction_is_no_error_and_a_wrong_one_is():
    """On the chain 0 -> 1 -> 2 the true CPDAG directs nothing."""
    dag = _marks(3, directed=[(0, 1), (1, 2)])
    cpdag = ages_cpu.scores(_marks(3, undirected=[(0, 1), (1, 2)]), dag)
    assert (cpdag["shd"], cpdag["shd_cpdag"], cpdag["shd_dag"]) == (0, 0, 2)
    assert (cpdag["correct"], cpdag["directed"], cpdag["true_edges"]) == (0, 0, 2)
    more = ages_cpu.scores(_marks(3, directed=[(0, 1)], undirected=[(1, 2)]), dag)
    assert (more["shd"], more["shd_cpdag"], more["shd_dag"]) == (0, 1, 1)
    assert (more["correct"], more["directed"]) == (1, 1)
    wrong = ages_cpu.scores(_marks(3, directed=[(1, 0)], undirected=[(1, 2)]), dag)
    assert (wrong["shd"], wrong["shd_cpdag"], wrong["shd_dag"]) == (1, 1, 2)
    assert (wrong["correct"], wrong["directed"]) == (0, 1)


def test_an_edge_the_true_cpdag_directs_must_be_directed():
    """On the collider 0 -> 2 <- 1 the true CPDAG is the DAG."""
    dag = _marks(3, directed=[(0, 2), (1, 2)])
    assert ages_cpu.scores(dag, dag)["shd"] == 0
    undirected = ages_cpu.scores(_marks(3, undirected=[(0, 2), (1, 2)]), dag)
    assert (undirected["shd"], undirected["shd_cpdag"], undirected["shd_dag"]) == (2, 2, 2)
    empty = ages_cpu.scores(_marks(3), dag)
    assert empty["shd"] == empty["shd_cpdag"] == empty["shd_dag"] == empty["empty_shd"] == 2
    extra = ages_cpu.scores(_marks(3, directed=[(0, 2), (1, 2)], undirected=[(0, 1)]), dag)
    assert extra["shd"] == 1


def test_compare_reads_the_side_of_the_interval_by_the_measures_direction():
    low, high = [1.0] * 12, [2.0, 3.0] * 6
    assert ages_summary.compare(high, low)["verdict"] == "worse"
    assert ages_summary.compare(low, high)["verdict"] == "better"
    assert ages_summary.compare(high, low, higher_is_better=True)["verdict"] == "better"
    same = ages_summary.compare(low, low)
    assert same["verdict"] == "tie" and same["bootstrap_95"] == [0.0, 0.0] and same["paired"] == 12
    mixed = ages_summary.compare([1.0, 2.0] * 6, [2.0, 1.0] * 6)
    assert mixed["verdict"] == "tie" and mixed["bootstrap_95"][0] < 0 < mixed["bootstrap_95"][1]


def test_colliders_lists_unshielded_ones_only():
    assert ages_summary.colliders(_marks(3, directed=[(0, 2), (1, 2)])) == {(0, 2, 1)}
    assert not ages_summary.colliders(_marks(3, directed=[(0, 2), (1, 2)], undirected=[(0, 1)]))


def _record(solution, seed, marks, dag, wall=1.0):
    return {
        "qs": 0.3, "qw": 0.3, "d": 3, "n": 100, "seed": seed, "solution": solution, "repeat": 0,
        "status": "ok", "wall_s": wall, "peak_rss_mb": 50.0, "marks": marks.tolist(),
        **ages_cpu.scores(marks, dag),
    }  # fmt: skip


def test_a_setting_with_the_references_graphs_is_identical_and_beats_the_floor(tmp_path):
    dag = _marks(3, directed=[(0, 1), (1, 2)])
    ages, ges = _marks(3, directed=[(0, 1), (1, 2)]), _marks(3, undirected=[(0, 1), (1, 2)])
    lines = []
    for seed in range(12):
        lines += [
            _record(ages_summary.ANDREY, seed, ages, dag, wall=0.5),
            _record(ages_summary.REFERENCE, seed, ages, dag, wall=2.0),
            _record(ages_summary.GES, seed, ges, dag, wall=0.1),
        ]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (cell,) = ages_summary.summarize(ages_summary.load([str(tmp_path)]))
    assert cell["reference"]["verdict"] == "identical" and cell["reference"]["same_graph"] == 12
    assert cell["reference"]["time_ratio"] == 4.0 and cell["ges"]["time_ratio"] == 5.0
    assert cell["floor"]["andrey_minus_empty"]["bootstrap_95"][1] < 0
    assert cell["ges"]["measures"]["correct"]["verdict"] == "better"
    assert cell["ges"]["measures"]["wrong"]["verdict"] == "tie"
    assert cell["ges"]["same_skeleton"] == 12 and cell["ges"]["new_colliders"] == 0
    assert cell["solutions"][ages_summary.ANDREY]["recall"] == 1.0
    assert cell["solutions"][ages_summary.GES]["precision"] is None


def test_andrey_directs_more_than_ges_on_one_skeleton():
    """The three Andrey graphs of one dataset: AGES keeps the GES skeleton and its directions."""
    X, _ = ages_cpu.sample(0.3, 0.7, 10000, 0)
    ages_arm, _, ges_arm = ages_cpu.SOLUTIONS
    for adapter in (ages_arm, ges_arm):
        adapter.setup()
    ages = ages_arm.to_structure(ages_arm.fit(X, ages_arm.params())).to_numpy()
    ges = ges_arm.to_structure(ges_arm.fit(X, ges_arm.params())).to_numpy()
    assert np.array_equal(ages != 0, ges != 0)
    directed = lambda m: (m == 1) & (m.T == 2)  # noqa: E731
    assert not (directed(ges) & ~directed(ages)).any()
    assert directed(ages).sum() > directed(ges).sum()


@needs_r()
def test_pcalg_returns_andreys_graph_in_andreys_encoding():
    X, _ = ages_cpu.sample(0.3, 0.7, 1000, 0)
    mine, reference = AndreyAGES(), PcalgAGES()
    mine.setup()
    reference.setup()
    expected = mine.to_structure(mine.fit(X, mine.params())).to_numpy()
    got = reference.to_structure(reference.fit(X, reference.params())).to_numpy()
    assert np.array_equal(got, expected)
    assert ((got == 1) & (got.T == 2)).any() and ((got == 1) & (got.T == 1)).any()


def test_the_driver_records_scores_and_rescores_them(tmp_path):
    out = tmp_path / "run"
    flags = ["--out", str(out), "--smoke", "--solutions", "andrey.ages.numpy,andrey.ges.numpy.bic"]
    assert ages_cpu.main(flags) == 0
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert [r["solution"] for r in records] == ["andrey.ages.numpy", "andrey.ges.numpy.bic"]
    assert all(
        r["status"] == "ok" and (r["qs"], r["qw"], r["n"]) == (0.3, 0.7, 1000) for r in records
    )
    assert records[0]["empty_shd"] == 45 and records[0]["shd"] < 45
    with pytest.raises(SystemExit, match="already holds a run"):
        ages_cpu.main(flags)
    before = [{k: r[k] for k in ("shd", "correct", "directed")} for r in records]
    assert ages_cpu.main(["--out", str(out), "--rescore"]) == 0
    after = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert [{k: r[k] for k in ("shd", "correct", "directed")} for r in after] == before
