"""The BOSS-FCI and GRaSP-FCI lane: the paper's simulation, the scores, the arms, and the summary.

The generator, the scores, and Andrey's child are checked in bench-env; the Tetrad arms by the
arguments they pass to causal-cmd, which the lane's dry run checks against causal-cmd's own parser.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import boss_grasp_fci_cpu as lane
import networkx as nx
import numpy as np
import pytest
from conftest import needs_r

from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure
from andrey.data.latent import marginal
from andrey_bench.adapters import andrey_boss_fci, tetrad_pag


def _cell(**changes) -> lane.Cell:
    return dataclasses.replace(lane.cell_named("P20|deg=4|L=4|setting=paper|n=200"), **changes)


def test_grid_has_the_designs_rows() -> None:
    cells = lane.grid()
    claimed = [c for c in cells if c.claim == "in"]
    assert len(claimed) == 73
    assert sorted({c.block for c in cells if c.claim == "out"}) == ["O1", "O2", "O3"]
    assert len({c.name for c in cells}) == len(cells)
    assert all(c.seeds == tuple(range(20)) for c in claimed)


@pytest.mark.parametrize("degree", [2, 4, 6])
def test_forward_graph_has_degree_times_d_over_2_edges_and_hides_at_most_l(degree) -> None:
    for seed in range(5):
        cell = _cell(degree=degree, latents=8, n=50)
        X, truth = lane.dataset(cell, seed)
        assert len(truth["arcs"]) == degree * 20 // 2
        assert truth["latents_hidden"] == min(8, truth["candidates"])
        assert truth["observed"] + truth["latents_hidden"] == truth["nodes"] == 28
        assert X.shape == (50, truth["observed"])


def test_a_cell_with_nothing_hidden_measures_every_node() -> None:
    X, truth = lane.dataset(_cell(latents=0, n=50), 0)
    assert truth["latents_hidden"] == 0 and truth["observed"] == truth["nodes"] == X.shape[1] == 20


def test_hiding_nodes_keeps_the_graph() -> None:
    """The draw with nothing hidden and the one with ``L`` hidden share the graph (the graph stream
    does not read the hidden count), so counting the candidates first is sound."""
    cell = _cell(n=50)
    for seed in range(3):
        plain = lane._forward(cell, 0).sample(n=50, seed=seed, scale="raw")
        hidden = lane._forward(cell, 4).sample(n=50, seed=seed, scale="raw")
        graphs = [nx.DiGraph(lane.arcs_of(ds.graph.to_numpy()).tolist()) for ds in (plain, hidden)]
        assert nx.is_isomorphic(*graphs)


def test_hidden_nodes_have_two_children_and_the_truth_is_their_marginal() -> None:
    _, truth = lane.dataset(_cell(latents=8, n=50), 1)
    arcs = np.asarray(truth["arcs"])
    for node in range(truth["observed"], truth["nodes"]):
        assert (arcs[:, 0] == node).sum() >= 2
    marks = np.zeros((truth["nodes"],) * 2, dtype=np.int8)
    marks[arcs[:, 0], arcs[:, 1]], marks[arcs[:, 1], arcs[:, 0]] = TAIL, ARROW
    types = np.r_[np.zeros(truth["observed"]), np.ones(truth["nodes"] - truth["observed"])]
    graph = GraphStructure.from_numpy(marks, kind="dag", node_types=types.astype(np.int8))
    assert np.array_equal(marginal(graph, "pag").to_numpy(), np.asarray(truth["pag"]))


def _truth() -> dict:
    """Hidden ``h`` (node 3) causes ``a`` and ``b``; ``a -> c``. The PAG is ``b o-o a o-o c``."""
    arcs = [[3, 0], [3, 1], [0, 2]]
    pag = np.zeros((3, 3), dtype=int)
    for i, j in ((0, 1), (0, 2)):
        pag[i, j] = pag[j, i] = CIRCLE
    return {"nodes": 4, "observed": 3, "arcs": arcs, "pag": pag.tolist(), "empty_shd": 4}


def _marks(*edges: tuple[int, int, int, int]) -> np.ndarray:
    marks = np.zeros((3, 3), dtype=int)
    for i, j, at_i, at_j in edges:
        marks[i, j], marks[j, i] = at_i, at_j
    return marks


def test_scores_of_the_truth_and_of_the_empty_graph() -> None:
    truth = _truth()
    exact = lane.pag_scores(truth["pag"], truth)
    assert exact["shd"] == 0 and exact["adjacency_precision"] == exact["adjacency_recall"] == 1
    assert exact["arrowhead_precision"] is None and exact["tail_f1"] is None
    empty = lane.pag_scores(np.zeros((3, 3), dtype=int), truth)
    assert empty["shd"] == truth["empty_shd"] and empty["edges"] == 0


def test_path_scores_read_the_dag() -> None:
    truth = _truth()
    right = lane.pag_scores(_marks((0, 1, ARROW, ARROW), (0, 2, TAIL, ARROW)), truth)
    assert right["bidirected_latent_precision"] == 1
    assert right["tail_path_precision"] == 1 and right["arrow_path_precision"] == 1
    wrong = lane.pag_scores(_marks((0, 2, ARROW, TAIL)), truth)  # c -> a
    assert wrong["tail_path_precision"] == 0 and wrong["arrow_path_precision"] == 0
    assert wrong["arrowhead_precision"] == 0 and wrong["arrowhead_common_precision"] == 0


def test_andrey_kwargs_map_the_harness_names() -> None:
    params = {"alpha": 0.01, "lambda_value": 2.0, "depth": -1, "score_func": "local_score_BIC"}
    boss = andrey_boss_fci.andrey_kwargs("boss_fci", params, draw=0)
    assert boss == {
        "alpha": 0.01,
        "indep_test": "fisherz",
        "max_depth": None,
        "score_func": "bic",
        "penalty_discount": 2.0,
        "seed": 0,
    }
    grasp = andrey_boss_fci.andrey_kwargs("grasp_fci", {**params, "depth": 7, "grasp_depth": 7}, 0)
    assert grasp["max_depth"] == 7 and grasp["grasp_depth"] == 7
    fci = andrey_boss_fci.andrey_kwargs("fci", params, 0)
    assert set(fci) == {"alpha", "indep_test", "max_depth"}


def test_counts_the_possible_dsep_searches_and_restores_the_step() -> None:
    """GFCI's step searches possible-d-separation sets; BOSS-FCI, without the step, searches none."""
    import andrey
    import andrey.constraint.fci as fci

    X, _ = lane.dataset(_cell(n=500), 0)
    before = fci._possible_dsep
    counts = {}
    for name in ("gfci", "boss_fci"):
        kwargs = andrey_boss_fci.andrey_kwargs(name, lane._params(_cell()), 0)
        with andrey_boss_fci.instrumented() as counts[name]:
            getattr(andrey, name)(X, **kwargs)
    assert fci._possible_dsep is before
    assert counts["gfci"]["possible_dsep_searches"] > 0 and counts["gfci"]["possible_dsep_max"] >= 1
    assert counts["boss_fci"] == {"possible_dsep_max": None, "possible_dsep_searches": 0}


def test_arms_per_cell() -> None:
    paper, default = _cell(), _cell(setting="default")
    assert [f["config"] for f in lane.andrey_fits(default)] == ["default", "default"]
    assert [f["algorithm"] for f in lane.andrey_fits(paper)[:2]] == ["boss_fci", "grasp_fci"]
    assert lane.andrey_fits(paper)[-1]["params"]["grasp_depth"] == 7
    gated = lane.tetrad_fits(paper, "tetrad-7.6.3")[:2]
    assert [f["solution"] for f in gated] == list(lane.GATED.values())
    assert all(f["draws"] == list(range(lane.DRAWS)) for f in gated)
    assert lane.tetrad_fits(default, "tetrad-7.6.10") == []
    o2 = lane.cell_named("O2|deg=6|L=20|setting=paper|n=1000")
    assert all(f["draws"] == [0] for f in lane.tetrad_fits(o2, "tetrad-7.6.3"))


@pytest.mark.parametrize("adapter", [tetrad_pag.TetradBFCI(), tetrad_pag.TetradCmdGraspFCI()])
def test_gated_reference_runs_tetrads_documented_defaults(adapter) -> None:
    """No seed, every boolean that is on by default named, and ``depth`` from ``params``."""
    arguments = adapter._arguments({**adapter.params(), "alpha": 0.01, "lambda_value": 2.0})
    pairs = dict(zip(arguments, arguments[1:]))
    assert pairs["--seed"] == "-1" and pairs["--depth"] == "-1"
    assert pairs["--alpha"] == "0.01" and pairs["--penaltyDiscount"] == "2.0"
    named = {"--completeRuleSetUsed", "--doDiscriminatingPathRule", "--precomputeCovariances"}
    if adapter.algorithm == "grasp_fci":
        named |= {"--graspOrderedAlg", "--useDataOrder"}
    assert named <= set(arguments) and "--possibleMsepDone" not in arguments
    capped = adapter._arguments({**adapter.params(), "depth": 7})
    assert capped[capped.index("--depth") + 1] == "7"


def _records(cell: lane.Cell, shift: float, seeds=range(20)) -> list[dict]:
    rows = []
    for seed in seeds:
        base = {"cell": cell.name, "seed": seed, "status": "ok", "empty_shd": 50, "marks": [[0]]}
        rows.append(
            {
                **base,
                "package": "andrey",
                "algorithm": "boss_fci",
                "config": cell.setting,
                "draw": 0,
                "shd": 30 + seed % 3 + shift,
                "wall_s": 0.1,
                "possible_dsep_max": 5,
            }
        )
        for draw in range(lane.DRAWS):
            rows.append(
                {
                    **base,
                    "package": "tetrad-7.6.3",
                    "algorithm": "boss_fci",
                    "config": cell.setting,
                    "draw": draw,
                    "shd": 30 + seed % 3,
                    "wall_s": 0.5,
                    "marks": [[draw % 2]],
                }
            )
    return rows


def test_summary_pairs_andrey_with_the_reference_mean() -> None:
    cell = _cell()
    worse = lane.summarize(_records(cell, 2.0), [])
    (entry,) = worse["methods"]
    assert entry["comparator"] == "tetrad-cmd-1.12.0.bfci" and entry["role"] == "gated"
    (row,) = entry["rows"]
    assert row["paired"] == 20 and row["verdict"] == "worse"
    assert row["multi_draw"]["mean_D"] == pytest.approx(2.0)
    assert row["floor"]["applies"] is True
    assert worse["reference_draws"][f"{cell.name}|tetrad-7.6.3|boss_fci"] == {
        "datasets": 20,
        "draws": lane.DRAWS,
        "datasets_with_distinct_draws": 20,
    }
    assert worse["possible_dsep_candidates"][f"{cell.name}|boss_fci|paper"]["max"] == 5
    tie = lane.summarize(_records(cell, 0.0), [])["methods"][0]["rows"][0]
    assert tie["verdict"] == "tie" and "replication" not in tie
    rerun = _records(cell, 2.0) + _records(cell, 0.0, seeds=range(100, 120))
    row = lane.summarize(rerun, [])["methods"][0]["rows"][0]
    assert row["paired"] == 20 and row["verdict"] == "worse"
    assert row["replication"]["seeds"] == "100-119" and row["replication"]["mean_D"] == 0
    json.dumps(worse)


@needs_r()
def test_truth_check_flags_a_pag_pcalg_does_not_give(tmp_path) -> None:
    lines = []
    for seed in range(2):
        _, truth = lane.dataset(_cell(n=50), seed)
        lines.append({"cell": _cell().name, "seed": seed, **truth})
    pag = np.asarray(lines[1]["pag"])
    i, j = np.argwhere(pag == CIRCLE)[0]
    pag[i, j] = TAIL
    lines[1]["pag"] = pag.tolist()
    (tmp_path / lane.DATASETS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    args = SimpleNamespace(runs=[str(tmp_path)], out=str(tmp_path), max_observed=30)
    assert lane.check_truth(args) == 1
    result = json.loads((tmp_path / "truth-check.json").read_text())
    assert result["agree"] == 1 and result["disagree"][0]["cells"] == [[_cell().name, 1]]


def test_a_call_past_its_cap_stops() -> None:
    import time

    with pytest.raises(andrey_boss_fci.FitTimeout):
        with andrey_boss_fci.capped(0.05):
            while True:
                time.sleep(0.01)
    with andrey_boss_fci.capped(5.0):
        pass
