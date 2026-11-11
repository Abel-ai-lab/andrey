"""The NOTEARS and DAGMA lane: the grid, the arms, the cycle repair, the driver, the summary.

The grid, Andrey's arms, and the summary are checked in bench-env. The references (the authors'
NOTEARS and DAGMA, CausalDisco) run in ``.venv-notears``; they are checked by running each on one
small dataset, and skipped when that environment has not been built.
"""

from __future__ import annotations

from pathlib import Path

import notears_cpu
import notears_summary
import numpy as np
import pandas as pd
import pytest

from andrey_bench import datasets, runner
from andrey_bench.adapters.andrey_notears import AndreyContinuous
from andrey_bench.adapters.weights_dag import dag_of_weights, has_cycle

NOTEARS_PYTHON = Path(runner.BENCH_ROOT) / ".venv-notears" / "bin" / "python"
NOTEARS_CODE = Path(runner.BENCH_ROOT) / ".notears"
needs_notears_env = pytest.mark.skipif(
    not NOTEARS_PYTHON.exists() or not (NOTEARS_CODE / "linear.py").exists(),
    reason="notears env not built - run benchmarks/start_notears.sh",
)


def _chain(n: int = 500, seed: int = 0) -> np.ndarray:
    """Rows of the chain ``0 -> 1 -> 2`` with weights 1.5 and unit noise."""
    x = np.random.default_rng(seed).standard_normal((n, 3))
    x[:, 1] += 1.5 * x[:, 0]
    x[:, 2] += 1.5 * x[:, 1]
    return x


def _edges(structure) -> set[tuple[int, int]]:
    """The directed edges ``i -> j``: a tail at ``i`` and an arrowhead at ``j``."""
    marks = np.asarray(structure.to_numpy())
    return {(int(i), int(j)) for i, j in zip(*np.nonzero((marks == 1) & (marks.T == 2)))}


# --- the cycle repair ------------------------------------------------------------------------------


def test_the_repair_drops_the_weakest_edges_until_no_cycle_is_left() -> None:
    w = np.zeros((4, 4))
    w[0, 1], w[1, 2], w[2, 0], w[2, 3] = 0.9, -0.5, 0.4, 0.31
    assert has_cycle(w) and not has_cycle(np.triu(np.ones((4, 4)), 1))
    adjacency, dropped = dag_of_weights(w)
    # 2 -> 3 is weaker than 2 -> 0 and leaves first, although it lies on no cycle: weakest first.
    assert dropped == 2 and not has_cycle(adjacency)
    assert {(int(i), int(j)) for i, j in zip(*np.nonzero(adjacency))} == {(0, 1), (1, 2)}


def test_the_repair_leaves_a_dag_as_it_is() -> None:
    w = np.triu(np.random.default_rng(0).uniform(0.5, 2.0, (5, 5)), 1)
    adjacency, dropped = dag_of_weights(w)
    assert dropped == 0 and np.array_equal(adjacency, (w != 0).astype(np.int8))


# --- the grid --------------------------------------------------------------------------------------


def test_the_default_plan_is_the_papers_grid() -> None:
    units = notears_cpu.plan()
    assert len(units) == 3 * 2 * 3 * 3 * 20
    keys = {u.key for u in units}
    assert {k.functional for k in keys} == {"linear"} and {k.n for k in keys} == {1000}
    assert {(k.topology, k.density) for k in keys} == {("er", 4.0), ("scale_free", 8.0)}
    assert {k.noise for k in keys} == {"gaussian", "exponential", "gumbel"}
    assert {k.standardize for k in keys} == {"raw", "standardized", "ancestral"}
    assert {k.d for k in keys} == {10, 20, 50} and {k.seed for k in keys} == set(range(20))
    assert len({k.digest for k in keys}) == len(units)


def test_the_graphs_have_the_papers_edge_counts() -> None:
    for graph, per_node in (("er2", 2), ("sf4", 4)):
        _, truth, _, _ = datasets.generate_one(
            notears_cpu.cell_key(graph, 20, "gaussian", "raw", 0)
        )
        edges = int((np.asarray(truth) != 0).sum()) // 2
        assert 0.75 * per_node * 20 <= edges <= 1.25 * per_node * 20, (graph, edges)


def test_every_cell_pairs_andrey_with_its_reference_the_controls_and_the_floor() -> None:
    always = {
        "andrey.notears",
        "andrey.dagma",
        "zheng.notears",
        "bello.dagma",
        "causaldisco.r2_sort_regress",
        "causaldisco.var_sort_regress",
        "baseline.empty",
    }
    scaled = {"andrey.notears.standardize", "andrey.dagma.standardize"}
    robust = {"andrey.notears.robust", "andrey.dagma.robust"}
    names = lambda *a, **k: {arm.name for arm in notears_cpu.arms(*a, **k)}  # noqa: E731
    assert names("standardized", 50) == always
    assert names("raw", 50) == names("rescaled", 10) == always | scaled
    assert names("standardized", 20, robust=True) == always | robust
    assert names("standardized", 50, robust=True) == always
    assert names("raw", 10, robust=True) == always | scaled
    paper = {"andrey.dagma.paper", "bello.dagma.paper"}
    assert set(notears_cpu.solution_names()) == always | scaled | robust | paper
    with_paper = {a.name for a in notears_cpu.arms("standardized", 50, paper=True)}
    assert with_paper == always | paper
    assert {a.name for a in notears_cpu.arms("raw", 50, paper=True)} == always | scaled


def test_paper_arms_set_the_paper_penalty_in_both_codes() -> None:
    andrey_arm, reference = (
        a for a in notears_cpu.arms("standardized", 10, paper=True) if a.name.endswith(".paper")
    )
    assert andrey_arm.params() == {"lambda1": 0.05}
    assert reference.params()["lambda1"] == 0.05
    default = next(a for a in notears_cpu.arms("standardized", 10) if a.name == "bello.dagma")
    assert default.params()["lambda1"] == 0.03
    # The controls run where CausalDisco is installed.
    controls = [a for a in notears_cpu.arms("raw", 10) if a.name.startswith("causaldisco")]
    assert {a.python_env for a in controls} == {notears_cpu.REFERENCE_ENV}


def test_the_parts_split_the_plan_and_a_rerun_takes_fresh_seeds() -> None:
    units = notears_cpu.plan(sizes=[10], seeds=2)
    parts = [units[part::5] for part in range(5)]
    assert sorted(u.key.digest for p in parts for u in p) == sorted(u.key.digest for u in units)
    again = notears_cpu.plan(sizes=[10], seeds=2, first_seed=20)
    assert {u.key.seed for u in again} == {20, 21}
    assert not {u.key.digest for u in again} & {u.key.digest for u in units}
    only = notears_cpu.plan(sizes=[10], seeds=1, solutions=["andrey.dagma", "bello.dagma"])
    assert {a.name for u in only for a in u.arms} == {"andrey.dagma", "bello.dagma"}


# --- Andrey's arms ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["notears", "dagma"])
def test_andreys_arm_fits_the_chain_and_standardize_ignores_the_units(method: str) -> None:
    x = _chain()
    plain = AndreyContinuous(method)
    assert plain.name == f"andrey.{method}" and plain.params() == {}
    assert _edges(plain.to_structure(plain.fit(x, plain.params()))) == {(0, 1), (1, 2)}
    scaled = AndreyContinuous(method, "standardize")
    assert scaled.name == f"andrey.{method}.standardize" and scaled.params() == {
        "standardize": True
    }
    graphs = [
        _edges(scaled.to_structure(scaled.fit(x * units, scaled.params())))
        for units in (np.ones(3), np.array([100.0, 1.0, 0.01]))
    ]
    assert graphs[0] == graphs[1]
    with pytest.raises(ValueError, match="options"):
        AndreyContinuous(method, "fast")


def test_the_driver_scores_andrey_and_the_floor_on_the_same_dataset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    argv = ["--out", str(tmp_path), "--sizes", "10", "--graphs", "er2", "--noises", "gaussian"]
    argv += ["--scales", "raw", "--seeds", "1", "--cores", "1"]
    solutions = ["andrey.notears", "andrey.notears.standardize", "baseline.empty"]
    assert notears_cpu.main([*argv, "--solutions", *solutions]) == 0
    assert (tmp_path / "done.json").exists()
    table = pd.read_parquet(tmp_path / "runs.parquet").set_index("name")
    assert sorted(table.index) == sorted(solutions) and set(table["status"]) == {"ok"}
    assert table["dataset_id"].nunique() == 1
    # On the generator's own scale the variances give the order away, and NOTEARS reads them.
    assert table.loc["andrey.notears", "shd"] < table.loc["baseline.empty", "shd"]
    rows = notears_summary.all_rows(notears_summary.load([str(tmp_path)]))
    assert [(r["method"], r["graph"], r["d"], r["scale"], r["decision"]) for r in rows] == [
        ("notears", "er2", 10, "raw", False)
    ]
    assert rows[0]["floor"]["paired"] == 1 and rows[0]["reference"] is None


def test_the_driver_stops_before_fitting_when_the_reference_environment_is_unset(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(notears_cpu.REFERENCE_ENV, raising=False)
    with pytest.raises(SystemExit, match=notears_cpu.REFERENCE_ENV):
        notears_cpu.main(["--out", str(tmp_path), "--sizes", "10", "--seeds", "1"])


# --- the summary -----------------------------------------------------------------------------------


def _records(shd: dict[str, list[float | None]], scale: str = "standardized") -> pd.DataFrame:
    """Records of one cell: per arm, the SHD of each seed, ``None`` for a timeout."""
    rows = []
    for name, values in shd.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "dataset_id": f"{scale}-{seed}",
                    "name": name,
                    "status": "ok" if value is not None else "timeout",
                    "output_type": "cpdag" if name == "baseline.empty" else "dag",
                    "topology": "er",
                    "density": 4.0,
                    "noise": "gaussian",
                    "standardize": notears_cpu.SCALES[scale],
                    "d": 10,
                    "data_seed": seed,
                    "shd": value,
                    "mec_shd": None if name == "baseline.empty" or value is None else value - 1,
                    "skeleton_f1": 0.5,
                    "wall_s": 1.0 if value is not None else None,
                    "structure_hash": f"{name if value != 7 else 'same'}-{seed}",
                    "varsortability": 0.5,
                    "r2_sortability": 0.5,
                }
            )
    return pd.DataFrame(rows)


def _summarize(tmp_path: Path, table: pd.DataFrame, name: str) -> str:
    folder = tmp_path / name
    folder.mkdir()
    table.to_parquet(folder / "runs.parquet", index=False)
    return str(folder)


def _run(tmp_path: Path, runs: str, replications: str | None = None) -> dict:
    import json

    out = tmp_path / "summary.json"
    argv = ["--runs", runs, "--out", str(out)]
    assert (
        notears_summary.main(argv + (["--replications", replications] if replications else [])) == 0
    )
    return json.loads(out.read_text())


def test_a_worse_cell_waits_for_its_rerun_and_counts_only_when_it_repeats(tmp_path) -> None:
    seeds = 12
    first = _records(
        {
            "andrey.notears": [8.0] * seeds,
            "zheng.notears": [6.0] * seeds,
            "andrey.dagma": [7.0] * seeds,
            "bello.dagma": [7.0] * seeds,
            "baseline.empty": [20.0] * seeds,
        }
    )
    runs = _summarize(tmp_path, first, "first")
    summary = _run(tmp_path, runs)
    notears, dagma = summary["decision"]["notears"], summary["decision"]["dagma"]
    assert notears["rerun"] == ["er2 10 gaussian standardized"] and notears["outcome"] == "rerun"
    assert notears["reference_bar"] is None and notears["floor_bar"] is True
    assert dagma["outcome"] == "supported" and dagma["reference"]["tie"] == 1
    row = next(r for r in summary["rows"] if r["method"] == "dagma")
    assert row["reference"]["same_graph"] == 1.0 and row["floor"]["mean_D"] == -13.0
    # The empty graph's class distance is its SHD; a DAG arm's is its own column.
    assert row["floor"]["mec"]["mean_D"] == -14.0

    tie = first.assign(shd=first["shd"].where(first["name"] != "andrey.notears", 6.0))
    summary = _run(tmp_path, runs, _summarize(tmp_path, tie, "tie"))
    assert summary["decision"]["notears"]["outcome"] == "supported"
    row = next(r for r in summary["rows"] if r["method"] == "notears")
    assert row["result"] == "tie (did not replicate)" and row["replication"]["datasets"] == seeds

    summary = _run(tmp_path, runs, _summarize(tmp_path, first, "again"))
    assert summary["decision"]["notears"]["worse"] == ["er2 10 gaussian standardized"]
    assert summary["decision"]["notears"]["outcome"] == "not promoted: reference"


def test_the_floor_fails_a_method_whatever_its_reference_does(tmp_path) -> None:
    seeds = 12
    table = _records(
        {
            "andrey.notears": [21.0, 19.0] * (seeds // 2),
            "zheng.notears": [22.0, 20.0] * (seeds // 2),
            "baseline.empty": [20.0] * seeds,
        }
    )
    result = _run(tmp_path, _summarize(tmp_path, table, "runs"))["decision"]["notears"]
    assert result["reference"]["better"] == 1 and result["below_empty"] == 0
    assert result["floor_bar"] is False and result["outcome"] == "not supported: floor"


def test_a_timeout_is_a_failure_and_never_a_graph(tmp_path) -> None:
    table = _records(
        {
            "andrey.notears": [5.0] * 4 + [None] * 8,
            "zheng.notears": [5.0] * 12,
            "baseline.empty": [20.0] * 12,
        }
    )
    summary = _run(tmp_path, _summarize(tmp_path, table, "runs"))
    row = summary["rows"][0]
    assert row["arms"]["andrey.notears"]["status"] == {"timeout": 8, "ok": 4}
    assert row["arms"]["andrey.notears"]["shd"] == 5.0
    assert row["reference"]["paired"] == 4 and row["reference"]["failures"] == 8
    assert row["reference"]["verdict"] is None
    result = summary["decision"]["notears"]
    assert result["unjudged"] == ["er2 10 gaussian standardized"] and result["andrey_failures"] == 8
    assert result["reference_bar"] is False and result["outcome"] == "not supported: floor"


def test_raw_cells_do_not_decide_and_report_the_units_check(tmp_path) -> None:
    standardized = _records({"andrey.notears": [7.0] * 12, "baseline.empty": [20.0] * 12})
    raw = _records(
        {
            "andrey.notears": [1.0] * 12,
            "andrey.notears.standardize": [7.0] * 12,
            "zheng.notears": [9.0] * 12,
            "causaldisco.var_sort_regress": [1.0] * 12,
            "baseline.empty": [20.0] * 12,
        },
        scale="raw",
    )
    runs = _summarize(tmp_path, pd.concat([standardized, raw]), "runs")
    summary = _run(tmp_path, runs)
    row = next(r for r in summary["rows"] if r["scale"] == "raw")
    assert row["decision"] is False and row["result"] is None
    assert row["var_control"]["verdict"] == "tie" and row["reference"]["verdict"] == "better"
    assert row["standardize"]["same_as_standardized"] == 1.0
    assert row["standardize"]["default"]["verdict"] == "worse"
    assert summary["decision"]["notears"]["cells"] == 1


# --- the references, in their own environment ------------------------------------------------------


@needs_notears_env
def test_the_references_and_controls_run_there_and_return_a_dag(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    monkeypatch.setenv(notears_cpu.REFERENCE_ENV, str(NOTEARS_PYTHON))
    monkeypatch.setenv("ANDREY_BENCH_NOTEARS", str(NOTEARS_CODE))
    argv = ["--out", str(tmp_path), "--sizes", "10", "--graphs", "er2", "--noises", "gumbel"]
    assert notears_cpu.main([*argv, "--scales", "raw", "--seeds", "1", "--cores", "1"]) == 0
    table = pd.read_parquet(tmp_path / "runs.parquet").set_index("name")
    assert set(table["status"]) == {"ok"} and len(table) == 9
    assert set(table.loc[table.index != "baseline.empty", "output_type"]) == {"dag"}
    # On the papers' own scale each method and its reference recover most of the graph.
    floor = table.loc["baseline.empty", "shd"]
    for name in ("andrey.notears", "zheng.notears", "andrey.dagma", "bello.dagma"):
        assert table.loc[name, "shd"] < floor / 2, name
    row = notears_summary.all_rows(notears_summary.load([str(tmp_path)]))[0]
    assert row["reference"]["paired"] == 1 and row["var_control"]["paired"] == 1
