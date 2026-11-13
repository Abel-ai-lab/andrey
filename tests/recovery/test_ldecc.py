"""Recovery tests for ``ldecc``: the graphs of Gupta et al. (2023) and Schubert et al. (2026) under
exact d-separation, random graphs against the CPDAG's local parent sets, and answers on
linear-Gaussian data drawn by the benchmark lane's sampler of Gupta et al.'s Appendix C.1.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), LDECC's parent sets are those the
treatment has over the DAGs of the CPDAG (Theorem 5 of Gupta et al.), or, where an eager collider
check cannot direct a parent (Example A.2 of Schubert et al.), a wider family that holds them:
never a family that misses one. Its neighbors are the treatment's adjacencies.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from andrey import GraphStructure, ldecc
from andrey.constraint import ldecc as engine
from andrey.constraint.blanket import LocalSearch
from andrey.core import adjustment as adj
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle
from andrey.metrics.graph import to_cpdag

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

#: Gupta et al. (2023), Figure 1(a): A -> W <- B is an unshielded collider far from X.
FIGURE_1 = [
    *(("C", "A"), ("C", "B"), ("A", "W"), ("B", "W"), ("W", "X"), ("X", "M")),
    *(("M", "Y"), ("C", "D"), ("D", "Y")),
]
#: Schubert et al. (2026), Example A.1: the spouse 4 descends from the treatment 0, and only
#: {1, 2} separates them, with the spouse 2 in it.
EXAMPLE_A1 = [(0, 1), (0, 3), (1, 4), (2, 1), (2, 4), (4, 3)]
#: Schubert et al. (2026), Example A.2: the CPDAG directs 2 -> 0 through 3 -> 2 <- 4 and Meek's
#: rules, with 3 and 4 adjacent to the treatment 0.
EXAMPLE_A2 = [(0, 1), (2, 0), (3, 0), (4, 0), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4)]


def oracle_search(edges, nodes) -> LocalSearch:
    """A search whose test is d-separation in the DAG ``edges`` over ``nodes``."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(nodes)
    return LocalSearch(DSepOracle(dag, nodes=list(nodes)), list(range(len(nodes))), 0.5)


def oracle_answer(edges, nodes, x, **kwargs) -> engine.Result:
    return engine.ldecc(oracle_search(edges, nodes), nodes.index(x), **kwargs)


def named(nodes, positions) -> set:
    return {nodes[k] for k in positions}


def among_neighbors(tests, x, mb):
    """The step that finds the neighbors as Fig. 2 of Gupta et al. prints it: each member is
    tested against subsets of the treatment's remaining neighbors."""
    adjacent = tests.by_rank(mb)
    sepsets = {}
    size = 0
    while len(adjacent) > size:
        for v in list(adjacent):
            found = tests.first_separating(x, v, [u for u in adjacent if u != v], size)
            if found is not None:
                adjacent.remove(v)
                sepsets[v] = found
        size += 1
    return adjacent, sepsets


def class_parent_sets(edges, nodes, x) -> set[tuple[int, ...]]:
    """The treatment's parent sets over the DAGs of the CPDAG (local IDA), by position."""
    ix = {v: k for k, v in enumerate(nodes)}
    marks = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a, b in edges:
        marks[ix[a], ix[b]], marks[ix[b], ix[a]] = 1, 2
    cpdag = to_cpdag(GraphStructure.from_numpy(marks, kind="dag"))
    return {tuple(s) for s in adj.local_parent_sets(cpdag, ix[x])}


def verdict(result: engine.Result, truth: set) -> str:
    got = set(result.parent_sets)
    return "equal" if got == truth else "wider" if got > truth else "other"


# --- worked examples ------------------------------------------------------------------------------


@pytest.mark.parametrize("check", [True, False])
def test_figure_1_directs_w_by_an_eager_collider_check(check):
    nodes = sorted({v for e in FIGURE_1 for v in e})
    result = oracle_answer(FIGURE_1, nodes, "X", check=check)
    assert named(nodes, result.neighbors) == {"W", "M"}
    assert named(nodes, result.parents) == {"W"}
    assert named(nodes, result.children) == {"M"}
    assert [named(nodes, s) for s in result.parent_sets] == [{"W"}]


def test_example_a1_neighbors_are_the_adjacencies():
    nodes = list(range(5))
    result = oracle_answer(EXAMPLE_A1, nodes, 0)
    assert set(result.neighbors) == {1, 3}
    assert set(result.children) == {1, 3}
    assert result.parent_sets == ((),)


def test_example_a1_conditioning_on_remaining_neighbors_keeps_the_spouse():
    """Fig. 2 as printed drops the spouse 2 at size 0 and never conditions on it again, so the
    spouse 4 stays a neighbor; testing against subsets of the blanket separates it."""
    nodes = list(range(5))
    printed = oracle_answer(EXAMPLE_A1, nodes, 0, adjacency=among_neighbors)
    assert 4 in printed.neighbors
    assert 4 not in oracle_answer(EXAMPLE_A1, nodes, 0).neighbors


def test_example_a2_leaves_a_parent_undirected_that_the_cpdag_directs():
    nodes = list(range(6))
    result = oracle_answer(EXAMPLE_A2, nodes, 0)
    assert set(result.parents) == {3, 4}
    assert set(result.children) == {1}
    assert set(result.unoriented) == {2}
    truth = class_parent_sets(EXAMPLE_A2, nodes, 0)
    assert truth == {(2, 3, 4)}
    assert verdict(result, truth) == "wider"


def test_the_outcome_as_a_parent_says_the_effect_is_zero():
    """In y -> x <- w, the outcome is a parent of the treatment: the effect is zero in every graph
    the answer allows, and no set is valid."""
    rng = np.random.default_rng(0)
    w, y = rng.standard_normal((2, 4000))
    x = w + y + rng.standard_normal(4000)
    out = ldecc(pd.DataFrame({"w": w, "y": y, "x": x}), treatment="x", outcome="y")
    record = out.metadata["effects"][0]
    assert record["nonzero"] is False and record["identifiable"] is False
    assert record["adjustment_sets"] == []
    assert set(out.structure.oriented_edges()) == {("w", "x", "directed"), ("y", "x", "directed")}


# --- random graphs under d-separation -------------------------------------------------------------


def random_dag(rng, d, p) -> nx.DiGraph:
    order = rng.permutation(d)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                dag.add_edge(int(order[i]), int(order[j]))
    return dag


def lane_models():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        return importlib.import_module("andrey_bench.adjustment_models")
    finally:
        sys.path.remove(str(BENCHMARKS))


def appendix_c1_graphs(seeds):
    lane = lane_models()
    for seed in seeds:
        model = lane.ldecc(seed)
        dag = np.asarray(model.dag)
        yield (
            [(int(a), int(b)) for a, b in zip(*np.nonzero(dag), strict=True)],
            dag.shape[0],
            (model.pairs[0].treatment),
        )


def er_graphs(seeds, d=12, p=0.25):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        dag = random_dag(rng, d, p)
        yield list(dag.edges), d, int(rng.integers(d))


GRAPHS = {
    "appendix-c1": lambda: appendix_c1_graphs(range(40)),
    "random": lambda: er_graphs(range(60)),
}


@pytest.mark.parametrize("family", list(GRAPHS))
@pytest.mark.parametrize("check", [True, False])
def test_parent_sets_hold_the_class_sets_on_random_graphs(family, check):
    """Theorem 5 of Gupta et al., up to the parents an eager collider check cannot direct: the
    family equals the CPDAG's or is wider, and the neighbors are the adjacencies."""
    counts = {"equal": 0, "wider": 0, "other": 0}
    for edges, d, x in GRAPHS[family]():
        nodes = list(range(d))
        result = oracle_answer(edges, nodes, x, check=check)
        dag = nx.DiGraph(edges)
        dag.add_nodes_from(nodes)
        assert set(result.neighbors) == set(dag.predecessors(x)) | set(dag.successors(x))
        assert set(result.parents) <= set(dag.predecessors(x))
        assert set(result.children) <= set(dag.successors(x))
        counts[verdict(result, class_parent_sets(edges, nodes, x))] += 1
    assert counts["other"] == 0
    assert counts["equal"] >= 0.9 * sum(counts.values())


def test_the_tests_stop_once_every_neighbor_is_directed():
    """In Figure 1(a) every neighbor is directed at size 1, after the collider check on A and B
    and the non-collider W - X - M: no pair away from X is tested given two other variables."""
    nodes = sorted({v for e in FIGURE_1 for v in e})
    dag = nx.DiGraph(FIGURE_1)
    x = nodes.index("X")
    search = LocalSearch(DSepOracle(dag, nodes=nodes), list(range(len(nodes))), 0.5, trace=True)
    engine.ldecc(search, x)
    assert search.counts.queries is not None
    away = [s for a, b, s in search.counts.queries if x not in (a, b) and x not in s]
    assert away and max(len(s) for s in away) == 1


# --- data ---------------------------------------------------------------------------------------


def population_effect(model, x, y, members) -> float:
    """The coefficient of ``x`` when ``y`` is regressed on ``x`` and ``members`` in the model's
    population, standardized by the two standard deviations."""
    cov = model.covariance()
    rows = [x, *members]
    beta = np.linalg.solve(cov[np.ix_(rows, rows)], cov[rows, y])
    return float(beta[0] * np.sqrt(cov[x, x] / cov[y, y]))


def hausdorff(estimates, truths) -> float:
    """The Hausdorff distance between two sets of effects (Gupta et al. 2023, Sect. 5)."""
    far = max(min(abs(u - v) for v in truths) for u in estimates)
    return max(far, max(min(abs(u - v) for u in estimates) for v in truths))


def test_appendix_c1_effects_beat_no_adjustment():
    """On the data of Gupta et al.'s Appendix C.1 at its smallest size and level, the effects of
    the returned sets lie closer to the class's effects (Hausdorff distance, estimated on a fresh
    sample) than the unadjusted estimate does."""
    lane = lane_models()
    n = 8000
    ours, empty = [], []
    for seed in range(8):
        model = lane.ldecc(seed)
        x, y = model.pairs[0].treatment, model.pairs[0].outcome
        out = ldecc(model.rows(n, lane.DATA), treatment=x, outcome=y, alpha=0.01 * 32000 / n)
        record = out.metadata["effects"][0]
        fresh = model.rows(n, lane.FRESH)
        cpdag_sets = effect_oracle(
            model.dag, x, y, set_kind="local_parents", identified_from="cpdag"
        )["adjustment_sets"]
        truths = [population_effect(model, x, y, s) for s in cpdag_sets or [[]]]
        if record["nonzero"] is False:
            estimates = [0.0]
        else:
            estimates = [ols(fresh, x, y, s) for s in record["adjustment_sets"] or [[]]]
        ours.append(hausdorff(estimates, truths))
        empty.append(hausdorff([ols(fresh, x, y, [])], truths))
    assert np.mean(ours) < np.mean(empty)


def ols(data, x, y, members) -> float:
    """The coefficient of ``x`` in the least-squares fit of ``y`` on ``x`` and ``members``,
    standardized by the sample standard deviations."""
    design = np.column_stack([np.ones(len(data)), data[:, x], *(data[:, v] for v in members)])
    beta = np.linalg.lstsq(design, data[:, y], rcond=None)[0]
    return float(beta[1] * data[:, x].std(ddof=1) / data[:, y].std(ddof=1))
