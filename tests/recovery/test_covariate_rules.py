"""Recovery tests for ``covariate_rules``: the examples and theorems of Entner, Hoyer, and Spirtes
(2013) under exact d-separation, and answers on linear-Gaussian data drawn by the benchmark lane's
sampler, scored against the truth.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), on random graphs whose covariates all
precede the treatment, hidden ones among them: the two rules never both apply; an effect is
claimed only where the treatment is an ancestor of the outcome, with every set returned a valid
adjustment set in the true graph (Theorem 2); no effect is claimed only where it is not one; a
zero effect with a valid set of observed covariates is always found (any such set satisfies R2
(i)); and where neither rule applies, a graph with the effect and one without entail the same
independences (Theorem 3). A cap on the conditioning sets keeps every answer right and loses only
answers.
"""

from __future__ import annotations

import importlib
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from andrey import GraphStructure, covariate_rules
from andrey.constraint import covariates
from andrey.constraint.blanket import LocalSearch
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle, effect_scores, pooled_answer_scores

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

#: Figure 1 of Entner et al. (2013): the treatment x, the outcome y, observed covariates w, and
#: hidden ones u.
FIGURE_1 = {
    "a": [("w1", "x"), ("w1", "y"), ("x", "y")],
    "b": [("u1", "x"), ("u1", "w1"), ("u2", "w1"), ("u2", "y"), ("x", "y")],
    "c": [
        *(("u1", "w1"), ("u1", "w3"), ("u1", "x"), ("w1", "x"), ("w3", "w2")),
        *(("u2", "w2"), ("w2", "x"), ("u2", "y"), ("x", "y")),
    ],
    "d": [
        *(("u1", "w1"), ("w2", "w1"), ("u1", "x"), ("w2", "y"), ("u2", "x")),
        *(("u2", "y"), ("w3", "y")),
    ],
}


def oracle_answer(edges, observed, x="x", y="y", max_depth=None):
    """The engine's answer under d-separation over ``observed``, the sets by name."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    search = LocalSearch(DSepOracle(dag, nodes=list(observed)), list(range(len(observed))), 0.5)
    ix = {v: k for k, v in enumerate(observed)}
    others = [k for k in range(len(observed)) if k not in (ix[x], ix[y])]
    result = covariates.covariate_rules(search, ix[x], ix[y], others, max_depth=max_depth)
    return result, [{observed[k] for k in z} for z in result.sets]


def true_graph(dag: nx.DiGraph) -> GraphStructure:
    nodes = list(dag)
    ix = {v: k for k, v in enumerate(nodes)}
    marks = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a, b in dag.edges:
        marks[ix[a], ix[b]], marks[ix[b], ix[a]] = 1, 2
    return GraphStructure.from_numpy(marks, kind="dag", labels=tuple(nodes))


def test_figure_1_a_and_b_are_undecided():
    # (a) and (b) entail no independence among x, y, and w1, though {w1} is the valid set in (a)
    # and the empty set in (b) (Sect. 2).
    for key in ("a", "b"):
        result, sets = oracle_answer(FIGURE_1[key], ["w1", "x", "y"])
        assert result.verdict == "?"
        assert result.counts["R1"] == result.counts["R2"] == 0


def test_figure_1_c_finds_the_effect_and_the_set_r1_names():
    result, sets = oracle_answer(FIGURE_1["c"], ["w1", "w2", "w3", "x", "y"])
    assert result.verdict == "+-"
    assert {"w2", "w3"} in sets  # w = w1, Z = {w2, w3} (Sect. 3)
    dag = nx.DiGraph(FIGURE_1["c"])
    true = true_graph(dag)
    assert all(true.is_adjustment_set("x", "y", z) for z in sets)


def test_figure_1_d_finds_the_zero_effect():
    result, _ = oracle_answer(FIGURE_1["d"], ["w1", "w2", "w3", "x", "y"])
    assert result.verdict == "0"
    assert result.counts["R1"] == 0 and result.counts["R2"] > 0


# --- random graphs under the paper's assumptions ------------------------------------------------


def pretreatment_dag(rng, w: int, u: int, p: float, *, effect: bool, after: bool):
    """A random DAG over ``w`` observed and ``u`` hidden covariates in a random causal order, all
    before ``x`` and ``y``; each possible arc with probability ``p``, ``x -> y`` when ``effect``.
    With ``after``, a hidden mediator ``m`` (``x -> m -> y``, arcs from covariates with
    probability ``p``) and a hidden child of the outcome, which the assumptions also allow."""
    names = [*(f"w{k}" for k in range(w)), *(f"u{k}" for k in range(u))]
    order = [names[k] for k in rng.permutation(len(names))]
    dag = nx.DiGraph()
    dag.add_nodes_from([*names, "x", "y"])
    for i, a in enumerate(order):
        dag.add_edges_from((a, b) for b in order[i + 1 :] if rng.random() < p)
        dag.add_edges_from((a, t) for t in ("x", "y") if rng.random() < p)
    if effect:
        dag.add_edge("x", "y")
    hidden = [f"u{k}" for k in range(u)]
    if after:
        dag.add_edges_from([("x", "m"), ("m", "y"), ("y", "k")])
        dag.add_edges_from((a, "m") for a in names if rng.random() < p)
        hidden += ["m", "k"]
    return dag, [*(f"w{k}" for k in range(w)), "x", "y"], hidden


def random_cases(n_graphs: int, seed: int):
    rng = np.random.default_rng(seed)
    for g in range(n_graphs):
        w, u = int(rng.integers(2, 7)), int(rng.integers(1, 4))
        dag, observed, hidden = pretreatment_dag(
            rng, w, u, float(rng.uniform(0.2, 0.5)), effect=bool(g % 2), after=g % 3 == 0
        )
        yield dag, observed, hidden


def truth(dag: nx.DiGraph, hidden) -> dict:
    """The record the effect calls for: whether the treatment is an ancestor of the outcome, and
    whether some set of observed covariates is a valid adjustment set."""
    return effect_oracle(
        true_graph(dag), "x", "y", set_kind="valid", identified_from="dag", latent=hidden
    )


def test_every_answer_on_random_pretreatment_graphs_is_right():
    verdicts = {"+-": 0, "0": 0, "?": 0}
    for dag, observed, hidden in random_cases(240, 0):
        result, sets = oracle_answer(dag.edges, observed)
        verdicts[result.verdict] += 1
        assert not (result.counts["R1"] and result.counts["R2"])  # the rules never contradict
        effect = nx.has_path(dag, "x", "y")
        record = truth(dag, hidden)
        true = true_graph(dag)
        if result.verdict == "+-":
            assert effect and sets
            assert all(true.is_adjustment_set("x", "y", z) for z in sets)
        elif result.verdict == "0":
            assert not effect
        if not effect and record["identifiable"]:
            assert result.verdict == "0"  # a valid set satisfies R2 (i)
        if effect and not record["identifiable"]:
            assert result.verdict == "?"  # no valid set, so no R1 set can be one
    assert min(verdicts.values()) >= 20, verdicts


def test_a_cap_keeps_every_answer_and_loses_only_answers():
    for dag, observed, _ in random_cases(80, 1):
        full, sets = oracle_answer(dag.edges, observed)
        for cap in (0, 1, 2):
            capped, capped_sets = oracle_answer(dag.edges, observed, max_depth=cap)
            assert capped.verdict in (full.verdict, "?")
            assert all(z in sets and len(z) < cap for z in capped_sets)


def independences(dag: nx.DiGraph, observed) -> frozenset:
    """Every independence among ``observed`` the DAG entails, as ``(a, b, Z)`` by position."""
    oracle = DSepOracle(dag, nodes=list(observed))
    found = set()
    for a, b in combinations(range(len(observed)), 2):
        rest = [k for k in range(len(observed)) if k not in (a, b)]
        for size in range(len(rest) + 1):
            found.update((a, b, z) for z in combinations(rest, size) if oracle(a, b, z) > 0.5)
    return frozenset(found)


def test_an_undecided_pair_has_a_graph_each_way_with_the_same_independences():
    # Theorem 3's constructions. No effect: adding x -> y changes no independence. An effect:
    # a hidden confounder of x and y, added where none is, changes none; then removing x -> y
    # and adding an arc into y from each parent of x changes none, and leaves no effect.
    checked = {True: 0, False: 0}
    rng = np.random.default_rng(3)
    for g in range(160):
        w, u = int(rng.integers(2, 6)), int(rng.integers(1, 3))
        effect = bool(g % 2)
        dag, observed, _ = pretreatment_dag(
            rng, w, u, float(rng.uniform(0.2, 0.5)), effect=effect, after=False
        )
        result, _ = oracle_answer(dag.edges, observed)
        if result.verdict != "?":
            continue
        model = independences(dag, observed)
        other = dag.copy()
        if not effect:
            other.add_edge("x", "y")
        else:
            if not set(dag.predecessors("x")) & set(dag.predecessors("y")) - set(observed):
                other.add_edges_from([("c", "x"), ("c", "y")])
                assert independences(other, observed) == model
            other.remove_edge("x", "y")
            other.add_edges_from((v, "y") for v in list(other.predecessors("x")))
            assert not nx.has_path(other, "x", "y")
        assert independences(other, observed) == model
        checked[effect] += 1
    assert min(checked.values()) >= 10, checked


# --- data -------------------------------------------------------------------------------------


def lane_models():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        return importlib.import_module("andrey_bench.adjustment_models")
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_on_pretreatment_data_the_answers_beat_the_floor():
    # Few covariates, where one level leaves the rules room to answer (see Limitations).
    lane = lane_models()
    pooled = {"nonzero": [], "adjustment": []}
    for task in ("effect-admissible", "effect-none", "zero-admissible", "zero-none"):
        for seed in range(5):
            model = lane.entner(seed, task=task, w=4, u=2, edge_prob=0.4)
            pair = model.pairs[0]
            out = covariate_rules(
                model.rows(5000, lane.DATA), treatment=pair.treatment, outcome=pair.outcome
            )
            oracle = effect_oracle(
                model.dag,
                pair.treatment,
                pair.outcome,
                set_kind="valid",
                identified_from="dag",
                latent=list(model.hidden),
            )
            got = effect_scores(out.metadata["effects"], [oracle], true=model.dag)
            for key in pooled:
                pooled[key].append(got[key])
    nonzero = pooled_answer_scores(pooled["nonzero"])
    adjustment = pooled_answer_scores(pooled["adjustment"])
    assert nonzero["net_rate"] > nonzero["floor_rate"]
    assert adjustment["net_rate"] > adjustment["floor_rate"]
