"""Recovery tests for ``ldp`` and ``ld3``: the graphs and theorems of Maasch et al. (2024, 2025)
under exact d-separation, and answers on linear-Gaussian data drawn from the papers' graphs by the
benchmark lane's sampler, scored against the truth.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), LDP labels the 10-node graph of its
Figure 3 exactly, and the set it returns is a valid adjustment set in the true graph whenever it
returns one, hidden variables included (Theorem 4.5); its ``Z4`` variables are never descendants
of the outcome and its ``Z7`` variables are descendants of the treatment (Theorem 4.6). LD3 finds
exactly the parents of a sink outcome whose parents are all observed (Theorem 1), even with other
variables hidden (Theorem 2); the graphs where its structural direct criterion is wrong are
listed by name.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from andrey import GraphStructure, ld3, ldp
from andrey.constraint import partition
from andrey.constraint.blanket import LocalSearch
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle, effect_scores, pooled_answer_scores

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

#: LDP's Figure 3: one node per partition.
FIGURE_3 = [
    *(("Z1", "X"), ("Z1", "Y"), ("Z5", "X"), ("Z4", "Y"), ("X", "Y"), ("X", "Z3")),
    *(("Z3", "Y"), ("X", "Z2"), ("Y", "Z2"), ("X", "Z7"), ("Y", "Z6")),
]

#: LDP's Figure E.5: Figure 3 with an M-structure, a butterfly, and second variables behind Z4 and
#: Z5; the paper hides its dashed nodes one at a time.
FIGURE_E5 = [
    *(("B1", "B3"), ("B2", "B3"), ("B1", "X"), ("B3", "X"), ("B3", "Y"), ("B2", "Y")),
    *(("Z1", "X"), ("Z1", "Y"), ("Z5a", "X"), ("Z5a", "Z5b"), ("Z4a", "Y"), ("Z4a", "Z4b")),
    *(("X", "Y"), ("X", "Z7"), ("X", "Z3"), ("Z3", "Y"), ("X", "Z2"), ("Y", "Z2")),
    *(("Y", "Z6"), ("Z6", "Z2"), ("M1", "X"), ("M1", "M3"), ("M2", "M3"), ("M2", "Y")),
]
DASHED = ("B1", "B2", "B3", "Z1", "Z5a", "Z4a", "M1", "M2")

#: LD3's Figure D.1 with the authors' Z7 and Z8: the outcome is a sink.
FIGURE_D1 = [
    *(("B1", "B3"), ("B2", "B3"), ("B1", "X"), ("B3", "X"), ("B3", "Y"), ("B2", "Y")),
    *(("Z1", "X"), ("Z1", "Y"), ("Z5", "X"), ("X", "Y"), ("X", "Z3a"), ("Z3a", "Z3b")),
    *(("Z3b", "Z3c"), ("Z3c", "Y"), ("X", "Z3d"), ("Z4a", "Z3d"), ("Z3d", "Y"), ("Z4a", "Z4b")),
    *(("Z4b", "Y"), ("M1", "X"), ("M1", "M3"), ("M2", "M3"), ("M2", "Y"), ("X", "Z7")),
]


def oracle_answer(method, edges, observed, x="X", y="Y"):
    """The engine's answer under d-separation over ``observed``, with every name mapped back."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    search = LocalSearch(DSepOracle(dag, nodes=list(observed)), list(observed), 0.5)
    ix = {v: k for k, v in enumerate(observed)}
    others = [k for k in range(len(observed)) if k not in (ix[x], ix[y])]
    return getattr(partition, method)(search, ix[x], ix[y], others)


def names(observed, positions):
    return {observed[k] for k in positions}


def true_graph(edges, nodes) -> GraphStructure:
    ix = {v: k for k, v in enumerate(nodes)}
    marks = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a, b in edges:
        marks[ix[a], ix[b]], marks[ix[b], ix[a]] = 1, 2
    return GraphStructure.from_numpy(marks, kind="dag", labels=tuple(nodes))


def all_nodes(edges, extra=()):
    return list(dict.fromkeys([v for e in edges for v in e] + list(extra)))


def test_ldp_labels_figure_3_exactly():
    observed = [*all_nodes(FIGURE_3), "Z8"]
    for effect in (True, False):
        edges = FIGURE_3 if effect else [e for e in FIGURE_3 if e != ("X", "Y")]
        result = oracle_answer("ldp", edges, observed)
        got = {k: names(observed, v) for k, v in result.partition.items() if v}
        assert got == {
            "Z1": {"Z1"},
            "Z4": {"Z4"},
            "Z5": {"Z5"},
            "Z7": {"Z7"},
            "Z8": {"Z8"},
            "Zpost": {"Z2", "Z3", "Z6"},
        }
        assert result.valid


def test_ldp_figure_e5_returns_only_valid_sets():
    nodes = all_nodes(FIGURE_E5)
    true = true_graph(FIGURE_E5, nodes)
    answered = 0
    for hidden in ((), *((v,) for v in DASHED)):
        observed = [v for v in nodes if v not in hidden]
        result = oracle_answer("ldp", FIGURE_E5, observed)
        if result.valid:
            answered += 1
            assert true.is_adjustment_set("X", "Y", names(observed, result.partition["Z1"]))
        assert_theorem_4_6(FIGURE_E5, observed, result)
    assert answered >= 5


def test_ldp_gives_no_set_for_a_zero_effect_a_set_identifies():
    # Schubert et al. (2026), Example A.3: {V2} is valid and the effect is zero, but the
    # instrument V1 is independent of the outcome, so Step 3 misses it and Step 8 has no Z5.
    edges = [("V1", "X"), ("V2", "X"), ("V2", "Y"), ("V3", "Y")]
    observed = ["V1", "X", "V2", "Y", "V3"]
    assert true_graph(edges, observed).is_adjustment_set("X", "Y", {"V2"})
    result = oracle_answer("ldp", edges, observed)
    assert not result.valid
    assert names(observed, result.partition["Z4"]) == {"V3"}


def assert_theorem_4_6(edges, observed, result, x="X", y="Y"):
    """Z4, Z7, and Z8 as Table 1 defines them: Z4 variables are not descendants of the outcome,
    depend on it, and are independent of the treatment; Z7 variables are descendants of the
    treatment that every path to the outcome through it separates; Z8 variables are independent
    of both."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    for z in names(observed, result.partition["Z4"]):
        assert z not in nx.descendants(dag, y)
        assert not nx.is_d_separator(dag, z, y, set())
        assert nx.is_d_separator(dag, z, x, set())
    for z in names(observed, result.partition["Z7"]):
        assert z in nx.descendants(dag, x)
        assert nx.is_d_separator(dag, z, y, {x})
    for z in names(observed, result.partition["Z8"]):
        assert nx.is_d_separator(dag, z, x, set()) and nx.is_d_separator(dag, z, y, set())


def random_dag(rng, d, p):
    order = rng.permutation(d)
    upper = np.triu(rng.random((d, d)) < p, 1)
    return [(f"V{order[i]}", f"V{order[j]}") for i, j in zip(*np.nonzero(upper), strict=True)]


def test_ldp_on_random_graphs_with_hidden_variables():
    # Random covariates around a pair with a planted confounder C, instrument I, and cause of
    # the outcome P, so that conditions C1 and C2 can hold; the effect is null half the time.
    rng = np.random.default_rng(7)
    answered = tried = 0
    for _ in range(80):
        d = int(rng.integers(6, 11))
        dag = nx.DiGraph(random_dag(rng, d, 0.2))
        dag.add_nodes_from(f"V{k}" for k in range(d))
        x, y = (f"V{k}" for k in rng.choice(d, 2, replace=False))
        if x in nx.descendants(dag, y):
            x, y = y, x
        dag.add_edges_from([("C", x), ("C", y), ("I", x), ("P", y)])
        if rng.random() < 0.5:
            dag.add_edge(x, y)
        hidden = {f"V{k}" for k in rng.choice(d, int(rng.integers(0, 3)), replace=False)} - {x, y}
        observed = [v for v in dag if v not in hidden]
        result = oracle_answer("ldp", list(dag.edges), observed, x, y)
        tried += 1
        if result.valid:
            answered += 1
            true = true_graph(list(dag.edges), list(dag))
            assert true.is_adjustment_set(x, y, names(observed, result.partition["Z1"]))
        assert_theorem_4_6(list(dag.edges), observed, result, x, y)
    assert answered >= tried / 4


def test_ld3_finds_the_parents_in_figure_d1():
    nodes = [*all_nodes(FIGURE_D1), "Z8"]
    parents = {"Z1", "Z3c", "Z3d", "Z4b", "M2", "B2", "B3"}
    for hidden in ((), ("B1",), ("Z4a",), ("M1",), ("Z3a",)):
        observed = [v for v in nodes if v not in hidden]
        result = oracle_answer("ld3", FIGURE_D1, observed)
        assert names(observed, result.parents) == parents
        assert result.direct


def test_ld3_criterion_fails_when_a_mediator_and_a_cause_of_the_outcome_collide():
    # X -> M <- W, M -> Y <- W: W is independent of X and so in Z4; the criterion's test
    # conditions on M alone, which opens X -> M <- W -> Y, so it says X is a parent of Y.
    edges = [("X", "M"), ("W", "M"), ("M", "Y"), ("W", "Y")]
    observed = ["X", "M", "W", "Y"]
    result = oracle_answer("ld3", edges, observed)
    assert names(observed, result.parents) == {"M", "W"}
    assert names(observed, result.partition["Z4"]) == {"W"}
    assert result.direct


def test_ld3_criterion_fails_on_figure_d1_without_the_direct_effect():
    # The same collider in the paper's own graph: X -> Z3d <- Z4a -> Z4b -> Y with Z3d -> Y.
    nodes = [*all_nodes(FIGURE_D1), "Z8"]
    result = oracle_answer("ld3", [e for e in FIGURE_D1 if e != ("X", "Y")], nodes)
    assert names(nodes, result.parents) == {"Z1", "Z3c", "Z3d", "Z4b", "M2", "B2", "B3"}
    assert result.direct


def sink_dag(rng, d, p):
    """A random DAG whose last node in the order, the outcome, has no children."""
    order = [f"V{k}" for k in rng.permutation(d)]
    upper = np.triu(rng.random((d, d)) < p, 1)
    edges = [(order[i], order[j]) for i, j in zip(*np.nonzero(upper), strict=True)]
    return edges, order[-1]


def test_ld3_parents_on_random_sink_graphs():
    rng = np.random.default_rng(11)
    checked = wrong_criterion = 0
    for _ in range(150):
        d = int(rng.integers(6, 13))
        edges, y = sink_dag(rng, d, 0.3)
        dag = nx.DiGraph(edges)
        dag.add_nodes_from(f"V{k}" for k in range(d))
        parents = set(dag.predecessors(y))
        if not parents:
            continue
        x = sorted(parents)[0] if rng.random() < 0.5 else str(rng.choice(sorted(set(dag) - {y})))
        free = sorted(set(dag) - parents - {x, y})
        hidden = set(rng.choice(free, min(len(free), int(rng.integers(0, 3))), replace=False))
        observed = [v for v in sorted(dag, key=lambda v: int(v[1:])) if v not in hidden]
        result = oracle_answer("ld3", list(dag.edges), observed, x, y)
        truth = effect_oracle(
            true_graph(list(dag.edges), list(dag)),
            x,
            y,
            estimand="direct",
            set_kind="parents_of_outcome",
            identified_from="dag",
            latent=sorted(hidden),
        )
        assert [names(observed, result.parents)] == [set(s) for s in truth["adjustment_sets"]]
        checked += 1
        wrong_criterion += result.direct != truth["nonzero"]
        if result.direct != truth["nonzero"]:
            # Only a false "direct": a Z4 parent of the outcome is a parent of a mediator too.
            assert result.direct and not truth["nonzero"]
    assert checked >= 100
    assert wrong_criterion < checked / 4


def test_ld3_test_count_is_linear():
    rng = np.random.default_rng(3)
    for d in (10, 20, 40):
        edges, y = sink_dag(rng, d, 3 / d)
        dag = nx.DiGraph(edges)
        dag.add_nodes_from(f"V{k}" for k in range(d))
        x = next(iter(set(dag) - {y}))
        observed = sorted(dag, key=lambda v: int(v[1:]))
        oracle = DSepOracle(dag, nodes=observed)
        search = LocalSearch(oracle, observed, 0.5)
        ix = observed.index
        partition.ld3(search, ix(x), ix(y), [k for k in range(d) if k not in (ix(x), ix(y))])
        # Two marginal tests, at most two given one of the pair, and one or two parent tests per
        # variable, then the criterion.
        assert search.counts.issued <= 5 * (d - 2) + 1


# --- data ---------------------------------------------------------------------------------------


def lane_models():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        return importlib.import_module("andrey_bench.adjustment_models")
    finally:
        sys.path.remove(str(BENCHMARKS))


def scores(method, models, seeds, n):
    """``effect_scores`` of ``method``'s records on each model ``models(seed)`` gives, pooled."""
    lane = lane_models()
    pooled = {"nonzero": [], "identifiable": [], "adjustment": []}
    for seed in seeds:
        for model in models(seed):
            pair = model.pairs[0]
            x, y, estimand = pair.treatment, pair.outcome, pair.estimand
            out = method(model.rows(n, lane.DATA), treatment=x, outcome=y)
            truth = effect_oracle(
                model.dag,
                x,
                y,
                estimand=estimand,
                set_kind="valid" if estimand == "total" else "parents_of_outcome",
                identified_from="dag",
                latent=list(model.hidden),
            )
            got = effect_scores(out.metadata["effects"], [truth], true=model.dag)
            for key in pooled:
                pooled[key].append(got[key])
    return {key: pooled_answer_scores(parts) for key, parts in pooled.items()}


def test_ldp_on_figure_3_data_beats_the_floor():
    lane = lane_models()
    got = scores(lambda *a, **k: ldp(*a, **k), lambda s: [lane.ldp(s)], range(5), 5000)
    got = got["adjustment"]
    assert got["coverage"] >= 0.6
    assert got["net_rate"] > got["floor_rate"]
    assert got["risk"] == 0


def test_ld3_on_figure_d1_data_beats_the_floor():
    lane = lane_models()
    figure = scores(
        lambda *a, **k: ld3(*a, **k),
        lambda s: [lane.ld3_dag(s, effect=e) for e in (1.25, 0.0)],
        range(4),
        5000,
    )
    assert figure["adjustment"]["net_rate"] > figure["adjustment"]["floor_rate"]
    # The criterion with and without the direct effect, on the graph of the authors' code, which
    # lacks the arc Z4a -> Z4b (see test_ld3_criterion_fails_when_...).
    edges = [e for e in lane.LD3_EDGES if e != ("Z4a", "Z4b")]

    def code_graph(seed):
        return [
            lane.from_edges(
                "ld3-dag",
                seed,
                [e for e in edges if effect or e != ("X", "Y")],
                ("Z8",),
                coefficient=lambda rng: float(rng.uniform(0.75, 1.1)),
                special={("X", "Y"): 1.25},
                pair=("X", "Y", "direct"),
            )
            for effect in (True, False)
        ]

    code = scores(ld3, code_graph, range(4), 5000)
    assert code["nonzero"]["net_rate"] > code["nonzero"]["floor_rate"]
    assert code["adjustment"]["net_rate"] > code["adjustment"]["floor_rate"]
