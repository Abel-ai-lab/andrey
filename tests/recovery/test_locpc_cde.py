"""Recovery tests for ``locpc_cde``: Loranchet and Assaad (2026) under exact d-separation, and
answers on linear-Gaussian data drawn by the benchmark lane's sampler, scored against the truth.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), every answer the record decides
equals the answer the true CPDAG gives (``andrey.metrics.effect_oracle``), and the set of an
identifiable effect is the outcome's parents, a valid set for the controlled direct effect
(Theorem 14), with one exception the tests name: the non-orientability criterion (Theorem 13)
stops on some graphs whose CPDAG directs every edge at the outcome.
"""

from __future__ import annotations

import importlib
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from andrey import locpc_cde
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.locpc import Memory, locpc
from andrey.constraint.locpc import locpc_cde as engine
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle, effect_scores, pooled_answer_scores, to_cpdag

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

#: The paper's Figure 1(a).
FIGURE_1 = [
    *(("X", "M"), ("M", "Y"), ("X", "Y"), ("Y", "Z1"), ("Z2", "X"), ("Z2", "Y"), ("Z2", "Z3")),
    *(("Z3", "X"), ("Z3", "Z1"), ("Z4", "Z1"), ("Z5", "Z3"), ("Z6", "Z3"), ("Z7", "Z4")),
]
#: The paper's Figure 2(a): the direct effect of X on Y is not identifiable.
FIGURE_2 = [
    *(("X", "A1"), ("W1", "A1"), ("X", "Y"), ("X", "D1"), ("D1", "Y"), ("Y", "D2")),
    *(("A1", "D2"), ("W1", "Z"), ("D2", "A2"), ("A2", "W2"), ("Z", "W2")),
]


def oracle_search(dag: nx.DiGraph, names: list, ranks=None) -> LocalSearch:
    ranks = list(range(len(names))) if ranks is None else ranks
    return LocalSearch(DSepOracle(dag, nodes=names), ranks, 0.5)


def random_dag(rng: np.random.Generator, d: int, p: float) -> np.ndarray:
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=np.int8)
    for i, j in combinations(range(d), 2):
        if rng.random() < p:
            A[order[i], order[j]] = 1
    return A


def collider_at_inner_end(dag: np.ndarray, answer) -> bool:
    """Whether a member's double-bar edge to an outer variable ``a`` ends in a collider ``b -> v
    <- a`` of the true DAG with ``b`` an inner neighbor of the member not adjacent to ``a``: the
    pattern that orients the edge into the member at the next hop."""
    graph, inside = answer.graph, set(answer.graph.neighborhood)
    adjacent = (dag + dag.T) > 0
    for v, a in graph.double_bar:
        if v not in answer.members or not dag[a, v]:
            continue
        for b in graph.neighbors(v):
            if b in inside and b != a and dag[b, v] and not adjacent[a, b]:
                return True
    return False


# --- exact tests --------------------------------------------------------------------------------


@pytest.mark.parametrize("pairs", ["parents", "all"])
def test_every_decided_answer_is_the_true_cpdags_but_the_criterions_named_stops(pairs):
    """On random DAGs with exact tests, an "oriented" answer gives exactly the outcome's parents
    and the oracle calls the effect identifiable; a zero stop is a true zero; a "not orientable"
    answer is right, or its criterion met a double-bar edge that a collider at its inner end
    directs at the next hop (the counterexample to Theorem 13); every non-identifiable effect is
    found."""
    seen = {"oriented": 0, "not orientable": 0, "zero": 0, "criterion": 0}
    for seed in range(40 if pairs == "parents" else 15):
        rng = np.random.default_rng(seed)
        d = int(rng.integers(5, 12))
        A = random_dag(rng, d, min(float(rng.choice([2.0, 3.0])) / (d - 1), 0.9))
        dag = nx.DiGraph()
        dag.add_nodes_from(range(d))
        dag.add_edges_from(zip(*np.nonzero(A), strict=True))
        ranks = rng.permutation(d).tolist()
        if pairs == "parents":
            asked = [(int(x), int(y)) for x, y in zip(*np.nonzero(A), strict=True)]
        else:
            asked = [(x, y) for x in range(d) for y in range(d) if x != y]
        for x, y in asked:
            answer = engine(oracle_search(dag, list(range(d)), ranks), x, y)
            truth = effect_oracle(A, x, y, estimand="direct")
            parents = np.flatnonzero(A[:, y]).tolist()
            if answer.reason == "oriented":
                assert truth["identifiable"]
                assert answer.graph.parents(y) == parents
                seen["oriented"] += 1
            elif answer.reason == "not orientable" and not truth["identifiable"]:
                seen["not orientable"] += 1
            elif answer.reason == "not orientable":
                assert collider_at_inner_end(A, answer), (seed, x, y)
                seen["criterion"] += 1
            else:
                assert not A[x, y]
                seen["zero"] += 1
                if not answer.graph.non_arrow(y):
                    assert answer.graph.parents(y) == parents
    assert min(seen["oriented"], seen["not orientable"]) > 20
    assert seen["criterion"] < seen["oriented"] / 10


def test_the_criterion_stops_on_an_identifiable_effect_when_a_collider_lies_past_the_hop():
    """Theorem 13 says a met criterion shows the effect is not identifiable. Here ``y -> x <- w``
    makes ``y -> x`` compelled (``x`` is the outcome's child, a zero effect, and the outcome's only
    edge is directed), but at hop 1 ``w`` is outside, ``x``'s edge to it is double-bar (``v`` is
    separated from ``x`` by the empty set), the criterion holds for ``{y, x}``, and the answer is
    "not orientable"."""
    dag = nx.DiGraph([("y", "x"), ("w", "x")])
    dag.add_node("v")
    names = ["v", "w", "x", "y"]
    answer = engine(oracle_search(dag, names), names.index("x"), names.index("y"))
    assert answer.reason == "not orientable" and answer.hop == 1
    assert (names.index("x"), names.index("w")) in answer.graph.double_bar
    cpdag = to_cpdag(nx.to_numpy_array(dag, nodelist=names, dtype=np.int8)).to_numpy()
    assert cpdag[names.index("y"), names.index("x")] == 1  # y -> x in the CPDAG
    larger = locpc(oracle_search(dag, names), names.index("y"), 2, Memory())
    assert larger.parents(names.index("x")) == [names.index("w"), names.index("y")]


def test_figure_1_orients_every_edge_at_the_outcome_and_returns_its_parents():
    """The effect of Figure 1 is identifiable and the outcome's parents are M, X, and Z2; at hop
    0 the outcome's edge to its child Z1 is double-bar, as the figure's L^{Y,0} draws it."""
    dag = nx.DiGraph(FIGURE_1)
    names = sorted(dag.nodes)
    ix = names.index
    answer = engine(oracle_search(dag, names), ix("X"), ix("Y"))
    assert answer.reason == "oriented"
    assert [names[v] for v in answer.graph.parents(ix("Y"))] == ["M", "X", "Z2"]
    hop0 = locpc(oracle_search(dag, names), ix("Y"), 0, Memory())
    assert (ix("Y"), ix("Z1")) in hop0.double_bar
    assert [names[v] for v in hop0.neighbors(ix("Y"))] == ["M", "X", "Z1", "Z2"]


def test_figure_2_is_not_orientable():
    """Figure 2's effect is not identifiable: X - Y is undirected in its CPDAG."""
    dag = nx.DiGraph(FIGURE_2)
    names = sorted(dag.nodes)
    answer = engine(oracle_search(dag, names), names.index("X"), names.index("Y"))
    assert answer.reason == "not orientable"
    A = nx.to_numpy_array(dag, nodelist=names, dtype=np.int8)
    assert not effect_oracle(A, names.index("X"), names.index("Y"), estimand="direct")[
        "identifiable"
    ]


def test_the_answer_does_not_depend_on_the_column_order_or_the_ranks_beyond_ties():
    """With exact tests, every relabeling of Figure 1 and Figure 2 and every rank order gives the
    same verdict and the same parents by name."""
    for edges in (FIGURE_1, FIGURE_2):
        dag = nx.DiGraph(edges)
        base = None
        for seed in range(6):
            rng = np.random.default_rng(seed)
            names = list(rng.permutation(sorted(dag.nodes)))
            ranks = rng.permutation(len(names)).tolist()
            answer = engine(oracle_search(dag, names, ranks), names.index("X"), names.index("Y"))
            got = (answer.reason, sorted(names[v] for v in answer.graph.parents(names.index("Y"))))
            base = got if base is None else base
            assert got == base


# --- data ---------------------------------------------------------------------------------------


def lane_models():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        return importlib.import_module("andrey_bench.adjustment_models")
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_on_the_papers_linear_gaussian_setting_the_answers_beat_the_base_rate():
    """On the paper's planted Erdos-Renyi graphs (10 nodes, n = 5000), half with an identifiable
    effect, the pooled ``identifiable`` and ``adjustment`` answers beat the best constant answer,
    no ``nonzero`` answer is wrong (every pair is a parent and its outcome), and the data's column
    order does not change a record."""
    lane = lane_models()
    pooled = {"identifiable": [], "adjustment": [], "nonzero": []}
    for seed in range(6):
        for identifiable in (True, False):
            model = lane.locpc(seed, identifiable=identifiable, v=10)
            pair = model.pairs[0]
            rows = model.rows(5000, lane.DATA)
            out = locpc_cde(rows, treatment=pair.treatment, outcome=pair.outcome)
            truth = effect_oracle(model.dag, pair.treatment, pair.outcome, estimand="direct")
            got = effect_scores(out.metadata["effects"], [truth], true=model.dag)
            for key in pooled:
                pooled[key].append(got[key])
            if seed < 2:
                order = np.random.default_rng(seed).permutation(rows.shape[1])
                labels = [f"v{j}" for j in range(rows.shape[1])]
                frame = pd.DataFrame(rows, columns=labels)[[labels[j] for j in order]]
                again = locpc_cde(
                    frame, treatment=labels[pair.treatment], outcome=labels[pair.outcome]
                )
                named = locpc_cde(
                    pd.DataFrame(rows, columns=labels),
                    treatment=labels[pair.treatment],
                    outcome=labels[pair.outcome],
                )
                assert again.metadata["effects"] == named.metadata["effects"]
    scores = {key: pooled_answer_scores(parts) for key, parts in pooled.items()}
    for key in ("identifiable", "adjustment"):
        assert scores[key]["net_rate"] > scores[key]["floor_rate"], (key, scores[key])
    assert scores["nonzero"]["wrong"] == 0
    assert scores["identifiable"]["risk"] <= 0.25
