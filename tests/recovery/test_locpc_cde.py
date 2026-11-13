"""Recovery tests for ``locpc_cde``: Loranchet and Assaad (2026) under exact d-separation, and
answers on linear-Gaussian data drawn by the benchmark lane's sampler, scored against the truth.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), every verdict equals the one the true
CPDAG gives (``andrey.metrics.effect_oracle``), and the set of an identifiable effect is the
outcome's parents, a valid set for the controlled direct effect.
"""

from __future__ import annotations

import importlib
import sys
from collections import Counter
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


def draw(seed: int, low: int, high: int) -> tuple[np.ndarray, nx.DiGraph, list[int]]:
    """A random DAG with ``low`` to ``high`` variables, two or three expected parents per
    variable, and a random rank order of its variables."""
    rng = np.random.default_rng(seed)
    d = int(rng.integers(low, high + 1))
    A = random_dag(rng, d, min(float(rng.choice([2.0, 3.0])) / (d - 1), 0.9))
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d))
    dag.add_edges_from(zip(*np.nonzero(A), strict=True))
    return A, dag, rng.permutation(d).tolist()


def assert_the_cpdags_verdict(A: np.ndarray, answer, x: int, y: int) -> None:
    """The answer for ``x`` and ``y`` is the true CPDAG's: a zero stop only for a true zero
    (``x`` not adjacent to ``y``, or its child), otherwise "oriented" exactly when the CPDAG
    directs every edge at ``y``, with ``y``'s true parents; a zero stop that directs every edge
    at ``y`` gives them too."""
    truth = effect_oracle(A, x, y, estimand="direct")
    parents = np.flatnonzero(A[:, y]).tolist()
    if answer.reason == "not adjacent":
        assert not A[x, y] and not A[y, x]
    elif answer.reason == "treatment is a child":
        assert A[y, x]
    else:
        assert (answer.reason == "oriented") == truth["identifiable"], answer.reason
    if not answer.graph.non_arrow(y):
        assert truth["identifiable"] and answer.graph.parents(y) == parents


# --- exact tests --------------------------------------------------------------------------------


@pytest.mark.parametrize("first", range(0, 300, 30))
def test_every_verdict_on_a_parent_pair_is_the_true_cpdags(first):
    """On 300 random DAGs of 6 to 15 variables, for every pair ``x -> y`` of the DAG, as the
    paper samples its pairs, the verdict and the set are the true CPDAG's."""
    for seed in range(first, first + 30):
        A, dag, ranks = draw(seed, 6, 15)
        for x, y in zip(*np.nonzero(A), strict=True):
            answer = engine(oracle_search(dag, list(range(len(A))), ranks), int(x), int(y))
            assert_the_cpdags_verdict(A, answer, int(x), int(y))


def test_every_verdict_on_any_pair_is_the_true_cpdags():
    """On random DAGs of 5 to 11 variables, for every ordered pair, the verdict is the true
    CPDAG's, and both verdicts and both zero stops occur."""
    seen = Counter()
    for seed in range(15):
        A, dag, ranks = draw(seed, 5, 11)
        d = len(A)
        for x, y in ((x, y) for x in range(d) for y in range(d) if x != y):
            answer = engine(oracle_search(dag, list(range(d)), ranks), x, y)
            assert_the_cpdags_verdict(A, answer, x, y)
            seen[answer.reason] += 1
    assert min(seen.values()) > 20 and len(seen) == 4, seen


def test_a_double_bar_edge_that_a_collider_past_the_hop_directs_is_learned_to_that_hop():
    """In ``y -> x <- w``, with ``v`` apart, the outcome's edge to ``x`` is undirected at hop 1
    and ``x``'s edge to ``w``, still outside, is double-bar (``v`` is separated from ``x`` by the
    empty set). At hop 2 the collider at ``x`` directs ``y -> x``, as the CPDAG does, and the
    answer is a zero effect."""
    dag = nx.DiGraph([("y", "x"), ("w", "x")])
    dag.add_node("v")
    names = ["v", "w", "x", "y"]
    v, w, x, y = range(4)
    hop1 = locpc(oracle_search(dag, names), y, 1, Memory())
    assert (x, w) in hop1.double_bar and hop1.non_arrow(y) == [x]
    answer = engine(oracle_search(dag, names), x, y)
    assert answer.reason == "treatment is a child" and answer.hop == 2
    assert answer.graph.parents(x) == [w, y]
    cpdag = to_cpdag(nx.to_numpy_array(dag, nodelist=names, dtype=np.int8)).to_numpy()
    assert cpdag[y, x] == 1  # y -> x in the CPDAG


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
