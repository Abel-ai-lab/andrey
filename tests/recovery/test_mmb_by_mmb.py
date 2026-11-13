"""Recovery checks for ``mmb_by_mmb``, MMB-by-MMB (Xie, Li, Wu, Zeng, Liu, and Geng 2024).

Under a d-separation oracle, on the graph where the paper's stop rule R3 stops early and 60 random
graphs with up to four hidden common causes, 657 targets, the answer at every target holds the
edges the target has in the PAG of the observed variables, each mark the PAG's or a circle, and
equals the PAG's at all but three targets. On simulated linear-Gaussian data with hidden common
causes, pooled over every target, the answer is scored against those edges
(``target_answer_scores``): its adjacencies beat the empty answer's, and its marks are right more
often than wrong, which the empty answer, with no mark, is not. The paper's blanket learner, total
conditioning, and IAMB both pass.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.sequential import MMB_BY_MMB, sequential
from andrey.core import CIRCLE
from andrey.data.latent import DSepOracle, causal_digraph
from andrey.metrics import pooled_answer_scores, target_answer_scores, target_oracle

#: The graph on which R3 stops at ``1 o-> 4`` after one blanket, where the PAG has ``1 -> 4``.
_R3 = [(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (0, 5), (1, 5), (5, 4)]


def _random_dag(seed: int, d: int, hidden: int, mediated: bool) -> nx.DiGraph:
    """An Erdos-Renyi DAG over ``d + hidden`` nodes, the last ``hidden`` of them hidden: each a
    common parent of two observed nodes and, with ``mediated``, a child of a third, earlier one."""
    rng = np.random.default_rng(seed)
    dag = causal_digraph(
        data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=2.5).graph
    )
    order = {v: k for k, v in enumerate(nx.topological_sort(dag))}
    for h in range(d, d + hidden):
        a, b, c = sorted(rng.choice(d, size=3, replace=False).tolist(), key=order.__getitem__)
        dag.add_edges_from([(h, b), (h, c)])
        if mediated:
            dag.add_edge(a, h)
    return dag


def _graphs():
    yield "r3", nx.DiGraph(_R3), 6
    for s in range(60):
        d = 7 + s % 9
        yield f"er{s}", _random_dag(s, d, s % 5, s % 2 == 1), d


def test_every_target_of_61_graphs_under_exact_tests():
    # The engine as the method runs it, with IAMB blankets and the local structures uncapped. The
    # three targets left with a circle are recorded; any other target that differs from the PAG
    # fails.
    circles, targets = set(), 0
    for name, dag, d in _graphs():
        dag.add_nodes_from(range(d))
        adjacency = nx.to_numpy_array(dag, nodelist=range(dag.number_of_nodes()))
        hidden = range(d, dag.number_of_nodes())
        for t in range(d):
            search = LocalSearch(DSepOracle(dag), list(range(d)), 0.05)
            answer = sequential(search, t, "iamb", MMB_BY_MMB).network.star(t, d)
            truth = target_oracle(adjacency, t, "pag", latent=hidden).to_numpy()
            targets += 1
            if np.array_equal(answer, truth):
                continue
            assert np.array_equal(answer != 0, truth != 0), (name, t)
            assert np.all((answer == truth) | (answer == CIRCLE)), (name, t)
            circles.add((name, t))
    assert targets == 657
    assert circles <= {("er26", 7), ("er26", 11), ("er35", 7)}


@pytest.mark.parametrize("blanket", ["total_conditioning", "iamb"])
def test_mmb_by_mmb_beats_the_empty_answer(blanket):
    # 20 nodes, 3 of them hidden common causes: 17 columns.
    sample = data.sample_scm(
        graph="erdos_renyi", d=20, n=5000, seed=0, density=2.0, latents=3, scale="standardize"
    )
    d = sample.data.shape[1]
    hidden = range(d, sample.graph.n_nodes)
    adjacency, marks = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        for t in range(d):
            out = andrey.mmb_by_mmb(sample.data, target=t, blanket=blanket)
            truth = target_oracle(sample.graph, t, "pag", latent=hidden)
            scores = target_answer_scores(out.structure, truth, t)
            adjacency.append(scores["adjacency"])
            marks.append(scores["marks"])
    adjacency, marks = pooled_answer_scores(adjacency), pooled_answer_scores(marks)
    # The empty answer says False for every pair; it gets the true adjacencies wrong.
    assert adjacency["net"] > adjacency["policy_net"]["False"]
    assert adjacency["correct"] > 0.9 * adjacency["queries"]
    # The empty answer has no edge, so no mark: a net of zero.
    assert marks["net"] > 0
