"""Recovery tests for ``y_structures``: the examples and the theorem of Versteeg, Zhang, and Mooij
(2022) under exact d-separation, and claims on linear-Gaussian data drawn under selection bias by
the benchmark lane's sampler, scored against the true ancestors.

With exact tests (:class:`~andrey.data.latent.DSepOracle`, every query also given the selection
nodes), every claim must hold in the graph: the cause is an ancestor of the effect, the effect is
not an ancestor of the cause, the cause is not an ancestor of a selection node, and no hidden
common cause joins the two (Proposition 4). The method must also claim every pair that a pattern
of four observed variables shows, found here by testing every ordered four-tuple directly.
"""

from __future__ import annotations

import importlib
import itertools
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from andrey import y_structures
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.ystructures import y_structures as engine
from andrey.data.latent import DSepOracle
from andrey.metrics import claim_scores

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def oracle_claims(edges, observed, selection=(), *, extended=False):
    """The engine's claims under d-separation, as ``{(x, y): patterns}`` by node name."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    search = LocalSearch(DSepOracle(dag, nodes=observed, given=selection), observed, 0.5)
    name = observed.__getitem__
    return {
        (name(c.cause), name(c.effect)): {(name(v), name(w)) for v, w in c.patterns}
        for c in engine(search, range(len(observed)), extended=extended)
    }


def pattern_pairs(edges, observed, selection=(), *, extended=False):
    """Every ``(x, y)`` some ordered ``(v, w, x, y)`` of observed variables makes a pattern of,
    by d-separation given ``selection``, tested tuple by tuple."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    separated = DSepOracle(dag, nodes=observed, given=selection)
    index = {name: k for k, name in enumerate(observed)}

    def apart(a, b, given=()):
        return separated(index[a], index[b], [index[z] for z in given]) == 1.0

    found = set()
    for v, w, x, y in itertools.permutations(observed, 4):
        minimal = not apart(v, y) and apart(v, y, (x,)) and apart(v, w) and not apart(v, w, (x,))
        if minimal and (extended or (not apart(w, y) and apart(w, y, (x,)))):
            found.add((x, y))
    return found


def holds(edges, observed, selection, x, y):
    """Whether Proposition 4's conclusion holds for ``x -> y`` in the DAG ``edges``."""
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(observed)
    hidden = set(dag) - set(observed) - set(selection)
    below_x = nx.descendants(dag, x)
    # A hidden node with a directed path of hidden nodes to each of x and y joins them.
    confounded = any(
        all(v in nx.descendants(dag.subgraph(hidden | {v}), h) for v in (x, y)) for h in hidden
    )
    return (
        y in below_x
        and x not in nx.descendants(dag, y)
        and not below_x & set(selection)
        and not confounded
    )


#: Figure 4(b): v <-> x <-> w, x -> y; the bidirected edges through hidden nodes.
FIG4B = ([("h1", "v"), ("h1", "x"), ("h2", "w"), ("h2", "x"), ("x", "y")], ["v", "w", "x", "y"])
#: Figure 4(c): v -> x <-> s <- w, x -> y, with s the selection node.
FIG4C = ([("v", "x"), ("h", "x"), ("h", "s"), ("w", "s"), ("x", "y")], ["v", "w", "x", "y"])
#: Figure 5(a), the paper's fixed graph: c -> x1, c -> x5 <- x4, x5 -> x6, x3 -> x2, and the
#: selection node s <- x1, x2.
FIG5A = (
    [("c", "x1"), ("c", "x5"), ("x4", "x5"), ("x5", "x6"), ("x3", "x2"), ("x2", "s"), ("x1", "s")],
    ["c", "x1", "x2", "x3", "x4", "x5", "x6"],
)


@pytest.mark.parametrize("extended", [False, True])
def test_the_papers_y_structure_examples_claim_x_causes_y(extended):
    """Figure 4(b) and 4(c) of the paper each claim ``x -> y`` from the pattern ``(v, w)``: in
    4(c) the selection opens the path from ``w`` to ``y`` the Y-structure needs."""
    for (edges, observed), selection in ((FIG4B, ()), (FIG4C, ("s",))):
        found = oracle_claims(edges, observed, selection, extended=extended)
        assert set(found) == {("x", "y")}
        assert found["x", "y"] == ({("v", "w"), ("w", "v")} if extended else {("v", "w")})


@pytest.mark.parametrize("selection", [("s",), ()])
@pytest.mark.parametrize("extended", [False, True])
def test_the_fixed_graph_claims_only_x5_causes_x6(selection, extended):
    """On the paper's fixed graph, the one claim is ``x5 -> x6`` (the Y-structure
    ``<c, x4, x5, x6>``), with or without the selection; nothing is claimed about ``x1`` and
    ``x2``, which selection makes dependent (where a three-variable rule claims ``x1 -> x2``)."""
    found = oracle_claims(*FIG5A, selection, extended=extended)
    assert set(found) == {("x5", "x6")}
    assert ("c", "x4") in found["x5", "x6"]
    # Selection makes x1, x2, x3 dependent on c, so each becomes a pattern's other variable.
    witnesses = {v for pair in found["x5", "x6"] for v in pair} - {"x4"}
    assert witnesses == ({"c", "x1", "x2", "x3"} if selection else {"c", "x1"})


def random_selection_graph(seed):
    """A random DAG over 12 nodes, 2 of them hidden, each ordered pair an edge with probability
    0.2, and a selection node ``s`` whose parents are 2 observed nodes."""
    rng = np.random.default_rng(seed)
    nodes = [f"n{i}" for i in range(12)]
    order = rng.permutation(12)
    edges = [
        (nodes[order[i]], nodes[order[j]])
        for i in range(12)
        for j in range(i + 1, 12)
        if rng.random() < 0.2
    ]
    hidden = {nodes[i] for i in rng.choice(12, size=2, replace=False)}
    observed = [n for n in nodes if n not in hidden]
    parents = rng.choice(observed, size=2, replace=False)
    return [*edges, *((str(p), "s") for p in parents)], observed


@pytest.mark.parametrize("extended", [False, True])
def test_every_claim_holds_and_every_pattern_is_claimed_under_d_separation(extended):
    """On 100 random graphs with hidden nodes, with and without selection: every claim holds in
    the graph (Proposition 4), and the claims are exactly the pairs a pattern shows."""
    claims = selected = 0
    for seed in range(100):
        edges, observed = random_selection_graph(seed)
        for selection in (("s",), ()):
            found = oracle_claims(edges, observed, selection, extended=extended)
            assert set(found) == pattern_pairs(edges, observed, selection, extended=extended)
            for x, y in found:
                assert holds(edges, observed, selection, x, y), (seed, selection, x, y)
            claims += len(found)
            selected += len(found) if selection else 0
    assert claims > 50 and selected > 10


def selection_sample(setting, seed):
    """One model of the benchmark lane's selection sampler, with the paper's 10,000 rows."""
    import sys

    sys.path.insert(0, str(BENCHMARKS))
    try:
        models = importlib.import_module("andrey_bench.selection_models")
    finally:
        sys.path.remove(str(BENCHMARKS))
    return models.sample(setting, n=10_000, seed=seed)


def scored(sample, data, *, extended=False):
    """A fit on ``data`` and its claim scores among the system variables, against the truth."""
    system = sample.system.tolist()
    fit = y_structures(data, variables=system, extended=extended)
    pairs = [(a, b) for a in system for b in system if a != b]
    return fit, claim_scores(fit, sample.graph.adjacency, pairs=pairs)


@pytest.mark.parametrize(
    ("setting", "rows", "extended"),
    [
        ("fixed", "biased", False),
        ("fixed", "biased", True),
        ("fixed", "unbiased", False),
        ("fixed", "unbiased", True),
        ("large", "unbiased", False),
    ],
)
def test_claims_beat_the_base_rate(setting, rows, extended):
    """On data drawn as in the paper's simulations, with selection bias (``biased``) or without,
    the claims pooled over 5 models are right more often than the share of true ancestor pairs
    and more often right than wrong."""
    true_claims = false_claims = 0
    base = []
    for seed in range(5):
        sample = selection_sample(setting, seed)
        _, scores = scored(sample, getattr(sample, rows), extended=extended)
        true_claims += scores["true_claims"]
        false_claims += scores["false_claims"]
        base.append(scores["base_rate"])
    assert true_claims > false_claims
    assert true_claims / (true_claims + false_claims) > np.mean(base)
