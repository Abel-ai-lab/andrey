"""Graph conversions and the ``mag`` kind: a CPDAG's DAG extension, a DAG's MAG and PAG, DOT text.

Fixed cases are small graphs whose answer is derived by hand. The properties hold on every input:
an extension lies in the CPDAG's equivalence class; a MAG is ancestral and maximal, carries the
ancestor marks, and has the DAG's independence model over the observed nodes; a PAG equals what
FCI returns with a d-separation test on the same DAG; DOT text parses back to the same marks. A
``mag`` holds no circle, scores as a PAG with the same marks, and survives every exchange format.
"""

from __future__ import annotations

import json
import re
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey import GraphStructure
from andrey.cli import graph_view
from andrey.core import ci
from andrey.core.structure import ARROW, CIRCLE, LATENT, NULL, TAIL
from andrey.metrics import markov_equivalent, score, shd, to_cpdag
from andrey.resample import BootstrapOutput
from andrey.spec.generate import to_json_schema
from andrey.spec.models import OutputSpec
from andrey.spec.registry import REGISTRY

_, T, A, C = NULL, TAIL, ARROW, CIRCLE


def _dag(edges: list[tuple[int, int]], n: int, labels=None, node_types=None) -> GraphStructure:
    marks = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        marks[i, j], marks[j, i] = TAIL, ARROW
    return GraphStructure.from_numpy(marks, kind="dag", labels=labels, node_types=node_types)


def _random_dag(rng: np.random.Generator, n: int, p: float) -> list[tuple[int, int]]:
    order = rng.permutation(n)
    return [
        (int(order[i]), int(order[j])) for i, j in combinations(range(n), 2) if rng.random() < p
    ]


# ---- to_dag -------------------------------------------------------------------------------------


def _chain() -> GraphStructure:
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[[0, 1, 1, 2], [1, 0, 2, 1]] = TAIL  # a -- b -- c
    return GraphStructure.from_numpy(marks, kind="cpdag", labels=("a", "b", "c"))


def _labeled_cpdag(seed: int, n: int = 9) -> GraphStructure:
    truth = _dag(_random_dag(np.random.default_rng(seed), n, 0.4), n)
    labels = tuple(f"x{k}" for k in range(n))
    return GraphStructure.from_numpy(to_cpdag(truth).to_numpy(), kind="cpdag", labels=labels)


def _in_class(dag: GraphStructure, cpdag: GraphStructure) -> bool:
    edges = nx.DiGraph([(s, t) for s, t, _ in dag.oriented_edges(index=True)])
    return (
        dag.kind == "dag"
        and nx.is_directed_acyclic_graph(edges)
        and np.array_equal(dag.to_numpy() != NULL, cpdag.to_numpy() != NULL)  # same skeleton
        and markov_equivalent(dag, cpdag)  # same v-structures
        and to_cpdag(dag).to_numpy().tolist() == cpdag.to_numpy().tolist()
    )


def test_to_dag_refuses_to_choose_by_default():
    with pytest.raises(ValueError) as raised:
        _chain().to_dag()
    assert "a -- b" in str(raised.value) and "b -- c" in str(raised.value)


def test_to_dag_returns_the_unique_dag_without_a_choice():
    dag = _dag([(0, 2), (1, 2)], 3)  # a -> c <- b: its CPDAG has no undirected edge
    assert to_cpdag(dag).to_dag() == dag
    assert andrey.StructureOutput.new(dag).structure.to_dag() == dag


def test_to_dag_keeps_each_directed_edge():
    # The PDAG a -> b <- c, b -> d, d -- e: to_dag directs d -- e without adding a collider.
    a, b, c, d, e = range(5)
    marks = _dag([(a, b), (c, b), (b, d)], 5).to_numpy()
    marks[d, e] = marks[e, d] = TAIL
    dag = GraphStructure.from_numpy(marks, kind="cpdag").to_dag(tie_break="label")
    directed = (marks == TAIL) & (marks.T == ARROW)
    assert np.array_equal(dag.to_numpy()[directed], marks[directed])
    assert markov_equivalent(dag, GraphStructure.from_numpy(marks, kind="cpdag"))


@pytest.mark.parametrize("seed", range(10))
def test_to_dag_lands_in_the_equivalence_class(seed):
    cpdag = _labeled_cpdag(seed)
    assert _in_class(cpdag.to_dag(tie_break="label"), cpdag)
    for draw in range(3):
        assert _in_class(cpdag.to_dag(seed=draw), cpdag)


@pytest.mark.parametrize("seed", range(4))
def test_label_tie_break_ignores_the_column_order(seed):
    cpdag = _labeled_cpdag(seed)
    want = sorted(cpdag.to_dag(tie_break="label").oriented_edges())
    rng = np.random.default_rng(seed)
    for _ in range(5):
        order = rng.permutation(cpdag.n_nodes)
        marks = cpdag.to_numpy()[np.ix_(order, order)]
        labels = tuple(cpdag.labels[k] for k in order)
        shuffled = GraphStructure.from_numpy(marks, kind="cpdag", labels=labels)
        assert sorted(shuffled.to_dag(tie_break="label").oriented_edges()) == want


def test_a_seed_repeats_its_dag_and_seeds_reach_several():
    cpdag = _chain()
    assert cpdag.to_dag(seed=3) == cpdag.to_dag(seed=3)
    dags = {tuple(cpdag.to_dag(seed=seed).oriented_edges()) for seed in range(20)}
    assert len(dags) == 3  # every DAG of the chain's class: a <- b <- c, a -> b -> c, a <- b -> c


@pytest.mark.parametrize("options", [{"tie_break": "label", "seed": 0}, {"tie_break": "random"}])
def test_to_dag_rejects_bad_options(options):
    with pytest.raises(ValueError):
        _chain().to_dag(**options)


@pytest.mark.parametrize("options", [{}, {"tie_break": "label"}])
def test_to_dag_raises_without_an_extension(options):
    # The chordless cycle a -- b -- c -- d -- a: every orientation adds a v-structure or a cycle.
    marks = np.zeros((4, 4), dtype=np.int8)
    for i in range(4):
        marks[i, (i + 1) % 4] = marks[(i + 1) % 4, i] = TAIL
    with pytest.raises(ValueError, match="no consistent DAG extension"):
        GraphStructure.from_numpy(marks, kind="cpdag").to_dag(**options)


def test_to_dag_rejects_a_pag():
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.array([[0, 3], [3, 0]]), kind="pag").to_dag()


# ---- to_mag -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edges", "n", "latent", "selection", "want"),
    [
        # L -> a, L -> b with L hidden: a <-> b.
        ([(2, 0), (2, 1)], 3, [2], [], [[_, A], [A, _]]),
        # a -> s <- b with s selected: conditioning on s joins a and b, both its ancestors.
        ([(0, 2), (1, 2)], 3, [], [2], [[_, T], [T, _]]),
        # a -> b -> s with s selected: a -- b.
        ([(0, 1), (1, 2)], 3, [], [2], [[_, T], [T, _]]),
        # a -> b <- L -> c, b -> c: the inducing path a -> b <- L -> c makes a -> c.
        ([(0, 1), (3, 1), (3, 2), (1, 2)], 4, [3], [], [[_, T, T], [A, _, T], [A, A, _]]),
    ],
)
def test_to_mag_fixed_cases(edges, n, latent, selection, want):
    mag = _dag(edges, n).to_mag(latent=latent, selection=selection)
    assert mag.kind == "mag"
    np.testing.assert_array_equal(mag.to_numpy(), np.array(want, dtype=np.int8))


def test_to_mag_reads_latents_from_node_types_and_keeps_labels():
    types = np.array([0, 0, 1, 0], dtype=np.int8)  # c is hidden
    dag = _dag([(2, 0), (2, 1), (1, 3)], 4, labels=("a", "b", "c", "d"), node_types=types)
    mag = dag.to_mag()
    assert mag.labels == ("a", "b", "d")
    assert sorted(mag.oriented_edges()) == [("a", "b", "bidirected"), ("b", "d", "directed")]
    assert dag.to_mag(latent=[]).n_nodes == 4  # an explicit latent replaces node_types


def _canonical(marks: np.ndarray) -> tuple[nx.DiGraph, set[int]]:
    """The canonical DAG of a MAG and its selection nodes."""
    n = marks.shape[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    selected = set()
    for i, j in combinations(range(n), 2):
        pair = (marks[i, j], marks[j, i])
        if pair == (T, A):
            dag.add_edge(i, j)
        elif pair == (A, T):
            dag.add_edge(j, i)
        elif pair == (A, A):
            dag.add_edges_from([(len(dag), i), (len(dag), j)])
        elif pair == (T, T):
            s = len(dag)
            dag.add_edges_from([(i, s), (j, s)])
            selected.add(s)
    return dag, selected


def _latent_case(seed: int) -> tuple[list[tuple[int, int]], int, list[int], list[int]]:
    """A random 8-node DAG with one or two common causes hidden and maybe a collider selected."""
    rng = np.random.default_rng(seed)
    n = 8
    edges = _random_dag(rng, n, 0.4)
    children = np.bincount([a for a, _ in edges], minlength=n)
    parents = np.bincount([b for _, b in edges], minlength=n)
    nodes = rng.permutation(n).tolist()
    latent = [k for k in nodes if children[k] >= 2][: 1 + seed % 2]
    selection = [k for k in nodes if parents[k] >= 2 and k not in latent][: (seed // 2) % 2]
    return edges, n, latent, selection


# Fixed cases for the comparison with FCI: a hidden confounder; a chordless cycle of selected
# pairs; and a graph where selection makes every PAG edge undirected through R5.
FIXED_CASES = {
    "confounder": ([(0, 1), (4, 1), (4, 2), (3, 2)], 5, [4], []),
    "selected cycle": (
        [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (0, 7)],
        8,
        [],
        [4, 5, 6, 7],
    ),
    "selected r5": (
        [(0, 2), (0, 3), (0, 10), (1, 5), (1, 7), (2, 9), (3, 7), (3, 8), (4, 2), (4, 8)]
        + [(5, 9), (5, 10), (6, 0), (6, 1)],
        11,
        [6],
        [7, 8, 9, 10],
    ),
}
CASES = {**FIXED_CASES, **{f"random {seed}": _latent_case(seed) for seed in range(16)}}


@pytest.mark.parametrize("name", CASES)
def test_mag_is_ancestral_maximal_and_keeps_the_independences(name):
    edges, n, latent, selection = CASES[name]
    mag = _dag(edges, n).to_mag(latent=latent, selection=selection).to_numpy()
    full = nx.DiGraph(edges)
    full.add_nodes_from(range(n))
    observed = [k for k in range(n) if k not in latent and k not in selection]
    m = len(observed)

    # Ancestral: the directed part is acyclic, no <-> joins an ancestor and its descendant, and
    # no arrowhead points into an end of a -- edge.
    directed = nx.DiGraph(
        [(i, j) for i in range(m) for j in range(m) if (mag[i, j], mag[j, i]) == (T, A)]
    )
    directed.add_nodes_from(range(m))
    assert nx.is_directed_acyclic_graph(directed)
    for i, j in combinations(range(m), 2):
        if (mag[i, j], mag[j, i]) == (A, A):
            assert not nx.has_path(directed, i, j) and not nx.has_path(directed, j, i)
        if (mag[i, j], mag[j, i]) == (T, T):
            assert not (mag[i] == A).any() and not (mag[j] == A).any()

    # The marks: a tail at a exactly when a is an ancestor of b or of the selection.
    up = set().union(*(nx.ancestors(full, s) for s in selection)) if selection else set()
    for i, j in combinations(range(m), 2):
        a, b = observed[i], observed[j]
        if mag[i, j] != NULL:
            assert (mag[i, j] == T) == (a in nx.ancestors(full, b) | up)
            assert (mag[j, i] == T) == (b in nx.ancestors(full, a) | up)

    # The independence model over the observed nodes is the DAG's given the selection, and every
    # non-adjacent pair has a separating set (maximal).
    canonical, selected = _canonical(mag)
    for i, j in combinations(range(m), 2):
        rest = [k for k in range(m) if k not in (i, j)]
        separated = False
        for size in range(len(rest) + 1):
            for z in combinations(rest, size):
                in_dag = nx.is_d_separator(
                    full, {observed[i]}, {observed[j]}, {observed[k] for k in z} | set(selection)
                )
                in_mag = nx.is_d_separator(canonical, {i}, {j}, set(z) | selected)
                assert in_dag == in_mag
                separated |= in_mag
        assert separated == (mag[i, j] == NULL)


# ---- to_pag -------------------------------------------------------------------------------------


def test_pag_of_a_hidden_confounder():
    # a -> b <- L -> c <- d with L hidden: a o-> b <-> c <-o d.
    edges, n, latent, _selection = FIXED_CASES["confounder"]
    dag, hidden = _dag(edges, n), latent[0]
    want = np.array([[_, C, _, _], [A, _, A, _], [_, A, _, A], [_, _, C, _]], dtype=np.int8)
    np.testing.assert_array_equal(dag.to_pag(latent=[hidden]).to_numpy(), want)
    np.testing.assert_array_equal(dag.to_mag(latent=[hidden]).to_pag().to_numpy(), want)


def test_pag_of_a_selected_cycle_has_only_tails():
    # Each edge of the chordless cycle a - b - c - d - a has a selected common child; R5 turns the
    # uncovered circle cycle into tails.
    edges, n, _latent, selection = FIXED_CASES["selected cycle"]
    pag = _dag(edges, n).to_pag(selection=selection).to_numpy()
    want = np.zeros((4, 4), dtype=np.int8)
    for i in range(4):
        want[i, (i + 1) % 4] = want[(i + 1) % 4, i] = TAIL
    np.testing.assert_array_equal(pag, want)


def _oracle_fci(edges, n, latent, selection, monkeypatch) -> np.ndarray:
    """FCI's PAG over the observed nodes with a d-separation test on the DAG ``edges``."""
    full = nx.DiGraph(edges)
    full.add_nodes_from(range(n))
    observed = [k for k in range(n) if k not in latent and k not in selection]
    data = np.random.default_rng(0).standard_normal((20, len(observed)))

    def factory(x):
        # FCI may hand the test its columns in another order; find each column's node.
        node = [observed[int(np.flatnonzero((data == col[:, None]).all(axis=0))[0])] for col in x.T]

        def test(i, j, condition_set=()):
            z = {node[k] for k in condition_set} | set(selection)
            return float(nx.is_d_separator(full, {node[i]}, {node[j]}, z))

        return test

    monkeypatch.setitem(ci._CI_REGISTRY, "dsep", factory)
    return andrey.fci(data, indep_test="dsep", alpha=0.5).structure.to_numpy()


@pytest.mark.parametrize("name", CASES)
def test_pag_equals_fci_with_a_dsep_test(name, monkeypatch):
    edges, n, latent, selection = CASES[name]
    dag = _dag(edges, n)
    pag = dag.to_pag(latent=latent, selection=selection)
    np.testing.assert_array_equal(
        pag.to_numpy(), _oracle_fci(edges, n, latent, selection, monkeypatch)
    )
    mag = dag.to_mag(latent=latent, selection=selection)
    assert mag.to_pag() == pag  # through the canonical DAG of the MAG
    resolved = (pag.to_numpy() != CIRCLE) & (pag.to_numpy() != NULL)
    assert np.array_equal(pag.to_numpy()[resolved], mag.to_numpy()[resolved])


def test_pag_cases_reach_every_edge_type():
    kinds = set()
    for edges, n, latent, selection in CASES.values():
        kinds |= {
            t
            for _, _, t in _dag(edges, n)
            .to_pag(latent=latent, selection=selection)
            .oriented_edges()
        }
    assert kinds >= {"directed", "bidirected", "undirected", "partially_directed", "circle"}


def test_pag_matches_marginal_on_latent_truths():
    for seed in range(4):
        ds = andrey.data.sample_scm(d=12, n=16, seed=seed, density=3.0, latents=2)
        assert np.array_equal(
            ds.graph.to_pag().to_numpy(), andrey.data.latent.marginal(ds.graph, "pag").to_numpy()
        )
        assert (ds.graph.node_types == LATENT).sum() == 2


@pytest.mark.parametrize(
    ("marks", "kind"),
    [
        # a -> b -> c with a <-> c: a is an ancestor of c, so not ancestral.
        ([[_, T, A], [A, _, T], [A, A, _]], "mag"),
        ([[_, T, A], [A, _, T], [A, A, _]], "pag"),
        # a <-> b <-> c <-> d with c -> a and b -> d: an inducing path joins a and d, so not
        # maximal.
        ([[_, A, A, _], [A, _, A, T], [T, A, _, A], [_, A, A, _]], "mag"),
        # a circle: a PAG, not a MAG.
        ([[_, C], [A, _]], "pag"),
    ],
)
def test_to_pag_rejects_a_graph_that_is_not_a_mag(marks, kind):
    with pytest.raises(ValueError, match="not a MAG"):
        GraphStructure.from_numpy(np.array(marks), kind=kind).to_pag()


@pytest.mark.parametrize(
    ("latent", "selection", "error"),
    [
        ("c", [], TypeError),
        (["z"], [], ValueError),
        ([5], [], ValueError),
        ([0, "a"], [], ValueError),
        ([0], [0], ValueError),
        ([True], [], TypeError),
    ],
)
def test_node_arguments_fail_fast(latent, selection, error):
    dag = _dag([(0, 1), (1, 2)], 3, labels=("a", "b", "c"))
    with pytest.raises(error):
        dag.to_mag(latent=latent, selection=selection)


def test_a_directed_cycle_is_rejected():
    marks = _dag([(0, 1), (1, 2)], 3).to_numpy()
    marks[2, 0], marks[0, 2] = TAIL, ARROW  # 2 -> 0 closes a cycle
    with pytest.raises(ValueError, match="cycle"):
        GraphStructure.from_numpy(marks, kind="dag").to_pag()


# ---- to_dot -------------------------------------------------------------------------------------

_SHAPES = {"none": TAIL, "normal": ARROW, "odot": CIRCLE}
_ID = r'"((?:[^"\\]|\\.)*)"'


def _parse_dot(text: str) -> tuple[list[str], np.ndarray]:
    """Node names and the endpoint-mark matrix of DOT text in ``to_dot``'s form."""
    unescape = lambda s: re.sub(r"\\(.)", r"\1", s)  # noqa: E731
    names = [unescape(m) for m in re.findall(rf"^  {_ID};$", text, re.M)]
    index = {name: k for k, name in enumerate(names)}
    marks = np.zeros((len(names), len(names)), dtype=np.int8)
    for u, v, attrs in re.findall(rf"^  {_ID} -> {_ID}(?: \[(.*)\])?;$", text, re.M):
        i, j = index[unescape(u)], index[unescape(v)]
        style = dict(item.split("=") for item in attrs.split(", ")) if attrs else {}
        if i == j:
            marks[i, i] = ARROW
        elif style.get("dir") == "none":
            marks[i, j] = marks[j, i] = TAIL
        elif style.get("dir") == "both":
            marks[i, j], marks[j, i] = _SHAPES[style["arrowtail"]], _SHAPES[style["arrowhead"]]
        else:
            marks[j, i] = ARROW
            marks[i, j] = marks[i, j] or TAIL
    return names, marks


def _every_pag_edge() -> GraphStructure:
    pairs = {(0, 1): (T, A), (2, 1): (A, T), (1, 3): (T, T), (3, 4): (A, A), (4, 5): (C, A)}
    pairs |= {(5, 6): (T, C), (6, 7): (C, C), (0, 7): (A, C)}
    marks = np.zeros((8, 8), dtype=np.int8)
    for (i, j), (mi, mj) in pairs.items():
        marks[i, j], marks[j, i] = mi, mj
    labels = ("a", 'say "hi"', "back\\slash", "node", "e", "f", "g", "h")
    return GraphStructure.from_numpy(marks, kind="pag", labels=labels)


def _digraph() -> GraphStructure:
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[0, 1] = marks[1, 0] = ARROW  # 0 -> 1 and 1 -> 0
    marks[1, 2], marks[2, 1] = TAIL, ARROW
    marks[2, 2] = ARROW  # a self-loop
    return GraphStructure.from_numpy(marks, kind="digraph", allow_self_loops=True)


@pytest.mark.parametrize(
    "graph",
    [
        _every_pag_edge(),
        _digraph(),
        to_cpdag(_dag([(0, 1), (1, 2), (3, 2)], 4)),
        GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag"),
    ],
    ids=["pag", "digraph", "cpdag", "no-edges"],
)
def test_to_dot_round_trips_every_mark(graph):
    names, marks = _parse_dot(graph.to_dot())
    want = list(graph.labels) if graph.labels else [str(k) for k in range(graph.n_nodes)]
    assert names == want
    np.testing.assert_array_equal(marks, graph.to_numpy())


# ---- the mag kind -------------------------------------------------------------------------------


def _hidden_cause_mag() -> GraphStructure:
    # a -> b <- L -> c <- d, b -> e with L hidden: a -> b <-> c <- d, b -> e.
    dag = _dag([(0, 1), (5, 1), (5, 2), (3, 2), (1, 4)], 6, labels=tuple("abcdeL"))
    return dag.to_mag(latent=["L"])


def test_marginal_and_to_mag_return_a_mag():
    mag = _hidden_cause_mag()
    assert mag.kind == "mag" and ("b", "c", "bidirected") in mag.oriented_edges()
    ds = andrey.data.sample_scm(d=12, n=16, seed=0, density=3.0, latents=2)
    assert andrey.data.latent.marginal(ds.graph, "mag").kind == "mag"


def test_a_mag_holds_no_circle():
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_numpy(np.array([[0, 3], [2, 0]]), kind="mag").validate()


def test_a_mag_scores_as_a_pag_with_the_same_marks():
    truth = _hidden_cause_mag()
    marks = truth.to_numpy()
    marks[0, 1] = ARROW  # a <-> b instead of a -> b
    marks[3, 2] = marks[2, 3] = NULL  # drop c <- d
    estimate = GraphStructure.from_numpy(marks, kind="mag")
    as_pag = GraphStructure.from_numpy(marks, kind="pag")
    truth_pag = GraphStructure.from_numpy(truth.to_numpy(), kind="pag")
    scores = score(estimate, truth)
    assert scores == score(as_pag, truth_pag) == score(estimate, truth_pag)
    assert scores["family"] == "pag" and scores["shd_endpoint"] > 0
    assert shd(estimate, truth, endpoint_aware=True) == shd(as_pag, truth_pag, endpoint_aware=True)
    digraph = GraphStructure.from_numpy(truth.to_numpy() * 0, kind="digraph")
    with pytest.raises(ValueError, match="digraph"):
        score(digraph, truth)


def test_a_mag_round_trips_through_pcalg_networkx_and_saved_results(tmp_path):
    mag = _hidden_cause_mag()
    amat = mag.to_pcalg()
    assert set(np.unique(amat)) <= {0, 2, 3}  # amat.pag codes, no circle
    assert GraphStructure.from_pcalg(amat, "mag", labels=mag.labels) == mag
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_pcalg(np.array([[0, 1], [2, 0]]), "mag")
    assert GraphStructure.from_networkx(mag.to_networkx(), kind="mag") == mag
    out = andrey.StructureOutput.new(mag)
    for fmt in ("json", "npz"):
        path = tmp_path / f"mag.{fmt}"
        out.save(path, fmt=fmt)
        assert andrey.StructureOutput.load(path).structure == mag


def test_the_command_line_view_names_the_mag_kind():
    view = graph_view(_hidden_cause_mag())
    assert view["kind"] == "mag"
    assert view["edge_types"] == {"directed": 3, "bidirected": 1}


def test_the_method_schema_carries_the_mag_kind():
    output = OutputSpec(structure_type="graph", graph_kind="mag")
    spec = REGISTRY["fci"].model_copy(update={"output": output})
    schema = json.loads(json.dumps(to_json_schema(spec)))
    assert schema["output"]["graph_kind"] == "mag"


def test_an_averaged_mag_reads_as_a_pag():
    mag = _hidden_cause_mag()
    average = BootstrapOutput.from_fits([andrey.StructureOutput.new(mag)] * 2).structure
    assert average.kind == "pag"
    assert average.to_numpy().tolist() == mag.to_numpy().tolist()
