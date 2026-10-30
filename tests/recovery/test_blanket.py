"""Recovery checks for the local-structure calls: IAMB, Inter-IAMB, and MMPC.

Under a d-separation oracle each call returns the true set of each variable, and without a target
each output its own truth: the true CPDAG, skeleton, and moral graph, with background knowledge
that fits the truth as well. On seeded linear-Gaussian data from sparse graphs the sets are close
to the true ones, and on one recorded case each output is the graph bnlearn returns.
"""

from __future__ import annotations

import functools
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.core import ci
from andrey.core.ci import register_indep_test
from andrey.core.orient import dag2cpdag
from andrey.spec import REGISTRY
from tests.recovery.helpers.oracle import DSeparation

METHODS = ["iamb", "inter_iamb", "mmpc"]
KEY = {"iamb": "markov_blanket", "inter_iamb": "markov_blanket", "mmpc": "parents_and_children"}
#: Each method's outputs, from its spec.
OUTPUTS = {m: next(p.choices for p in REGISTRY[m].params if p.name == "output") for m in METHODS}
CELLS = [(m, o) for m in METHODS for o in OUTPUTS[m]]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def _er_sample(d: int, seed: int, n: int = 2000):
    return data.sample_scm(
        graph="erdos_renyi", d=d, n=n, seed=seed, density=2.0, scale="standardize"
    )


def _truth(oracle: DSeparation, method: str, v: int) -> set[int]:
    return oracle.parents_children(v) if method == "mmpc" else oracle.markov_blanket(v)


def _adjacency(out: andrey.StructureOutput) -> np.ndarray:
    return out.structure.to_numpy() != 0


def _true_output(oracle: DSeparation, sample, output: str) -> np.ndarray:
    """The truth an output is judged against: endpoint marks for the CPDAG, else adjacency."""
    if output == "cpdag":
        return dag2cpdag(sample.graph.to_numpy())
    return oracle.moral_graph() if output == "moral" else oracle.skeleton()


def _got_output(out: andrey.StructureOutput, output: str) -> np.ndarray:
    return out.structure.to_numpy() if output == "cpdag" else _adjacency(out)


def _named_edges(out: andrey.StructureOutput) -> set:
    """Each edge by its end labels: a directed edge as ``(tail, head)``, another as a set."""
    return {
        (a, b) if kind == "directed" else frozenset((a, b))
        for a, b, kind in out.structure.oriented_edges()
    }


@pytest.fixture
def oracle_test():
    box: dict[str, DSeparation] = {}
    register_indep_test("dsep", lambda x: box["oracle"])
    yield box
    ci._CI_REGISTRY.pop("dsep", None)


def _collider() -> pd.DataFrame:
    """a -> c <- b, c -> d."""
    rng = np.random.default_rng(0)
    a, b, e, f = rng.standard_normal((4, 1000))
    c = a + b + 0.5 * e
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": c + f})


@pytest.mark.parametrize(
    ("method", "target", "want"),
    [
        ("iamb", "a", {"b", "c"}),
        ("inter_iamb", "a", {"b", "c"}),
        ("mmpc", "a", {"c"}),
        ("iamb", "c", {"a", "b", "d"}),
        ("mmpc", "d", {"c"}),
    ],
)
def test_a_collider_gives_each_set(method, target, want):
    out = getattr(andrey, method)(_collider(), target=target)
    assert out.metadata["target"] == target and set(out.metadata[KEY[method]]) == want
    assert {frozenset(e[:2]) for e in out.structure.oriented_edges()} == {
        frozenset((target, m)) for m in want
    }
    assert all(kind == "undirected" for *_, kind in out.structure.oriented_edges())


@pytest.mark.parametrize("method", METHODS)
def test_a_target_by_position_is_the_target_by_label(method):
    frame = _collider()
    by_label = getattr(andrey, method)(frame, target="c")
    by_position = getattr(andrey, method)(frame, target=2)
    assert by_position.metadata == by_label.metadata
    unnamed = getattr(andrey, method)(frame.to_numpy(), target=2)
    assert unnamed.metadata[KEY[method]] == [
        list(frame).index(m) for m in by_label.metadata[KEY[method]]
    ]


@pytest.mark.parametrize("method", METHODS)
def test_independent_columns_give_empty_sets(method):
    x = np.random.default_rng(0).standard_normal((500, 6))
    assert getattr(andrey, method)(x, target=0).metadata[KEY[method]] == []
    assert getattr(andrey, method)(x).structure.oriented_edges() == []


@pytest.mark.parametrize("method", METHODS)
def test_a_missing_target_raises(method):
    with pytest.raises(ValueError, match="target"):
        getattr(andrey, method)(_collider(), target="z")
    with pytest.raises(ValueError, match="target"):
        getattr(andrey, method)(_collider(), target=4)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", METHODS)
def test_each_set_is_the_true_set_under_an_oracle(method, seed, oracle_test):
    sample = _er_sample(15, seed)
    oracle = oracle_test["oracle"] = DSeparation(sample.graph)
    for v in range(15):
        out = getattr(andrey, method)(sample.data, target=v, indep_test="dsep", alpha=0.5)
        assert set(out.metadata[KEY[method]]) == _truth(oracle, method, v), v


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("method", "output"), CELLS)
def test_each_output_is_its_own_truth_under_an_oracle(method, output, seed, oracle_test):
    """The CPDAG is the true CPDAG, the skeleton the true skeleton, the moral graph the true moral
    graph, and the kind says which: ``cpdag``, else ``undirected``."""
    sample = _er_sample(15, seed)
    oracle = oracle_test["oracle"] = DSeparation(sample.graph)
    out = getattr(andrey, method)(sample.data, output=output, indep_test="dsep", alpha=0.5)
    np.testing.assert_array_equal(_got_output(out, output), _true_output(oracle, sample, output))
    assert out.structure.kind == ("cpdag" if output == "cpdag" else "undirected")


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("method", "output"), CELLS)
def test_each_output_is_close_to_its_truth_on_sparse_graphs(method, output, seed):
    """On 20 variables with 2,000 rows, an undirected output has at most a third as many wrong
    pairs as its truth has pairs. A CPDAG pair is wrong when its edge or its marks differ, and one
    wrong collider misdirects the edges Meek's rules orient from it, so the CPDAG has at most half
    as many."""
    sample = _er_sample(20, seed)
    want = _true_output(DSeparation(sample.graph), sample, output)
    got = _got_output(getattr(andrey, method)(sample.data, output=output), output)
    wrong = (got != want) | (got.T != want.T)
    share = 1 / 2 if output == "cpdag" else 1 / 3
    assert np.triu(wrong, 1).sum() <= np.triu(want != 0, 1).sum() * share


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", ["iamb", "inter_iamb"])
def test_the_skeleton_is_a_subgraph_of_the_moral_graph(method, seed):
    sample = _er_sample(20, seed)
    fit = functools.partial(getattr(andrey, method), sample.data)
    skeleton, moral = _adjacency(fit(output="skeleton")), _adjacency(fit(output="moral"))
    assert not (skeleton & ~moral).any()
    np.testing.assert_array_equal(_adjacency(fit()), skeleton)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(("method", "output"), CELLS)
def test_a_column_shuffle_leaves_every_set(method, output, seed):
    frame = pd.DataFrame(_er_sample(20, 1).data, columns=[f"x{j}" for j in range(20)])
    order = list(np.random.default_rng(seed).permutation(20))
    shuffled = frame.iloc[:, order]
    for target in ("x3", "x10"):
        base = getattr(andrey, method)(frame, target=target, output=output).metadata
        assert getattr(andrey, method)(shuffled, target=target, output=output).metadata == base
    edges = _named_edges(getattr(andrey, method)(frame, output=output))
    assert _named_edges(getattr(andrey, method)(shuffled, output=output)) == edges


@pytest.mark.parametrize("method", METHODS)
def test_the_cpdag_orients_a_collider_and_its_consequence(method):
    """a -> c <- b, c -> d: the collider at c, then c -> d by Meek's first rule."""
    edges = set(getattr(andrey, method)(_collider()).structure.oriented_edges())
    assert edges == {("a", "c", "directed"), ("b", "c", "directed"), ("c", "d", "directed")}


def test_the_moral_graph_joins_a_childs_parents_and_the_skeleton_does_not():
    """The blankets' graph joins a and b, the parents of c; the skeleton and MMPC's do not."""
    frame = _collider()
    true_edges = {frozenset("ac"), frozenset("bc"), frozenset("cd")}
    for out in (andrey.iamb(frame, output="moral"), andrey.inter_iamb(frame, output="moral")):
        assert frozenset("ab") in {frozenset(e[:2]) for e in out.structure.oriented_edges()}
    for method in METHODS:
        for output in OUTPUTS[method]:
            out = getattr(andrey, method)(frame, output=output)
            edges = {frozenset(e[:2]) for e in out.structure.oriented_edges()}
            assert edges == (true_edges | {frozenset("ab")} if output == "moral" else true_edges)


def _two_colliders() -> pd.DataFrame:
    """a -> c <- b, c -> d, d -> f <- e."""
    rng = np.random.default_rng(1)
    a, b, e, u, v, w = rng.standard_normal((6, 2000))
    c = a + b + 0.5 * u
    d = 0.8 * c + v
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": d - e + 0.5 * w})


#: bnlearn 5.2.1's graphs on :func:`_two_colliders` with ``test="zf"`` and ``alpha=0.05``: each
#: variable's ``learn.mb`` joined by the AND rule, ``undirected=TRUE``, and the default.
BNLEARN_RECORDED = {
    "moral": {"a - b", "a - c", "b - c", "c - d", "d - e", "d - f", "e - f"},
    "skeleton": {"a - c", "b - c", "c - d", "d - f", "e - f"},
    "cpdag": {"a -> c", "b -> c", "c -> d", "d -> f", "e -> f"},
}


@pytest.mark.parametrize(("method", "output"), CELLS)
def test_each_output_matches_bnlearn_on_a_recorded_case(method, output):
    out = getattr(andrey, method)(_two_colliders(), output=output)
    edges = {
        f"{a} -> {b}" if kind == "directed" else f"{min(a, b)} - {max(a, b)}"
        for a, b, kind in out.structure.oriented_edges()
    }
    assert edges == BNLEARN_RECORDED[output]


def test_mmpc_has_no_moral_graph():
    with pytest.raises(ValueError, match="output"):
        andrey.mmpc(_collider(), output="moral")


@pytest.mark.parametrize("method", METHODS)
def test_a_target_gives_its_set_whatever_the_output(method):
    frame = _collider()
    base = getattr(andrey, method)(frame, target="c")
    for output in OUTPUTS[method]:
        out = getattr(andrey, method)(frame, target="c", output=output)
        assert out.metadata == base.metadata
        assert out.structure.kind == "undirected"


# ---- background knowledge ------------------------------------------------------------------------

# On _collider: a - d is not a true edge and c - d is; the tiers constrain orientation only.
KNOWLEDGE = andrey.Knowledge(
    required=[("a", "d")], forbidden=[("c", "d"), ("d", "c")], tiers=[["a", "b"], ["c", "d"]]
)


def _edges(out: andrey.StructureOutput) -> dict[frozenset, tuple[str, str, str]]:
    return {frozenset(e[:2]): e for e in out.structure.oriented_edges()}


@pytest.mark.parametrize("method", METHODS)
def test_a_required_pair_joins_and_a_pair_kept_apart_leaves_with_and_without_a_target(method):
    plain = _edges(getattr(andrey, method)(_collider()))
    assert frozenset("cd") in plain and frozenset("ad") not in plain
    for output in OUTPUTS[method]:
        edges = _edges(getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE, output=output))
        assert frozenset("ad") in edges and frozenset("cd") not in edges
    out = getattr(andrey, method)(_collider(), target="d", knowledge=KNOWLEDGE)
    assert out.metadata[KEY[method]] == ["a"]
    assert out.structure.oriented_edges() == [("a", "d", "undirected")]


@pytest.mark.parametrize("method", METHODS)
def test_tiers_direct_the_cpdag_and_leave_the_undirected_outputs(method):
    """The tiers put a and b before c and d: the CPDAG directs every edge between the tiers, and the
    skeleton stays undirected."""
    edges = _edges(getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE))
    assert edges[frozenset("ad")] == ("a", "d", "directed")
    assert edges[frozenset("ac")] == ("a", "c", "directed")
    out = getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE, output="skeleton")
    assert {kind for *_, kind in _edges(out).values()} == {"undirected"}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", METHODS)
def test_true_knowledge_leaves_the_true_sets_under_an_oracle(method, seed, oracle_test):
    """Knowledge that fits the truth: required true edges, the tiers of a causal order, and every
    pair of a child's parents that are not adjacent forbidden both ways. MMPC returns the true
    sets; IAMB and Inter-IAMB the true blankets less those pairs, which they still condition on."""
    sample = _er_sample(15, seed)
    oracle = oracle_test["oracle"] = DSeparation(sample.graph)
    edges = [(i, j) for i, j, _ in sample.graph.oriented_edges()]
    order = list(nx.topological_sort(oracle.graph))
    apart = np.argwhere(np.triu(oracle.moral_graph() & ~oracle.skeleton()))
    assert len(apart), "the graph has no unshielded collider"
    knowledge = andrey.Knowledge(
        required=edges[:3],
        forbidden=[(int(i), int(j)) for i, j in apart] + [(int(j), int(i)) for i, j in apart],
        tiers=[order[:5], order[5:]],
    )
    kept = oracle.skeleton()
    for v in range(15):
        out = getattr(andrey, method)(
            sample.data, target=v, indep_test="dsep", alpha=0.5, knowledge=knowledge
        )
        want = {u for u in _truth(oracle, method, v) if kept[v, u]}
        assert set(out.metadata[KEY[method]]) == want, v
    for output in OUTPUTS[method]:
        out = getattr(andrey, method)(
            sample.data, indep_test="dsep", alpha=0.5, knowledge=knowledge, output=output
        )
        np.testing.assert_array_equal(_adjacency(out), kept)
