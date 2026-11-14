"""Recovery checks for the local-structure calls: IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC,
and HITON-MB.

Under a d-separation oracle each call returns the true set of each variable, and without a target
each structure its own truth: the true CPDAG, skeleton, and moral graph, with background knowledge
that fits the truth as well. On seeded linear-Gaussian data from sparse graphs the sets are close
to the true ones, and on one recorded case each structure is the graph bnlearn returns. The sets of
GS and Fast-IAMB, the parents and children of HITON-PC, and the blankets of HITON-MB beat the empty
answer on continuous and categorical data, and a fit repeats. HITON-MB's parents and children are
HITON-PC's.
"""

from __future__ import annotations

import functools
import graphlib

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

METHODS = ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc", "hiton_mb", "mmmb", "pcmb"]
#: The learners of parents and children; the others learn Markov blankets.
PC_LEARNERS = ("mmpc", "hiton_pc")
#: The learners that add spouses to a parents-and-children learner's sets, and that learner.
SPOUSE_LEARNERS = {"hiton_mb": "hiton_pc", "mmmb": "mmpc", "pcmb": "get_pcd"}
KEY = {m: "parents_and_children" if m in PC_LEARNERS else "markov_blanket" for m in METHODS}
#: The settings under which the oracle checks run each method: the setting of its correctness
#: theorem, without a conditioning-set cap.
ORACLE = {m: {"indep_test": "dsep", "alpha": 0.5} for m in METHODS}
ORACLE["hiton_pc"]["max_depth"] = ORACLE["hiton_mb"]["max_depth"] = None
#: Each method's outputs, from its spec.
STRUCTURES = {
    m: next(p.choices for p in REGISTRY[m].params if p.name == "structure") for m in METHODS
}
CELLS = [(m, o) for m in METHODS for o in STRUCTURES[m]]


def _er_sample(d: int, seed: int, n: int = 2000):
    return data.sample_scm(
        graph="erdos_renyi", d=d, n=n, seed=seed, density=2.0, scale="standardize"
    )


def _truth(oracle: DSeparation, method: str, v: int) -> set[int]:
    return oracle.parents_children(v) if method in PC_LEARNERS else oracle.markov_blanket(v)


def _adjacency(out: andrey.StructureOutput) -> np.ndarray:
    return out.structure.to_numpy() != 0


def _true_structure(oracle: DSeparation, sample, structure: str) -> np.ndarray:
    """The truth a structure is judged against: endpoint marks for the CPDAG, else adjacency."""
    if structure == "cpdag":
        return dag2cpdag(sample.graph.to_numpy())
    return oracle.moral_graph() if structure == "moral" else oracle.skeleton()


def _got_structure(out: andrey.StructureOutput, structure: str) -> np.ndarray:
    return out.structure.to_numpy() if structure == "cpdag" else _adjacency(out)


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
        ("gs", "a", {"b", "c"}),
        ("gs", "c", {"a", "b", "d"}),
        ("gs", "d", {"c"}),
        ("fast_iamb", "a", {"b", "c"}),
        ("fast_iamb", "c", {"a", "b", "d"}),
        ("fast_iamb", "d", {"c"}),
        ("mmpc", "d", {"c"}),
        ("hiton_pc", "a", {"c"}),
        ("hiton_pc", "c", {"a", "b", "d"}),
        ("hiton_pc", "d", {"c"}),
        ("hiton_mb", "a", {"b", "c"}),
        ("hiton_mb", "c", {"a", "b", "d"}),
        ("hiton_mb", "d", {"c"}),
        ("mmmb", "a", {"b", "c"}),
        ("mmmb", "c", {"a", "b", "d"}),
        ("mmmb", "d", {"c"}),
        ("pcmb", "a", {"b", "c"}),
        ("pcmb", "c", {"a", "b", "d"}),
        ("pcmb", "d", {"c"}),
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
        out = getattr(andrey, method)(sample.data, target=v, **ORACLE[method])
        assert set(out.metadata[KEY[method]]) == _truth(oracle, method, v), v
        pc = out.metadata.get("parents_and_children", oracle.parents_children(v))
        assert set(pc) == oracle.parents_children(v), v


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("method", "structure"), CELLS)
def test_each_structure_is_its_own_truth_under_an_oracle(method, structure, seed, oracle_test):
    """The CPDAG is the true CPDAG, the skeleton the true skeleton, the moral graph the true moral
    graph, and the kind says which: ``cpdag``, else ``undirected``."""
    sample = _er_sample(15, seed)
    oracle = oracle_test["oracle"] = DSeparation(sample.graph)
    out = getattr(andrey, method)(sample.data, structure=structure, **ORACLE[method])
    np.testing.assert_array_equal(
        _got_structure(out, structure), _true_structure(oracle, sample, structure)
    )
    assert out.structure.kind == ("cpdag" if structure == "cpdag" else "undirected")


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("method", "structure"), CELLS)
def test_each_structure_is_close_to_its_truth_on_sparse_graphs(method, structure, seed):
    """On 20 variables with 2,000 rows, at the level 0.05 for every learner, an undirected structure
    has at most a third as many wrong pairs as its truth has pairs. A CPDAG pair is wrong when its
    edge or its marks differ, and one wrong collider misdirects the edges Meek's rules orient from
    it, so the CPDAG has at most half as many. HITON-MB, MMMB, and PCMB reach a spouse only through
    a child their parents-and-children search found, so an edge it misses also drops the spouses
    behind it from both blankets, and their moral graphs have at most half as many."""
    sample = _er_sample(20, seed)
    want = _true_structure(DSeparation(sample.graph), sample, structure)
    fit = getattr(andrey, method)(sample.data, structure=structure, alpha=0.05)
    got = _got_structure(fit, structure)
    wrong = (got != want) | (got.T != want.T)
    half = structure == "cpdag" or (method in SPOUSE_LEARNERS and structure == "moral")
    share = 1 / 2 if half else 1 / 3
    assert np.triu(wrong, 1).sum() <= np.triu(want != 0, 1).sum() * share


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "gs", "fast_iamb"])
def test_the_skeleton_is_a_subgraph_of_the_moral_graph(method, seed):
    sample = _er_sample(20, seed)
    fit = functools.partial(getattr(andrey, method), sample.data)
    skeleton, moral = _adjacency(fit(structure="skeleton")), _adjacency(fit(structure="moral"))
    assert not (skeleton & ~moral).any()
    np.testing.assert_array_equal(_adjacency(fit()), skeleton)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(("method", "structure"), CELLS)
def test_a_column_shuffle_leaves_every_set(method, structure, seed):
    frame = pd.DataFrame(_er_sample(20, 1).data, columns=[f"x{j}" for j in range(20)])
    order = list(np.random.default_rng(seed).permutation(20))
    shuffled = frame.iloc[:, order]
    for target in ("x3", "x10"):
        base = getattr(andrey, method)(frame, target=target, structure=structure).metadata
        assert (
            getattr(andrey, method)(shuffled, target=target, structure=structure).metadata == base
        )
    edges = _named_edges(getattr(andrey, method)(frame, structure=structure))
    assert _named_edges(getattr(andrey, method)(shuffled, structure=structure)) == edges


@pytest.mark.parametrize("method", METHODS)
def test_the_cpdag_orients_a_collider_and_its_consequence(method):
    """a -> c <- b, c -> d: the collider at c, then c -> d by Meek's first rule."""
    out = getattr(andrey, method)(_collider(), structure="cpdag")
    edges = set(out.structure.oriented_edges())
    assert edges == {("a", "c", "directed"), ("b", "c", "directed"), ("c", "d", "directed")}


def test_the_moral_graph_joins_a_childs_parents_and_the_skeleton_does_not():
    """The blankets' graph joins a and b, the parents of c; the skeleton and MMPC's do not."""
    frame = _collider()
    true_edges = {frozenset("ac"), frozenset("bc"), frozenset("cd")}
    for out in (
        andrey.iamb(frame, structure="moral"),
        andrey.inter_iamb(frame, structure="moral"),
        andrey.gs(frame, structure="moral"),
        andrey.fast_iamb(frame, structure="moral"),
    ):
        assert frozenset("ab") in {frozenset(e[:2]) for e in out.structure.oriented_edges()}
    for method in METHODS:
        for structure in STRUCTURES[method]:
            out = getattr(andrey, method)(frame, structure=structure)
            edges = {frozenset(e[:2]) for e in out.structure.oriented_edges()}
            assert edges == (true_edges | {frozenset("ab")} if structure == "moral" else true_edges)


def _two_colliders() -> pd.DataFrame:
    """a -> c <- b, c -> d, d -> f <- e."""
    rng = np.random.default_rng(1)
    a, b, e, u, v, w = rng.standard_normal((6, 2000))
    c = a + b + 0.5 * u
    d = 0.8 * c + v
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": d - e + 0.5 * w})


#: bnlearn 5.2.1's graphs on :func:`_two_colliders` with ``test="zf"`` and ``alpha=0.05``: each
#: variable's ``learn.mb`` joined by the AND rule, ``undirected=TRUE``, and the default; ``gs``
#: and ``fast.iamb`` give the same graphs, and ``si.hiton.pc`` the skeleton (its default,
#: ``undirected=TRUE``) and the CPDAG (``undirected=FALSE``).
BNLEARN_RECORDED = {
    "moral": {"a - b", "a - c", "b - c", "c - d", "d - e", "d - f", "e - f"},
    "skeleton": {"a - c", "b - c", "c - d", "d - f", "e - f"},
    "cpdag": {"a -> c", "b -> c", "c -> d", "d -> f", "e -> f"},
}


@pytest.mark.parametrize(("method", "structure"), CELLS)
def test_each_structure_matches_bnlearn_on_a_recorded_case(method, structure):
    out = getattr(andrey, method)(_two_colliders(), structure=structure)
    edges = {
        f"{a} -> {b}" if kind == "directed" else f"{min(a, b)} - {max(a, b)}"
        for a, b, kind in out.structure.oriented_edges()
    }
    assert edges == BNLEARN_RECORDED[structure]


#: The graph each of bnlearn 5.2.1's learners returns by default: ``undirected = FALSE`` for
#: ``iamb`` and ``inter.iamb``, ``TRUE`` for ``mmpc``.
BNLEARN_DEFAULT = {m: "skeleton" if m in PC_LEARNERS else "cpdag" for m in METHODS}


@pytest.mark.parametrize("method", METHODS)
def test_a_call_without_structure_returns_the_graph_bnlearn_returns_by_default(method):
    out = getattr(andrey, method)(_two_colliders())
    edges = {
        f"{a} -> {b}" if kind == "directed" else f"{min(a, b)} - {max(a, b)}"
        for a, b, kind in out.structure.oriented_edges()
    }
    assert edges == BNLEARN_RECORDED[BNLEARN_DEFAULT[method]]
    assert out.structure.kind == REGISTRY[method].output.graph_kind


@pytest.mark.parametrize("method", PC_LEARNERS)
def test_a_parents_and_children_learner_has_no_moral_graph(method):
    with pytest.raises(ValueError, match="structure"):
        getattr(andrey, method)(_collider(), structure="moral")


@pytest.mark.parametrize("method", METHODS)
def test_a_target_gives_its_set_whatever_the_structure(method):
    frame = _collider()
    base = getattr(andrey, method)(frame, target="c")
    for structure in STRUCTURES[method]:
        out = getattr(andrey, method)(frame, target="c", structure=structure)
        assert out.metadata == base.metadata
        assert out.structure.kind == "undirected"


def _sample_of(kind: str):
    """Linear-Gaussian data on 20 variables, or categorical data on 10, with their settings."""
    if kind == "gaussian":
        return _er_sample(20, 0), {}
    return data.sample_discrete(d=10, n=3000, seed=1, density=2.0), {"indep_test": "chisq"}


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_gs_sets_beat_the_empty_answer(kind):
    """Summed over every target, the members GS gets wrong, missing or extra, are at most half the
    true blanket members, which is what the empty answer gets wrong."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        got = set(andrey.gs(sample.data, target=v, **settings).metadata["markov_blanket"])
        want = oracle.markov_blanket(v)
        wrong, members = wrong + len(got ^ want), members + len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_gs_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.gs(sample.data, target=target, **settings)
        again = andrey.gs(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_fast_iamb_sets_beat_the_empty_answer(kind):
    """As for GS: summed over every target, at most half as many wrong members as the true
    blankets hold."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        got = set(andrey.fast_iamb(sample.data, target=v, **settings).metadata["markov_blanket"])
        want = oracle.markov_blanket(v)
        wrong, members = wrong + len(got ^ want), members + len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_fast_iamb_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.fast_iamb(sample.data, target=target, **settings)
        again = andrey.fast_iamb(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_hiton_pc_sets_beat_the_empty_answer(kind):
    """At its defaults, summed over every target, at most half as many wrong members as the true
    parents and children hold, which is what the empty answer gets wrong."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        out = andrey.hiton_pc(sample.data, target=v, **settings)
        want = oracle.parents_children(v)
        wrong += len(set(out.metadata["parents_and_children"]) ^ want)
        members += len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_hiton_pc_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.hiton_pc(sample.data, target=target, **settings)
        again = andrey.hiton_pc(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_hiton_mb_sets_beat_the_empty_answer(kind):
    """At its defaults, summed over every target, at most half as many wrong members as the true
    blankets hold, which is what the empty answer gets wrong."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        out = andrey.hiton_mb(sample.data, target=v, **settings)
        want = oracle.markov_blanket(v)
        wrong += len(set(out.metadata["markov_blanket"]) ^ want)
        members += len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_hiton_mb_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.hiton_mb(sample.data, target=target, **settings)
        again = andrey.hiton_mb(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_hiton_mb_parents_and_children_are_hiton_pcs(kind):
    """HITON-MB's parents and children are HITON-PC's at the same settings and open its blanket,
    and its skeleton is HITON-PC's."""
    sample, settings = _sample_of(kind)
    for v in range(sample.data.shape[1]):
        got = andrey.hiton_mb(sample.data, target=v, **settings).metadata
        pc = andrey.hiton_pc(sample.data, target=v, **settings).metadata["parents_and_children"]
        assert got["parents_and_children"] == pc == got["markov_blanket"][: len(pc)], v
    skeleton = andrey.hiton_mb(sample.data, structure="skeleton", **settings)
    assert _named_edges(skeleton) == _named_edges(andrey.hiton_pc(sample.data, **settings))


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_mmmb_sets_beat_the_empty_answer(kind):
    """At its defaults, summed over every target, at most half as many wrong members as the true
    blankets hold, which is what the empty answer gets wrong."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        out = andrey.mmmb(sample.data, target=v, **settings)
        want = oracle.markov_blanket(v)
        wrong += len(set(out.metadata["markov_blanket"]) ^ want)
        members += len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_mmmb_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.mmmb(sample.data, target=target, **settings)
        again = andrey.mmmb(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_mmmb_parents_and_children_are_mmpcs(kind):
    """MMMB's parents and children are MMPC's at the same settings and degrees of freedom and open
    its blanket, and its skeleton is MMPC's."""
    sample, settings = _sample_of(kind)
    same = settings | ({"degrees_of_freedom": "nonempty"} if kind == "categorical" else {})
    for v in range(sample.data.shape[1]):
        got = andrey.mmmb(sample.data, target=v, **settings).metadata
        pc = andrey.mmpc(sample.data, target=v, **same).metadata["parents_and_children"]
        assert got["parents_and_children"] == pc == got["markov_blanket"][: len(pc)], v
    skeleton = andrey.mmmb(sample.data, structure="skeleton", **settings)
    assert _named_edges(skeleton) == _named_edges(andrey.mmpc(sample.data, **same))


def test_mmmb_finds_the_blanket_on_the_graph_the_2003_form_gets_wrong():
    """Pena et al. (2007), DAG (b), P -> Q -> T -> S <- R <- P, on 5,000 rows of linear-Gaussian
    data: T's blanket is {Q, S, R}. P is dependent on T given {Q, S}, so the 2003 MMMB, which tests
    P with every parent and child of T, admits it; MMMB tests P only with Q, its neighbor."""
    rng = np.random.default_rng(0)
    p = rng.standard_normal(5000)
    q = p + rng.standard_normal(5000)
    t = q + rng.standard_normal(5000)
    r = p + rng.standard_normal(5000)
    s = t + r + rng.standard_normal(5000)
    frame = pd.DataFrame({"P": p, "Q": q, "T": t, "S": s, "R": r})
    out = andrey.mmmb(frame, target="T")
    assert set(out.metadata["parents_and_children"]) == {"Q", "S"}
    assert set(out.metadata["markov_blanket"]) == {"Q", "S", "R"}


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_pcmb_sets_beat_the_empty_answer(kind):
    """At its defaults, summed over every target, at most half as many wrong members as the true
    blankets hold, which is what the empty answer gets wrong."""
    sample, settings = _sample_of(kind)
    oracle = DSeparation(sample.graph)
    wrong = members = 0
    for v in range(sample.data.shape[1]):
        out = andrey.pcmb(sample.data, target=v, **settings)
        want = oracle.markov_blanket(v)
        wrong += len(set(out.metadata["markov_blanket"]) ^ want)
        members += len(want)
    assert members and wrong <= members / 2


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_a_pcmb_fit_repeats(kind):
    sample, settings = _sample_of(kind)
    for target in (None, 3):
        first = andrey.pcmb(sample.data, target=target, **settings)
        again = andrey.pcmb(sample.data, target=target, **settings)
        assert first.structure.oriented_edges() == again.structure.oriented_edges()
        assert first.metadata == again.metadata


@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_pcmb_skeleton_joins_each_variable_to_its_parents_and_children(kind):
    """PCMB's parents and children open its blanket, each variable's are the variables the
    skeleton joins it to, and every edge of the skeleton is in the moral graph."""
    sample, settings = _sample_of(kind)
    d = sample.data.shape[1]
    skeleton = _adjacency(andrey.pcmb(sample.data, structure="skeleton", **settings))
    moral = _adjacency(andrey.pcmb(sample.data, structure="moral", **settings))
    for v in range(d):
        got = andrey.pcmb(sample.data, target=v, **settings).metadata
        pc = got["parents_and_children"]
        assert pc == got["markov_blanket"][: len(pc)], v
        assert set(pc) == set(np.flatnonzero(skeleton[v]).tolist()), v
    assert not (skeleton & ~moral).any()


def _pena(edges: str, seed: int = 0, n: int = 5000) -> pd.DataFrame:
    """Linear-Gaussian data on a network of Pena et al. (2007) whose edges are ``"P>Q Q>T ..."``:
    each node is 0.8 times the sum of its parents plus standard Gaussian noise. With weights of 1,
    DAG (a) is unfaithful: P and Q are independent given S."""
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    pairs = [tuple(e.split(">")) for e in edges.split()]
    order = graphlib.TopologicalSorter({b: {a for a, c in pairs if c == b} for _, b in pairs})
    for node in order.static_order():
        parents = [columns[a] for a, b in pairs if b == node]
        columns[node] = 0.8 * sum(parents, np.zeros(n)) + rng.standard_normal(n)
    return pd.DataFrame(columns)


@pytest.mark.parametrize(
    ("edges", "pc", "mb"),
    [
        ("T>Q P>Q P>R Q>S R>S", {"Q"}, {"Q", "P"}),
        ("P>Q Q>T T>S P>R R>S", {"Q", "S"}, {"Q", "S", "R"}),
    ],
    ids=["dag_a", "dag_b"],
)
def test_pcmb_finds_the_blanket_on_the_graphs_of_its_paper(edges, pc, mb):
    """Pena et al. (2007), DAGs (a) and (b), on 5,000 rows of linear-Gaussian data: on (a), GetPCD
    keeps T's grandchild S, which the symmetry check drops; on (b), P is dependent on T given
    {Q, S}, and PCMB tests it only with Q, its neighbor in PC(T)."""
    out = andrey.pcmb(_pena(edges), target="T")
    assert set(out.metadata["parents_and_children"]) == pc
    assert set(out.metadata["markov_blanket"]) == mb


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
    for structure in STRUCTURES[method]:
        edges = _edges(
            getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE, structure=structure)
        )
        assert frozenset("ad") in edges and frozenset("cd") not in edges
    out = getattr(andrey, method)(_collider(), target="d", knowledge=KNOWLEDGE)
    assert out.metadata[KEY[method]] == ["a"]
    assert out.structure.oriented_edges() == [("a", "d", "undirected")]


@pytest.mark.parametrize("method", METHODS)
def test_tiers_direct_the_cpdag_and_leave_the_undirected_structures(method):
    """The tiers put a and b before c and d: the CPDAG directs every edge between the tiers, and the
    skeleton stays undirected."""
    edges = _edges(getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE, structure="cpdag"))
    assert edges[frozenset("ad")] == ("a", "d", "directed")
    assert edges[frozenset("ac")] == ("a", "c", "directed")
    out = getattr(andrey, method)(_collider(), knowledge=KNOWLEDGE, structure="skeleton")
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
        out = getattr(andrey, method)(sample.data, target=v, knowledge=knowledge, **ORACLE[method])
        want = {u for u in _truth(oracle, method, v) if kept[v, u]}
        assert set(out.metadata[KEY[method]]) == want, v
    for structure in STRUCTURES[method]:
        out = getattr(andrey, method)(
            sample.data, knowledge=knowledge, structure=structure, **ORACLE[method]
        )
        np.testing.assert_array_equal(_adjacency(out), kept)
