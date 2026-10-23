"""Recovery checks for the local-structure calls: IAMB, Inter-IAMB, and MMPC.

Under a d-separation oracle each call returns the true set of each variable, and without a target
the moral graph or the skeleton. On seeded linear-Gaussian data from sparse graphs the sets are
close to the true ones.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.core import ci
from andrey.core.ci import register_indep_test
from tests.recovery.helpers.oracle import DSeparation

METHODS = ["iamb", "inter_iamb", "mmpc"]
KEY = {"iamb": "markov_blanket", "inter_iamb": "markov_blanket", "mmpc": "parents_and_children"}


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
    graph = getattr(andrey, method)(sample.data, indep_test="dsep", alpha=0.5)
    want = oracle.skeleton() if method == "mmpc" else oracle.moral_graph()
    np.testing.assert_array_equal(_adjacency(graph), want)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("method", METHODS)
def test_sets_are_close_to_the_true_sets_on_sparse_graphs(method, seed):
    """On 20 variables with 2,000 rows, the graph of all sets has at most a third as many wrong
    pairs as the true graph has pairs."""
    sample = _er_sample(20, seed)
    oracle = DSeparation(sample.graph)
    want = oracle.skeleton() if method == "mmpc" else oracle.moral_graph()
    got = _adjacency(getattr(andrey, method)(sample.data))
    assert np.triu(got != want).sum() <= np.triu(want).sum() / 3


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method", METHODS)
def test_a_column_shuffle_leaves_every_set(method, seed):
    frame = pd.DataFrame(_er_sample(20, 1).data, columns=[f"x{j}" for j in range(20)])
    order = list(np.random.default_rng(seed).permutation(20))
    shuffled = frame.iloc[:, order]
    for target in ("x3", "x10"):
        base = getattr(andrey, method)(frame, target=target).metadata
        assert getattr(andrey, method)(shuffled, target=target).metadata == base
    edges = {frozenset(e[:2]) for e in getattr(andrey, method)(frame).structure.oriented_edges()}
    moved = getattr(andrey, method)(shuffled).structure.oriented_edges()
    assert {frozenset(e[:2]) for e in moved} == edges


def test_without_a_target_the_moral_graph_joins_a_childs_parents():
    """The blankets' graph joins a and b, the parents of c; MMPC's skeleton does not."""
    frame = _collider()
    spouses = frozenset(("a", "b"))
    for out in (andrey.iamb(frame), andrey.inter_iamb(frame)):
        assert spouses in {frozenset(e[:2]) for e in out.structure.oriented_edges()}
    skeleton = {frozenset(e[:2]) for e in andrey.mmpc(frame).structure.oriented_edges()}
    assert skeleton == {frozenset(("a", "c")), frozenset(("b", "c")), frozenset(("c", "d"))}
