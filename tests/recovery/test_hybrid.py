"""Recovery checks for the restrict-maximize searches: MMHC and RSMAX2.

A restrict phase of conditional-independence tests keeps a set of candidate pairs, and a score
search joins only those pairs. Under a d-separation oracle the candidates are the true skeleton;
on seeded linear-Gaussian data from sparse graphs, the result is close to the true equivalence
class.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core.ci import register_indep_test
from andrey.metrics import markov_equivalent, shd, to_cpdag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.oracle import DSeparation
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def _er_sample(d: int, seed: int, n: int = 2000):
    """A standardized linear-Gaussian sample on an Erdos-Renyi DAG of average degree 2."""
    return data.sample_scm(
        graph="erdos_renyi", d=d, n=n, seed=seed, density=2.0, scale="standardize"
    )


def _adjacent(out: andrey.StructureOutput) -> set[frozenset[int]]:
    return {frozenset((i, j)) for i, j, _ in out.structure.oriented_edges(index=True)}


def _edges(out: andrey.StructureOutput, order: np.ndarray | None = None) -> set[tuple]:
    """The edges of ``out``, with column ``k`` read as ``order[k]``; an undirected edge as a set."""
    at = (lambda k: k) if order is None else (lambda k: int(order[k]))
    return {
        (at(i), at(j), kind) if kind == "directed" else (frozenset((at(i), at(j))), kind)
        for i, j, kind in out.structure.oriented_edges(index=True)
    }


def test_mmhc_reaches_the_score_optimum_on_gauss_5v():
    """MMHC returns the class exact search finds, and the graph and score match the recorded
    output."""
    x = datagen.generate("gauss_5v")
    out = andrey.mmhc(x)
    assert out.structure.kind == "cpdag"
    assert markov_equivalent(out.structure, to_cpdag(andrey.exact_search(x).structure))
    assert out.metadata["restrict"] == "mmpc" and out.metadata["maximize"] == "hc"
    assert_recorded_output(out, "MMHC", case="gauss_5v")


def test_mmhc_finds_no_edges_between_independent_columns():
    x = np.random.default_rng(0).standard_normal((500, 6))
    assert andrey.mmhc(x).structure.oriented_edges() == []


@pytest.mark.parametrize("seed", range(4))
def test_mmhc_recovers_sparse_graphs(seed):
    """On 20 variables with 2,000 rows, MMHC makes at most half the empty graph's errors."""
    sample = _er_sample(20, seed)
    out = andrey.mmhc(sample.data)
    assert shd(out.structure, to_cpdag(sample.graph)) <= 10


@pytest.fixture
def oracle_test():
    """Register ``"dsep"``, a d-separation oracle of the DAG ``truth`` the test sets."""
    from andrey.core import ci

    box: dict[str, DSeparation] = {}
    register_indep_test("dsep", lambda x: box["oracle"])
    yield box
    ci._CI_REGISTRY.pop("dsep", None)


@pytest.mark.parametrize("seed", range(3))
def test_mmhc_searches_only_the_true_skeleton_under_an_oracle(seed, oracle_test):
    """With exact tests, every edge MMHC returns is a true adjacency."""
    sample = _er_sample(15, seed)
    oracle_test["oracle"] = DSeparation(sample.graph)
    out = andrey.mmhc(sample.data, indep_test="dsep")
    true = {frozenset((i, j)) for i, j, _ in sample.graph.oriented_edges(index=True)}
    assert _adjacent(out) <= true


@pytest.mark.parametrize("seed", range(5))
def test_mmhc_relabels_under_a_column_shuffle(seed):
    sample = _er_sample(20, 0)
    order = np.random.default_rng(seed).permutation(20)
    base = andrey.mmhc(sample.data)
    moved = andrey.mmhc(sample.data[:, order])
    assert _edges(moved, order) == _edges(base)
    assert moved.metadata["score"] == pytest.approx(base.metadata["score"], rel=1e-12)
