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
    assert markov_equivalent(out.structure, andrey.exact_search(x).structure)
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


PHASES = [(r, m) for r in ("mmpc", "iamb", "inter_iamb", "pc") for m in ("hc", "tabu")]


def test_mmhc_is_rsmax2_with_mmpc_and_hill_climbing():
    x = _er_sample(20, 1).data
    mmhc, rsmax2 = andrey.mmhc(x), andrey.rsmax2(x)
    assert _edges(mmhc) == _edges(rsmax2)
    assert mmhc.metadata["score"] == rsmax2.metadata["score"]


def test_rsmax2_reaches_the_score_optimum_on_gauss_5v():
    x = datagen.generate("gauss_5v")
    out = andrey.rsmax2(x, restrict="pc", maximize="tabu")
    assert markov_equivalent(out.structure, andrey.exact_search(x).structure)
    assert_recorded_output(out, "RSMAX2", case="gauss_5v")


@pytest.mark.parametrize(("restrict", "maximize"), PHASES)
def test_rsmax2_joins_only_candidate_pairs_and_recovers_sparse_graphs(restrict, maximize):
    from andrey.search.hybrid import candidate_graph

    sample = _er_sample(20, 0)
    out = andrey.rsmax2(sample.data, restrict=restrict, maximize=maximize)
    candidates = candidate_graph(sample.data, restrict, alpha=0.05, indep_test="fisherz")
    assert all(candidates[i, j] for i, j in map(tuple, _adjacent(out)))
    assert shd(out.structure, to_cpdag(sample.graph)) <= 10


@pytest.mark.parametrize("restrict", ["mmpc", "iamb", "inter_iamb", "pc"])
def test_tabu_never_scores_worse_than_hill_climbing_on_the_same_candidates(restrict):
    x = _er_sample(20, 1).data
    tabu = andrey.rsmax2(x, restrict=restrict, maximize="tabu").metadata["score"]
    assert tabu <= andrey.rsmax2(x, restrict=restrict, maximize="hc").metadata["score"]


@pytest.mark.parametrize("seed", range(3))
def test_each_restrict_phase_finds_its_graph_under_an_oracle(seed, oracle_test):
    """The skeleton phases return the true skeleton; the blanket phases the true moral graph."""
    from andrey.search.hybrid import candidate_graph

    sample = _er_sample(15, seed)
    oracle = oracle_test["oracle"] = DSeparation(sample.graph)
    for restrict, want in [
        ("mmpc", oracle.skeleton()),
        ("pc", oracle.skeleton()),
        ("iamb", oracle.moral_graph()),
        ("inter_iamb", oracle.moral_graph()),
    ]:
        got = candidate_graph(sample.data, restrict, alpha=0.5, indep_test="dsep")
        np.testing.assert_array_equal(got, want, err_msg=restrict)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("restrict", ["iamb", "pc"])
def test_rsmax2_relabels_under_a_column_shuffle(seed, restrict):
    x = _er_sample(20, 1).data
    order = np.random.default_rng(seed).permutation(20)
    base = andrey.rsmax2(x, restrict=restrict, maximize="tabu")
    moved = andrey.rsmax2(x[:, order], restrict=restrict, maximize="tabu")
    assert _edges(moved, order) == _edges(base)


def test_rsmax2_with_pc_gives_the_same_graph_on_more_workers():
    from andrey.core import backend

    x = _er_sample(20, 1).data
    first = andrey.rsmax2(x, restrict="pc")
    with backend.config(n_jobs=3):
        again = andrey.rsmax2(x, restrict="pc")
    assert _edges(again) == _edges(first) and again.metadata == first.metadata


@pytest.mark.parametrize(
    ("setting", "error"),
    [
        ({"restrict": "gs"}, ValueError),
        ({"maximize": "ges"}, ValueError),
        ({"maximize": "tabu", "score_func": "cond_gauss"}, ValueError),
        ({"score_func": "bdeu"}, ValueError),  # a categorical score with fisherz
    ],
)
def test_rsmax2_rejects_settings_it_does_not_take(setting, error):
    with pytest.raises(error):
        andrey.rsmax2(_er_sample(5, 0, n=200).data, **setting)


def test_categorical_data_with_the_chi_square_test_and_bdeu():
    """On a categorical collider, MMHC keeps the independent causes apart, which hill climbing
    alone joins; tabu search within the same skeleton orients the collider."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, (2, 3000))
    c = np.where(rng.random(3000) < 0.85, a + b, rng.integers(0, 3, 3000))
    x = np.column_stack([a, b, c]).astype(float)
    setting = {"indep_test": "chisq", "score_func": "bdeu"}
    assert frozenset((0, 1)) in _adjacent(andrey.hc(x, score_func="bdeu"))
    assert _adjacent(andrey.mmhc(x, **setting)) == {frozenset((0, 2)), frozenset((1, 2))}
    out = andrey.rsmax2(x, maximize="tabu", **setting)
    assert _edges(out) == {(0, 2, "directed"), (1, 2, "directed")}
