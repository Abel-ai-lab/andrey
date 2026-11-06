"""``andrey.markov_check``: its p-values against the definition, and its error rates against the
level it promises."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import andrey.data as ad
from andrey import GraphStructure, SampleSizeWarning, markov_check
from andrey.markov import anderson_darling_uniform
from andrey.metrics import to_cpdag

#: A count is accepted when the rate it should follow lies inside its 99.9% binomial interval, so a
#: sound result fails 1 time in 1,000. The seeds are fixed, so a run is the same every time.
LEVEL = 1e-3


def _sample(d: int, n: int, seed: int, avg_degree: float = 2.0):
    scm = ad.SCM(
        graph=ad.graphs.erdos_renyi(d=d, avg_degree=avg_degree),
        functional=ad.functional.linear(),
        noise=ad.noise.gaussian(),
    )
    return scm.sample(n=n, seed=seed)


def _without_edge(graph: GraphStructure, k: int) -> GraphStructure:
    M = graph.to_numpy().copy()
    i, j = np.argwhere(np.triu(M != 0))[k % int(np.triu(M != 0).sum())]
    M[i, j] = M[j, i] = 0
    return GraphStructure.from_numpy(M, kind="dag")


def test_the_uniformity_p_value_is_exact_for_one_p_value():
    """At one value the statistic is -1 - ln(u (1 - u)), whose distribution is known."""
    p = [anderson_darling_uniform(u)[1] for u in np.random.default_rng(1).uniform(size=(5000, 1))]
    assert stats.kstest(p, "uniform").pvalue > LEVEL


@functools.cache
def uniformity_p_values(m: int) -> np.ndarray:
    """The uniformity p-values of 10,000 sets of ``m`` uniform p-values, drawn from seed ``m``."""
    sets = np.random.default_rng(m).uniform(size=(10_000, m))
    return np.array([anderson_darling_uniform(u)[1] for u in sets])


@pytest.mark.parametrize("m", [1, 2, 5, 150])
@pytest.mark.parametrize("level", [0.01, 0.05, 0.2])
def test_uniform_p_values_are_rejected_at_the_level(m, level):
    """The share of uniform p-value sets that the test rejects at ``level`` is ``level``, to the
    binomial error of the number of sets (the verdict is made at these levels)."""
    p_values = uniformity_p_values(m)
    rejected = int((p_values <= level).sum())
    assert stats.binomtest(rejected, len(p_values), level).pvalue > LEVEL


def test_the_statistic_is_the_definition():
    u = np.array([0.1, 0.4, 0.45, 0.8])
    n = len(u)
    a2 = (
        -n
        - sum((2 * i - 1) * (np.log(u[i - 1]) + np.log(1 - u[n - i])) for i in range(1, n + 1)) / n
    )
    assert anderson_darling_uniform(u)[0] == pytest.approx(a2)


def test_the_tests_are_the_non_parent_pairs_given_the_parents_whatever_the_column_order():
    sample = _sample(8, 300, 0)
    dag = sample.graph
    labels = tuple("abcdefgh")
    named = GraphStructure.from_numpy(dag.to_numpy(), kind="dag", labels=labels)
    frame = pd.DataFrame(sample.data, columns=list(labels))
    check = markov_check(named, frame, sample_fraction=1.0)
    assert check.n_tests == 8 * 7 // 2 - len(dag.oriented_edges())
    parents = {x: set() for x in labels}
    for source, target, _ in named.oriented_edges():
        parents[target].add(source)
    for row in check.table():
        assert set(row["given"]) == parents[row["x"]] and row["y"] not in parents[row["x"]]
        assert named.d_separated(row["x"], row["y"], row["given"])
    shuffled = frame[list(np.random.default_rng(1).permutation(labels))]
    again = markov_check(named, shuffled, sample_fraction=1.0)
    assert {(r["x"], r["y"]): r["p_value"] for r in again.table()} == pytest.approx(
        {(r["x"], r["y"]): r["p_value"] for r in check.table()}
    )
    assert again.uniformity_p_value == pytest.approx(check.uniformity_p_value)


@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_a_true_graph_is_rejected_about_as_often_as_alpha_with_the_default(alpha):
    fits = 200
    rejected = 0
    for seed in range(fits):
        sample = _sample(10, 500, seed)
        rejected += not markov_check(sample.graph, sample.data, alpha=alpha, seed=seed).passed
    assert stats.binomtest(rejected, fits, alpha).pvalue > LEVEL


def test_a_graph_with_an_edge_removed_is_rejected_at_the_documented_rate():
    """One missing edge was rejected in 96% to 99% of the documented simulations at 1,000 rows, so
    the count must not be significantly below 96%."""
    fits = 100
    rejected = 0
    for seed in range(fits):
        sample = _sample(10, 1000, seed)
        broken = _without_edge(sample.graph, seed)
        rejected += not markov_check(broken, sample.data, seed=seed).passed
    assert stats.binomtest(rejected, fits, 0.96, alternative="less").pvalue > LEVEL


def test_a_cpdag_is_checked_through_a_dag_of_its_class():
    sample = _sample(8, 400, 3)
    check = markov_check(to_cpdag(sample.graph), sample.data)
    assert check.dag.kind == "dag" and to_cpdag(check.dag).to_numpy().tolist() == (
        to_cpdag(sample.graph).to_numpy().tolist()
    )
    assert check.n_tests == markov_check(sample.graph, sample.data).n_tests


def test_seed_sets_the_row_draws_and_sample_fraction_one_draws_none():
    sample = _sample(8, 400, 4)
    a = markov_check(sample.graph, sample.data, seed=1)
    assert np.array_equal(a.p_values, markov_check(sample.graph, sample.data, seed=1).p_values)
    assert not np.array_equal(a.p_values, markov_check(sample.graph, sample.data, seed=2).p_values)
    one = [markov_check(sample.graph, sample.data, sample_fraction=1.0, seed=s) for s in (1, 2)]
    assert np.array_equal(one[0].p_values, one[1].p_values)
    assert a.metadata["seed"] == 1 and a.metadata["rows_per_test"] == 200


def test_a_graph_that_implies_nothing_cannot_be_rejected():
    complete = GraphStructure.from_numpy(np.array([[0, 1, 1], [2, 0, 1], [2, 2, 0]]), kind="dag")
    check = markov_check(complete, np.random.default_rng(0).standard_normal((50, 3)))
    assert check.n_tests == 0 and check.passed and check.uniformity_p_value is None


def test_graphs_and_data_that_cannot_be_checked_raise():
    sample = _sample(5, 100, 0)
    pag = sample.graph.to_pag(latent=[0])
    with pytest.raises(ValueError, match="dag or a cpdag"):
        markov_check(pag, sample.data[:, 1:])
    with pytest.raises(ValueError, match="columns"):
        markov_check(sample.graph, sample.data[:, :4])
    named = GraphStructure.from_numpy(sample.graph.to_numpy(), kind="dag", labels=tuple("abcde"))
    with pytest.raises(ValueError, match="differ"):
        markov_check(named, pd.DataFrame(sample.data, columns=list("abcdz")))
    with pytest.raises(ValueError, match="alpha"):
        markov_check(sample.graph, sample.data, alpha=0.0)
    with pytest.raises(ValueError, match="sample_fraction"):
        markov_check(sample.graph, sample.data, sample_fraction=1.5)
    with pytest.raises(ValueError, match="rows"):
        markov_check(sample.graph, sample.data[:6], sample_fraction=0.5)


def _fisher_z(block: np.ndarray) -> float:
    """The Fisher-Z p-value of the first two columns of ``block`` given the others."""
    precision = np.linalg.inv(np.corrcoef(block, rowvar=False))
    r = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    z = np.sqrt(len(block) - (block.shape[1] - 2) - 3) * abs(np.arctanh(r))
    return float(2 * stats.norm.sf(z))


def test_missing_values_are_deleted_test_by_test():
    """Each test reads the rows observed in its own columns, so its p-value is the Fisher-Z test
    on those rows alone, and the row counts summarize the tests' rows."""
    sample = _sample(6, 400, 5)
    data = sample.data.copy()
    rng = np.random.default_rng(0)
    for j in range(6):  # a different share missing in each column
        data[rng.random(400) < 0.02 * (j + 1), j] = np.nan
    check = markov_check(sample.graph, data, sample_fraction=1.0)
    assert check.metadata["indep_test"] == "fisherz"
    rows = []
    for test in check.table(index=True):
        columns = data[:, [test["x"], test["y"], *test["given"]]]
        block = columns[~np.isnan(columns).any(axis=1)]
        rows.append(len(block))
        assert test["p_value"] == pytest.approx(_fisher_z(block), rel=1e-9)
    assert len(set(rows)) > 1
    assert check.metadata["rows_tested"] == {"min": min(rows), "median": float(np.median(rows))}


def _categorical(n: int, levels: int, seed: int) -> tuple[GraphStructure, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, levels, n)
    b = np.where(rng.random(n) < 0.7, a, rng.integers(0, levels, n))
    c = np.where(rng.random(n) < 0.7, b, rng.integers(0, levels, n))
    M = np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]])
    return GraphStructure.from_numpy(M, kind="dag"), np.column_stack([a, b, c])


def test_sparse_categorical_tables_warn_and_dense_ones_do_not():
    graph, sparse = _categorical(60, 5, 0)
    with pytest.warns(SampleSizeWarning, match="not evidence"):
        markov_check(graph, sparse, indep_test="gsq", categorical=[0, 1, 2])
    graph, dense = _categorical(4000, 2, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SampleSizeWarning)
        markov_check(graph, dense, indep_test="gsq", categorical=[0, 1, 2])
