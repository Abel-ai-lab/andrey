"""The selection-bias sampler: the fixed graph, the random-graph rule, and the two datasets."""

from __future__ import annotations

import numpy as np
import pytest

from andrey_bench import selection_models as S


def test_the_fixed_graph_has_the_five_edges_of_the_authors_simulation_and_four_ancestral_pairs():
    g = S.fixed_graph()
    edges = {(int(i), int(j)) for i, j in zip(*np.nonzero(g.adjacency))}
    # C -> X1, C -> X4, X6 -> X4, X4 -> X5, X3 -> X2 (columns 0 to 5 are X1 to X6, 6 the context).
    assert edges == {(6, 0), (6, 3), (5, 3), (3, 4), (2, 1)}
    assert g.selection == (0, 1) and g.context == 6
    ancestral = S.ancestral_matrix(g.adjacency)
    system = {(int(i), int(j)) for i, j in zip(*np.nonzero(ancestral[:6, :6]))}
    assert system == {(5, 3), (3, 4), (5, 4), (2, 1)}  # X6 -> X4 -> X5 is transitive
    assert ancestral[6, 4]  # the context reaches X5 through X4


def test_ancestry_is_the_transitive_closure_without_the_diagonal():
    chain = np.zeros((6, 6), dtype=np.int8)
    for i in range(5):
        chain[i, i + 1] = 1
    anc = S.ancestral_matrix(chain)
    assert anc[0, 5] and not anc[5, 0] and not anc.diagonal().any()
    assert int(anc.sum()) == 15


@pytest.mark.parametrize("setting", ["small", "large"])
def test_a_random_graph_follows_the_rule_of_the_papers_appendix(setting):
    spec = S.SETTINGS[setting]
    for seed in range(25):
        rng = np.random.default_rng(seed)
        g = S.random_graph(spec, rng)
        a = g.adjacency.astype(bool)
        assert a.shape == (spec.system + 1,) * 2
        assert not S.ancestral_matrix(a).diagonal().any()  # acyclic
        context = g.context
        assert a[:, context].sum() == 0 and a[context].sum() >= 1  # a source with children
        # At least the minimum number of colliders, nodes with two or more parents.
        assert (a.sum(0) >= 2).sum() >= spec.min_colliders
        assert len(g.selection) == spec.selection == len(set(g.selection))
        colliders = np.flatnonzero(a.sum(0) >= 2)
        below = S.ancestral_matrix(a)[colliders].any(0) | np.isin(np.arange(len(a)), colliders)
        for node in g.selection:  # a leaf among the colliders' descendants
            assert node != context and a[node].sum() == 0 and below[node]


def test_the_edge_probability_is_the_settings():
    spec = S.SETTINGS["large"]
    # Before the rule's rejections: the share of the pairs i < j that carry an edge.
    rates = []
    for seed in range(60):
        g = S.random_graph(spec, np.random.default_rng(seed))
        rates.append(g.adjacency.sum() / (17 * 16 / 2))
    assert 0.06 < np.mean(rates) < 0.2  # near 0.09, raised by the rule's collider minimum


def test_a_sample_is_a_function_of_the_setting_seed_and_n():
    a = S.sample("fixed", n=500, seed=3)
    b = S.sample("fixed", n=500, seed=3)
    np.testing.assert_array_equal(a.biased, b.biased)
    np.testing.assert_array_equal(a.unbiased, b.unbiased)
    longer = S.sample("fixed", n=800, seed=3)
    np.testing.assert_array_equal(a.weights, longer.weights)
    np.testing.assert_array_equal(a.graph.adjacency, longer.graph.adjacency)
    assert not np.array_equal(a.biased, S.sample("fixed", n=500, seed=4).biased)


def test_the_context_stays_last_and_the_truth_is_permuted_with_the_columns():
    for seed in range(6):
        d = S.sample("small", n=200, seed=seed)
        p = d.graph.adjacency.shape[0]
        assert d.biased.shape == d.unbiased.shape == (200, p) and p == 9 and d.graph.context == 8
        assert d.graph.adjacency[:, 8].sum() == 0  # the context is a source
        np.testing.assert_array_equal(d.ancestral, S.ancestral_matrix(d.graph.adjacency))
        np.testing.assert_array_equal(d.system, np.arange(8))
    orders = {
        tuple(S.sample("fixed", n=50, seed=s).graph.adjacency[6]) for s in range(12)
    }  # the context's children are in different columns
    assert len(orders) > 1
    plain = S.sample("fixed", n=50, seed=0, shuffle=False)
    np.testing.assert_array_equal(plain.graph.adjacency, S.fixed_graph().adjacency)


def test_weights_have_random_signs_and_the_rescaled_magnitudes():
    d = S.sample("large", n=100, seed=0)
    w = d.weights
    assert np.all((w != 0) == d.graph.adjacency.astype(bool))
    # Each column was divided by sqrt(sum of the squared system weights + 1): the system part of
    # the column now holds less than 1 of the squared weight.
    assert np.all((w[:16] ** 2).sum(0) < 1.0)
    signs = set()
    for seed in range(10):
        ws = S.sample("large", n=50, seed=seed).weights
        signs |= set(np.sign(ws[ws != 0]))
    assert signs == {-1.0, 1.0}


def test_unbiased_rows_have_the_covariance_of_the_model():
    d = S.sample("fixed", n=200_000, seed=1)
    inverse = np.linalg.inv(np.eye(7) - d.weights)
    np.testing.assert_allclose(np.cov(d.unbiased, rowvar=False), inverse.T @ inverse, atol=0.04)


def test_selected_rows_hold_the_sum_of_the_selection_variables_inside_the_interval():
    d = S.sample("fixed", n=4000, seed=2)
    total = d.biased[:, list(d.graph.selection)].sum(1)
    assert np.all((total > S.INTERVAL[0]) & (total < S.INTERVAL[1]))
    unselected = d.unbiased[:, list(d.graph.selection)].sum(1)
    assert not np.all((unselected > S.INTERVAL[0]) & (unselected < S.INTERVAL[1]))
    assert 0.005 < d.acceptance < 0.2  # the share of draws that were kept


def test_selection_induces_the_dependence_between_the_selection_variables():
    """The two selection variables of the fixed graph are independent until they are selected on."""
    d = S.sample("fixed", n=6000, seed=5)
    a, b = d.graph.selection
    assert abs(np.corrcoef(d.unbiased[:, a], d.unbiased[:, b])[0, 1]) < 0.05
    assert np.corrcoef(d.biased[:, a], d.biased[:, b])[0, 1] < -0.3


def test_an_interval_the_model_never_reaches_is_an_error(monkeypatch):
    monkeypatch.setattr(S, "INTERVAL", (1e6, 1e6 + 1))
    monkeypatch.setattr(S, "MAX_CHUNKS", 3)
    with pytest.raises(RuntimeError, match="rows kept"):
        S.sample("fixed", n=100, seed=0)
