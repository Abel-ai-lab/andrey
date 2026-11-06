"""The mixed-network sampler: categorical columns first in causal order, seeded, validated, and
its interventions on both kinds of column.

The checks that compare one sample with another are exact: environment 0 with the rows of a
sample without ``envs`` draws its noise, so a column an intervention does not reach is equal in
the two.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from andrey.data import sample_mixed
from andrey.data.interventions import hard, soft

# Column 3 is categorical, with a parent and a child; column 2 is continuous, with a continuous
# and a categorical parent and a child.
NETWORK = {"d": 10, "density": 3.0, "categories": (3, 3), "seed": 0}
CATEGORICAL, CONTINUOUS = 3, 2


def test_categorical_columns_hold_codes_and_have_only_categorical_parents():
    ds = sample_mixed(d=12, n=500, seed=1, categorical_share=0.5, categories=(2, 4))
    config = ds.provenance["config"]
    columns, cards = config["categorical"], config["categories"]
    assert len(columns) == 6 and all(2 <= k <= 4 for k in cards)
    for column, k in zip(columns, cards):
        assert set(np.unique(ds.data[:, column])) <= set(range(k))
    edges = ds.params.edges
    into_categorical = np.isin(edges[:, 1], columns)
    assert np.isin(edges[into_categorical, 0], columns).all()
    assert np.isnan(ds.params.weights[into_categorical]).all()
    assert np.isfinite(ds.params.weights[~into_categorical]).all()


def test_the_same_seed_gives_the_same_data():
    first, second = sample_mixed(d=6, n=100, seed=7), sample_mixed(d=6, n=100, seed=7)
    assert np.array_equal(first.data, second.data)
    assert not np.array_equal(first.data, sample_mixed(d=6, n=100, seed=8).data)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_the_share_of_categorical_columns_runs_from_none_to_all(share):
    ds = sample_mixed(d=5, n=50, seed=0, categorical_share=share)
    assert len(ds.provenance["config"]["categorical"]) == round(5 * share)


@pytest.mark.parametrize(
    "settings",
    [
        {"categorical_share": 1.5},
        {"categories": (1, 3)},
        {"categories": (3, 2)},
        {"concentration": 0.0},
    ],
)
def test_bad_settings_raise(settings):
    with pytest.raises(ValueError):
        sample_mixed(d=5, n=50, seed=0, **settings)


def _digraph(ds) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(ds.data.shape[1]))
    g.add_edges_from(ds.params.edges.tolist())
    return g


def _design(ds, rows, target):
    """The target's parents as regressors: one indicator per category of a categorical parent,
    each continuous parent's values, and a constant."""
    columns = [np.ones(int(rows.sum()))]
    for p in sorted(_digraph(ds).predecessors(target)):
        x = ds.data[rows, p]
        if p in ds.provenance["config"]["categorical"]:
            columns += [x == k for k in np.unique(ds.data[:, p])[1:]]
        else:
            columns.append(x)
    return np.column_stack(columns).astype(float)


def _fit(ds, k, target):
    """The least-squares coefficients of ``target`` on its parents in environment ``k``, and the
    share of its variance they explain."""
    rows = ds.env == k
    design, y = _design(ds, rows, target), ds.data[rows, target]
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return coefficients, 1.0 - residual.var() / y.var()


def test_the_network_has_the_columns_the_tests_name():
    ds = sample_mixed(n=10, **NETWORK)
    graph, categorical = _digraph(ds), ds.provenance["config"]["categorical"]
    assert CATEGORICAL in categorical and CONTINUOUS not in categorical
    parents = set(graph.predecessors(CONTINUOUS))
    assert parents & set(categorical) and parents - set(categorical)
    for column in (CATEGORICAL, CONTINUOUS):
        assert graph.in_degree(column) and graph.out_degree(column)


def test_a_model_seed_fixes_the_network_and_the_seed_draws_the_rows():
    a, b = (sample_mixed(d=8, n=200, seed=seed, model_seed=3) for seed in (1, 2))
    own = sample_mixed(d=8, n=200, seed=3)
    for other in (b, own):
        assert other.graph == a.graph
        assert other.provenance["column_order"] == a.provenance["column_order"]
        for key in ("categorical", "categories"):
            assert other.provenance["config"][key] == a.provenance["config"][key]
        for field in ("edges", "weights", "noise_scales"):
            assert np.array_equal(getattr(other.params, field), getattr(a.params, field), True)
    assert not np.array_equal(a.data, b.data)
    assert "model_entropy" in a.provenance and "model_entropy" not in own.provenance
    assert np.array_equal(
        sample_mixed(d=8, n=200, seed=1, model_seed=1).data, sample_mixed(d=8, n=200, seed=1).data
    )


def test_without_envs_is_unchanged():
    plain = sample_mixed(d=6, n=200, seed=2)
    same = sample_mixed(d=6, n=200, seed=2, envs=[{}])
    assert np.array_equal(plain.data, same.data)
    assert plain.env is None and plain.targets is None
    assert same.env.tolist() == [0] * 200 and same.targets == {0: []}


@pytest.mark.parametrize(
    ("column", "intervention", "check"),
    [
        (CATEGORICAL, hard(1), lambda values, _: np.all(values == 1)),
        (CATEGORICAL, hard(probs=[0.0, 0.0, 1.0]), lambda values, _: np.all(values == 2)),
        (CATEGORICAL, soft(), lambda values, _: set(np.unique(values)) <= {0, 1, 2}),
        (CONTINUOUS, hard(2.5), lambda values, _: np.all(values == 2.5)),
        (CONTINUOUS, soft(shift=1.5), lambda values, plain: np.allclose(values - plain, 1.5)),
    ],
)
def test_an_intervention_changes_only_its_column_and_descendants(column, intervention, check):
    plain = sample_mixed(n=2000, **NETWORK)
    under = sample_mixed(n=2000, envs=[{column: intervention}], **NETWORK)
    reached = nx.descendants(_digraph(plain), column) | {column}
    for j in set(range(plain.data.shape[1])) - reached:
        assert np.array_equal(under.data[:, j], plain.data[:, j])
    values = under.data[:, column]
    assert not np.array_equal(values, plain.data[:, column])
    assert check(values, plain.data[:, column])
    assert under.targets == {0: [column]}


def test_soft_scales_and_redraws_the_effects_of_both_kinds_of_parent():
    envs = [
        {},
        {CONTINUOUS: soft(weight_scale=0.0)},
        {CONTINUOUS: soft(noise_scale=0.0)},
        {CONTINUOUS: soft(weights="redraw")},
    ]
    pooled = sample_mixed(n=20000, envs=envs, model_seed=7, **NETWORK)
    plain, r2 = _fit(pooled, 0, CONTINUOUS)
    assert r2 > 0.3
    assert _fit(pooled, 1, CONTINUOUS)[1] < 0.01  # no parent acts
    assert _fit(pooled, 2, CONTINUOUS)[1] == pytest.approx(1.0)  # no noise
    redrawn = [
        _fit(
            sample_mixed(n=20000, envs=envs, **{**NETWORK, "seed": seed}, model_seed=7),
            3,
            CONTINUOUS,
        )[0]
        for seed in (1, 2)
    ]
    assert np.allclose(redrawn[0][1:], redrawn[1][1:], atol=0.1)  # the model's new effects
    assert not np.allclose(redrawn[0][1:], plain[1:], atol=0.1)


def test_a_noise_scale_range_and_an_sd_draw_per_row():
    envs = [{CONTINUOUS: soft(noise_scale=(1.0, 3.0))}, {CONTINUOUS: hard(1.0, sd=0.5)}]
    pooled = sample_mixed(n=20000, envs=envs, **NETWORK)
    residual = (
        pooled.data[pooled.env == 0, CONTINUOUS]
        - _design(pooled, pooled.env == 0, CONTINUOUS) @ _fit(pooled, 0, CONTINUOUS)[0]
    )
    assert residual.var() == pytest.approx(13 / 3, rel=0.05)  # E[U[1, 3]^2] for unit noise
    values = pooled.data[pooled.env == 1, CONTINUOUS]
    assert values.mean() == pytest.approx(1.0, abs=0.02)
    assert values.std() == pytest.approx(0.5, rel=0.05)


def test_pooled_env_targets_and_record():
    envs = [{}, {CATEGORICAL: 2}, {CONTINUOUS: soft(shift=1.0), CATEGORICAL: soft()}]
    pooled = sample_mixed(n=[50, 20, 30], envs=envs, **NETWORK)
    assert pooled.data.shape == (100, 10)
    assert pooled.env.tolist() == [0] * 50 + [1] * 20 + [2] * 30
    assert pooled.targets == {0: [], 1: [CATEGORICAL], 2: [CONTINUOUS, CATEGORICAL]}
    assert np.all(pooled.data[pooled.env == 1, CATEGORICAL] == 2)
    assert [e["n"] for e in pooled.provenance["config"]["envs"]] == [50, 20, 30]
    assert pooled.provenance["config"]["n"] == 100
    again = sample_mixed(n=[50, 20, 30], envs=envs, **NETWORK)
    assert np.array_equal(pooled.data, again.data)


@pytest.mark.parametrize(
    ("envs", "match"),
    [
        ([{CATEGORICAL: hard(3)}], "categories 0..2"),
        ([{CATEGORICAL: hard(1.0, sd=0.2)}], "no sd"),
        ([{CATEGORICAL: soft(shift=1.0)}], "takes no arguments"),
        ([{CONTINUOUS: hard()}], "takes a value"),
        ([{CONTINUOUS: hard(probs=[0.5, 0.5])}], "takes a value"),
        ([{CONTINUOUS: soft()}], "changes nothing"),
        ([{10: hard(1.0)}], "not one of the observed columns"),
        ([], "non-empty"),
    ],
)
def test_bad_envs_raise(envs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        sample_mixed(n=20, envs=envs, **NETWORK)


def test_bad_row_counts_raise():
    with pytest.raises(ValueError, match="row counts"):
        sample_mixed(n=[10, 10], envs=[{}], **NETWORK)
    with pytest.raises(TypeError, match="one row count"):
        sample_mixed(n=[10], **NETWORK)
