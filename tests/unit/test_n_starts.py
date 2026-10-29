"""``n_starts``: one start is the search without the argument, more starts never end on a worse
score than one, and the random edge changes before a later start of ``hc`` and ``tabu`` keep the
knowledge, ``max_parents``, the mixed-data rule, and acyclicity."""

from __future__ import annotations

import random

import numpy as np
import pytest

import andrey
from andrey import Knowledge, data
from andrey.search import hc as hc_engine
from andrey.search import tabu as tabu_engine

METHODS = ("boss", "grasp", "hc", "tabu")
#: The searches whose later starts change the best DAG so far at random.
RESTARTED = ("hc", "tabu")
#: Dense small graphs on which more starts reach a better score for each method on some seeds.
DATASETS = [
    data.sample_scm(
        graph="erdos_renyi", d=10, n=300, seed=seed, density=4.0, scale="standardize"
    ).data
    for seed in range(4)
]
KNOWLEDGE = Knowledge(tiers=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], required=[(1, 7)])
PERTURB = hc_engine.perturb


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kwargs", [{}, {"knowledge": KNOWLEDGE}, {"ebic_gamma": 0.0}])
def test_one_start_is_the_search_without_the_argument(method, kwargs):
    fit = getattr(andrey, method)
    for x in DATASETS:
        plain = fit(x, seed=3, **kwargs)
        one = fit(x, seed=3, n_starts=1, **kwargs)
        assert np.array_equal(plain.structure.to_numpy(), one.structure.to_numpy())
        assert plain.metadata == one.metadata  # the same score, bit for bit


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kwargs", [{}, {"knowledge": KNOWLEDGE}])
def test_more_starts_never_end_on_a_worse_score(method, kwargs):
    fit = getattr(andrey, method)
    gains = []
    for x in DATASETS:
        scores = [fit(x, seed=0, n_starts=k, **kwargs).metadata["score"] for k in (1, 2, 4)]
        assert scores[0] >= scores[1] >= scores[2]
        gains.append(scores[0] - scores[2])
    if not kwargs:  # the later starts are searched: some dataset ends on a better score
        assert max(gains) > 1e-6


@pytest.mark.parametrize("method", RESTARTED)
def test_one_start_draws_nothing_and_records_no_seed(method):
    fit = getattr(andrey, method)
    x = DATASETS[0]
    plain = fit(x)
    other = fit(x, seed=9, n_perturbations=2)
    assert np.array_equal(plain.structure.to_numpy(), other.structure.to_numpy())
    assert plain.metadata == other.metadata
    assert not {"seed", "n_perturbations"} & set(plain.metadata)


@pytest.mark.parametrize("method", RESTARTED)
def test_more_starts_record_the_seed_and_the_changes(method, monkeypatch):
    fit = getattr(andrey, method)
    x = DATASETS[0]
    monkeypatch.setenv("ANDREY_SEED", "5")
    out = fit(x, n_starts=3)
    assert (out.metadata["seed"], out.metadata["n_perturbations"]) == (5, x.shape[1])
    out = fit(x, n_starts=3, seed=2, n_perturbations=4)
    assert (out.metadata["seed"], out.metadata["n_perturbations"]) == (2, 4)


@pytest.mark.parametrize("method", RESTARTED)
@pytest.mark.parametrize(("value", "error"), [(0, ValueError), (1.5, TypeError), (True, TypeError)])
def test_n_perturbations_is_a_count_of_at_least_one(method, value, error):
    with pytest.raises(error, match=rf"^{method}: n_perturbations must be "):
        getattr(andrey, method)(DATASETS[0], n_starts=2, n_perturbations=value)


def dag_rules_hold(parents_of, d, forbidden, required, max_parents) -> None:
    """Assert that the DAG keeps every rule a move keeps; ``forbidden[v, u]`` bars ``u -> v`` and
    ``required[v, u]`` holds it."""
    edge = np.zeros((d, d), dtype=bool)  # [v, u]: u -> v
    for v, pa in parents_of.items():
        assert pa == sorted(set(pa))
        edge[v, pa] = True
    assert not (edge & forbidden).any()
    assert (edge[required]).all()
    if max_parents is not None:
        assert edge.sum(axis=1).max() <= max_parents
    order, left = [], set(range(d))
    while left:  # peel the nodes without parents left: a cycle leaves none
        roots = [v for v in left if not set(parents_of[v]) & left]
        assert roots
        order += roots
        left -= set(roots)


@pytest.mark.parametrize("max_parents", [None, 1, 2])
def test_random_changes_keep_every_rule_of_a_move(max_parents):
    d = 8
    rng = np.random.default_rng(max_parents or 0)
    required = np.zeros((d, d), dtype=bool)
    required[3, 1] = required[5, 4] = True  # 1 -> 3 and 4 -> 5
    forbidden = (rng.random((d, d)) < 0.3) | required.T
    np.fill_diagonal(forbidden, False)
    forbidden &= ~required
    adj_int = [0] * d
    parents_of: dict[int, list[int]] = {v: [] for v in range(d)}
    for v, u in zip(*np.nonzero(required)):
        hc_engine._apply_move(adj_int, parents_of, ("add", int(u), int(v)))
    draws = random.Random(0)
    seen = set()
    for _ in range(300):
        PERTURB(
            adj_int,
            parents_of,
            3,
            draws,
            ranks=rng.permutation(d),
            forbidden=forbidden,
            required=required,
            max_parents=max_parents,
        )
        assert adj_int == [sum(1 << v for v in range(d) if u in parents_of[v]) for u in range(d)]
        dag_rules_hold(parents_of, d, forbidden, required, max_parents)
        seen.add(tuple(adj_int))
    assert len(seen) > 100  # the changes do move the DAG


@pytest.mark.parametrize("method", RESTARTED)
@pytest.mark.parametrize("max_parents", [None, 2])
def test_every_later_start_begins_from_a_dag_that_keeps_the_knowledge(
    method, max_parents, monkeypatch
):
    """Every start's DAG after its random changes keeps the knowledge and ``max_parents``, and so
    does the result."""
    engine = {"hc": hc_engine, "tabu": tabu_engine}[method]
    starts = []

    def recorded(adj_int, parents_of, changes, draws, **rules):
        PERTURB(adj_int, parents_of, changes, draws, **rules)
        starts.append(({v: list(pa) for v, pa in parents_of.items()}, rules))

    monkeypatch.setattr(engine, "perturb", recorded)
    knowledge = Knowledge(
        tiers=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], required=[(1, 7)], forbidden=[(2, 3), (8, 9)]
    )
    out = getattr(andrey, method)(
        DATASETS[1], n_starts=6, knowledge=knowledge, max_parents=max_parents
    )
    assert len(starts) == 5
    for parents_of, rules in starts:
        dag_rules_hold(parents_of, 10, rules["forbidden"], rules["required"], max_parents)
        edges = {(u, v) for v, pa in parents_of.items() for u in pa}
        assert (1, 7) in edges and not {(2, 3), (8, 9)} & edges
        assert not {(u, v) for u, v in edges if u >= 5 > v}  # no edge into an earlier tier
    marks = out.structure.to_numpy()
    assert marks[1, 7] == andrey.core.TAIL and marks[7, 1] == andrey.core.ARROW


@pytest.mark.parametrize("method", RESTARTED)
def test_random_changes_keep_the_mixed_data_rule(method, monkeypatch):
    """Under a mixed-data score no start's DAG gives a categorical column a continuous parent."""
    engine = {"hc": hc_engine, "tabu": tabu_engine}[method]
    starts = []

    def recorded(adj_int, parents_of, changes, draws, **rules):
        PERTURB(adj_int, parents_of, changes, draws, **rules)
        starts.append({v: list(pa) for v, pa in parents_of.items()})

    monkeypatch.setattr(engine, "perturb", recorded)
    x = DATASETS[2].copy()
    categorical = [0, 3, 6]
    for c in categorical:
        x[:, c] = (x[:, c] > 0).astype(float)
    getattr(andrey, method)(x, score_func="cond_gauss", categorical=categorical, n_starts=4)
    assert len(starts) == 3
    for parents_of in starts:
        for v in categorical:
            assert set(parents_of[v]) <= set(categorical)


@pytest.mark.parametrize("method", RESTARTED)
def test_a_new_column_order_relabels_the_result_of_many_starts(method):
    """The random changes pick edges in the columns' rank order, so they move with the columns."""
    x = DATASETS[3]
    order = np.random.default_rng(1).permutation(x.shape[1])
    fit = getattr(andrey, method)
    base = fit(x, n_starts=4).structure.to_numpy()
    moved = fit(x[:, order], n_starts=4).structure.to_numpy()
    assert np.array_equal(moved, base[np.ix_(order, order)])
