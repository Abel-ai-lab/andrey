"""``n_starts`` on the permutation searches: one start is the search without the argument, and more
starts never end on a worse score than one."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import Knowledge, data

METHODS = ("boss", "grasp")
#: Dense small graphs on which more starts reach a better score for each method on some seeds.
DATASETS = [
    data.sample_scm(
        graph="erdos_renyi", d=10, n=300, seed=seed, density=4.0, scale="standardize"
    ).data
    for seed in range(4)
]
KNOWLEDGE = Knowledge(tiers=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], required=[(1, 7)])


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
