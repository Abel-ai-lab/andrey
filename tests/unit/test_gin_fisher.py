"""GIN pools the p-values of its independence tests by Fisher's method."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.stats import chi2

from andrey.latent.gin import _fisher_combine


def test_gin_pools_p_values_by_fishers_method() -> None:
    """Fisher's statistic ``-2 sum(log p)`` against chi-square with ``2c`` degrees of freedom.

    One p-value pools to itself, and one strong rejection among many weak ones keeps its weight.
    """
    for p in (0.5, 1e-3, 1e-6, 1e-12, 1e-50):
        assert _fisher_combine([p]) == pytest.approx(p, rel=1e-9)
    pvals = [0.5] * 19 + [1e-12]
    expected = chi2.sf(-2.0 * np.sum(np.log(pvals)), 2 * len(pvals))
    assert _fisher_combine(pvals) == pytest.approx(expected, rel=1e-9)
    assert _fisher_combine(pvals) < 0.05
    assert _fisher_combine([0.0, 0.5]) < 1e-300


def test_gin_root_test_puts_every_member_in_the_surrogate(monkeypatch) -> None:
    """The root test runs once per member of the largest cluster and pools the runs by Fisher.

    Across the runs every member of the candidate, of the other cluster, and of each ordered
    cluster enters the surrogate; a run's reference set holds the candidate's other members and one
    member of each ordered cluster, none of them in that run's surrogate.
    """
    engine = importlib.import_module("andrey.latent.gin")
    candidate, other, ordered = [0, 1, 2], [3, 4], [[5, 6], [7, 8, 9, 10]]
    runs: list[tuple[list[int], list[int]]] = []

    def pooled(data, cov, grams, x, z):
        runs.append((x, z))
        return 0.1 * len(runs)

    monkeypatch.setattr(engine, "_independence_pvalue", pooled)
    pvalue = engine._root_pvalue(np.zeros((1, 11)), np.eye(11), [], candidate, other, ordered)

    assert len(runs) == 4
    for cluster in [candidate, other, *ordered]:
        assert {m for x, _ in runs for m in x if m in cluster} == set(cluster)
    for x, z in runs:
        assert not set(x) & set(z)
        assert sorted(m for m in z if m in candidate) == sorted(set(candidate) - set(x))
        assert all(sum(m in prefix for m in z) == 1 for prefix in ordered)
    assert pvalue == pytest.approx(_fisher_combine([0.1, 0.2, 0.3, 0.4]), rel=1e-12)
