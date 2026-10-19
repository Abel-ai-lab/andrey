"""GIN pools the p-values of its independence tests by Fisher's method."""

from __future__ import annotations

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
