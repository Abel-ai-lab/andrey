"""RCD's Pearson p-value, which skips SciPy's input checks, equals SciPy's test."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from andrey.irregular.rcd import _pearson_pvalue_fast


@pytest.mark.parametrize("n", [3, 10, 100, 2000])
def test_pearson_pvalue_matches_scipy(n):
    rng = np.random.default_rng(n)
    for slope in np.linspace(-3, 3, 9) / np.sqrt(n):  # p-values from about 1 down to 1e-7
        a = rng.standard_normal(n)
        b = slope * a + rng.standard_normal(n)
        expected = pearsonr(a, b)[1]
        if expected > 1e-6:  # the range where a verdict at a usual cor_alpha can turn
            assert _pearson_pvalue_fast(a, b) == pytest.approx(expected, rel=1e-12)


def test_a_correlation_rounding_to_one_gives_scipys_zero_p_value():
    # r is 0.9999999999999999 here; SciPy's tail rounds (1 + r) / 2 to 1 and returns p = 0
    x = np.array([0.0, 1.0, 4.0])
    assert _pearson_pvalue_fast(x, x) == 0.0
