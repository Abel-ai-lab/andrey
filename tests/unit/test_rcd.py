"""RCD's Pearson p-value, which skips SciPy's input checks, equals SciPy's test."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from andrey.irregular.rcd import _pearson_pvalue_fast


@pytest.mark.parametrize("n", [3, 10, 100, 2000])
def test_pearson_pvalue_matches_scipy(n):
    rng = np.random.default_rng(n)
    for slope in np.linspace(-1, 1, 9):
        a = rng.standard_normal(n)
        b = slope * a + rng.standard_normal(n)
        expected = pearsonr(a, b)[1]
        if expected > 1e-6:  # the range where a verdict at a usual cor_alpha can turn
            assert _pearson_pvalue_fast(a, b) == pytest.approx(expected, rel=1e-12)
