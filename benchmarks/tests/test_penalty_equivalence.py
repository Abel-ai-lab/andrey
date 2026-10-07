"""Verify equivalent BIC penalties in Andrey and causal-learn.

causal-learn scores log-likelihood while Andrey scores deviance, so Andrey's coefficient must be
twice causal-learn's. Score differences remove constants independent of the parent set.
Checked on the score functions rather than through a search, so a failure points at the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core.score import local_score_bic
from andrey_bench.adapters.bic_penalty import (
    ANDREY_LAMBDA,
    CAUSAL_LEARN_LAMBDA,
    andrey_lambda,
)

# causal-learn is installed in the bench venv only; elsewhere this module skips.
local_score_BIC_from_cov = pytest.importorskip(
    "causallearn.score.LocalScoreFunction"
).local_score_BIC_from_cov

#: Parent sets to compare; the first removes each scorer's parent-set-independent constant.
_PARENT_SETS = ([], [0], [1, 2], [0, 1, 2, 4], [2, 3, 4, 5, 6])

_D, _N = 8, 500


def _sample() -> np.ndarray:
    """A fixed correlated sample."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((_N, _D))
    x[:, 3] += 2 * x[:, 1]
    x[:, 5] += x[:, 2] - x[:, 0]
    return x


def _covariances() -> tuple[np.ndarray, np.ndarray]:
    """Return covariance matrices using each package's ``ddof`` convention.

    causal-learn uses ``ddof=0``; Andrey uses ``ddof=1``. Their residual variances differ by
    ``(n - 1) / n``, which adds the same ``n * log((n - 1) / n)`` term to every parent set. Score
    differences remove it.
    Each scorer gets the matrix its own production path builds, so the test covers that difference
    rather than assuming it away.
    """
    x = _sample()
    return np.cov(x, rowvar=False, ddof=0), np.cov(x, rowvar=False, ddof=1)


def _score_deltas(node: int, covs, cl_lambda: float, andrey_lam: float):
    """Yield each package's score difference from the reference parent set.

    Remove ``node`` from its own parent set to avoid a zero residual and ``log(0)``.
    """
    cl_cov, andrey_cov = covs
    sets = [[p for p in parents if p != node] for parents in _PARENT_SETS]
    cl_base = local_score_BIC_from_cov((cl_cov, _N), node, sets[0], {"lambda_value": cl_lambda})
    andrey_base = local_score_bic(andrey_cov, _N, node, sets[0], lambda_value=andrey_lam)
    for parents in sets[1:]:
        cl = local_score_BIC_from_cov((cl_cov, _N), node, parents, {"lambda_value": cl_lambda})
        andrey = local_score_bic(andrey_cov, _N, node, parents, lambda_value=andrey_lam)
        yield cl - cl_base, andrey - andrey_base


def test_the_conversion_is_a_factor_of_two() -> None:
    """Convert causal-learn's coefficient to twice its value for Andrey."""
    assert andrey_lambda(CAUSAL_LEARN_LAMBDA) == ANDREY_LAMBDA
    assert ANDREY_LAMBDA == 2 * CAUSAL_LEARN_LAMBDA


@pytest.mark.parametrize("node", range(_D))
def test_matched_lambdas_rank_parent_sets_identically(node: int) -> None:
    """Match Andrey score differences to ``-2`` times causal-learn's."""
    covs = _covariances()
    for cl_delta, andrey_delta in _score_deltas(node, covs, CAUSAL_LEARN_LAMBDA, ANDREY_LAMBDA):
        assert -2 * cl_delta == pytest.approx(andrey_delta, abs=1e-9)


def test_equal_coefficients_are_not_equal_penalties() -> None:
    """Negative control: equal coefficients must disagree, or an identity conversion would pass."""
    covs = _covariances()
    worst = max(
        abs(-2 * cl_delta - andrey_delta)
        for node in range(_D)
        for cl_delta, andrey_delta in _score_deltas(
            node, covs, CAUSAL_LEARN_LAMBDA, CAUSAL_LEARN_LAMBDA
        )
    )
    # The deviance gap is well above the matched pair's `1e-9` tolerance.
    assert worst > 1.0
