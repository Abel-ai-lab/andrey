from __future__ import annotations

import numpy as np
import pytest

from andrey.api._irregular import likelihood_output
from andrey.api.irregular import _pair_matrix, loci
from andrey.data import sample_scm
from andrey.irregular.loci import _gradient_hessian, _maximize
from tests.recovery.helpers import datagen


def test_alpha_is_refused_under_the_likelihood_criterion():
    with pytest.raises(ValueError, match="alpha must be None under criterion='likelihood'"):
        loci(datagen.generate("lsnm_pair"), alpha=0.05, criterion="likelihood")


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="criterion"):
        loci(datagen.generate("lsnm_pair"), criterion="hsic")


def test_each_criterion_returns_its_metadata():
    pair = datagen.generate("lsnm_pair")
    independence = loci(pair).metadata
    likelihood = loci(pair, criterion="likelihood").metadata
    assert {"p_values", "statistics", "log_likelihoods"} <= independence.keys()
    assert "p_values" not in likelihood
    assert likelihood["log_likelihoods"] == independence["log_likelihoods"]


def test_equal_likelihoods_give_no_edge():
    assert likelihood_output(-1.0, -1.0).structure.oriented_edges() == []
    assert likelihood_output(-1.0, -2.0).structure.oriented_edges() == [(0, 1, "directed")]


@pytest.mark.parametrize("seed", [None, 16, 24, 27])
def test_the_fit_ends_at_the_maximum(seed):
    from sklearn.preprocessing import SplineTransformer

    if seed is None:
        pair = datagen.generate("lsnm_pair")
    else:
        pair = sample_scm(
            functional="location_scale", d=2, n=500, seed=seed, density=1.0, scale="rescale"
        ).data
    columns, _, _ = _pair_matrix("loci", pair, standardize=True)
    for cause, effect in (columns.T, columns.T[::-1]):
        features = np.abs(SplineTransformer(n_knots=25, degree=5).fit_transform(cause[:, None]))
        weights = _maximize(features, effect)
        gradient, _ = _gradient_hessian(weights, features, effect)
        dim = features.shape[1]
        # Karush-Kuhn-Tucker conditions of the bound w_2 >= 0: a zero gradient off the bound and a
        # nonnegative one on it, so that no feasible move lowers the objective.
        on_bound = np.r_[np.zeros(dim, dtype=bool), weights[dim:] == 0]
        assert np.abs(gradient[~on_bound]).max() < 1e-5
        assert gradient[on_bound].min(initial=0.0) > -1e-5
