from __future__ import annotations

import numpy as np
import pytest

from andrey.api._irregular import likelihood_output
from andrey.api.irregular import loci
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


def test_the_fit_ends_at_the_maximum():
    from sklearn.preprocessing import SplineTransformer

    x, y = datagen.generate("lsnm_pair").T
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    features = SplineTransformer(n_knots=25, degree=5).fit_transform(x.reshape(-1, 1))
    weights = _maximize(np.abs(features), y)
    gradient, _ = _gradient_hessian(weights, np.abs(features), y)
    dim = features.shape[1]
    # Karush-Kuhn-Tucker conditions of the bound w_2 >= 0: a zero gradient off the bound, and one
    # that points out of the feasible set on it.
    on_bound = np.r_[np.zeros(dim, dtype=bool), weights[dim:] == 0]
    assert np.abs(gradient[~on_bound]).max() < 1e-3
    assert gradient[on_bound].min() > -1e-3
