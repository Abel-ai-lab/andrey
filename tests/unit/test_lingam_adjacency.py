"""The LiNGAM adaptive Lasso selects the same parents for the data in any unit.

scikit-learn's LARS ends its Lasso path at a penalty of ``2**-23`` in the target's squared units,
and breaks for a target with a standard deviation near ``2**124``. The shared Lasso takes BIC over
the whole path at any scale, and returns the plain ``LassoLarsIC`` fit wherever that path is whole.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LassoLarsIC, LinearRegression

import andrey
import andrey.data as data
from andrey.core import TemporalStructure
from andrey.lingam.adjacency import predict_adaptive_lasso
from tests.unit.test_longitudinal import _panel

# 2**-128 to 2**126: the floor's end, unit scale, and the end where LARS's tiny divisor matters.
_SCALES = [-128, -64, -32, -10, 0, 10, 32, 64, 100, 124, 126]


def _nearly_noiseless(seed: int, n: int = 2000) -> np.ndarray:
    """Standardized ``x1``, ``x2 = 100 x1 + e2``, ``y = x1 + x2 + e`` with ``var(e) = 1e-7 var(y)``.

    The share of ``y``'s variance that ``x1`` explains beyond ``x2`` is about ``1e-8``, below the
    penalty at which LARS ends its path at unit scale; with so little noise, BIC keeps ``x1``.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = 100.0 * x1 + rng.uniform(-1.0, 1.0, n)
    signal = x1 + x2
    noise = rng.uniform(-1.0, 1.0, n)
    y = signal + noise * np.sqrt(1e-7 / (1.0 - 1e-7) * signal.var() / noise.var())
    X = np.column_stack([x1, x2, y])
    return (X - X.mean(axis=0)) / X.std(axis=0)


@pytest.mark.parametrize("seed", range(3))
def test_a_nearly_noiseless_target_keeps_both_true_parents_at_every_scale(seed):
    X = _nearly_noiseless(seed)
    for k in _SCALES:
        assert np.all(predict_adaptive_lasso(X * 2.0**k, [0, 1], 2) != 0), k


@pytest.mark.parametrize("seed", range(3))
def test_direct_lingam_returns_one_graph_for_the_data_in_any_unit(seed):
    # Strong weights on a dense graph leave some targets almost noiseless after standardizing.
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=10, avg_degree=4),
        functional=data.functional.linear(weight_range=(2.0, 5.0)),
        noise=data.noise.uniform(),
    )
    ds = scm.sample(n=500, seed=seed)
    graphs = [andrey.direct_lingam(ds.data * 2.0**k).weighted_adjacency != 0 for k in _SCALES]
    for k, graph in zip(_SCALES, graphs, strict=True):
        np.testing.assert_array_equal(graph, graphs[_SCALES.index(0)], err_msg=f"2**{k}")


def test_a_path_that_reaches_the_least_squares_fit_keeps_the_plain_lasso_fit():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (500, 4))
    X[:, 3] = X[:, :3] @ [0.8, 0.6, -0.5] + rng.uniform(-1.0, 1.0, 500)
    P = X[:, [0, 1, 2]]
    weight = np.abs(LinearRegression().fit(P, X[:, 3]).coef_)
    plain = LassoLarsIC(criterion="bic").fit(P * weight, X[:, 3])
    assert plain.alphas_[-1] == 0.0  # the path ran to the least-squares fit
    np.testing.assert_array_equal(predict_adaptive_lasso(X, [0, 1, 2], 3), plain.coef_ * weight)


def test_longitudinal_lingam_on_a_panel_near_the_largest_floats_keeps_its_graph():
    panel = _panel(3, n=100, p=4, T=3)

    def edges(scale: float) -> np.ndarray:
        structure = andrey.longitudinal_lingam([x * scale for x in panel]).structure
        assert isinstance(structure, TemporalStructure)
        return np.nan_to_num(np.asarray(structure.time_weights, dtype=np.float64)) != 0

    np.testing.assert_array_equal(edges(2.0**124), edges(1.0))
