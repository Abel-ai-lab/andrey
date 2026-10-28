from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.irregular.anm import anm
from andrey.irregular.pnl import pnl
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


# PNL draws no random state, so a fixed input pins its output. The tolerance is loose on purpose:
# these are p-values from a kernel statistic, and holding them to the last bit would gate on the
# host's SIMD kernels rather than on the method.
_PNL_CUBIC_PVALUES = (0.22597081473440506, 0.01989273405111902)
_ANM_CUBIC_PVALUES = (0.07269113995001186, 0.0)
# The columns of ``anm_pair_cubic``'s cause and effect.
_CAUSE, _EFFECT = (int(column) for column in datagen.column_of("anm_pair_cubic"))
_PLANTED = [(_CAUSE, _EFFECT, "directed")]


def _pnl_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One ground-truth post-nonlinear pair ``x -> y`` with standardized, non-Gaussian variables."""
    n = int(rng.integers(200, 351))

    def nongaussian(m: int) -> np.ndarray:
        kind = rng.integers(0, 3)
        if kind == 0:
            v = rng.uniform(-2.0, 2.0, m)
        elif kind == 1:
            v = rng.laplace(0.0, 1.0, m)
        else:
            u = rng.normal(0.0, 1.0, m)
            v = np.sign(u) * np.abs(u) ** 1.5
        return (v - v.mean()) / (v.std() + 1e-12)

    inner_maps = (
        lambda z: z**3 + z,
        lambda z: np.tanh(2 * z),
        lambda z: np.sinh(z),
        lambda z: z + 0.5 * np.sin(3 * z),
    )
    outer_maps = (lambda z: z**3 + z, lambda z: np.tanh(2 * z), lambda z: np.sinh(z))
    x = nongaussian(n)
    inner = inner_maps[int(rng.integers(0, len(inner_maps)))](x)
    inner = (inner - inner.mean()) / (inner.std() + 1e-12)
    noise = 0.5 * nongaussian(n)
    y = outer_maps[int(rng.integers(0, len(outer_maps)))](inner + noise)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return x, y


def test_pnl_resolves_the_planted_direction():
    x = datagen.generate("anm_pair_cubic")
    pval_forward, pval_backward, *_ = pnl(x[:, _CAUSE], x[:, _EFFECT])
    assert pval_forward > pval_backward  # anm_pair_cubic plants x -> y
    assert np.allclose((pval_forward, pval_backward), _PNL_CUBIC_PVALUES, atol=1e-6)


def test_pnl_matches_the_recorded_output():
    out = andrey.pnl(datagen.generate("anm_pair_cubic"))
    assert_recorded_output(out, "PNL", case="anm_pair_cubic")


def test_pnl_is_deterministic_over_varied_pairs():
    rng = np.random.default_rng(0)
    for _ in range(6):
        x, y = _pnl_pair(rng)
        first = pnl(x, y)
        assert all(0.0 <= p <= 1.0 for p in first[:2])
        assert pnl(x, y) == first, "PNL is not deterministic"


def test_pnl_output_contract():
    from andrey.api.irregular import pnl as pnl_facade

    x = datagen.generate("anm_pair_cubic")
    out = pnl_facade(x)
    assert out.structure.type == "graph"
    keys = ("pval_forward", "pval_backward", "stat_forward", "stat_backward")
    assert set(out.metadata) == {"algorithm", *keys}
    assert tuple(out.metadata[key] for key in keys) == pnl(x[:, 0], x[:, 1])


@pytest.mark.parametrize("method", ["anm", "pnl"])
def test_a_zero_pvalue_tie_is_broken_by_the_statistic(method):
    # Both coordinates of a noiseless circle are functions of its angle, so neither direction fits
    # and both p-values round to zero; the smaller HSIC statistic over its null mean decides.
    from andrey.api import irregular

    angle = np.pi * np.random.default_rng(0).uniform(-1, 1, 200)
    out = getattr(irregular, method)(np.column_stack([np.cos(angle), np.sin(angle)]))
    md = out.metadata
    assert md["pval_forward"] == md["pval_backward"] == 0.0
    assert md["stat_forward"] != md["stat_backward"]
    edge = (0, 1) if md["stat_forward"] < md["stat_backward"] else (1, 0)
    assert out.structure.oriented_edges() == [(*edge, "directed")]


def test_pnl_draws_no_edge_when_the_statistics_tie_too():
    # Two equal columns fit the same model to the same data in both directions.
    from andrey.api.irregular import pnl as pnl_facade

    x = datagen.generate("anm_pair_cubic")[:, 0]
    out = pnl_facade(np.column_stack([x, x]))
    md = out.metadata
    assert md["pval_forward"] == md["pval_backward"]
    assert md["stat_forward"] == md["stat_backward"]
    assert out.structure.oriented_edges() == []


def _pairs() -> dict[str, np.ndarray]:
    """Return three sample pairs: directed, unrelated, and linear Gaussian."""
    planted = datagen.generate("anm_pair_cubic")
    rng = np.random.default_rng(0)
    n = planted.shape[0]
    unrelated = np.column_stack([rng.laplace(size=n), rng.laplace(size=n)])
    g = rng.standard_normal(n)
    gaussian = np.column_stack([g, 0.8 * g + 0.6 * rng.standard_normal(n)])
    return {"planted": planted, "unrelated": unrelated, "gaussian": gaussian}


# The measured p-values, the planted pair's from cause to effect first, are (0.226, 0.020), (0.594,
# 0.298), and (0.881, 0.922): the comparison rule always draws an edge; the threshold rule draws one
# only for the planted pair.
@pytest.mark.parametrize(
    ("pair", "alpha", "edges"),
    [
        ("planted", None, _PLANTED),
        ("planted", 0.05, _PLANTED),
        ("unrelated", None, [(0, 1, "directed")]),
        ("unrelated", 0.05, []),
        ("gaussian", None, [(1, 0, "directed")]),
        ("gaussian", 0.05, []),
    ],
)
def test_pnl_decision_rules_on_measured_pairs(pair, alpha, edges):
    from andrey.api.irregular import pnl as pnl_facade

    assert pnl_facade(_pairs()[pair], alpha=alpha).structure.oriented_edges() == edges


def test_pnl_names_the_direction_whatever_the_column_order():
    import pandas as pd

    from andrey.api.irregular import pnl as pnl_facade

    names = [("cause", "effect")[v] for v in datagen.column_order("anm_pair_cubic")]
    df = pd.DataFrame(datagen.generate("anm_pair_cubic"), columns=names)
    for columns in (["cause", "effect"], ["effect", "cause"]):
        edges = pnl_facade(df[columns]).structure.oriented_edges()
        assert edges == [("cause", "effect", "directed")]


def test_anm_resolves_the_planted_direction():
    from andrey.api._irregular import _adapt_anm

    x = datagen.generate("anm_pair_cubic")
    pval_forward, pval_backward, stat_forward, stat_backward = anm(x[:, _CAUSE], x[:, _EFFECT])
    assert pval_forward > pval_backward
    # Loosely pinned as well: the direction alone would survive a numerical regression in the GP
    # residual path that still left the forward p-value above zero.
    assert np.allclose((pval_forward, pval_backward), _ANM_CUBIC_PVALUES, atol=1e-4)
    out = _adapt_anm(
        pval_forward, pval_backward, stat_forward=stat_forward, stat_backward=stat_backward
    )
    assert out.structure.type == "graph"
    assert out.structure.oriented_edges() == [(0, 1, "directed")]
    keys = {"pval_forward", "pval_backward", "stat_forward", "stat_backward"}
    assert set(out.metadata) == keys  # the method wrapper adds "algorithm"


def test_anm_matches_the_recorded_output():
    out = andrey.anm(datagen.generate("anm_pair_cubic"))
    assert_recorded_output(out, "ANM", case="anm_pair_cubic")


def test_anm_hyperparameter_objective_is_scikit_learns():
    """The search's objective is scikit-learn's log marginal likelihood, and so is its gradient."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    from andrey.irregular.anm import _kernel, _negative_log_marginal_likelihood

    x = datagen.generate("anm_pair_cubic")
    cause, effect = np.ascontiguousarray(x[:, [_CAUSE]]), np.ascontiguousarray(x[:, [_EFFECT]])
    gp = GaussianProcessRegressor(kernel=_kernel(), optimizer=None).fit(cause, effect)
    # The start, an interior point, and the signal variance at its floor (y unrelated to x).
    for theta in (gp.kernel_.theta, np.log([2.0, 0.3, 0.05]), np.log([1e-3, 1.0, 1.0])):
        value, gradient = gp.log_marginal_likelihood(theta, eval_gradient=True)
        negative, negative_gradient = _negative_log_marginal_likelihood(theta, cause, effect)
        assert -negative == value
        np.testing.assert_allclose(-negative_gradient, gradient, rtol=1e-9, atol=1e-9)


def test_anm_matches_scikit_learns_own_search():
    """The p-values of scikit-learn's default L-BFGS-B fit, up to rounding."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor

    from andrey.core.independence import hsic_test
    from andrey.irregular.anm import _kernel

    x = datagen.generate("anm_pair_cubic")
    expected = []
    for cause, effect in ((x[:, :1], x[:, 1:]), (x[:, 1:], x[:, :1])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=_kernel()).fit(cause, effect)
        expected.append(hsic_test(cause, effect - gp.predict(cause).reshape(-1, 1)))
    assert np.allclose(anm(x[:, 0], x[:, 1])[:2], expected, rtol=0, atol=1e-8)


def test_anm_names_the_planted_direction_in_any_units_and_column_order():
    import pandas as pd

    from andrey.api.irregular import anm as anm_facade

    planted = datagen.generate("anm_pair_cubic")
    first = anm_facade(planted)
    assert first.structure.oriented_edges() == _PLANTED
    keys = {"pval_forward", "pval_backward", "stat_forward", "stat_backward"}
    assert set(first.metadata) == {"algorithm", *keys}
    pvalues = (first.metadata["pval_forward"], first.metadata["pval_backward"])
    for scale in (100.0, 0.01, 1e-200):  # 1e-200: squared deviations would underflow
        out = anm_facade(planted * np.array([scale, 1 / scale]))
        assert np.allclose((out.metadata["pval_forward"], out.metadata["pval_backward"]), pvalues)
    names = [("cause", "effect")[v] for v in datagen.column_order("anm_pair_cubic")]
    df = pd.DataFrame(planted, columns=names)
    for columns in (["cause", "effect"], ["effect", "cause"]):
        edges = anm_facade(df[columns]).structure.oriented_edges()
        assert edges == [("cause", "effect", "directed")]


# The measured p-values, the planted pair's from cause to effect first, are (0.057, 0.0), (0.818,
# 0.478), and (0.795, 0.785): the comparison rule draws an edge between unrelated variables; the
# threshold rule draws one only for the planted pair.
@pytest.mark.parametrize(
    ("pair", "alpha", "edges"),
    [
        ("planted", 0.02, _PLANTED),
        ("unrelated", None, [(0, 1, "directed")]),
        ("unrelated", 0.05, []),
        ("gaussian", 0.05, []),
    ],
)
def test_anm_decision_rules_on_measured_pairs(pair, alpha, edges):
    from andrey.api.irregular import anm as anm_facade

    assert anm_facade(_pairs()[pair], alpha=alpha).structure.oriented_edges() == edges
