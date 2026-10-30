"""``fdr_method`` on the methods that test a family of lagged links (``pcmci``, ``pcmciplus``,
``granger``): ``"fdr_bh"``, the call without the argument, adjusts the lagged links' p-values over
the links tested, by the Benjamini-Hochberg procedure, which is computed here by SciPy, and
``"none"`` leaves them as tested."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import andrey
from andrey import Knowledge
from andrey.core.fdr import FDR_METHODS, benjamini_hochberg, corrected_pvalues

TAU_MAX = 2
N_VARS = 5
ALPHA = 0.05


def series(seed: int, n_steps: int = 300) -> np.ndarray:
    """Five variables with a few lagged links, some weak enough for a correction to remove."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_steps, N_VARS))
    e = rng.standard_normal((n_steps, N_VARS))
    for t in range(2, n_steps):
        x[t] = e[t]
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += 0.4 * x[t - 1, 1] + 0.45 * x[t - 1, 0]
        x[t, 2] += 0.3 * x[t - 2, 1] + 0.1 * x[t - 1, 2]
        x[t, 3] += 0.18 * x[t - 1, 2] + 0.15 * x[t - 1, 0]
        x[t, 4] += 0.14 * x[t - 2, 3]
    return x


DATA = series(0)
METHODS = ("pcmci", "pcmciplus", "granger")
KNOWLEDGE = Knowledge(required=[(0, 1, 1)], forbidden=[(4, 0), (2, 4)])


def fit(method: str, **kwargs):
    if method == "granger":
        return andrey.granger(DATA, tau_max=TAU_MAX, alpha=ALPHA, **kwargs)
    alpha = {"alpha": ALPHA} if method == "pcmci" else {"alpha": 0.01}
    return getattr(andrey, method)(DATA, tau_max=TAU_MAX, indep_test="student_t", **alpha, **kwargs)


def lagged(out) -> list[np.ndarray]:
    """The p-values of lags 1 to ``tau_max``, each ``[cause, effect]``."""
    p = np.asarray(out.metadata["p_values"])
    return p if len(p) == TAU_MAX else p[1:]


def graphs(out) -> list[np.ndarray]:
    """The mark matrix of every lag graph."""
    return [out.structure.lag(k).to_numpy() for k in out.structure.lags]


def links(out) -> set[tuple[int, int, int]]:
    """The lagged links ``(cause, effect, lag)`` of the graph."""
    return {
        (i, j, k)
        for k in range(1, TAU_MAX + 1)
        for i, j, _ in out.structure.lag(k).oriented_edges()
    }


def test_the_choices_are_the_two_tigramite_names():
    assert FDR_METHODS == ("none", "fdr_bh")


@pytest.mark.parametrize("seed", range(4))
def test_benjamini_hochberg_is_the_scipy_adjustment(seed):
    rng = np.random.default_rng(seed)
    p = np.round(rng.uniform(size=50) ** 2, 2)  # rounded, so some values tie
    np.testing.assert_allclose(
        benjamini_hochberg(p), stats.false_discovery_control(p, method="bh"), rtol=1e-12
    )


def test_benjamini_hochberg_on_a_hand_case():
    # sorted 0.005, 0.01, 0.03, 0.04, 0.5 -> p * 5 / rank = 0.025, 0.025, 0.05, 0.05, 0.5
    p = np.array([0.04, 0.01, 0.5, 0.03, 0.005])
    expected = np.array([0.05, 0.025, 0.5, 0.05, 0.025])
    np.testing.assert_allclose(benjamini_hochberg(p), expected)


def test_an_excluded_entry_is_not_counted_and_keeps_its_value():
    p = np.array([[0.04, 0.01], [1.0, 0.03]])
    excluded = np.array([[False, False], [True, False]])
    out = corrected_pvalues(p, "fdr_bh", excluded)
    assert out[1, 0] == 1.0
    np.testing.assert_allclose(out[~excluded], stats.false_discovery_control(p[~excluded]))
    assert p[0, 0] == 0.04  # the input is not changed


def test_none_returns_the_p_values_themselves():
    p = np.array([0.2, 0.01])
    assert corrected_pvalues(p, "none") is p


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("knowledge", [None, KNOWLEDGE])
def test_fdr_bh_is_the_call_without_the_argument(method, knowledge):
    plain = fit(method, knowledge=knowledge)
    named = fit(method, knowledge=knowledge, fdr_method="fdr_bh")
    assert all(map(np.array_equal, graphs(plain), graphs(named)))
    assert plain.metadata.keys() == named.metadata.keys()
    for key in plain.metadata.keys() & {"p_values", "statistics"}:
        assert np.array_equal(plain.metadata[key], named.metadata[key])
    if method == "granger":
        assert np.array_equal(plain.structure.lag_weights, named.structure.lag_weights)


@pytest.mark.parametrize("method", METHODS)
def test_an_unknown_correction_raises_before_any_test_runs(method):
    with pytest.raises(ValueError, match="fdr_method"):
        fit(method, fdr_method="bonferroni")


def test_pcmci_adjusts_the_p_values_of_all_its_links_together():
    raw, adjusted = fit("pcmci", fdr_method="none"), fit("pcmci")
    expected = stats.false_discovery_control(np.asarray(raw.metadata["p_values"]).ravel())
    np.testing.assert_allclose(
        np.asarray(adjusted.metadata["p_values"]).ravel(), expected, rtol=1e-12
    )
    assert np.array_equal(raw.metadata["statistics"], adjusted.metadata["statistics"])
    assert raw.metadata["parents"] == adjusted.metadata["parents"]  # the PC1 step is not corrected


def test_pcmci_keeps_the_links_whose_adjusted_p_value_is_at_most_alpha():
    raw, adjusted = fit("pcmci", fdr_method="none"), fit("pcmci")
    p = np.asarray(adjusted.metadata["p_values"])
    expected = {(i, j, k + 1) for k, i, j in np.argwhere(p <= ALPHA)}
    assert links(adjusted) == expected
    assert links(adjusted) < links(raw)  # a correction removes links on this series, adds none


@pytest.mark.parametrize("method", METHODS)
def test_an_adjusted_p_value_is_never_below_the_tested_one(method):
    raw, adjusted = fit(method, fdr_method="none"), fit(method)
    assert np.all(lagged(adjusted) >= lagged(raw))
    assert np.all(lagged(adjusted) <= 1.0)


def test_pcmciplus_adjusts_the_lagged_p_values_only_and_keeps_its_links():
    raw, adjusted = fit("pcmciplus", fdr_method="none"), fit("pcmciplus")
    p_raw, p_adjusted = (
        np.asarray(raw.metadata["p_values"]),
        np.asarray(adjusted.metadata["p_values"]),
    )
    assert np.array_equal(p_raw[0], p_adjusted[0])  # the same-step p-values stay as tested
    np.testing.assert_allclose(
        p_adjusted[1:].ravel(), stats.false_discovery_control(p_raw[1:].ravel()), rtol=1e-12
    )
    assert not np.array_equal(p_raw[1:], p_adjusted[1:])
    assert all(map(np.array_equal, graphs(raw), graphs(adjusted)))
    assert np.array_equal(raw.metadata["statistics"], adjusted.metadata["statistics"])


@pytest.mark.parametrize("method", METHODS)
def test_a_forbidden_link_is_not_counted_and_a_required_one_is(method):
    raw = fit(method, knowledge=KNOWLEDGE, fdr_method="none")
    adjusted = fit(method, knowledge=KNOWLEDGE)
    p_raw, p_adjusted = lagged(raw), lagged(adjusted)
    forbidden = np.zeros(p_raw.shape, dtype=bool)
    forbidden[:, 4, 0] = forbidden[:, 2, 4] = True
    assert np.all(p_adjusted[forbidden] == 1.0)
    assert p_adjusted[0, 0, 1] == 0.0  # required: p-value 0, counted among the links
    expected = stats.false_discovery_control(p_raw[~forbidden])
    np.testing.assert_allclose(p_adjusted[~forbidden], expected, rtol=1e-12)


def test_granger_adjusts_the_p_values_of_all_its_links_together():
    raw, adjusted = fit("granger", fdr_method="none"), fit("granger")
    expected = stats.false_discovery_control(np.asarray(raw.metadata["p_values"]).ravel())
    np.testing.assert_allclose(
        np.asarray(adjusted.metadata["p_values"]).ravel(), expected, rtol=1e-12
    )


def test_granger_keeps_the_links_whose_adjusted_p_value_is_at_most_alpha():
    raw, adjusted = fit("granger", fdr_method="none"), fit("granger")
    p = np.asarray(adjusted.metadata["p_values"])
    assert links(adjusted) == {(i, j, k + 1) for k, i, j in np.argwhere(p <= ALPHA)}
    assert links(adjusted) < links(raw)  # a correction removes links on this series, adds none
    for weights_raw, weights_adjusted in zip(
        raw.structure.lag_weights, adjusted.structure.lag_weights, strict=True
    ):
        kept = weights_adjusted != 0
        assert np.array_equal(weights_adjusted[kept], weights_raw[kept])  # the fit is not changed


def test_granger_on_noise_returns_the_links_that_chance_adds_only_without_a_correction():
    noise = np.random.default_rng(1).standard_normal((1000, 20))
    raw = andrey.granger(noise, tau_max=TAU_MAX, fdr_method="none")
    adjusted = andrey.granger(noise, tau_max=TAU_MAX)
    n_links = 20 * 20 * TAU_MAX
    assert len(links(raw)) > ALPHA * n_links / 2  # about alpha of the absent links appear
    assert len(links(adjusted)) <= 1
