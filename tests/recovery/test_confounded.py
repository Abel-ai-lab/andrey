"""Quality and behavior checks for CAM-UV, RCD, and BottomUpParceLiNGAM.

Fixed cases assert exact, deterministic answers against known SEMs. Random models check execution,
repeatability, and well-formed output; recovery rates are measured separately because they depend
on sample size and regime.

CAM-UV and BottomUpParceLiNGAM do not report the planted confounder in the fixed family.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.recovery.helpers import datagen


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


# The observed parent sets exclude the latent row. The planted pair also has a direct edge, so
# methods that skip adjacent pairs cannot identify it as confounded. RCD flags the pair only when it
# misses that edge, which makes a fixed confounder assertion unstable.
_CONFOUNDED_6V_PARENTS = [[], [], [1], [0], [2], [3, 4]]


def test_camuv_recovers_the_observed_structure():
    """CAM-UV recovers every observed parent set on ``confounded_6v``.

    The expected parent sets come from the planted SEM. CAM-UV does not report its confounded pair.
    """
    from andrey.irregular.camuv import camuv

    x = datagen.generate("confounded_6v")
    parents, _confounded = camuv(x, alpha=0.01, max_explanatory_vars=3)
    assert [sorted(p) for p in parents] == _CONFOUNDED_6V_PARENTS


def test_confounded_output_contract():
    from andrey.irregular.bottom_up_parce import bottom_up_parce_lingam
    from andrey.irregular.rcd import fit_rcd

    x = datagen.generate("confounded_6v")
    d = x.shape[1]
    assert fit_rcd(x).adjacency_matrix_.shape == (d, d)
    assert bottom_up_parce_lingam(x).adjacency_matrix_.shape == (d, d)


def test_hsic_gamma_separates_dependence_from_independence():
    """Check gamma HSIC on seeded independent and nonlinear dependent samples.

    Repeated calls on the same input must return the same result.
    """
    from andrey.core.independence import hsic_gamma_test

    rng = np.random.default_rng(0)
    for _ in range(6):
        a = rng.standard_normal(200)
        independent = rng.standard_normal(200)
        dependent = a**2 + 0.3 * rng.standard_normal(200)
        _stat, p_independent = hsic_gamma_test(a, independent)
        _stat, p_dependent = hsic_gamma_test(a, dependent)
        # One seeded null p-value falls below 0.05; the dependent p-values are zero.
        assert p_independent > 0.01, f"independent draws rejected at p={p_independent}"
        assert p_dependent < 0.01, f"quadratic dependence missed at p={p_dependent}"
    assert hsic_gamma_test(a, dependent) == hsic_gamma_test(a, dependent)


def _confounded_sem(seed: int, n: int = 250, d: int = 5) -> np.ndarray:
    """Sample a linear non-Gaussian SEM with one latent confounder.

    Uniform noise makes the model LiNGAM-identifiable.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    coef = np.zeros((d, d))
    for pos in range(d):
        for prev in range(pos):
            if rng.random() < 0.4:
                coef[order[pos], order[prev]] = rng.uniform(0.3, 1.5) * rng.choice([-1.0, 1.0])
    latent = rng.uniform(-1.0, 1.0, n)
    confounded = rng.choice(d, 2, replace=False)
    noise = rng.uniform(-1.0, 1.0, (n, d))
    data = np.zeros((n, d))
    for pos, idx in enumerate(order):
        col = noise[:, idx].copy()
        for prev in range(pos):
            col += coef[idx, order[prev]] * data[:, order[prev]]
        if idx in confounded:
            col += 0.8 * latent
        data[:, idx] = col
    return data


# --- general-input behavior -----------------------------------------------------------------------
#
# Random models check execution, repeatability, shape, and relabel invariance. RCD is invariant
# bit-for-bit. BottomUpParceLiNGAM preserves its causal order and non-identifiable mask exactly, but
# column-mean reduction introduces roundoff in its weights, so those use a tolerance.


def _stable(adjacency: np.ndarray) -> np.ndarray:
    """Replace non-identifiable ``NaN`` entries with a stable comparison value."""
    return np.nan_to_num(np.asarray(adjacency, dtype=np.float64), nan=-9.0)


def test_rcd_is_deterministic_and_order_invariant():
    """RCD repeats itself exactly and does not depend on the order of the columns."""
    from andrey.irregular.rcd import fit_rcd

    for seed in range(6):
        x = _confounded_sem(seed)
        first = _stable(fit_rcd(x).adjacency_matrix_)
        assert first.shape == (x.shape[1], x.shape[1])
        assert np.array_equal(_stable(fit_rcd(x).adjacency_matrix_), first), "not deterministic"

        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = _stable(fit_rcd(x[:, order]).adjacency_matrix_)
        assert np.array_equal(permuted[np.ix_(back, back)], first), "depends on column order"


def test_bottom_up_parce_is_deterministic_and_order_invariant():
    """Check BottomUpParceLiNGAM determinism and relabel invariance.

    The non-identifiable mask matches exactly. Weights allow reduction-order roundoff.
    """
    from andrey.irregular.bottom_up_parce import bottom_up_parce_lingam

    for seed in range(6):
        x = _confounded_sem(seed)
        first = _stable(bottom_up_parce_lingam(x).adjacency_matrix_)
        assert first.shape == (x.shape[1], x.shape[1])
        assert np.array_equal(_stable(bottom_up_parce_lingam(x).adjacency_matrix_), first), (
            "not deterministic"
        )

        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = _stable(bottom_up_parce_lingam(x[:, order]).adjacency_matrix_)[
            np.ix_(back, back)
        ]
        assert np.array_equal(np.isnan(permuted), np.isnan(first)), "non-identifiable mask moved"
        assert np.allclose(permuted, first, atol=1e-8), "answer depends on column order"
