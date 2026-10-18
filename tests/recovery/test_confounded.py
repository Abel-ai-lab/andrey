"""Quality and behavior checks for RCD and BottomUpParceLiNGAM.

Recovery tests run the public functions on SEMs where a hidden common cause drives two observed
variables that share no direct edge. With a direct edge as well (``confounded_6v``), whether a
method marks the pair or orients the edge depends on its tests' power. The planted pair and
the direct edges come from ``datagen.planted_structure``. Each case has a control with the same
graph and the hidden variable observed, where no pair is confounded. RCD and BottomUpParceLiNGAM
also match their recorded outputs on the linear case. Random models check execution,
repeatability, and well-formed output; recovery rates are measured separately because they depend
on sample size and regime.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


# --- confounder recovery ----------------------------------------------------------------------
#
# Linear cases (2000 rows, uniform noise) serve RCD and BottomUpParceLiNGAM. Each test draws three
# independent samples of its case (replicates 0-2).

_REPLICATES = (0, 1, 2)
_FORK = "confounded_fork_4v"
_FORK_OBSERVED = "fork_5v_observed"


def _assert_pairs_and_edges(out, case: str, mark: str, *, exact_edges: bool = True) -> None:
    """Check ``out`` against the planted structure of ``case``.

    The confounded pairs match the structure's ``mark`` edges, and ``metadata`` where the method
    lists them; no other kind of edge appears. The planted direct edges
    are all present, and with ``exact_edges`` no other direct edge appears.
    """
    edges, pairs = datagen.planted_structure(case)
    found = out.structure.oriented_edges()
    if mark == "bidirected":  # BottomUpParceLiNGAM lists no pairs: its o-o block is unordered
        assert out.metadata["confounded_pairs"] == [list(pair) for pair in sorted(pairs)]
    assert {(i, j) for i, j, kind in found if kind == mark} == pairs
    assert {kind for _i, _j, kind in found} <= {"directed", mark}
    directed = {(i, j) for i, j, kind in found if kind == "directed"}
    assert directed == edges if exact_edges else directed >= edges


@pytest.mark.parametrize("replicate", _REPLICATES)
def test_rcd_reports_the_confounded_pair(replicate):
    out = andrey.rcd(datagen.generate(_FORK, replicate=replicate))
    _assert_pairs_and_edges(out, _FORK, "bidirected")


@pytest.mark.parametrize("replicate", _REPLICATES)
def test_rcd_reports_no_pair_when_the_cause_is_observed(replicate):
    out = andrey.rcd(datagen.generate(_FORK_OBSERVED, replicate=replicate))
    _assert_pairs_and_edges(out, _FORK_OBSERVED, "bidirected")


# BottomUpParceLiNGAM runs at alpha=0.01: at the default 0.1 its Bonferroni-corrected sink test
# sometimes rejects a true sink by chance, which leaves extra variables unordered. Its adaptive
# Lasso can add a weak direct edge, so the checks require the planted edges and not an exact set.


@pytest.mark.parametrize("replicate", _REPLICATES)
def test_bottom_up_parce_lingam_leaves_the_confounded_pair_unordered(replicate):
    out = andrey.bottom_up_parce_lingam(datagen.generate(_FORK, replicate=replicate), alpha=0.01)
    _assert_pairs_and_edges(out, _FORK, "circle", exact_edges=False)
    _edges, pairs = datagen.planted_structure(_FORK)
    unordered = [set(item) for item in out.metadata["causal_order"] if isinstance(item, list)]
    assert unordered == [set(pair) for pair in sorted(pairs)]


@pytest.mark.parametrize("replicate", _REPLICATES)
def test_bottom_up_parce_lingam_orders_everything_when_the_cause_is_observed(replicate):
    x = datagen.generate(_FORK_OBSERVED, replicate=replicate)
    out = andrey.bottom_up_parce_lingam(x, alpha=0.01)
    _assert_pairs_and_edges(out, _FORK_OBSERVED, "circle", exact_edges=False)
    assert not any(isinstance(item, list) for item in out.metadata["causal_order"])


def test_rcd_matches_its_recorded_output():
    out = andrey.rcd(datagen.generate(_FORK))
    assert_recorded_output(out, "RCD", case=_FORK)


def test_bottom_up_parce_lingam_matches_its_recorded_output():
    out = andrey.bottom_up_parce_lingam(datagen.generate(_FORK), alpha=0.01)
    assert_recorded_output(out, "BottomUpParceLiNGAM", case=_FORK)


def test_a_planted_pair_has_no_direct_edge():
    """No planted pair is also a direct edge, so each method's answer for the pair is stable."""
    edges, pairs = datagen.planted_structure(_FORK)
    assert pairs
    assert all((i, j) not in edges and (j, i) not in edges for i, j in pairs)
    assert not datagen.planted_structure(_FORK_OBSERVED)[1]


def test_confounded_6v_hides_its_pair_behind_a_direct_edge():
    """``confounded_6v`` hides its confounded pair behind a direct edge, so no pair is planted."""
    edges, pairs = datagen.planted_structure("confounded_6v")
    assert (1, 2) in edges
    assert not pairs


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
