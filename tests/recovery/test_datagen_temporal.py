"""Validate seeded inputs and capture wiring independently of the committed baselines."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.data.sortability import index_correlation
from qa import capture_baselines as capture
from tests.recovery.helpers import datagen
from tests.recovery.helpers.tolerances import SPEC

# Temporal / irregular cases, with their expected (n_samples, d) shape.
NEW_CASES = {
    "var_4v_stable": (500, 4),
    "svar_4v_gauss": (500, 4),
    "varma_3v": (500, 3),
    "var_3v_lag2": (1000, 3),
    "anm_pair_cubic": (200, 2),
    "confounded_6v": (300, 6),
}


@pytest.mark.parametrize(("case_id", "shape"), NEW_CASES.items(), ids=list(NEW_CASES))
def test_new_case_shape(case_id, shape):
    X = datagen.generate(case_id)
    assert X.shape == shape
    assert X.dtype == np.float64
    assert np.isfinite(X).all(), "generator produced non-finite values"


def test_var_processes_are_stationary():
    """The VAR/VARMA reduced-form spectral radius is < 1 (bounded, stationary)."""
    for case_id in ("var_4v_stable", "svar_4v_gauss", "varma_3v"):
        spec = datagen.CASES[case_id]
        B0 = np.asarray(spec["B0"], dtype=np.float64)
        A1 = np.asarray(spec["A1"], dtype=np.float64)
        d = B0.shape[0]
        reduced = np.linalg.inv(np.eye(d) - B0) @ A1
        assert np.max(np.abs(np.linalg.eigvals(reduced))) < 1.0


def test_confounder_is_dropped():
    spec = datagen.CASES["confounded_6v"]
    assert 0 not in spec["observed"], "latent column 0 must not be observed"
    assert datagen.generate("confounded_6v").shape[1] == len(spec["observed"])


# --- capture <-> tolerances wiring ---------------------------------------------


def test_every_algorithm_has_a_spec_and_case():
    for name, (_runner, case) in capture.ALGORITHMS.items():
        assert name in SPEC, f"{name} missing a tolerances.SPEC entry"
        assert case in datagen.CASES, f"{name} references unknown case {case!r}"


def test_every_spec_entry_compares_something():
    for name, spec in SPEC.items():
        assert set(spec["structural"]) or set(spec["numeric"]), f"{name} would compare nothing"


# --- column order ------------------------------------------------------------------------------

#: Replicates whose column orders the test draws; only the permutations are drawn, not the data.
_REPLICATES = 500


def _correlation(case_id: str, order: np.ndarray) -> float:
    """The rank correlation of column index and causal depth, column ``j`` holding ``order[j]``."""
    k, d, edges = datagen.defined_edges(case_id)
    node = np.concatenate([np.argsort(order), np.arange(k, d)])  # the column of each variable
    return index_correlation(node[edges], d, n_observed=k)


_ORDERED = [
    case
    for case in (*datagen.CASES, *datagen.GROUP_CASES)
    if np.isfinite(_correlation(case, np.arange(datagen.defined_edges(case)[0])))
]


@pytest.mark.parametrize("case_id", _ORDERED)
def test_a_fixture_column_order_says_nothing_of_the_causal_order(case_id):
    """Each case defines its variables in causal order; its columns come in an order drawn from its
    seed. Under a uniform order the rank correlation of column index and causal depth has mean 0
    and variance ``1 / (k - 1)`` on ``k`` columns, so over the replicates' orders its mean stays
    within four standard deviations of 0. The order the case defines, the positive control, is
    outside that bound."""
    k = datagen.defined_edges(case_id)[0]
    bound = 4 / np.sqrt(_REPLICATES * (k - 1))
    drawn = [
        _correlation(case_id, datagen.column_order(case_id, replicate=r))
        for r in range(_REPLICATES)
    ]
    assert abs(np.mean(drawn)) < bound
    assert abs(_correlation(case_id, np.arange(k))) > bound


def test_the_input_lists_the_variables_in_the_drawn_order():
    """The shuffle moves whole columns: column ``j`` is the defined order's ``column_order[j]``."""
    for case_id in ("gauss_5v", "var_6v_hidden", "measurement_6v"):
        order = datagen.column_order(case_id)
        plain = datagen.generate(case_id, shuffle=False)
        np.testing.assert_array_equal(datagen.generate(case_id), plain[:, order])
    groups = datagen.generate_groups("mgdl_2groups")
    plain = datagen.generate_groups("mgdl_2groups", shuffle=False)
    order = datagen.column_order("mgdl_2groups")
    assert all(np.array_equal(g, p[:, order]) for g, p in zip(groups, plain, strict=True))
