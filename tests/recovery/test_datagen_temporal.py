"""Validate seeded inputs and capture wiring independently of the committed baselines."""

from __future__ import annotations

import numpy as np
import pytest

from qa import capture_baselines as capture
from tests.recovery.helpers import datagen
from tests.recovery.helpers.tolerances import SPEC

# Temporal / irregular cases, with their expected (n_samples, d) shape.
NEW_CASES = {
    "var_4v_stable": (500, 4),
    "varma_3v": (500, 3),
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
    for case_id in ("var_4v_stable", "varma_3v"):
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
