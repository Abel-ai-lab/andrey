"""RESIT recovers nonlinear additive-noise graphs: fixed cases, then seeded ground truth.

The fixed cases have one certain answer: independent columns give no edge, a cause with a
nonlinear effect and uniform noise points to it in either column order, and a collider of two
nonlinear causes is found with both edges directed into it. The ground truth is additive-noise data
from ``andrey.data`` on random graphs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm
from andrey.metrics import shd

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.resit(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, kind in out.structure.oriented_edges() if kind == "directed"}


def _pair(seed: int) -> np.ndarray:
    """``x -> y`` with ``y = sin(2 x) + x`` plus uniform noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, N)
    return np.column_stack([x, np.sin(2 * x) + x + rng.uniform(-0.5, 0.5, N)])


def _collider(seed: int) -> np.ndarray:
    """``x0 -> x2 <- x1`` with ``x2 = x0**2 + cos(2 x1)`` plus uniform noise."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.uniform(-2, 2, (2, N))
    return np.column_stack([x0, x1, x0**2 + np.cos(2 * x1) + rng.uniform(-0.5, 0.5, N)])


def test_independent_columns_give_no_edges():
    out = _fit(np.random.default_rng(0).standard_normal((N, 4)))
    assert out.structure.oriented_edges() == []
    assert sorted(out.ordering) == [0, 1, 2, 3]


@pytest.mark.parametrize("reverse", [False, True], ids=["cause first", "effect first"])
def test_a_nonlinear_cause_points_to_its_effect(reverse):
    X = _pair(1)
    out = _fit(X[:, ::-1] if reverse else X)
    cause, effect = (1, 0) if reverse else (0, 1)
    assert _edges(out) == {(cause, effect)}
    assert out.ordering == (cause, effect)


def test_a_collider_of_two_nonlinear_causes_is_found():
    out = _fit(_collider(2))
    assert _edges(out) == {(0, 2), (1, 2)}
    assert out.ordering[-1] == 2


@pytest.fixture(scope="module")
def additive_fits():
    """Five additive-noise graphs over 6 variables, 400 rows each, and RESIT's fits."""
    cases = [
        sample_scm(functional="additive_noise", d=6, n=N, seed=seed, density=2, scale="rescale")
        for seed in range(5)
    ]
    return [(case, _fit(case.data)) for case in cases]


def test_additive_noise_graphs_are_recovered(additive_fits):
    errors = [shd(out.structure, case.graph) for case, out in additive_fits]
    empty = [len(case.graph.oriented_edges()) for case, _ in additive_fits]
    assert all(e < m for e, m in zip(errors, empty, strict=True)), (errors, empty)
    assert np.median(errors) <= 2, errors


def test_a_fit_repeats(additive_fits):
    case, out = additive_fits[0]
    again = _fit(case.data)
    assert again.structure == out.structure
    assert again.ordering == out.ordering
