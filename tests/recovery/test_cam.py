"""CAM recovers nonlinear additive-noise graphs: fixed cases, then seeded ground truth.

The fixed cases have one certain answer: independent columns give no edge, a cause with a
nonlinear effect points to it, and a collider of two nonlinear causes is found with both edges
directed into it. The ground truth is additive-noise data from ``andrey.data`` on a random graph.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

N = 500


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, kind in out.structure.oriented_edges() if kind == "directed"}


def _pair(seed: int) -> np.ndarray:
    """``x -> y`` with ``y = sin(2 x) + x`` plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, N)
    return np.column_stack([x, np.sin(2 * x) + x + 0.3 * rng.standard_normal(N)])


def _collider(seed: int) -> np.ndarray:
    """``x0 -> x2 <- x1`` with ``x2 = x0**2 + cos(2 x1)`` plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.uniform(-2, 2, (2, N))
    return np.column_stack([x0, x1, x0**2 + np.cos(2 * x1) + 0.3 * rng.standard_normal(N)])


def test_independent_columns_give_no_edges():
    out = andrey.cam(np.random.default_rng(0).standard_normal((N, 4)))
    assert out.structure.oriented_edges() == []
    assert sorted(out.ordering) == [0, 1, 2, 3]


@pytest.mark.parametrize("reverse", [False, True], ids=["cause first", "effect first"])
def test_a_nonlinear_cause_points_to_its_effect(reverse):
    X = _pair(1)
    out = andrey.cam(X[:, ::-1] if reverse else X)
    cause, effect = (1, 0) if reverse else (0, 1)
    assert _edges(out) == {(cause, effect)}
    assert out.ordering == (cause, effect)


def test_a_collider_of_two_nonlinear_causes_is_found():
    out = andrey.cam(_collider(2))
    assert _edges(out) == {(0, 2), (1, 2)}
    assert out.ordering[-1] == 2


def test_an_additive_noise_graph_is_recovered():
    """A random additive-noise graph over 10 variables and 500 rows: CAM finds it to within 3
    edits, where the empty graph is 10 away."""
    case = sample_scm(functional="additive_noise", d=10, n=N, seed=0, density=2, scale="rescale")
    assert shd(andrey.cam(case.data).structure, case.graph) <= 3


def test_cam_matches_the_recorded_output():
    out = andrey.cam(datagen.generate("nonmonotone_4v"))
    assert_recorded_output(out, "CAM", case="nonmonotone_4v")
