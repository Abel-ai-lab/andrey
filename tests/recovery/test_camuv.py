"""CAM-UV recovers nonlinear additive graphs with hidden variables: fixed cases, then seeded ground
truth.

The fixed cases have one certain answer: independent columns give no edge and no pair, a cause
with a nonlinear effect points to it in either column order, a hidden common cause of two
variables marks the pair and leaves the observed edges directed, the same model with that cause
observed marks no pair, and a hidden variable between a cause and its effect marks the pair. The
ground truth is additive-noise data from ``andrey.data`` on random graphs with no hidden variable,
and the recorded output is the marks of a nonlinear hidden fork.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

N = 1000


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.camuv(X, **kwargs)


def _edges(out: andrey.StructureOutput, kind: str = "directed") -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, k in out.structure.oriented_edges() if k == kind}


def _fork(seed: int) -> np.ndarray:
    """``x0 <- h -> x1``, ``x1 -> x2``, ``x0 -> x3`` with nonlinear effects and uniform noise; ``h``
    is the last column."""
    rng = np.random.default_rng(seed)
    h, *noise = rng.uniform(-1, 1, (5, N))
    x0 = np.sin(3 * h) + 0.3 * noise[0]
    x1 = np.cos(3 * h) + 0.3 * noise[1]
    return np.column_stack(
        [x0, x1, np.tanh(2 * x1) + 0.3 * noise[2], np.tanh(2 * x0) + 0.3 * noise[3], h]
    )


def test_independent_columns_give_no_edges():
    out = _fit(np.random.default_rng(0).standard_normal((N, 4)))
    assert out.structure.oriented_edges() == []
    assert out.metadata["latent_pairs"] == []


@pytest.mark.parametrize("reverse", [False, True], ids=["cause first", "effect first"])
def test_a_nonlinear_cause_points_to_its_effect(reverse):
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, N)
    X = np.column_stack([x, np.sin(2 * x) + x + rng.uniform(-0.5, 0.5, N)])
    out = _fit(X[:, ::-1] if reverse else X)
    assert _edges(out) == ({(1, 0)} if reverse else {(0, 1)})
    assert out.metadata["latent_pairs"] == []


def test_a_hidden_common_cause_marks_its_pair():
    out = _fit(_fork(2)[:, :4])
    assert _edges(out) == {(0, 3), (1, 2)}
    assert _edges(out, "circle") == {(0, 1)}
    assert out.metadata["latent_pairs"] == [[0, 1]]


def test_the_observed_common_cause_marks_no_pair():
    out = _fit(_fork(2))
    assert _edges(out) == {(0, 3), (1, 2), (4, 0), (4, 1)}
    assert out.metadata["latent_pairs"] == []


def test_a_hidden_variable_on_a_causal_path_marks_its_pair():
    """``x0 -> h -> x1`` with ``x1 = h**2`` plus noise: no function of ``x0`` removes the effect of
    ``h``'s own noise, so neither direction leaves an independent residual."""
    rng = np.random.default_rng(3)
    x0, *noise = rng.uniform(-1, 1, (3, N))
    h = x0 + 0.5 * noise[0]
    out = _fit(np.column_stack([x0, h**2 + 0.1 * noise[1]]))
    assert _edges(out) == set()
    assert out.metadata["latent_pairs"] == [[0, 1]]


@pytest.fixture(scope="module")
def additive_fits():
    """Five additive-noise graphs over 6 variables, 1,000 rows each, and CAM-UV's fits."""
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
    assert again.metadata == out.metadata


def test_camuv_matches_the_recorded_output():
    out = _fit(datagen.generate("nonlinear_fork_4v"))
    assert_recorded_output(out, "CAMUV", case="nonlinear_fork_4v")
