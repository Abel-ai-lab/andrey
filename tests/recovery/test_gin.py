"""Test GIN cluster recovery and latent order against a known measurement model.

Clustering can drift across numeric libraries, so no baseline is committed.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import LATENT, OBSERVED, GraphStructure, StructureOutput


def _two_latent_model() -> tuple[np.ndarray, list[list[int]]]:
    """A known two-latent chain L1 -> L2, each with three pure non-Gaussian indicators.

    Returns the ``(n, 6)`` sample matrix and the true observed-variable clustering; L1 is the root,
    so the true causal order is cluster ``[0, 1, 2]`` before cluster ``[3, 4, 5]``.
    """
    rng = np.random.default_rng(1)
    n = 1500

    def ng() -> np.ndarray:  # super-Gaussian source (LiNGAM identifiability)
        g = rng.standard_normal(n)
        return np.sign(g) * np.abs(g) ** 1.5

    latent1 = ng()
    latent2 = 0.9 * latent1 + 0.4 * ng()
    columns = [
        1.0 * latent1 + 0.3 * ng(),
        0.9 * latent1 + 0.3 * ng(),
        1.1 * latent1 + 0.3 * ng(),
        0.8 * latent2 + 0.3 * ng(),
        1.2 * latent2 + 0.3 * ng(),
        0.9 * latent2 + 0.3 * ng(),
    ]
    return np.column_stack(columns), [[0, 1, 2], [3, 4, 5]]


@pytest.fixture(scope="module")
def fit() -> tuple[np.ndarray, StructureOutput]:
    data, _ = _two_latent_model()
    return data, andrey.gin(data, alpha=0.05)


def test_gin_recovers_latent_clusters(fit) -> None:
    _, out = fit
    _, truth = _two_latent_model()

    clusters = {frozenset(c) for c in out.metadata["clusters"]}
    assert clusters == {frozenset(c) for c in truth}, out.metadata["clusters"]

    order = out.metadata["causal_order"]
    assert [sorted(c) for c in order] == [[0, 1, 2], [3, 4, 5]], order

    s = out.structure
    assert isinstance(s, GraphStructure)
    assert s.kind == "dag"
    assert s.n_nodes == 8  # six observed indicators + two discovered latents
    assert s.node_types is not None
    assert list(s.node_types) == [OBSERVED] * 6 + [LATENT] * 2
    assert s.labels == ("X1", "X2", "X3", "X4", "X5", "X6", "L1", "L2")


def test_gin_is_deterministic(fit) -> None:
    data, out = fit
    assert andrey.gin(data, alpha=0.05) == out


def test_gin_output_contract(fit) -> None:
    _, out = fit
    assert out.metadata["algorithm"] == "GIN"
    assert out.metadata["n_latents"] >= 1
