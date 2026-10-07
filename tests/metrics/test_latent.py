"""Latent-cluster metrics: adjusted Rand index (checked vs sklearn) and GIN clustering."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import LATENT, OBSERVED, GraphStructure
from andrey.metrics import adjusted_rand_index, latent_cluster_ari, number_of_latents

# ---- adjusted Rand index ------------------------------------------------------------------------


def test_ari_identical_partitions():
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0  # label-invariant


def test_ari_single_cluster_is_perfect():
    assert adjusted_rand_index([0, 0, 0], [0, 0, 0]) == 1.0


def test_ari_single_element_is_perfect_not_nan():
    # n < 2 has no pairs; the partition is trivially perfect (must not divide by zero -> nan).
    assert adjusted_rand_index([0], [0]) == 1.0
    assert adjusted_rand_index([], []) == 1.0


@pytest.mark.parametrize(
    "a, b",
    [
        ([0, 0, 1, 1, 2], [0, 0, 1, 2, 2]),
        ([0, 1, 2, 3], [0, 0, 1, 1]),
        ([1, 1, 1, 2, 2, 3], [1, 2, 1, 2, 3, 3]),
    ],
)
def test_ari_matches_sklearn(a, b):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    assert adjusted_rand_index(a, b) == pytest.approx(sklearn_metrics.adjusted_rand_score(a, b))


# ---- GIN latent clustering ----------------------------------------------------------------------


def _latent_graph(marks: np.ndarray, types: np.ndarray) -> GraphStructure:
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)


def _cluster_graph(children_of_latent: dict[int, list[int]], n: int) -> GraphStructure:
    """A graph where each listed latent points to its observed children."""
    marks = np.zeros((n, n), dtype=np.int8)
    types = np.full(n, OBSERVED, dtype=np.int8)
    for latent, children in children_of_latent.items():
        types[latent] = LATENT
        for child in children:
            marks[latent, child] = 1  # TAIL at the latent
            marks[child, latent] = 2  # ARROW at the observed child
    return _latent_graph(marks, types)


def test_latent_cluster_ari_perfect_recovery():
    # latents 3 and 4 cluster {0,1} and {2}; the estimate relabels the latents, same groups.
    truth = _cluster_graph({3: [0, 1], 4: [2]}, n=5)
    estimate = _cluster_graph({4: [0, 1], 3: [2]}, n=5)
    assert latent_cluster_ari(estimate, truth) == 1.0


def test_latent_cluster_ari_wrong_grouping():
    truth = _cluster_graph({3: [0, 1], 4: [2]}, n=5)
    scrambled = _cluster_graph({3: [0], 4: [1, 2]}, n=5)
    assert latent_cluster_ari(scrambled, truth) < 1.0


def test_number_of_latents():
    truth = _cluster_graph({3: [0, 1], 4: [2]}, n=5)
    assert number_of_latents(truth) == 2
    plain = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag")
    assert number_of_latents(plain) == 0


def test_single_observed_variable_scores_one_not_nan():
    # a 2-node latent graph (1 observed, 1 latent): ARI over one node must be 1.0, not nan.
    one_observed = _cluster_graph({1: [0]}, n=2)
    assert latent_cluster_ari(one_observed, one_observed) == 1.0


def test_different_latent_counts_are_comparable():
    # same observed variables {0, 1}, different latent counts -> allowed, scores over the observed.
    two_latents = _cluster_graph({2: [0], 3: [1]}, n=4)  # observed {0, 1}, latents {2, 3}
    one_latent = _cluster_graph({2: [0, 1]}, n=3)  # observed {0, 1}, latent {2}
    assert latent_cluster_ari(two_latents, one_latent) < 1.0  # different groupings, but comparable


def test_different_observed_sets_raise():
    est = _cluster_graph({2: [0, 1]}, n=3)  # observed {0, 1}
    truth = _cluster_graph({3: [0, 1, 2]}, n=4)  # observed {0, 1, 2}
    with pytest.raises(ValueError, match="observed variables"):
        latent_cluster_ari(est, truth)


def test_observed_label_mismatch_raises():
    marks = np.array([[0, 0, 2], [0, 0, 2], [1, 1, 0]], dtype=np.int8)  # latent 2 -> {0, 1}
    types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    est = GraphStructure.from_numpy(marks, kind="dag", node_types=types, labels=("x", "y", "L"))
    truth = GraphStructure.from_numpy(marks, kind="dag", node_types=types, labels=("y", "x", "L"))
    with pytest.raises(ValueError, match="label"):
        latent_cluster_ari(est, truth)
