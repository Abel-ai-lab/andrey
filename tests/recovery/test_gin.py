"""Test GIN on fixed cases, on a planted measurement model, and against its recorded output."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey.core import LATENT, OBSERVED, GraphStructure, StructureOutput
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

_CASE = "measurement_6v"


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture(scope="module")
def fit() -> tuple[np.ndarray, StructureOutput]:
    data = datagen.generate(_CASE)
    return data, andrey.gin(data)


def test_gin_recovers_the_planted_clusters_and_their_order(fit) -> None:
    _, out = fit
    planted = datagen.clusters(_CASE)  # earliest latent first

    assert out.metadata["algorithm"] == "gin"
    assert out.metadata["n_latents"] == 2
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert [sorted(c) for c in out.metadata["causal_order"]] == planted

    s = out.structure
    assert isinstance(s, GraphStructure)
    assert s.kind == "dag"
    assert s.n_nodes == 8  # six observed indicators + two discovered latents
    assert list(s.node_types) == [OBSERVED] * 6 + [LATENT] * 2
    assert s.labels == ("X1", "X2", "X3", "X4", "X5", "X6", "L1", "L2")
    latent_edges = [(6, child, "directed") for child in planted[0]]
    latent_edges += [(6, 7, "directed")]
    latent_edges += [(7, child, "directed") for child in planted[1]]
    assert sorted(s.oriented_edges(index=True)) == sorted(latent_edges)


def test_gin_orders_a_chain_of_three_latents() -> None:
    case = "measurement_chain_9v"
    planted = datagen.clusters(case)  # earliest latent first
    out = andrey.gin(datagen.generate(case))
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert [sorted(c) for c in out.metadata["causal_order"]] == planted


def test_gin_finds_the_same_latents_for_reversed_columns() -> None:
    """Reversing the columns relabels the clusters and the latent order and changes nothing else."""
    data = datagen.generate("measurement_chain_9v")
    last = data.shape[1] - 1
    out = andrey.gin(data).metadata
    reversed_out = andrey.gin(data[:, ::-1]).metadata

    def back(groups: list[list[int]]) -> list[list[int]]:
        return [sorted(last - i for i in group) for group in groups]

    assert sorted(back(reversed_out["clusters"])) == sorted(out["clusters"])
    assert back(reversed_out["causal_order"]) == out["causal_order"]
    assert sorted(back(reversed_out["unordered"])) == sorted(out["unordered"])


def test_gin_matches_its_recorded_output(fit) -> None:
    _, out = fit
    assert_recorded_output(out, "GIN", case=_CASE)


def test_gin_is_deterministic(fit) -> None:
    data, out = fit
    assert andrey.gin(data) == out


def test_gin_finds_one_latent_behind_three_indicators() -> None:
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, 500)
    data = np.column_stack(
        [w * latent + 0.1 * rng.uniform(-1.0, 1.0, 500) for w in (0.8, 1.2, -0.9)]
    )
    out = andrey.gin(data)
    assert out.metadata["clusters"] == [[0, 1, 2]]
    assert sorted(out.structure.oriented_edges(index=True)) == [
        (3, i, "directed") for i in range(3)
    ]


def test_gin_needs_a_variable_outside_a_cluster() -> None:
    """Two columns leave no variable to test a pair against, so no latent is found."""
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, 500)
    data = np.column_stack([latent + 0.1 * rng.uniform(-1.0, 1.0, 500) for _ in range(2)])
    out = andrey.gin(data)
    assert out.metadata["n_latents"] == 0
    assert out.structure.n_nodes == 2
    assert out.structure.oriented_edges(index=True) == []


def test_gin_leaves_latents_unordered_when_no_root_passes() -> None:
    """Two latents share a hidden cause, so neither is a root and the order stays empty."""
    case = "measurement_hidden_cause_6v"
    planted = datagen.clusters(case)
    out = andrey.gin(datagen.generate(case))
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert out.metadata["causal_order"] == []
    unordered = sorted(planted, key=min)  # by smallest member, as GIN lists them
    assert out.metadata["unordered"] == unordered
    latent_edges = [(6, child, "directed") for child in unordered[0]]
    latent_edges += [(7, child, "directed") for child in unordered[1]]
    assert sorted(out.structure.oriented_edges(index=True)) == sorted(latent_edges)


def test_gin_root_is_the_passing_candidate_with_the_highest_p_value(monkeypatch) -> None:
    """A root passes the test against every other cluster; ties go to the highest lowest p-value.

    The stub gives each root test a p-value by the candidate and the other cluster. Cluster
    ``[4, 5]`` passes with lowest p-value 0.3 and ``[2, 3]`` with 0.6, so ``[2, 3]`` is the root;
    then ``[0, 1]`` and ``[4, 5]`` both fail, so the ordering stops.
    """
    engine = importlib.import_module("andrey.latent.gin")

    first_step = {(0, 2): 0.9, (0, 4): 0.01, (2, 0): 0.6, (2, 4): 0.8, (4, 0): 0.3, (4, 2): 0.7}
    later_step = {(0, 4): 0.04, (4, 0): 0.001}

    def pooled(data, cov, grams, candidate, other, ordered):
        table = later_step if ordered else first_step
        return table[(candidate[0], other[0])]

    monkeypatch.setattr(engine, "_root_pvalue", pooled)
    clusters = [[0, 1], [2, 3], [4, 5]]
    ordered, unordered = engine._order_clusters(np.zeros((1, 6)), np.eye(6), [], clusters, 0.05)
    assert ordered == [[2, 3]]
    assert unordered == [[0, 1], [4, 5]]
