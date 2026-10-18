"""The ANM adapters, the pairs they run on, and the direction score.

causal-learn's ANM returns two p-values; its adapter must decide the edge as Andrey decides its own,
so that the packages differ only in what they compute. Every pair the driver generates must hold
exactly one edge, or the direction score has nothing to score.
"""

from __future__ import annotations

import importlib
import json
import pickle

import anm_pairs
import numpy as np
import pytest

from andrey import GraphStructure
from andrey.api._irregular import _adapt_anm
from andrey_bench import datasets
from andrey_bench.adapters.andrey_anm import AndreyANM
from andrey_bench.adapters.causal_learn_anm import CausalLearnANM

ADAPTERS = [AndreyANM(), CausalLearnANM()]


def _pair(n: int = 150, seed: int = 0) -> np.ndarray:
    """A cubic additive-noise pair ``x -> y``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return np.column_stack([x, x * x * x + 0.3 * rng.uniform(-1, 1, n)])


def test_both_packages_report_one_method_the_same_way() -> None:
    andrey_adapter, other = ADAPTERS
    for field in ("algorithm", "output_type", "mode"):
        assert getattr(andrey_adapter, field) == getattr(other, field), field
    assert andrey_adapter.params() == other.params()


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_an_adapter_survives_the_record_and_the_worker(adapter) -> None:
    assert json.loads(json.dumps(adapter.params(), sort_keys=True)) == adapter.params()
    assert pickle.loads(pickle.dumps(adapter)).name == adapter.name


@pytest.mark.parametrize(
    ("forward", "backward", "alpha"),
    [(0.4, 0.01, None), (0.01, 0.4, None), (0.3, 0.3, None), (0.6, 0.5, 0.05)],
)
def test_causal_learn_decides_the_edge_as_andrey_does(
    monkeypatch, forward, backward, alpha
) -> None:
    module = importlib.import_module("causallearn.search.FCMBased.ANM.ANM")

    calls: list = []

    class Fake:
        def cause_or_effect(self, x, y):
            calls.append((x.shape, y.shape))
            return forward, backward

    monkeypatch.setattr(module, "ANM", Fake)
    adapter = CausalLearnANM()
    marks = adapter.fit(np.zeros((10, 2)), {"alpha": alpha})
    expected = _adapt_anm(forward, backward, alpha=alpha).structure.to_numpy()
    np.testing.assert_array_equal(marks, expected)
    assert calls == [((10, 1), (10, 1))]  # one column each, as causal-learn's ANM takes them


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_a_fit_converts_to_the_planted_two_node_dag(adapter) -> None:
    adapter.setup()
    native = np.asarray(adapter.fit(_pair(), adapter.params()))
    structure = adapter.to_structure(native)
    assert isinstance(structure, GraphStructure)
    assert (structure.kind, structure.n_nodes) == ("dag", 2)
    assert structure.oriented_edges() == [(0, 1, "directed")]


def test_every_generated_pair_has_one_edge() -> None:
    directions = set()
    for key in anm_pairs.pair_keys([50], seeds=10):
        _, marks, kind, _ = datasets.generate_one(key)
        [(source, target, _)] = GraphStructure.from_numpy(marks, kind=kind).oriented_edges()
        directions.add((source, target))
    assert directions == {(0, 1), (1, 0)}  # the seed draws the direction


def _dag(*edges: tuple[int, int]) -> GraphStructure:
    marks = np.zeros((2, 2), dtype=np.int8)
    for source, target in edges:
        marks[source, target], marks[target, source] = 1, 2  # TAIL at the source, ARROW at target
    return GraphStructure.from_numpy(marks, kind="dag")


@pytest.mark.parametrize(
    ("estimate", "column"),
    [
        (_dag((0, 1)), "direction_correct"),
        (_dag((1, 0)), "direction_reversed"),
        (_dag(), "direction_none"),
    ],
)
def test_a_pair_is_scored_by_the_direction_it_names(estimate, column) -> None:
    scores = anm_pairs.direction_scores(estimate, _dag((0, 1)))
    assert scores == {c: float(c == column) for c in anm_pairs.SCORE_COLUMNS}
