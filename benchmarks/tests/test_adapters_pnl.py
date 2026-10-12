"""The PNL adapters, the pairs they run on, and the driver's agreement column.

causal-learn's PNL returns two p-values; its adapter must decide the edge as Andrey decides its own,
so that the packages differ only in what they compute. Every pair the driver generates must hold
exactly one edge, or the direction score has nothing to score.
"""

from __future__ import annotations

import importlib
import json
import pickle

import numpy as np
import pandas as pd
import pnl_pairs
import pytest

from andrey import GraphStructure
from andrey.api._irregular import _adapt_pnl
from andrey_bench import datasets
from andrey_bench.adapters.andrey_pnl import AndreyPNL
from andrey_bench.adapters.causal_learn_pnl import CausalLearnPNL

ADAPTERS = [AndreyPNL(), CausalLearnPNL()]


def _pair(n: int = 300, seed: int = 2) -> np.ndarray:
    """A post-nonlinear pair ``x -> y``: ``y = tanh(x**3 + e)``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return np.column_stack([x, np.tanh(x**3 + 0.3 * rng.uniform(-1, 1, n))])


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
    [
        (np.array([0.4]), np.array([0.01]), None),
        (np.array([0.01]), np.array([0.4]), None),
        (np.array([0.3]), np.array([0.3]), None),
        (np.array([0.6]), np.array([0.5]), 0.05),
    ],
)
def test_causal_learn_decides_the_edge_as_andrey_does(
    monkeypatch, forward, backward, alpha
) -> None:
    module = importlib.import_module("causallearn.search.FCMBased.PNL.PNL")

    calls: list = []

    class Fake:
        def cause_or_effect(self, x, y):
            calls.append((x.shape, y.shape))
            return forward, backward

    monkeypatch.setattr(module, "PNL", Fake)
    adapter = CausalLearnPNL()
    marks = adapter.fit(np.zeros((10, 2)), {"alpha": alpha})
    expected = _adapt_pnl(forward, backward, alpha=alpha).structure.to_numpy()
    np.testing.assert_array_equal(marks, expected)
    assert calls == [((10, 1), (10, 1))]  # one column each, as causal-learn's PNL takes them


def test_andrey_finds_the_planted_direction() -> None:
    adapter = AndreyPNL()
    adapter.setup()
    structure = adapter.to_structure(np.asarray(adapter.fit(_pair(), adapter.params())))
    assert isinstance(structure, GraphStructure)
    assert (structure.kind, structure.n_nodes) == ("dag", 2)
    assert structure.oriented_edges() == [(0, 1, "directed")]


@pytest.fixture
def torch_float32():
    """Run with torch's default dtype at float32, as in a fresh fit process.

    causal-learn's CALM module sets the default to float64 on import, and PNL's networks then
    reject its float32 data. A benchmark fit runs in its own process, which never imports CALM.
    """
    torch = pytest.importorskip("torch")
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(before)


@pytest.mark.usefixtures("torch_float32")
def test_causal_learn_returns_a_two_node_dag() -> None:
    # Its decision rests on a t-test of means, so the test asserts the output's shape, not the
    # direction. 150 rows trains one batch per epoch.
    adapter = CausalLearnPNL()
    adapter.setup()
    structure = adapter.to_structure(np.asarray(adapter.fit(_pair(150), adapter.params())))
    assert isinstance(structure, GraphStructure)
    assert (structure.kind, structure.n_nodes) == ("dag", 2)
    assert len(structure.oriented_edges()) <= 1


def test_every_generated_pair_has_one_edge() -> None:
    directions = set()
    for key in pnl_pairs.pair_keys([50], seeds=10):
        _, marks, kind = datasets.generate_one(key)
        [(source, target, _)] = GraphStructure.from_numpy(marks, kind=kind).oriented_edges()
        directions.add((source, target))
    assert directions == {(0, 1), (1, 0)}  # the seed draws the direction


def test_agreement_leaves_the_baseline_out(capsys) -> None:
    # Both PNL solutions agree on one pair and disagree on the other; the baseline agrees with
    # neither. The agreement is one half whatever the baseline returns.
    rows = []
    for dataset_id, hashes in (("a", ("x", "x", "e")), ("b", ("x", "y", "e"))):
        for name, structure_hash in zip(
            ("andrey.pnl.numpy", "causal-learn.pnl", pnl_pairs.BASELINE.name), hashes, strict=True
        ):
            rows.append(
                {
                    "n": 200,
                    "name": name,
                    "status": "ok",
                    "dataset_id": dataset_id,
                    "structure_hash": structure_hash,
                    "wall_s": 1.0,
                    "direction_correct": 1.0,
                    "direction_reversed": 0.0,
                    "direction_none": 0.0,
                }
            )
    pnl_pairs._summarize(pd.DataFrame(rows))
    agree = {
        line.split()[1]: float(line.split()[-1])
        for line in capsys.readouterr().out.splitlines()[2:]
    }
    assert set(agree.values()) == {0.5}
