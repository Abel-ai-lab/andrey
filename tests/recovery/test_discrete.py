"""Recovery on categorical data: every method that takes a discrete test or score.

Fixed cases: independent columns give no edges, and a noisy OR collider is oriented. Ground truth:
rows sampled from the ASIA network's own probability tables, scored against its CPDAG.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import andrey
from andrey.core.orient import dag2cpdag, from_structure, to_structure
from andrey.data.discrete import read_bif_network
from andrey.metrics import shd

_ASIA = Path(__file__).parents[1] / "unit" / "fixtures" / "asia.bif.gz"

_METHODS = [
    ("pc", {"indep_test": "chisq"}),
    ("pc", {"indep_test": "gsq"}),
    ("fci", {"indep_test": "gsq"}),
    ("gfci", {"indep_test": "chisq", "score_func": "local_score_BDeu"}),
    ("ges", {"score_func": "local_score_BDeu"}),
    ("ges", {"score_func": "local_score_BIC_discrete"}),
    ("hc", {"score_func": "local_score_BIC_discrete"}),
    ("tabu", {"score_func": "local_score_BDeu"}),
    ("boss", {"score_func": "local_score_BDeu"}),
    ("grasp", {"score_func": "local_score_BIC_discrete"}),
    ("exact_search", {"score_func": "local_score_BDeu"}),
]
_SCORE_METHODS = [(m, kw) for m, kw in _METHODS if "indep_test" not in kw]


def _fit(method: str, x: np.ndarray, settings: dict) -> andrey.StructureOutput:
    return getattr(andrey, method)(x, **settings)


def _collider() -> np.ndarray:
    """``a -> b <- c`` with binary ``a`` and ``c`` independent and ``b = a OR c``, flipped 10%."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 2, n)
    c = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < 0.9, a | c, 1 - (a | c))
    return np.column_stack([a, b, c]).astype(np.float64)


@pytest.mark.parametrize(("method", "settings"), _METHODS)
def test_independent_columns_give_no_edges(method, settings):
    x = np.random.default_rng(1).integers(0, 3, (500, 4)).astype(np.float64)
    assert _fit(method, x, settings).structure.oriented_edges() == []


@pytest.mark.parametrize(("method", "settings"), _METHODS)
def test_a_collider_is_oriented(method, settings):
    edges = _fit(method, _collider(), settings).structure.oriented_edges()
    into_b = "partially_directed" if method in ("fci", "gfci") else "directed"
    assert edges == [(0, 1, into_b), (2, 1, into_b)]


def test_cdnod_orients_from_a_categorical_context():
    # x's distribution moves with the three-level context; y copies x 85% of the time.
    rng = np.random.default_rng(0)
    context = np.repeat([0, 1, 2], 400)
    x = np.where(rng.random(1200) < 0.2 + 0.3 * context, 1, 0)
    y = np.where(rng.random(1200) < 0.85, x, 1 - x)
    out = andrey.cdnod(
        np.column_stack([x, y]).astype(np.float64), context.astype(np.float64), indep_test="chisq"
    )
    assert out.structure.oriented_edges() == [(0, 1, "directed")]


@pytest.fixture(scope="module")
def asia() -> tuple[np.ndarray, andrey.GraphStructure]:
    """5,000 rows from ASIA's probability tables, and ASIA's CPDAG."""
    network = read_bif_network(gzip.decompress(_ASIA.read_bytes()).decode("utf-8"))
    d = len(network.nodes)
    marks = np.zeros((d, d), dtype=np.int8)
    for child, parents in enumerate(network.parents):
        for p in parents:
            marks[p, child], marks[child, p] = 1, 2
    rows = network.sample(5000, np.random.default_rng(1)).astype(np.float64)
    return rows, to_structure(dag2cpdag(marks), kind="cpdag")


@pytest.mark.parametrize(("method", "settings"), _SCORE_METHODS)
def test_score_searches_recover_asia(method, settings, asia):
    rows, cpdag = asia
    found = _fit(method, rows, settings).structure
    if found.kind == "dag":
        found = to_structure(dag2cpdag(from_structure(found)), kind="cpdag")
    assert shd(found, cpdag) == 0


@pytest.mark.parametrize("indep_test", ["chisq", "gsq"])
def test_pc_beats_the_empty_graph_on_asia(indep_test, asia):
    # ASIA's either = lung OR tub exactly, a deterministic relation that hides edges from tests.
    rows, cpdag = asia
    found = andrey.pc(rows, indep_test=indep_test).structure
    assert shd(found, cpdag) < len(cpdag.oriented_edges())


def test_dataframe_labels_and_categories_carry_through():
    import pandas as pd

    x = _collider()
    frame = pd.DataFrame(
        {
            "a": pd.Categorical(np.where(x[:, 0] == 1, "on", "off")),
            "b": x[:, 1].astype(bool),
            "c": np.where(x[:, 2] == 1, "on", "off"),
        }
    )
    edges = andrey.ges(frame, score_func="local_score_BDeu").structure.oriented_edges()
    assert edges == [("a", "b", "directed"), ("c", "b", "directed")]
