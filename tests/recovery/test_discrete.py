"""Recovery on categorical data: the discrete tests and scores in each search that takes them.

Fixed cases only. Exactly independent columns give no edges through every facade that takes a
discrete setting, since each forwards it on its own. A noisy OR collider is oriented once per code
path that scores discrete data: the constraint skeletons with a test that has no batched or
prescreen surface, GES's full-score arm, the hill-climbing move tables (Tabu search shares them),
the permutation grow-shrink trees (GRaSP shares the engine), and exact search's parent graphs.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey

_FACADES = [
    ("pc", {"indep_test": "chisq"}),
    ("fci", {"indep_test": "gsq"}),
    ("gfci", {"indep_test": "chisq", "score_func": "local_score_BDeu"}),
    ("ges", {"score_func": "local_score_BDeu"}),
    ("hc", {"score_func": "local_score_BIC_discrete"}),
    ("tabu", {"score_func": "local_score_BDeu"}),
    ("boss", {"score_func": "local_score_BDeu"}),
    ("grasp", {"score_func": "local_score_BIC_discrete"}),
    ("exact_search", {"score_func": "local_score_BDeu"}),
]
_PATHS = [
    ("pc", {"indep_test": "chisq"}),
    ("fci", {"indep_test": "gsq"}),
    ("ges", {"score_func": "local_score_BDeu"}),
    ("hc", {"score_func": "local_score_BIC_discrete"}),
    ("boss", {"score_func": "local_score_BDeu"}),
    ("exact_search", {"score_func": "local_score_BIC_discrete"}),
]


@pytest.fixture(scope="module")
def collider() -> np.ndarray:
    """``a -> b <- c`` with binary ``a`` and ``c`` independent and ``b = a OR c``, flipped 10%."""
    rng = np.random.default_rng(0)
    n = 1000
    a = rng.integers(0, 2, n)
    c = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < 0.9, a | c, 1 - (a | c))
    return np.column_stack([a, b, c]).astype(np.float64)


@pytest.mark.parametrize(("method", "settings"), _FACADES)
def test_independent_columns_give_no_edges(method, settings):
    # A full factorial design: every combination of three 3-level columns, ten times each, so every
    # table is exactly balanced and no test or score can find a dependence.
    levels = np.indices((3, 3, 3)).reshape(3, -1).T
    x = np.tile(levels, (10, 1)).astype(np.float64)
    assert getattr(andrey, method)(x, **settings).structure.oriented_edges() == []


@pytest.mark.parametrize(("method", "settings"), _PATHS)
def test_a_collider_is_oriented(method, settings, collider):
    edges = getattr(andrey, method)(collider, **settings).structure.oriented_edges()
    into_b = "partially_directed" if method == "fci" else "directed"
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
