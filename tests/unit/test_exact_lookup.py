"""Exact search's subset lookups: the parent-graph scan a fit over the byte budget uses gives the
same parent graphs, scores, and DAGs as the lookup tables."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import Knowledge
from andrey.core.knowledge import resolve
from andrey.core.score import make_score
from andrey.search import exact


def _chain(seed: int, n: int, d: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    for j in range(1, d):
        x[:, j] += 0.8 * x[:, j - 1]
    return x


def _relabeled() -> np.ndarray:
    """Binary columns where column 1 relabels column 0, so both split the rows alike and every
    parent set holding one scores the same as the set holding the other under BDeu."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 300)
    c = np.where(rng.random(300) < 0.85, a, 1 - a)
    return np.column_stack([a, 1 - a, c, np.where(rng.random(300) < 0.8, c, 1 - c)])


BDEU = "bdeu"
# (data, knowledge, score): a chain; the chain with required and forbidden edges; and tied scores.
CASES = {
    "plain": (_chain(0, 400), None, "bic"),
    "knowledge": (
        _chain(1, 400),
        Knowledge(required=[(0, 2)], forbidden=[(3, 4), (5, 1)]),
        "bic",
    ),
    "ties": (_relabeled(), None, BDEU),
}


def _graphs(x: np.ndarray, knowledge: Knowledge | None, score_func: str, tables: bool) -> list:
    score = exact._GramBICScore(x) if score_func == "bic" else make_score(score_func, x)
    d = x.shape[1]
    constraints = resolve(knowledge, d, None, method="exact_search")
    return [exact._generate_parent_graph(score, i, d, constraints, tables) for i in range(d)]


def _bits(parent_graphs: list) -> list:
    """Each entry's parent set, bitmask, and the bytes of its score."""
    return [[(p, b, np.float64(s).tobytes()) for p, s, b in pg] for pg in parent_graphs]


@pytest.mark.parametrize("case", CASES)
def test_scan_and_tables_build_the_same_parent_graphs(case: str) -> None:
    x, knowledge, score_func = CASES[case]
    tabled = _graphs(x, knowledge, score_func, tables=True)
    assert _bits(_graphs(x, knowledge, score_func, tables=False)) == _bits(tabled)
    d = x.shape[1]
    for search in (exact._astar_shortest_path, exact._dp_shortest_path):
        assert search(tabled, d, tables=False) == search(tabled, d, tables=True)


def test_ties_case_has_tied_scores() -> None:
    """The tie case's parent graphs hold equal scores, so the first of equal entries decides."""
    tied = [len({s for _, s, _ in pg}) < len(pg) for pg in _graphs(*CASES["ties"], tables=True)]
    assert any(tied)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_a_fit_over_the_byte_budget_returns_the_same_dag(
    case: str, search_method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, knowledge, score_func = CASES[case]
    kwargs = {"search_method": search_method, "knowledge": knowledge, "score_func": score_func}
    fast = andrey.exact_search(x, **kwargs)
    monkeypatch.setattr(exact, "_LOOKUP_BYTES", 0)
    with monkeypatch.context() as patch:
        calls = []
        scan = exact._best_subset
        patch.setattr(exact, "_best_subset", lambda *a: calls.append(1) or scan(*a))
        slow = andrey.exact_search(x, **kwargs)
    assert calls  # the fit scanned
    np.testing.assert_array_equal(slow.structure.to_numpy(), fast.structure.to_numpy())
    assert slow.metadata == fast.metadata
